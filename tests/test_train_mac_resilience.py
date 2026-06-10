import os
import tempfile
import types
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import train_mac


class DummyDataset(Dataset):
    def __init__(self, fail_idx=None):
        self.fail_idx = fail_idx

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        if self.fail_idx is not None and idx == self.fail_idx:
            raise RuntimeError(f"bad sample {idx}")
        return {
            "video": torch.ones(1, 3, 16, 2, 2, dtype=torch.float32),
            "action": torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
        }


class ConstantLogitsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(10))

    def forward(self, x):
        return self.logits.expand(x.size(0), -1)


def make_args(tmpdir, save_best_only=False, history_path=None):
    return types.SimpleNamespace(
        base_model_name="r2plus1d_multiclass",
        lr=1e-4,
        start_epoch=1,
        model_path=tmpdir,
        history_path=history_path or os.path.join(tmpdir, "history.txt"),
        save_best_only=save_best_only,
        epochs=1,
        device="cpu",
    )


class TrainMacResilienceTest(unittest.TestCase):
    def test_safe_dataloader_catches_dataset_exception(self):
        loader = DataLoader(
            train_mac.SafeDataset(DummyDataset(fail_idx=2), "train"),
            batch_size=2,
            shuffle=False,
            collate_fn=train_mac.make_safe_collate_fn(),
        )

        batches = list(loader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]["video"]), 2)
        self.assertFalse(batches[1].get("_skip_batch"))
        self.assertEqual(int(batches[1]["_skip_count"]), 1)
        self.assertEqual(batches[1]["video"].shape[0], 1)

    def test_augmented_annotation_missing_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = os.path.join(tmpdir, "annotation_dict.json")
            with open(annotation_path, "w") as fp:
                fp.write('{"sample_1": 0}')

            from dataset import BasketballDataset

            ds = BasketballDataset(annotation_dict=annotation_path, augmented_dict=os.path.join(tmpdir, "missing.json"))
            self.assertEqual(len(ds), 1)

    def test_save_best_only_keeps_latest_off_and_saves_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = make_args(tmpdir, save_best_only=True, history_path=os.path.join(tmpdir, "history.txt"))
            criterion = torch.nn.CrossEntropyLoss()
            model = ConstantLogitsModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

            train_loader = DataLoader(
                train_mac.SafeDataset(DummyDataset()),
                batch_size=1,
                shuffle=False,
                collate_fn=train_mac.make_safe_collate_fn(),
            )
            val_loader = DataLoader(
                train_mac.SafeDataset(DummyDataset()),
                batch_size=1,
                shuffle=False,
                collate_fn=train_mac.make_safe_collate_fn(),
            )
            model, _, _, _, _, _, _, _ = train_mac.train_model(
                model,
                {"train": train_loader, "val": val_loader},
                criterion,
                optimizer,
                torch.device("cpu"),
                args,
                start_epoch=1,
                num_epochs=1,
                initial_best_acc=-1.0,
                initial_best_epoch=0,
                initial_best_weights=None,
            )

            best_ckpt = os.path.join(tmpdir, "best.pt")
            latest_ckpt = os.path.join(tmpdir, "latest.pt")
            self.assertTrue(os.path.exists(best_ckpt))
            self.assertFalse(os.path.exists(latest_ckpt))
            self.assertEqual(torch.load(best_ckpt)["best_epoch"], 1)

    def test_no_epoch_run_does_not_touch_best_ckpt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            best_ckpt = os.path.join(tmpdir, "best.pt")
            torch.save({"epoch": 4, "best_val_acc": 0.99, "best_epoch": 4}, best_ckpt)
            mtime_before = os.path.getmtime(best_ckpt)

            args = make_args(tmpdir, save_best_only=True, history_path=os.path.join(tmpdir, "history.txt"))
            train_loader = DataLoader([], batch_size=1, shuffle=False, collate_fn=train_mac.make_safe_collate_fn())
            val_loader = DataLoader([], batch_size=1, shuffle=False, collate_fn=train_mac.make_safe_collate_fn())

            model = ConstantLogitsModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            criterion = torch.nn.CrossEntropyLoss()

            train_mac.train_model(
                model,
                {"train": train_loader, "val": val_loader},
                criterion,
                optimizer,
                torch.device("cpu"),
                args,
                start_epoch=5,
                num_epochs=1,
                initial_best_acc=0.99,
                initial_best_epoch=4,
                initial_best_weights=None,
            )

            self.assertAlmostEqual(os.path.getmtime(best_ckpt), mtime_before)
