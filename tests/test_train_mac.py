import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import train_mac


class _TwoClassDataset(Dataset):
    def __init__(self, bad_first=False):
        self.bad_first = bad_first

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        if idx == 0 and self.bad_first:
            return {
                "video": "broken_sample",
                "action": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            }

        return {
            "video": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "action": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }


class _ThrowingDataset(Dataset):
    def __init__(self, bad_first=False):
        self.bad_first = bad_first

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        if idx == 0 and self.bad_first:
            raise RuntimeError("bad sample")
        return {
            "video": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "action": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }


class _CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 10)
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return self.linear(x[:, :3])


class _AdaptiveEpochModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(10))
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        logits = self.bias.view(1, 10).repeat(x.shape[0], 1).clone()
        # Calls are ordered: train1,train2,val3,val4,train5,train6,val7,val8,...
        # Force wrong-class predictions on epoch1 and epoch3 val phases.
        if self.forward_calls in (3, 4, 11, 12):
            logits[:, 1] = 5.0
        else:
            # correct class for train steps and val epoch 2
            logits[:, 0] = 5.0
        return logits


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 10)

    def forward(self, x):
        # keep implementation simple and deterministic; dataset is mocked in most tests
        return self.fc(x[:, :1])


class _ForwardCounterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(10))
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return self.logit.view(1, 10).repeat(x.size(0), 1) + torch.tensor(
            [[5.0] + [0.0] * 9], dtype=x.dtype, device=x.device
        )


class TrainMacTests(TestCase):
    def test_train_loop_skips_bad_sample_and_continues(self):
        dataset = _ThrowingDataset(bad_first=True)
        dataloader = DataLoader(
            train_mac.SafeDataset(dataset, "train"),
            batch_size=1,
            shuffle=False,
            collate_fn=train_mac.make_safe_collate_fn(),
        )
        dataloaders = {"train": dataloader, "val": dataloader}

        model = _CountingModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        args = SimpleNamespace(
            base_model_name="r2plus1d_multiclass",
            lr=1e-4,
            model_path=tempfile.mkdtemp(),
            history_path=os.path.join(tempfile.gettempdir(), "train_mac_history_tmp.txt"),
            save_best_only=False,
            best_val_acc=0.0,
            best_epoch=0,
        )

        with patch("train_mac.init_session_history"), \
            patch("train_mac.save_weights", return_value="skip"), \
            patch("train_mac.write_history"), \
            patch("train_mac.validate_bn_stats", return_value=True), \
            patch("train_mac.os.makedirs"), \
            patch("train_mac.torch.save"):

            train_mac.train_model(
                model,
                dataloaders,
                criterion,
                optimizer,
                torch.device("cpu"),
                args,
                start_epoch=1,
                num_epochs=1,
            )

        # One valid sample per phase should run, so forward is called exactly twice.
        self.assertEqual(model.forward_calls, 2)

    def test_check_accuracy_skips_broken_samples(self):
        dataset = _ThrowingDataset(bad_first=True)
        dataloader = DataLoader(
            train_mac.SafeDataset(dataset, "test"),
            batch_size=1,
            shuffle=False,
            collate_fn=train_mac.make_safe_collate_fn(),
        )
        model = _ForwardCounterModel()

        accuracy = train_mac.check_accuracy(dataloader, model, torch.device("cpu"))

        self.assertGreater(accuracy, 0.0)
        # One good sample contributes to accuracy; one sample is skipped.
        self.assertEqual(model.forward_calls, 1)

    def test_main_wraps_train_val_test_loaders_with_safe_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ann_path = os.path.join(tmpdir, "annotation_dict.json")
            with open(ann_path, "w", encoding="utf-8") as fp:
                json.dump({f"video_{i}": i % 10 for i in range(20)}, fp)

            class FakeDataset(Dataset):
                def __len__(self):
                    return 20

                def __getitem__(self, idx):
                    if idx < 0:
                        raise RuntimeError("negative")
                    return {
                        "video": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
                        "action": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    }

            class FakeModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(1, 10)

                def forward(self, x):
                    return self.fc(x[:, :1])

            class FakeVideoModule:
                class R2Plus1D_18_Weights:
                    DEFAULT = object()

                @staticmethod
                def r2plus1d_18(weights=None):
                    return FakeModel()

            args = SimpleNamespace(
                device="cpu",
                batch_size=2,
                epochs=1,
                lr=1e-4,
                start_epoch=1,
                layers=["layer3", "layer4", "fc"],
                resume=None,
                num_workers=0,
                annotation_path=ann_path,
                augmented_path="does-not-exist.json",
                video_dir="dataset/examples/",
                augmented_dir="dataset/augmented-examples/",
                model_dir=os.path.join(tmpdir, "model_checkpoints"),
                history_path=os.path.join(tmpdir, "history.txt"),
                save_best_only=False,
            )

            loader_datasets = []
            loader_kwargs = []

            def fake_dataloader(*dargs, **dkwargs):
                loader_datasets.append(dargs[0])
                loader_kwargs.append(dkwargs)
                return DataLoader(*dargs, **dkwargs)

            with patch("train_mac.parse_args", return_value=args), \
                patch("train_mac.auto_device", return_value=torch.device("cpu")), \
                patch("train_mac.BasketballDataset", return_value=FakeDataset()), \
                patch("train_mac.train_model", return_value=(FakeModel(), [], [], [], [], [], [], [])), \
                patch("train_mac.check_accuracy", return_value=0.0), \
                patch("train_mac.models.video", FakeVideoModule), \
                patch("train_mac.DataLoader", side_effect=fake_dataloader):

                train_mac.main()

            # main() should build three loaders and each must consume SafeDataset + safe collate.
            self.assertEqual(len(loader_datasets), 3)
            self.assertTrue(all(isinstance(ds, train_mac.SafeDataset) for ds in loader_datasets))
            self.assertTrue(all(kwargs.get("collate_fn", None).__name__ == "safe_collate_fn" for kwargs in loader_kwargs))

    def test_val_loader_skips_broken_samples_during_training_loop(self):
        bad_dataset = _ThrowingDataset(bad_first=True)
        good_dataset = _ThrowingDataset(bad_first=False)
        train_loader = DataLoader(
            train_mac.SafeDataset(good_dataset, "train"),
            batch_size=1,
            shuffle=False,
            collate_fn=train_mac.make_safe_collate_fn(),
        )
        val_loader = DataLoader(
            train_mac.SafeDataset(bad_dataset, "val"),
            batch_size=1,
            shuffle=False,
            collate_fn=train_mac.make_safe_collate_fn(),
        )
        dataloaders = {"train": train_loader, "val": val_loader}

        model = _CountingModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        args = SimpleNamespace(
            base_model_name="r2plus1d_multiclass",
            lr=1e-4,
            model_path=tempfile.mkdtemp(),
            history_path=os.path.join(tempfile.gettempdir(), "train_mac_history_tmp.txt"),
            save_best_only=False,
        )

        with patch("train_mac.init_session_history"), \
            patch("train_mac.save_weights", return_value="skip"), \
            patch("train_mac.write_history"), \
            patch("train_mac.validate_bn_stats", return_value=True), \
            patch("train_mac.os.makedirs"), \
            patch("train_mac.torch.save"):

            train_mac.train_model(
                model,
                dataloaders,
                criterion,
                optimizer,
                torch.device("cpu"),
                args,
                start_epoch=1,
                num_epochs=1,
            )

        # Train runs all train samples; val skips one bad sample and still runs one valid sample.
        self.assertEqual(model.forward_calls, 3)

    def test_main_builds_dataset_from_original_when_augmented_annotations_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ann_path = os.path.join(tmpdir, "annotation_dict.json")
            aug_path = os.path.join(tmpdir, "augmented_annotation_dict_missing.json")
            with open(ann_path, "w", encoding="utf-8") as fp:
                json.dump({f"video_{i}": i % 10 for i in range(100)}, fp)

            model_dir = os.path.join(tmpdir, "model_checkpoints")
            history_path = os.path.join(tmpdir, "history.txt")

            random_split_calls = []
            fake_dataset_args = {}

            class FakeDataset:
                def __init__(self, annotation_dict, augmented_dict, video_dir="dataset/examples/",
                             augmented_dir="dataset/augmented-examples/", augment=True, transform=None, poseData=False):
                    fake_dataset_args["annotation_dict"] = annotation_dict
                    fake_dataset_args["augmented_dict"] = augmented_dict
                    fake_dataset_args["augment_flag"] = augment
                    with open(annotation_dict, "r", encoding="utf-8") as fp:
                        base_items = list(json.load(fp).items())
                    extra_items = []
                    if os.path.exists(augmented_dict):
                        with open(augmented_dict, "r", encoding="utf-8") as fp:
                            extra_items = list(json.load(fp).items())
                    self.video_list = base_items + extra_items
                    self.video_dir = video_dir
                    self.augmented_dir = augmented_dir

                def __len__(self):
                    return len(self.video_list)

                def __getitem__(self, idx):
                    return {
                        "video": torch.ones((1,), dtype=torch.float32),
                        "action": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    }

            class FakeModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(1, 10)

                def forward(self, x):  # pragma: no cover - patched out of training pipeline
                    return self.fc(x[:, :1])

            class FakeVideoModule:
                class R2Plus1D_18_Weights:
                    DEFAULT = object()

                @staticmethod
                def r2plus1d_18(weights=None):
                    return FakeModel()

            def fake_random_split(dataset, lengths, generator=None):
                random_split_calls.append(list(lengths))
                return dataset, dataset

            args = SimpleNamespace(
                device="cpu",
                batch_size=2,
                epochs=1,
                lr=1e-4,
                start_epoch=1,
                layers=["layer3", "layer4", "fc"],
                resume=None,
                num_workers=0,
                annotation_path=ann_path,
                augmented_path=aug_path,
                video_dir="dataset/examples/",
                augmented_dir="dataset/augmented-examples/",
                model_dir=model_dir,
                history_path=history_path,
                save_best_only=False,
            )

            with patch("train_mac.parse_args", return_value=args), \
                patch("train_mac.auto_device", return_value=torch.device("cpu")), \
                patch("train_mac.BasketballDataset", FakeDataset), \
                patch("train_mac.random_split", fake_random_split), \
                patch("train_mac.train_model", return_value=(FakeModel(), [], [], [], [], [], [], [])), \
                patch("train_mac.check_accuracy"), \
                patch("train_mac.models.video", FakeVideoModule):

                train_mac.main()

            # 100 entries, no augmented annotations: test_n=10, val_n=20, train_n=70.
            self.assertEqual(fake_dataset_args["annotation_dict"], ann_path)
            self.assertEqual(fake_dataset_args["augmented_dict"], aug_path)
            self.assertFalse(os.path.exists(aug_path))
            self.assertEqual(random_split_calls[0], [90, 10])
            self.assertEqual(random_split_calls[1], [70, 20])

    def test_resume_noop_does_not_overwrite_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _TwoClassDataset()
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            dataloaders = {"train": dataloader, "val": dataloader}

            model = _SimpleModel()
            optimizer = optim.SGD(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            args = SimpleNamespace(
                base_model_name="r2plus1d_multiclass",
                model_path=tmpdir,
                history_path=os.path.join(tmpdir, "history.txt"),
                lr=1e-4,
                save_best_only=False,
                best_val_acc=0.74,
                best_epoch=2,
            )

            with patch("train_mac.init_session_history"), \
                patch("train_mac.save_weights", return_value="skip"), \
                patch("train_mac.write_history"), \
                patch("train_mac.validate_bn_stats", return_value=True), \
                patch("train_mac.os.makedirs"), \
                patch("train_mac.torch.save") as mock_torch_save:

                train_mac.train_model(
                    model=model,
                    dataloaders=dataloaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    args=args,
                    start_epoch=4,
                    num_epochs=3,
                )

            mock_torch_save.assert_not_called()

    def test_best_epoch_metadata_stays_on_best_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _TwoClassDataset()
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            dataloaders = {"train": dataloader, "val": dataloader}
            model = _AdaptiveEpochModel()
            optimizer = optim.SGD(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            args = SimpleNamespace(
                base_model_name="r2plus1d_multiclass",
                model_path=tmpdir,
                history_path=os.path.join(tmpdir, "history.txt"),
                lr=1e-4,
                save_best_only=False,
            )

            with patch("train_mac.init_session_history"), \
                patch("train_mac.save_weights", return_value="skip"), \
                patch("train_mac.write_history"), \
                patch("train_mac.validate_bn_stats", return_value=True), \
                patch("train_mac.os.makedirs"), \
                patch("train_mac.torch.save") as mock_torch_save:

                train_mac.train_model(
                    model=model,
                    dataloaders=dataloaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    args=args,
                    start_epoch=1,
                    num_epochs=3,
                    initial_best_acc=0.5,
                )

            best_calls = []
            for call in mock_torch_save.call_args_list:
                if len(call.args) >= 2 and str(call.args[1]).endswith("best.pt"):
                    best_calls.append(call.args[0])
            self.assertGreaterEqual(len(best_calls), 1)
            self.assertTrue(all(item.get("best_epoch") == 2 for item in best_calls))
            self.assertTrue(all(item.get("epoch") == 2 for item in best_calls))

    def test_save_best_only_skips_epoch_weights_and_latest_ckpt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _TwoClassDataset(bad_first=False)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            dataloaders = {"train": dataloader, "val": dataloader}
            model = _AdaptiveEpochModel()
            optimizer = optim.SGD(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            args = SimpleNamespace(
                base_model_name="r2plus1d_multiclass",
                model_path=tmpdir,
                history_path=os.path.join(tmpdir, "history.txt"),
                lr=1e-4,
                save_best_only=True,
            )

            with patch("train_mac.init_session_history"), \
                patch("train_mac.save_weights") as mock_save_weights, \
                patch("train_mac.write_history"), \
                patch("train_mac.validate_bn_stats", return_value=True), \
                patch("train_mac.os.makedirs"), \
                patch("train_mac.torch.save") as mock_torch_save:

                train_mac.train_model(
                    model=model,
                    dataloaders=dataloaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    args=args,
                    start_epoch=1,
                    num_epochs=1,
                )

            # save_best_only: no regular epoch checkpoints, no latest checkpoint.
            mock_save_weights.assert_not_called()
            torch_save_calls = [str(c.args[1]) for c in mock_torch_save.call_args_list if c.args]
            self.assertTrue(any(path.endswith("best.pt") for path in torch_save_calls))
            self.assertFalse(any(path.endswith("latest.pt") for path in torch_save_calls))

    def test_pytest_discovery_does_not_collect_root_experiment_scripts(self):
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        output = result.stdout
        self.assertNotIn("test_steps.py::test_tracking", output)
        self.assertNotIn("test_run.py::", output)
        self.assertNotIn("test_pretrained.py::", output)
