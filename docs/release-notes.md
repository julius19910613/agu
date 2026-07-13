# Public Release Notes

This document records the source attribution, licensing boundaries, dataset policy, and model weight distribution policy required before publishing AGU as an open-source project.

## Source Statement

AGU is an open-source basketball video action understanding engine maintained as an independent Python/FastAPI project.

The repository contains:

- AGU application code under `app/`.
- Training and data utilities such as `train_mac.py`, `dataset.py`, and `scripts/`.
- Documentation under `docs/`.
- Public examples under `examples/`.
- Docker and dependency files for self-hosted deployment.

The repository does not contain:

- Private basketball videos.
- Private production outputs.
- Business database schemas or credentials.
- `visual_coach`, `basketball`, or `player_grouping` business logic.
- Model checkpoints or training datasets.

## License Statement

AGU source code is released under the MIT License. See `LICENSE`.

The MIT License applies to AGU source code and documentation that are authored for this repository. It does not automatically apply to third-party datasets, pretrained model weights, external model files, or dependency packages.

Users are responsible for complying with the licenses of:

- PyTorch and TorchVision.
- OpenCV.
- Ultralytics YOLO.
- SpaceJam or any other dataset used for training.
- Any external model checkpoints or VLM services.

The maintained dependency/model/data matrix is in `THIRD_PARTY_NOTICES.md`.
Generate a release-environment SBOM with:

```bash
python scripts/generate_sbom.py --output build/sbom.json
```

## Dataset Policy

AGU does not redistribute datasets. Training data must be obtained by users from the original dataset owners or official release channels.

If users train with SpaceJam or another basketball dataset, they must:

- Follow the original dataset license and usage terms.
- Keep raw videos and annotations outside the AGU git repository.
- Document the dataset version and source in their experiment logs.
- Avoid committing private, copyrighted, or personally identifiable videos.

Recommended local layout:

```text
dataset/
├── annotation_dict.json
├── augmented_annotation_dict.json
├── examples/
└── augmented-examples/
```

The `dataset/` directory is treated as runtime/local data and should not be committed.

## Weight Distribution Policy

AGU does not ship model weights by default.

If weights are published later, each release must include:

- Download URL.
- SHA256 checksum.
- Model card.
- Training dataset source and license summary.
- Label order.
- Preprocessing contract.
- Evaluation metrics.
- Known limitations.

Recommended release artifact layout:

```text
r2plus1d_v3/
├── best.pt
├── CHECKSUMS.txt
├── MODEL_CARD.md
└── README.md
```

Do not commit weight files into git. Use GitHub Releases, Hugging Face, object storage, or another artifact host instead.

## Public Release Checklist

- `LICENSE` exists.
- `CONTRIBUTING.md` exists.
- `docs/datasets.md` exists.
- `docs/checkpoints.md` exists.
- `docs/model-card.md` exists.
- `docs/api.md` exists.
- `.env.example` contains no secrets.
- `dataset/`, `model_checkpoints/`, `analysis_outputs/`, and `output_videos/` are not committed.
- `python scripts/smoke_open_source.py` passes.
- `python scripts/verify_harness.py` passes.
- `python scripts/evaluate_public_benchmark.py --strict` passes.
- `python -m build` produces a wheel and sdist; an isolated wheel install runs
  `agu --version` and `agu plugins doctor`.
- GitHub CI passes on supported Python versions.
- `THIRD_PARTY_NOTICES.md`, generated SBOM, `SECURITY.md`, `CHANGELOG.md`, and
  `CITATION.cff` match the release.
- Every shipped model/data artifact has source, license, checksum, and an
  explicit redistribution decision.
