# Checkpoints and Weights

AGU does not ship model weights in the repository.

Default runtime configuration expects:

```text
BASKETBALL_MODEL_PATH=model_checkpoints/r2plus1d_v3/
BASKETBALL_BASE_MODEL_NAME=best
```

This resolves to:

```text
model_checkpoints/r2plus1d_v3/best.pt
```

## Recommended Layout

```text
model_checkpoints/
└── r2plus1d_v3/
    ├── best.pt
    ├── CHECKSUMS.txt
    └── MODEL_CARD.md
```

## Checksum Convention

Use `sha256sum` or `shasum -a 256`:

```bash
shasum -a 256 model_checkpoints/r2plus1d_v3/best.pt
```

Store the result as:

```text
<sha256>  best.pt
```

## Publishing Weights

Before publishing weights, document:

- Training dataset source and license.
- Label order.
- Preprocessing contract.
- Validation metrics.
- Known limitations and failure cases.
- SHA256 checksum.

Recommended public hosts:

- GitHub Releases for small artifacts.
- Hugging Face Hub for model cards and versioned weights.
- Object storage for private or large internal artifacts.

Do not commit `.pt`, `.pth`, or other weight files into git.

## Download Instructions Template

When public weights are available, document them like this:

```text
Model: r2plus1d_v3
File: best.pt
URL: <release-url>
SHA256: <sha256>
License: <weight-license>
Training data: <dataset-name-and-license>
```

Then install locally:

```bash
mkdir -p model_checkpoints/r2plus1d_v3
curl -L <release-url> -o model_checkpoints/r2plus1d_v3/best.pt
shasum -a 256 model_checkpoints/r2plus1d_v3/best.pt
```

Compare the checksum with the published `SHA256` before running inference.

## Weight License

The AGU source code license does not automatically grant rights to model weights. Each published checkpoint must declare its own license or usage terms.

If a checkpoint was trained on data that restricts commercial use or redistribution, the checkpoint release must carry the same restrictions when required.

## Loading Behavior

When `BASKETBALL_BASE_MODEL_NAME=best`, AGU loads `best.pt` directly. Other values use the legacy checkpoint naming convention from `utils.checkpoints`.
