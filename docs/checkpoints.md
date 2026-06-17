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

## Loading Behavior

When `BASKETBALL_BASE_MODEL_NAME=best`, AGU loads `best.pt` directly. Other values use the legacy checkpoint naming convention from `utils.checkpoints`.
