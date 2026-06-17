# AGU R(2+1)D v3 Model Card

## Model Summary

AGU currently targets a R(2+1)D-18 action classifier for basketball video clips. The default deployment checkpoint path is:

```text
model_checkpoints/r2plus1d_v3/best.pt
```

The repository does not ship model weights. Users must train their own checkpoint or provide a compatible file.

## Intended Use

The model is intended for clip-level basketball action understanding after player tracking and crop extraction.

Supported labels:

```text
0 block
1 pass
2 run
3 dribble
4 shoot
5 ball in hand
6 defense
7 pick
8 no_action
9 walk
```

## Inputs

The deployed v3 inference preprocessing contract is intentionally preserved:

- OpenCV BGR channel order.
- Spatial resize to `112x112`.
- Pixel value range `[0, 255]`.
- No `/255` scaling.
- No RGB conversion.
- No Kinetics mean/std normalization.

This is not the ideal long-term preprocessing strategy, but it is the current deployed contract. Changing it requires retraining and regression tests.

## Outputs

The model produces:

- `action_id`
- `action`
- `confidence`
- `probabilities`

AGU then combines this with motion features, optional VLM review, and temporal smoothing to produce `final`.

## Known Limitations

- Performance depends heavily on tracking quality.
- Rare classes such as `shoot` and `pick` require careful class balancing and evaluation.
- The current v3 preprocessing does not match standard Kinetics pretrained transforms.
- Public benchmark results are not yet established.
- Long videos should be processed asynchronously and polled by `task_id`.

## Recommended Evaluation Metrics

Do not rely only on overall accuracy. For basketball actions, report:

- accuracy
- macro-F1
- balanced accuracy
- per-class precision
- per-class recall
- confusion matrix
- `shoot` recall
- `pick` recall

## Reproducibility Checklist

- Record checkpoint path and checksum.
- Record git commit.
- Record label order.
- Record preprocessing contract.
- Record train/val/test split.
- Record dataset source and license.
- Record evaluation command and metrics.

## Release Status

This model card documents the current AGU v3 deployment contract. A public checkpoint and benchmark table should be added before a formal open-source release.

## Source and License Notes

The AGU source code is MIT-licensed. Model weights are not included in the repository and must declare their own license when published.

Before distributing a checkpoint, verify that the training dataset license allows derived weight distribution. Include dataset source, dataset license, checkpoint checksum, and evaluation metrics with the release.
