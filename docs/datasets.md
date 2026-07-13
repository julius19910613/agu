# Dataset Guide

AGU does not include datasets in the repository. This keeps the project lightweight and avoids redistributing videos or annotations whose licenses may differ from AGU's MIT-licensed source code.

## Supported Training Layout

The current training scripts expect a local dataset layout similar to:

```text
dataset/
├── annotation_dict.json
├── augmented_annotation_dict.json
├── examples/
│   ├── clip_0001.mp4
│   └── ...
└── augmented-examples/
    ├── clip_aug_0001.mp4
    └── ...
```

`dataset/` is ignored as local runtime data and should not be committed.

## SpaceJam Notes

AGU training utilities were designed around a SpaceJam-style basketball action dataset.

Before using SpaceJam or any other third-party dataset:

- Read the original dataset license and terms.
- Download data from the original source or official mirror.
- Keep attribution in experiment notes.
- Do not redistribute raw videos through this repository.
- Do not publish trained weights unless the dataset terms allow derived model distribution.

Suggested local preparation flow:

```bash
mkdir -p dataset
# Place annotation_dict.json under dataset/
# Place video clips under dataset/examples/
python scripts/gen_splits.py --annotation-path dataset/annotation_dict.json
```

## Smoke Data

For CI and public examples, prefer tiny synthetic or explicitly licensed smoke data.

Recommended rules:

- Keep files small.
- Avoid private footage.
- Avoid copyrighted broadcast clips unless redistribution is allowed.
- Document the source of every sample video.

`examples/benchmark/` is AGU's checked-in public contract fixture. It contains
only authored JSON/CSV labels and predictions. A deterministic license-free MP4
can be generated locally with `scripts/make_public_benchmark_fixture.py`; the
generated video remains under ignored `analysis_outputs/`. This validates
evaluation plumbing and must not be reported as model accuracy.

## Publishing Dataset References

If you add a dataset reference, include:

- Dataset name.
- Official URL.
- License or usage terms URL.
- Expected local directory layout.
- Required annotation format.
- Any preprocessing steps.

## Annotation Format

AGU's training utilities expect annotations that can be mapped to the 10 action labels:

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

If you adapt a dataset with different labels, document the mapping and keep the mapping script outside generated output directories.
