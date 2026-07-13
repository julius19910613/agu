# Public Contract Benchmark

This small, redistributable fixture verifies AGU's evaluation and output
contracts without private videos, checkpoints, or external services. It is not
a claim about production model accuracy.

Run:

```bash
python scripts/evaluate_public_benchmark.py --strict
python scripts/make_public_benchmark_fixture.py
```

The generator creates an ignored synthetic MP4 with two colored player boxes
and a scoreboard panel under `analysis_outputs/public_benchmark/`. The checked-in
labels and predictions deliberately cover:

- final scoreboard agreement;
- event type/time/player matching;
- same-player and different-player identity pair decisions.

Real model releases must publish a separate dataset/model card with licensed
video, checkpoint hashes, hardware, thresholds, IDF1/HOTA, event F1, scoreboard
accuracy, and runtime.
