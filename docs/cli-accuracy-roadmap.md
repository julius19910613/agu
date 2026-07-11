# AGU CLI Accuracy Roadmap

This document records the P0-P5 implementation plan for making AGU usable as a
basketball video analysis CLI while keeping the repository clean and reusable.

## Architecture Direction

The CLI follows the same separation used by mature Python analysis frameworks:

- CLI adapters parse arguments and call library functions.
- Domain logic lives under `app/analysis/`.
- Offline artifact builders stay in `scripts/` until they become stable CLI
  commands.
- Generated analysis outputs, videos, caches, and local review artifacts stay
  out of git.

## P0 Repeatable Evaluation Loop

Implemented:

- `app.analysis.evaluation` loads event CSV labels, normalizes event names,
  extracts predictions from AGU JSON, and computes precision/recall/F1.
- `python -m app.cli evaluate` wraps that library and can emit JSON and Markdown.
- Strict player matching is available through `--require-player`.

Architecture review:

- Evaluation logic is not embedded in the CLI, so future API, notebook, or test
  harness callers can reuse it.
- CSV decoding accepts UTF-8 and common Chinese spreadsheet encodings.
- The evaluator is intentionally generic and does not reintroduce first-MOV-only
  scripts into the repository.

## P1 CLI Analysis Workbench

Implemented:

- `python -m app.cli analyze --preset fast|accurate|vlm-full`.
- CLI options now expose segmented analysis, VLM audit, tracker/ReID, segment
  slicing, confidence thresholds, and stride controls.
- `--summary` avoids dumping huge result payloads while polling.
- `--save-result` writes completed analysis JSON only when the task completes.
- `python -m app.cli report` wraps the durable player Markdown report builder.

Architecture review:

- Presets are thin payload defaults. Explicit CLI arguments override presets.
- The CLI does not import heavy OpenCV/report dependencies until `report` is
  called.
- Long-running task UX is improved without changing API behavior.

## P2 Event Owner Candidate Layer

Implemented:

- `app.analysis.event_owner` ranks nearby players for every event candidate.
- `EventCandidateResponse.owner_candidates[]` exposes top candidates with
  score, rank, support counts, temporal gap, and evidence.

Architecture review:

- The owner scorer is deterministic and dependency-light.
- It exposes candidate sets instead of pretending event ownership is solved.
- This creates a stable bridge for VLM review, human review, or future supervised
  actor selection.

## P3 Identity Graph Review Summary

Implemented:

- `LongVideoAnalysisResponse.identity_graph_summary` summarizes player identity
  graph nodes, duplicate candidates, confirmed merges, and VLM merge decisions.

Architecture review:

- Original segment-local players remain immutable.
- Confirmed merges remain explicit through `confirmed_identity_merges[]` and
  `merged_players[]`.
- This mirrors graph-review designs in tracking systems: propose edges first,
  confirm before mutation.

## P4 Box Score Confidence Contract

Implemented:

- `PlayerBoxScoreEstimateResponse` now includes `status`,
  `estimated_fields`, and `candidate_fields`.
- Existing point/assist estimates remain backward-compatible.
- Blocks, rebounds, and steals are explicitly marked as candidate/confirmation
  fields.

Architecture review:

- The schema distinguishes estimates from confirmed official stats.
- This keeps downstream products from mistaking `action_proxy_v1` for a final
  game book.

## P5 Repository Structure And Clean Submission

Implemented:

- CLI functionality is centralized in `app/cli.py`.
- Reusable evaluation and owner-scoring logic lives in `app/analysis/`.
- Tests cover CLI payload behavior, event evaluation, owner candidates, identity
  graph summary, and statistics contract.

Submission rules:

- Do not stage `analysis_outputs/`, `output_videos/`, local `.mov` files,
  Ollama caches, model checkpoints, or generated review packages.
- Keep one-off experiments out of the repository unless they become reusable
  framework tools.
- When behavior changes, update `README.md`, `docs/harness/TASK-BOARD.md`, and
  this roadmap or an equivalent durable doc.

## Next Accuracy Work

The next accuracy gain should come from candidate recall and event actor
selection, not from changing v3 preprocessing.

Recommended next iteration:

1. Run `analyze --preset accurate` on a labeled clip.
2. Run `evaluate --require-player` and inspect false negatives.
3. Compare candidate recall in `event_candidates[].owner_candidates[]`.
4. Add candidate-level labels for actor selection.
5. Train or calibrate a lightweight actor selector using owner-candidate
   features before promoting automatic player+action claims.
