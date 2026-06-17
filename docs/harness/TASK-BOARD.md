# AGU Task Board

This board records non-trivial Codex-assisted work so future sessions can see what changed, where task artifacts live, and what remains blocked.

Update this file when a task uses the full workflow, changes public behavior, or produces durable task artifacts under `docs/specs/`.

## Status Legend

- `W1 Requirement`: clarifying objective and acceptance criteria.
- `W2 Solution`: designing implementation and verification plan.
- `W3 Gate Review`: checking readiness to implement.
- `W4 Development`: implementing changes.
- `W5 Code Review`: reviewing risks and defects.
- `W6 Testing`: verifying and closing.
- `Done`: completed and verified.
- `Paused`: intentionally stopped.
- `Blocked`: cannot proceed without user input or external state.

## In Progress

| Task ID | Task Name | Phase | Owner | Blockers | Docs | Last Updated |
| --- | --- | --- | --- | --- | --- | --- |
| _none_ |  |  |  |  |  |  |

## Completed

| Task ID | Task Name | Completed | Docs | Verification | Delivery Notes |
| --- | --- | --- | --- | --- | --- |
| TASK-0001 | Initialize Codex workflow docs | 2026-06-14 | `docs/harness/WORKFLOW.md`, `docs/harness/TASK-BOARD.md` | Readback plus key-section search | Added phase workflow and task board foundation. |
| TASK-0002 | Add Codex harness rules and gates | 2026-06-14 | `AGENTS.md`, `.agents/skills/agu-test/SKILL.md`, `.agents/skills/agu-verify/SKILL.md`, `scripts/verify_harness.py` | `python scripts/verify_harness.py`; `python scripts/verify_harness.py --test-command "pytest tests/test_inference.py"` | Added project rules, repo skills, and first structural verification gate. |
| TASK-0003 | Document AGU deployment verification | 2026-06-17 | `docs/deploy-and-verify.md`, `README.md`, `Dockerfile`, `requirements-service.txt`, `.dockerignore` | `python scripts/verify_harness.py` | Added local, Docker self-hosted online deployment, basketball database writeback evaluation, and CloudBase integration notes for AGU-only analysis service. |
| TASK-0004 | Assess AGU open source scope | 2026-06-17 | `docs/open-source-scope-assessment.md` | `python scripts/verify_harness.py` | Added value assessment, open-source boundaries, retained modules, and extraction roadmap. |
| TASK-0005 | Implement open source phase 1 | 2026-06-17 | `README.md`, `app/cli.py`, `examples/`, `requirements-service.txt`, `requirements-training.txt`, `requirements-dev.txt` | `python scripts/verify_harness.py`; `python -m app.cli --help` | Added open-source positioning, sample request/output, API CLI client, dependency split, and legacy entrypoint guidance. |
| TASK-0006 | Implement reproducible baseline phase 2 | 2026-06-17 | `docs/api.md`, `docs/model-card.md`, `scripts/validate_open_source_baseline.py`, `scripts/verify_harness.py` | `python scripts/validate_open_source_baseline.py`; `python scripts/verify_harness.py` | Added schema-validated public examples, API contract, model card, and lightweight open-source baseline gate. |
| TASK-0007 | Implement open-source phases 3 and 4 | 2026-06-17 | `LICENSE`, `CONTRIBUTING.md`, `docs/checkpoints.md`, `docs/extensions.md`, `scripts/smoke_open_source.py`, `app/models/registry.py`, `app/analysis/tracker_registry.py`, `app/storage/backends.py`, `tests/test_open_source_extensions.py` | `python scripts/smoke_open_source.py`; `python scripts/validate_open_source_baseline.py`; `python scripts/verify_harness.py`; `pytest tests/test_open_source_extensions.py` | Added open-source contribution/license/checkpoint docs, smoke check, and first model/tracker/storage extension points. |
| TASK-0008 | Add public release policy docs | 2026-06-17 | `docs/release-notes.md`, `docs/datasets.md`, `docs/checkpoints.md`, `docs/model-card.md`, `README.md`, `scripts/validate_open_source_baseline.py` | `python scripts/smoke_open_source.py`; `python scripts/validate_open_source_baseline.py`; `python scripts/verify_harness.py` | Added source statement, license boundary, dataset acquisition guidance, and weight distribution policy for public release readiness. |

## Paused Or Blocked

| Task ID | Task Name | Status | Reason | Docs | Last Updated |
| --- | --- | --- | --- | --- | --- |
| _none_ |  |  |  |  |  |

## Maintenance Rules

- Use stable task IDs such as `TASK-0001`.
- Keep task names short and behavior-oriented.
- Link task docs when a task has a `docs/specs/TASK-*` folder.
- Move rows from `In Progress` to `Completed` only after relevant verification has run or the skipped verification is documented.
- Remove the `_none_` placeholder row when adding the first real row in a section.
