# Testing

## Verification

- `venv/bin/python -m ruff check ...` on new/changed framework modules: passed.
- Focused mypy on plugin, pipeline, and provenance public primitives: passed
  with no issues.
- `git diff --check`: passed.
- `venv/bin/python -m pytest -q`: `153 passed`, 15 non-failing existing library
  warnings.
- `venv/bin/python scripts/verify_harness.py --run-tests`: passed.
- `venv/bin/python scripts/validate_open_source_baseline.py`: passed.
- `venv/bin/python scripts/evaluate_public_benchmark.py --strict`: all five
  checked metrics `1.0` for the authored contract fixture.
- `venv/bin/python scripts/smoke_open_source.py`: passed.
- `venv/bin/python -m build`: built wheel and sdist without metadata warnings.
- Isolated wheel install: `agu 0.1.0`; non-strict plugin doctor correctly
  reported available core stages and missing optional heavy adapters.
- Local uvicorn on `127.0.0.1:8793`: `/health` and `/ready` returned 200; task
  `e767f6d8d8074d068c50e6f9ab12df42` completed with 18 clips, schema `1.0`,
  validate/dispatch/finalize trace, adapter manifest, and checkpoint SHA256.
- Runtime task `95112ea3fc794eebbb7abf277115f656` cancelled at 100%; retry
  `9f19539a25714fb5be640e6d4285063d` was created from its stored request and was
  independently cancellable.
