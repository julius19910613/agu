# Contributing to AGU

Thanks for helping improve AGU. The project is intentionally scoped as a basketball video action understanding engine, not a full basketball SaaS or mini-program backend.

## Good First Contributions

- Improve docs, examples, and API contract clarity.
- Add tests around preprocessing, output schema, and task behavior.
- Improve CLI ergonomics.
- Add model adapters behind `app.models.registry`.
- Add tracker adapters behind `app.analysis.tracker_registry`.
- Add storage adapters behind `app.storage.backends`.

## Project Boundaries

Please keep these out of AGU:

- Business database writes.
- User login, RBAC, API gateway, or rate limiting.
- `visual_coach`, `basketball`, or `player_grouping` business logic.
- Private CloudBase environment IDs, domains, tokens, or secrets.
- Large datasets, model weights, generated videos, and generated JSON outputs.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Run lightweight checks:

```bash
python scripts/smoke_open_source.py
python scripts/verify_harness.py
```

Run focused tests:

```bash
python -m pytest tests/test_inference.py tests/test_hybrid_analysis.py
```

## Pull Request Checklist

- Keep changes inside AGU's analysis-engine boundary.
- Add or update tests for behavioral changes.
- Update `docs/api.md` when request or response fields change.
- Update `docs/model-card.md` when model or preprocessing behavior changes.
- Update `.env.example` when adding `BASKETBALL_` settings.
- Do not commit secrets, datasets, checkpoints, or generated outputs.

## Model and Data Notes

AGU does not ship model weights or datasets. If you add links to weights or datasets, include license notes, checksum information, and clear usage constraints.
