# Development

## Implemented

- Added `pyproject.toml`, version `0.1.0`, optional dependency groups, wheel and
  sdist build metadata, and the `agu` console entry point while retaining
  `python -m app.cli`.
- Added dependency-light plugin metadata/discovery through `agu.plugins`, CLI
  list/doctor commands, capability and availability reporting, and a real
  external stage example.
- Added typed pipeline context/stage/runner primitives with validate,
  before-dispatch hooks, existing dispatch, after-dispatch hooks, and finalize.
- Added response `schema_version` and a pipeline manifest containing stage
  trace, adapter selection, checkpoint SHA256, and discovery status.
- Added TOML profiles, cooperative task cancellation/deadline, failed/cancelled
  task retry, request IDs, and task timestamps.
- Added a deterministic public contract benchmark covering scoreboard, event,
  and identity-pair evaluation plus an optional synthetic MP4 generator.
- Added third-party license boundaries, SBOM generation, CI/release workflows,
  security/conduct/citation/changelog files, issue/PR templates, and a docs map.

No change was made to `app/models/preprocessing.py`; the v3 contract and
existing single/segmented analysis implementations remain intact behind the
dispatch stage.
