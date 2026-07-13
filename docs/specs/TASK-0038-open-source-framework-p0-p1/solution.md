# Solution

## Design

- Keep `app` as the import package for backward compatibility; publish the
  distribution as `agu-basketball` and expose the `agu` console script.
- Introduce a dependency-light plugin catalog that records kind, capabilities,
  requirements, source, version, availability, and registration errors.
- Preserve the existing model/tracker registries and mirror their built-ins into
  the catalog. Discover third-party registration callables through the
  `agu.plugins` Python entry-point group.
- Add a small typed pipeline runner. The current analysis dispatch becomes a
  built-in `analysis.dispatch` stage; later algorithm extraction can happen one
  stage at a time without another public contract migration.
- Add response manifest fields with defaults so historical fixtures and callers
  remain valid.
- Treat public benchmark fixtures as contract/evaluator validation, not as a
  model quality claim.

## Verification Plan

- Focused framework, plugin, CLI, schema, and benchmark tests.
- Existing hybrid analysis and CLI tests.
- Full pytest and AGU harness with tests.
- Build wheel/sdist, install wheel into an isolated virtual environment, run
  `agu --version` and plugin diagnostics.
- Open-source smoke and baseline validation scripts.
- Local `/health`, `/ready`, `/analysis/run`, and `/analysis/status` curl hook.
