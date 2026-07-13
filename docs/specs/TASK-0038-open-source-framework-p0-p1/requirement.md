# Requirement

## Goal

Complete the open-source framework P0 and P1 work identified in the 2026-07-13
GitHub comparison while preserving AGU's current API, CLI, nine-MOV scoreboard
and identity behavior, and the v3 inference preprocessing contract.

## In Scope

- Installable/versioned Python distribution and `agu` console command.
- Optional dependency groups with a small core installation.
- Typed pipeline stage/context/runner foundation around existing analysis.
- Discoverable plugin metadata, capability checks, and CLI diagnostics.
- Stable output schema version and runtime pipeline manifest.
- Third-party code/model/data license governance and SBOM tooling.
- Reproducible public contract benchmark and golden fixtures.
- GitHub CI, security/community/release artifacts, configuration profiles,
  plugin example, and contributor documentation.

## Non-scope

- Replacing the R(2+1)D model, YOLO tracker, OCR, VLM, or identity algorithms.
- Changing v3 preprocessing or retraining checkpoints.
- Moving BFF, authentication, grouping, or product logic into AGU.
- Claiming model accuracy from the public contract fixture.

## Acceptance Criteria

1. `pip install -e .` metadata builds and `agu --version` works.
2. Existing `python -m app.cli` commands remain compatible.
3. External plugins can be discovered through `agu.plugins` entry points and
   inspected through `agu plugins list|doctor` without importing optional heavy
   backends unnecessarily.
4. `AnalysisService.run_analysis` uses the typed pipeline runner without
   changing its dispatch behavior.
5. Successful analysis output exposes a defaulted `schema_version` and
   `pipeline_manifest` without breaking older constructors.
6. Public benchmark fixtures and evaluator run without private video/model data.
7. CI and release/community/license governance artifacts exist and validate.
8. Focused tests, full pytest, harness, open-source smoke, package build/install,
   and local FastAPI curl smoke pass.

## Context

Read `/Users/ppt/wiki/queries/agu-ground-truth-baseline-evaluator-2026-07-02.md`,
especially the open-source cleanup, CLI roadmap, and nine-MOV generalization
entries. The repository-local open-source assessment and extension docs were
also reviewed. Current uncommitted nine-MOV changes are user work and must be
preserved.
