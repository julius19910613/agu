---
name: agu-test
description: Use when Codex needs to choose or run AGU pytest verification after code changes, especially for inference, training, API, task resilience, or video analysis behavior.
---

# AGU Test Skill

Use this skill to select focused verification for AGU changes.

## Test Selection

- Inference or preprocessing changes: run `pytest tests/test_inference.py`.
- Training loop, checkpoint, dataset, sampler, or Mac training changes: run `pytest tests/test_train_mac.py tests/test_train_mac_resilience.py`.
- Analysis service, router, task manager, fusion, tracking, VLM, or video writer changes: run `pytest tests/test_hybrid_analysis.py`.
- Cross-cutting behavior or uncertainty: run `pytest`.

## Procedure

1. Identify the changed files and affected behavior.
2. Pick the narrowest test command that covers the risk.
3. Run the command from the repository root.
4. If a focused test fails, inspect whether the failure is introduced, pre-existing, or environment-related.
5. Broaden to `pytest` when the change touches shared contracts or multiple subsystems.
6. Report the exact command and result in the final response.

## Notes

- Prefer lightweight fixtures and smoke data over large videos or model files.
- Do not change the v3 preprocessing contract without updating training and regression tests together.
- If tests cannot run because dependencies or model files are missing, record the blocker and run any available structural verification.
