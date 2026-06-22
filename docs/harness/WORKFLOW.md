# AGU Codex Workflow

This workflow is the project-level harness for using Codex on AGU. It keeps AI-assisted work traceable, testable, and aligned with the basketball video analysis pipeline.

Use the full workflow for changes that affect API behavior, model inference, training, preprocessing, configuration, output schemas, generated videos, or task orchestration. For small documentation-only or single-test fixes, use the compact workflow at the end.

## Core Principles

- Keep requirements, implementation, review, and testing as separate thinking steps.
- Write durable decisions into repository files, not only into chat.
- Do not treat a task as complete until the relevant verification has run or the reason for not running it is recorded.
- Prefer focused tests while iterating, then broaden verification when the change touches shared behavior.
- If a phase discovers a blocking issue, return to the phase that owns the issue instead of pushing forward.

## Full Workflow

### W1 Requirement

Goal: turn the user request into a clear task.

Inputs:
- User request.
- Existing code and docs.
- Relevant entries in `docs/harness/TASK-BOARD.md`.
- Relevant `llm-wiki` entries when the `llm-wiki` skill is available.

Output:
- A short task statement.
- Scope and non-scope.
- Acceptance criteria.
- Known risks or open questions.
- A note of what wiki context was read, or that `llm-wiki` was unavailable.

Exit criteria:
- The objective is specific enough to implement.
- Unclear or risky assumptions are called out.
- The task is added to `docs/harness/TASK-BOARD.md` when it is non-trivial.

Return here when:
- Implementation reveals that the original request was ambiguous.
- A test or review shows the selected behavior does not match user intent.

### W2 Solution

Goal: choose a conservative implementation path.

Inputs:
- W1 requirement.
- Relevant modules, schemas, tests, and existing patterns.

Output:
- Files likely to change.
- Implementation approach.
- Verification plan.
- Documentation/map updates needed.

Exit criteria:
- The plan respects existing module boundaries.
- The verification plan names concrete commands or checks.
- API, preprocessing, model, and output contract impacts are identified.

Return here when:
- The first approach would require broad refactoring.
- A shared contract changes unexpectedly.
- Verification exposes a design flaw rather than a coding mistake.

### W3 Gate Review

Goal: decide whether implementation can start.

Checks:
- Is the task scoped tightly enough?
- Are generated outputs, checkpoints, datasets, and secrets kept out of commits?
- Is the v3 inference preprocessing contract protected unless training and tests change together?
- Are API schema or configuration changes paired with docs updates?
- Is the verification plan realistic for the current environment?

Exit criteria:
- PASS: proceed to W4.
- REVISE: return to W1 or W2 with the reason.
- BLOCKED: stop and ask the user only when local context cannot resolve the blocker safely.

### W4 Development

Goal: implement the approved change.

Rules:
- Follow existing AGU patterns before adding new abstractions.
- Keep edits scoped to the task.
- Preserve user changes and unrelated dirty worktree state.
- Update docs/maps when changing public behavior, configuration, model contracts, or outputs.

Exit criteria:
- Code or docs are changed.
- The implementation matches W1 acceptance criteria.
- Obvious local formatting or import issues are resolved.

Return here when:
- W5 or W6 finds implementation defects.

### W5 Code Review

Goal: review the change before declaring it ready.

Review focus:
- Behavioral regressions.
- API-facing compatibility.
- Training/inference preprocessing drift.
- Task resilience and error handling.
- Missing tests or insufficient verification.
- Accidental large files, generated outputs, or secrets.

Exit criteria:
- Findings are either fixed or explicitly documented as residual risk.
- The verification plan is still appropriate after the final diff.

Return here when:
- W4 changes materially after review.

### W6 Testing

Goal: prove the result with the narrowest meaningful verification.

Default commands:
- `pytest` for broad changes.
- Focused runs such as `pytest tests/test_inference.py` while iterating.
- Manual API or video analysis smoke checks when behavior is user-visible and test fixtures are insufficient.

Exit criteria:
- Relevant checks pass, or failures are recorded with whether they are pre-existing, environment-related, or introduced by the change.
- For service/API/config/inference/tracking/VLM/output-contract changes, the local service curl hook in `docs/harness/LOCAL-SERVICE-CURL-HOOK.md` has run, or the blocker is recorded.
- `README.md` and `docs/api.md` have been checked against the changed code and updated when public startup, request, response, configuration, output, or checkpoint behavior changed.
- Development decisions and lessons are written to `llm-wiki`, or queued in `docs/harness/LLM-WIKI-PENDING.md` if the skill is unavailable.
- Final response reports what was changed and what verification ran.
- `docs/harness/TASK-BOARD.md` is updated for non-trivial tasks.

Return here when:
- Tests fail due to implementation defects: return to W4.
- Tests reveal design mismatch: return to W2.
- Tests reveal unclear requirement: return to W1.

## Compact Workflow

Use this for small, low-risk changes:

1. Clarify the request and affected files.
2. Read relevant `llm-wiki` context when available.
3. Implement the scoped change.
4. Run a focused verification or explain why it was not run.
5. For runtime-facing changes, run the local service curl hook or record why it does not apply.
6. Check README/API docs against code when public behavior or setup changed.
7. Write lessons to `llm-wiki` when available, or queue them in `docs/harness/LLM-WIKI-PENDING.md`.
8. Summarize the result.

Do not use the compact workflow for changes involving API contracts, inference preprocessing, checkpoints, training behavior, output JSON/video formats, or security-sensitive configuration.

## Task Artifacts

For larger tasks, create a folder under `docs/specs/`:

```text
docs/specs/TASK-0001-short-name/
├── requirement.md
├── solution.md
├── gate-review.md
├── development.md
├── code-review.md
└── testing.md
```

These files do not need to be verbose. Their job is to preserve decisions and handoff state when a task spans multiple phases or sessions.

## Completion Definition

A task is complete only when all of the following are true:

- The requested behavior or document change is implemented.
- Relevant tests or checks have run, or an explicit reason is recorded.
- Local service curl verification has run for runtime-facing changes, or an explicit blocker is recorded.
- README/API documentation has been checked against current code and updated if needed.
- Any changed public contract is reflected in docs or harness maps.
- The final response names the verification performed and any residual risk.
