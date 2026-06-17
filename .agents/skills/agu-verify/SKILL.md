---
name: agu-verify
description: Use when Codex needs to run AGU's harness verification gate, check workflow/documentation consistency, or validate that a task is ready to close.
---

# AGU Verify Skill

Use this skill as the final gate for non-trivial AGU work.

## Default Command

Run structural harness checks:

```bash
python scripts/verify_harness.py
```

Run structural checks plus the full pytest suite:

```bash
python scripts/verify_harness.py --run-tests
```

Run structural checks plus a focused test command:

```bash
python scripts/verify_harness.py --test-command "pytest tests/test_inference.py"
```

## What The Gate Checks

- Required harness documents exist.
- Required workflow sections exist.
- Task board has the expected sections.
- Generated output and large-data directories are not staged.
- Python source files compile.
- `.env.example` documents all `BASKETBALL_` settings declared in `app/config.py`.
- Optional pytest command passes when requested.

## Procedure

1. Run the default command after documentation-only harness changes.
2. Add `--test-command` for focused code changes.
3. Add `--run-tests` for shared or high-risk changes.
4. Fix failures introduced by the current task before closing.
5. If a failure is pre-existing or environment-related, record the reason and remaining risk.
