# LLM Wiki Pending Imports

Use this file only when the `llm-wiki` skill is unavailable.

Each pending entry should be imported into `llm-wiki` later, then removed or marked imported.

## Pending

### 2026-06-17 - Add AGU Codex LLM Wiki hooks

- Task objective: add two Codex workflow hooks for AGU development tasks.
- Changed files: `AGENTS.md`, `docs/harness/WORKFLOW.md`, `docs/harness/LLM-WIKI-HOOKS.md`, `docs/harness/LLM-WIKI-PENDING.md`, `docs/harness/TASK-BOARD.md`.
- Key decisions: use `llm-wiki` before development for context and after verification for knowledge writeback; if unavailable, queue summaries in this file.
- Verification: `python scripts/verify_harness.py` passed.
- Status: imported/installed follow-up completed on 2026-06-17.
- Notes: `llm-wiki` is now installed globally at `/Users/ppt/.codex/skills/llm-wiki` from the Hermes Agent skill at `/Users/ppt/.hermes/skills/research/llm-wiki`; Codex config includes `WIKI_PATH=/Users/ppt/Projects/wiki` to match the Hermes daily review job.
