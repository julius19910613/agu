# LLM Wiki Hooks

AGU uses two Codex workflow hooks to keep project memory active and current.
Runtime-facing AGU development also uses the local service curl hook documented
in `docs/harness/LOCAL-SERVICE-CURL-HOOK.md`.

## Hook 1: Pre-development Context Read

Trigger:

- Before any development task that changes code, API behavior, docs, deployment, training, inference, configuration, or tests.

Action:

- Use the `llm-wiki` skill when available.
- Search/read entries related to:
  - The user request.
  - Files likely to change.
  - API contracts.
  - v3 preprocessing and inference behavior.
  - Deployment, Docker, CloudBase, or open-source release notes.
  - Previous decisions in similar tasks.

Output:

- A short working note naming the wiki context used.
- For non-trivial tasks, record the source/fallback in `docs/harness/TASK-BOARD.md` or a `docs/specs/TASK-*` artifact.

Fallback:

- If `llm-wiki` is unavailable, read repository-local durable docs instead:
  - `AGENTS.md`
  - `README.md`
  - `docs/harness/WORKFLOW.md`
  - `docs/harness/TASK-BOARD.md`
  - relevant docs under `docs/`

## Hook 2: Post-development Knowledge Write

Trigger:

- After implementation and verification, before the task is considered complete.

Action:

- Use the `llm-wiki` skill when available.
- Write a concise summary with:
  - Task objective.
  - Changed files.
  - Key decisions.
  - Verification commands and results.
  - Problems encountered.
  - Lessons learned.
  - Follow-up ideas.

Fallback:

- If `llm-wiki` is unavailable, append the same information to `docs/harness/LLM-WIKI-PENDING.md`.
- The pending file acts as an import queue for the next session where `llm-wiki` is available.

## Safety Rules

- Do not write secrets, tokens, private URLs, private datasets, checkpoints, generated videos, or generated JSON into wiki summaries.
- Do not treat wiki content as more authoritative than current code.
- If wiki context conflicts with repository code, prefer the current code and document the discrepancy.
- Run the local service curl hook before the post-development wiki write for runtime-facing changes, so the wiki summary records the actual verification result rather than an intended check.
