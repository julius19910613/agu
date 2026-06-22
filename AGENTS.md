# Repository Guidelines

## Project Structure & Module Organization

AGU is a Python basketball video analysis project. The FastAPI service lives in `app/`: `app/main.py` starts the API, `app/analysis/` contains tracking, inference, fusion, VLM review, and task orchestration, `app/models/` contains R(2+1)D model and preprocessing code, and `app/video/` writes annotated outputs. Training and dataset utilities are at the repository root (`train_mac.py`, `train.py`, `dataset.py`) with support scripts in `scripts/` and shared helpers in `utils/`. Tests live in `tests/`. Example media is in `examples/`; generated outputs and large local data should stay in `dataset/`, `model_checkpoints/`, `analysis_outputs/`, and `output_videos/`.

## Build, Test, and Development Commands

Create or activate a virtual environment before running commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API locally:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Run the full test suite with `pytest`. Use focused runs such as `pytest tests/test_inference.py` while iterating. For Mac-oriented training, prefer `python train_mac.py`; older scripts are kept for compatibility.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, clear type-friendly names, and small functions that separate IO, preprocessing, model inference, and orchestration. Follow existing module naming: lowercase snake_case files, snake_case functions, PascalCase classes, and UPPER_SNAKE_CASE constants. Keep service schemas in `app/analysis/schemas.py` and configuration in `app/config.py`. Avoid changing the v3 inference preprocessing contract unless training and tests are updated together.

## Testing Guidelines

`pytest.ini` sets `tests/` as the test root. Name test files `test_*.py` and test functions `test_*`. Add regression tests for checkpoint loading, preprocessing, task resilience, and API-facing behavior when changing those paths. Prefer lightweight fixtures and smoke data over large videos or model files in unit tests.

## Commit & Pull Request Guidelines

Recent history uses concise subjects with optional Conventional Commit prefixes, for example `docs: rename project to AGU in README`, `fix: ...`, and `feat: ...`. Prefer `feat:`, `fix:`, `docs:`, `chore:`, or a short imperative subject. PRs should describe the behavior change, list tests run, call out model/checkpoint or preprocessing impacts, and include screenshots or sample JSON/video outputs for user-visible analysis changes.

## Security & Configuration Tips

Copy `.env.example` to `.env` for local settings and keep secrets, generated outputs, datasets, checkpoints, and downloaded weights out of commits. Configuration uses the `BASKETBALL_` prefix; document new variables in `.env.example` and `README.md`.

## Implementation Scope and Boundaries (Current Delivery)

- AGU（本仓库）边界：
  - `app/` 提供分析引擎 API：异步任务创建、任务查询、任务结果输出。
  - `analysis/` 相关推理链路保持为 `v3` 预处理与推理契约，不在未同步测试/数据变更时修改。
  - `/analysis/run` 与 `/analysis/status/{task_id}` 为当前核心能力。
- BFF/API 网关边界：
  - AGU 仓库不承载统一 BFF、网关、鉴权、限流、ops、grouping、player、match 聚合路由。
  - 统一 BFF/API 网关归属 `visual_coach`，并在其整体重构中使用 Rust 实现。
  - AGU 仅提供被调用的分析服务接口和稳定返回，不主动编排其他项目。
- 跨项目前端/应用：
  - `visual_coach`、`player_grouping`、`basketball` 视为外部/待接入系统，当前代码层仅做网关契约兼容。
  - 它们的 UI 与业务编排变更不在 AGU 仓库实现范围内。

## 开源组件能力治理（Open Source Component Capability Governance）

- AGU 作为开源篮球视频分析组件，功能开发必须优先评估可引用、可配置、可调用的开源能力，而不是默认自研完整实现。
- 新增能力前必须先评估：
  - 是否已有成熟开源组件可直接调用，例如检测、跟踪、ReID、OCR、姿态估计、球/篮筐检测、动作识别、事件检测、VLM audit、视频 IO 与可视化工具。
  - 组件许可证是否兼容 AGU 的开源发布目标，是否允许商用、再分发和模型权重使用。
  - 组件是否能在 AGU 的 Python/FastAPI 主栈内稳定调用；如需外部服务或 CLI，必须通过 adapter/registry/config 接入，避免把业务逻辑绑定到单一实现。
  - 组件是否支持本地离线运行、可配置模型路径、可替换后端、CPU/MPS/CUDA 等不同硬件条件。
  - 组件的输入输出是否能映射到 AGU 的稳定 API/schema，且不会破坏 v3 推理预处理契约。
- AGU 内部实现边界：
  - AGU 可以封装开源组件，提供统一分析 API、任务状态、标准 JSON 输出、VLM 抽检对账与置信度说明。
  - AGU 不应把外部组件的临时 CLI 参数、模型路径或服务地址硬编码进业务流程；必须进入 `app/config.py`、`.env.example` 和 README/API 文档。
  - AGU 不应为了单一实验引入长期维护成本过高的重型依赖；先用可选依赖、adapter、feature flag 或离线脚本验证。
- 功能方案评审要求：
  - 每个非平凡功能方案必须包含“可复用开源能力评估”和“AGU 自研/封装边界”两部分。
  - 如果选择自研，必须说明现有开源组件为何不满足：许可证不兼容、无法本地部署、接口不可控、性能不足、模型不可获得、或与 AGU 输出契约冲突。
  - 如果选择引用组件，必须说明 fallback、配置项、测试策略、输出兼容性和去耦方案。

## 语言治理与项目归一（Language Governance）

- 禁止单仓库语言混用：
  - 在同一个仓库内不新增“并行主语言栈”；例如 AGU 不新增长期维护的 Rust 业务服务。
  - 若确需引入新语言，必须先在该仓库对应的 `AGENTS.md`、`README.md`、`docs/harness` 明确迁移边界、接口接入和验收退出标准。
  - 除统一构建/运维脚本和环境配置外，不允许用另一门语言替代主服务职责。
- 各项目语言归一策略（待各自仓库同步补充）：
  - AGU：主服务固定为 Python（FastAPI + 推理/任务编排）；只提供分析能力返回，不承接 BFF 或 Rust 业务层。
  - `player_grouping`：主服务固定为 TypeScript/Node。
  - `visual_coach`：前端固定为 TypeScript；Rust 用于其后续整体重构（BFF/网关与高性能服务治理层）。
  - `basketball`：主服务固定为 TypeScript（小程序前端）。
- Rust/BFF 归位规则：
  - 历史 `gateway-rs` 是临时网关承接与迁移期产物，应从 AGU 迁出或清理。
  - Rust BFF/网关逻辑应在 `visual_coach` 重构里完成统一归集，不在 AGU 继续形成长期并行业务实现。
  - AGU 目录内如因迁移窗口临时保留 Rust 文件，仅允许用于对照，不允许新增下述功能：
    - 视频推理或训练主流程
    - 数据库事务主逻辑
    - 跨服务核心业务聚合逻辑（应在可归属服务完成）

## 环境变量与配置治理（禁止硬编码）

- 不得在代码中硬编码：
  - 线上 URL、密钥、数据库连接串、token、租户标识、实例地址、服务端口（除非是可运行 fallback）。
- 禁止通过 hardcoding 定死环境变量：
  - 不允许在业务代码、脚本或测试中用常量替代应由环境决定的值。
  - 不允许绕过配置层直接写死某个环境的 URL、端口、路径、模型名、token、租户或实例标识。
  - 所有可变环境配置必须从对应的 env 文件或平台环境变量读取，再经项目配置层使用。
  - AGU 本地默认读取 `.env` / `.env.example` 对应的 `BASKETBALL_` 变量；线上由平台环境变量覆盖。
  - 外部 BFF/API 网关配置必须从其归属项目的 env 文件读取，例如 `visual_coach` 使用 `BFF_*`、`AGU_BASE_URL`、`GROUPING_BASE_URL`。
- 统一读取规则：
  - AGU：统一通过 `app/config.py` 的 `Settings`（`BASKETBALL_` 前缀）读取。
  - BFF：归属 `visual_coach`，由其 Rust 服务统一读取本地和线上环境变量（统一入口 `BFF_*`、`AGU_BASE_URL`、`GROUPING_BASE_URL`）。
  - 本地运行优先读取 `.env`，线上运行读取平台环境变量并覆盖本地值。
- 变更要求：
  - 新增变量必须同时更新 `.env.example`、`README.md`、AGENTS（如有边界影响则更新）。
- 禁止直接在代码中拼接环境特定路径；需要新增配置时应先入配置层再使用。

## 本地启动服务验收计划（Local Start-up & Smoke）

1. 启动基线
   - 准备 `.env`，确认端口不冲突（AGU 默认 8765）。
   - 启动顺序：AGU -> visual_coach Rust BFF -> 依赖服务（如 player_grouping）。
   - 验收项：
     - AGU: `curl http://127.0.0.1:8765/health` 返回 `ok`。
     - visual_coach Rust BFF: `curl http://127.0.0.1:8080/health` 返回网关健康状态，`/ready` 可达。
2. 鉴权与路由验收
   - 调 `/api/v1/auth/login` 获取 token；未携带鉴权访问 protected 路由返回 `401`/统一错误码。
   - `POST /api/v1/analysis/tasks` 可提交任务，返回 `task_id`。
   - `GET /api/v1/analysis/tasks/{task_id}` 可查询状态，结果流包含 `trace_id/request_id`。
3. 功能可用性验收
   - 成功率：本地连续提交 10 个分析任务，至少 90% 在设定超时内返回可查询状态。
   - 成功率门槛：分析查询成功率 ≥ 95%，网关响应 2xx ≥ 99%（单次冒烟窗口）。
   - 网关错误码：验证 `AUTH_401`、`VALIDATION_422`、`UPSTREAM_502`/`UPSTREAM_TIMEOUT` 可返回。
4. 配置一致性验收
   - 删除 `.env` 中任一非必需变量，服务应回退默认行为但不崩溃；敏感变量缺失应阻断启动并给出明确日志。

## 上线部署验收计划（Staging / Production）

1. 上线前自检
   - 检查镜像标签版本、依赖一致性、`env` key 集合与 `.env.example` 同步。
   - 校验 Gateway 与 AGU 连接性、跨服务 DNS/内网地址可达。
2. 分阶段灰度
   - 预发：10% -> 30% -> 100%，每阶段保留 20 分钟观察窗。
   - 关键指标窗口：任务成功率、任务排队时延、网关 5xx、鉴权失败率、超时率。
3. 验收门禁（必须满足）
   - `ops/health` 中 AGU 与 grouping 健康检查全部通过。
   - 分析链路端到端无严重告警（新建任务、查询、结果可见）。
   - 关键错误率门槛：分析任务失败率 ≤ 2%，`UPSTREAM_TIMEOUT` ≤ 1%，`rate_limit` 告警为可控且有告警阈值命中日志。
4. 回滚条件与演练
   - 同一阶段任一核心指标连续 3 分钟异常触发 => 立即暂停放量并回退旧网关地址。
   - 回退路径需可在 5 分钟内切换完成，并保留 `request_id` 与 trace 日志用于对账。

## Codex Harness Rules

## Codex LLM Wiki Hooks

Every AGU development task must use the following two Codex hooks.

1. Pre-development context hook
   - Before implementing a development task, use the `llm-wiki` skill when it is available.
   - Read wiki entries relevant to the user request, touched modules, API contracts, model/preprocessing behavior, deployment notes, and previous task decisions.
   - Treat wiki content as project memory and context, not as a replacement for reading current code.
   - If `llm-wiki` is unavailable, record that in the working notes or final response and continue with repository-local docs.
   - For non-trivial work, mention the wiki context source or fallback in `docs/harness/TASK-BOARD.md` or the matching task artifact.

2. Post-development knowledge hook
   - After development and verification, summarize the implementation, decisions, changed files, verification commands, failures, and lessons learned.
   - Write that summary back into `llm-wiki` using the `llm-wiki` skill when it is available.
   - If `llm-wiki` is unavailable, append a pending entry to `docs/harness/LLM-WIKI-PENDING.md` so the knowledge can be imported later.
   - Keep secrets, private paths, tokens, datasets, checkpoints, and generated outputs out of wiki summaries.

## Codex Local Runtime Hook

After AGU development tasks that change service code, API behavior, configuration, task orchestration, inference/tracking/VLM behavior, output schemas, or README/API contract content:

1. Start the local FastAPI service.
2. Verify `/health` and `/ready` with `curl`.
3. Submit a lightweight `POST /api/v1/analysis/run` request with `curl` and confirm the response has the expected `task_id` / `status` shape.
4. Query the task status endpoint with `curl`; for user-visible analysis behavior changes, poll until `completed` or record the exact failure.
5. Check `README.md` against current code and `docs/api.md`; update README/API docs when request fields, response fields, startup commands, environment variables, output behavior, or model/checkpoint assumptions changed.

Use the repeatable procedure in `docs/harness/LOCAL-SERVICE-CURL-HOOK.md`. If local service startup or curl verification cannot run because of environment constraints, record the blocker and the closest verification that did run.

Use the workflow in `docs/harness/WORKFLOW.md` for changes that affect API contracts, inference preprocessing, training behavior, configuration, output JSON/video formats, or task orchestration. Small documentation-only or narrowly scoped test changes may use the compact workflow.

Keep `AGENTS.md` focused on durable rules. Put repeatable procedures in repo skills under `.agents/skills/`, and put task state or project maps under `docs/harness/`.

Do not mark implementation work complete until relevant verification has run or the reason for not running it is recorded. Prefer focused pytest runs while iterating; run broader checks when shared behavior changes.

When changing public behavior, update the matching harness documentation or task board entry. In particular, API, model/preprocessing, training, configuration, and output-contract changes should leave a durable note outside the chat.

Preserve the v3 inference preprocessing contract unless training code and regression tests are updated together.
