# AGU 本地与线上部署验证方案

本文只覆盖 AGU 分析服务自身。统一 BFF、鉴权、限流、分组聚合和运营后台 API 归属 `visual_coach` Rust 重构，不在 AGU 仓库验收。

## 验收目标

- AGU 服务可启动并完成模型加载。
- 健康检查、任务提交、任务查询和结果读取可用。
- 本地与线上均通过环境变量配置，不在代码中硬编码路径、端口、密钥或线上地址。
- 部署后可用真实线上 API 地址完成一次端到端分析调用。

## 本地验证

### 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

确认 `.env` 至少包含：

```text
BASKETBALL_MODEL_PATH=model_checkpoints/r2plus1d_v3/
BASKETBALL_BASE_MODEL_NAME=best
BASKETBALL_DEFAULT_VIDEO=examples/lebron_shoots.mp4
BASKETBALL_OUTPUT_DIR=analysis_outputs
BASKETBALL_VIDEO_OUTPUT_DIR=output_videos
BASKETBALL_HOST=127.0.0.1
BASKETBALL_PORT=8765
```

### 2. 启动服务

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

预期：

- 控制台出现模型加载完成日志。
- `analysis_outputs/` 与 `output_videos/` 可按配置创建。
- `/health` 返回 `{"status":"ok"}`。
- `/ready` 返回 `{"status":"ready"}`。

### 3. 健康检查

```bash
curl -sS http://127.0.0.1:8765/health
curl -sS http://127.0.0.1:8765/ready
```

验收标准：

- HTTP 状态码为 `200`。
- 返回 JSON 可解析。
- 不出现模型加载异常、checkpoint 缺失或端口冲突。

### 4. 提交分析任务

```bash
curl -sS -X POST http://127.0.0.1:8765/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "examples/lebron_shoots.mp4",
    "vlm_mode": "off",
    "max_frames": 120,
    "generate_video": false
  }'
```

预期返回：

```json
{
  "task_id": "...",
  "status": "pending",
  "message": "Analysis started asynchronously. Please poll the status endpoint to query progress."
}
```

验收标准：

- 返回 `task_id`。
- `status` 为 `pending` 或后续可查询状态。
- 无 `Video file not found`、checkpoint 加载错误或推理异常。

### 5. 查询任务状态

```bash
curl -sS http://127.0.0.1:8765/api/v1/analysis/status/<task_id>
```

验收标准：

- `task_id` 与提交返回一致。
- `status` 在 `pending`、`processing`、`completed`、`failed` 之一。
- 成功完成时 `result.summary.clip_count`、`result.records` 等字段存在。
- 失败时 `error` 字段必须包含可定位原因。

### 6. 外部 BFF 兼容接口

AGU 也提供兼容外部 BFF 的分析别名：

```bash
curl -sS -X POST http://127.0.0.1:8765/api/v1/analysis/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "examples/lebron_shoots.mp4",
    "vlm_mode": "off",
    "max_frames": 120,
    "generate_video": false
  }'

curl -sS http://127.0.0.1:8765/api/v1/analysis/tasks/<task_id>
curl -sS http://127.0.0.1:8765/api/v1/analysis/tasks/<task_id>/result
```

验收标准：

- 行为与 `/analysis/run`、`/analysis/status/{task_id}` 一致。
- 这些接口只提供分析结果，不承担 BFF 鉴权、限流或聚合职责。

## 线上部署方式

### 推荐方式：Docker 镜像本地/自有服务器运行

目标是把 AGU 打包成 Docker 镜像，在本地机房、训练机、GPU/CPU 工作站或自有云主机上运行。微信小程序、`basketball` 客户端和 `visual_coach` Rust BFF 不直接依赖 AGU 所在机器的本地路径，而是通过 API、数据库记录和文件存储集成。

推荐原因：

- AGU 依赖 PyTorch、OpenCV、YOLO、checkpoint 和可能的本地 VLM，容器比 CloudBase 直部署更容易控制运行环境。
- 视频分析任务可能超过 CloudBase 云托管 `60s` 请求超时，Docker 自托管更适合长任务和异步轮询。
- 输入视频、checkpoint 和输出视频体积较大，本地 volume 挂载更可控。
- 后续如果需要 GPU，Docker 自托管比 CloudBase 云托管更容易接入 GPU 节点。
- 小程序仍可通过 `visual_coach` Rust BFF 或 CloudBase HTTP 网关访问，不要求 AGU 自身部署在 CloudBase。

镜像要求：

- 安装 Python 运行时、系统 OpenCV 依赖、PyTorch、YOLO/ultralytics 依赖。
- checkpoint、样例数据和输出目录通过 volume 或对象存储挂载。
- 容器启动命令使用 `uvicorn app.main:app --host 0.0.0.0 --port ${BASKETBALL_PORT:-8765}`。

部署要求：

- 镜像 tag 使用不可变版本号，例如 `agu:<commit_sha>`。
- readiness probe 使用 `/ready`。
- liveness probe 使用 `/health`。
- 资源配置至少覆盖模型加载内存和单任务推理峰值。
- `analysis_outputs/`、`output_videos/`、`model_checkpoints/` 必须映射到宿主机 volume 或外部存储。
- AGU 服务端口只暴露给内网或 `visual_coach` Rust BFF，不建议直接暴露给小程序公网调用。

示例镜像构建：

```bash
docker build -t agu:<commit_sha> .
```

示例本地/服务器运行：

```bash
docker run -d --name agu \
  --restart unless-stopped \
  -p 8765:8765 \
  -e BASKETBALL_HOST=0.0.0.0 \
  -e BASKETBALL_PORT=8765 \
  -e BASKETBALL_MODEL_PATH=/app/model_checkpoints/r2plus1d_v3/ \
  -e BASKETBALL_OUTPUT_DIR=/app/analysis_outputs \
  -e BASKETBALL_VIDEO_OUTPUT_DIR=/app/output_videos \
  -v /srv/agu/model_checkpoints:/app/model_checkpoints \
  -v /srv/agu/analysis_outputs:/app/analysis_outputs \
  -v /srv/agu/output_videos:/app/output_videos \
  -v /srv/agu/service_inputs:/app/service_inputs \
  agu:<commit_sha>
```

部署后验证：

```bash
export AGU_BASE_URL=http://<server-ip-or-domain>:8765

curl -sS "$AGU_BASE_URL/health"
curl -sS "$AGU_BASE_URL/ready"
```

### 数据写入 basketball 数据库方案评估

目标：AGU 分析完成后，结果能够进入 `basketball` 当前使用的数据库，供小程序展示和 `visual_coach` 管理端复核。

推荐结论：不建议 AGU 直接承担 basketball 业务数据库写入。更稳妥的方式是 AGU 只产出分析结果，由 `visual_coach` Rust BFF 或独立同步 worker 写入 basketball 数据库。

推荐链路：

```text
basketball 小程序 / visual_coach
  -> visual_coach Rust BFF
  -> AGU Docker 服务
  -> AGU 返回 task_id / result
  -> Rust BFF 或 result-sync worker 写入 basketball 数据库
  -> basketball 小程序读取数据库展示
```

可选链路：

```text
AGU Docker 服务
  -> 分析完成后触发 callback
  -> visual_coach Rust BFF / result-sync worker
  -> basketball 数据库
```

不推荐链路：

```text
AGU Docker 服务
  -> 直接持有 basketball 数据库密钥
  -> 直接写 basketball 数据库业务表
```

评估：

| 方案 | 优点 | 风险 | 结论 |
| --- | --- | --- | --- |
| Rust BFF 写库 | 权限、鉴权、字段映射、幂等和审计集中；符合项目边界 | 需要 BFF 实现结果落库接口 | 推荐 |
| result-sync worker 写库 | AGU 与业务库解耦；失败可重试；适合异步任务 | 多一个部署组件 | 推荐 |
| AGU 直接写库 | 实现最快，少一跳 | AGU 需要持有业务库密钥；耦合 basketball schema；违反 AGU 只负责分析边界 | 仅限临时内测 |

建议写入的数据模型：

- `analysis_tasks`：`task_id`、`match_id`、`player_id`、`source_video_id`、`status`、`progress`、`error`、`created_at`、`updated_at`
- `analysis_results`：`task_id`、`summary`、`records`、`output_json_url`、`output_video_url`、`model_version`、`completed_at`
- `analysis_segments`：`task_id`、`player_id`、`clip_index`、`start_frame`、`end_frame`、`action`、`confidence`、`needs_review`
- `analysis_feedback`：人工修订与复核记录，由 `visual_coach` 写入

字段映射原则：

- AGU 返回原始 `result`，不直接理解 basketball 页面展示字段。
- BFF/worker 负责把 AGU `records`、`summary` 映射到 basketball 数据库 schema。
- 写库必须按 `task_id` 幂等 upsert，避免重复轮询导致重复记录。
- 失败任务也要写入状态和 `error`，方便小程序展示“分析失败/重试”。
- 输出视频和 JSON 文件应先落到对象存储或可访问文件服务，再把 URL/fileID 写入数据库。

环境变量建议：

```text
AGU_BASE_URL=http://agu.internal:8765
BASKETBALL_DB_PROVIDER=cloudbase
BASKETBALL_DB_ENV_ID=<env-id>
BASKETBALL_DB_COLLECTION_TASKS=analysis_tasks
BASKETBALL_DB_COLLECTION_RESULTS=analysis_results
```

说明：

- 上述 `BASKETBALL_DB_*` 不应放进 AGU，建议放在 `visual_coach` Rust BFF 或 result-sync worker。
- 若短期必须由 AGU 直接写库，必须先在 `AGENTS.md` 和任务板中记录临时例外、退出时间和密钥治理方案。

### 可选方式：CloudBase 云托管部署

结论：CloudBase 更适合作为小程序生态入口、数据库、云存储和 HTTP 网关；AGU 本体推荐先用 Docker 在本地/自有服务器运行。CloudBase 云托管仍可作为 staging 或轻量分析验证方案，但不是当前生产首选。

依据 CloudBase 官方文档：

- 云托管支持托管任意语言和框架编写的容器化应用，适合 FastAPI 这类 Python HTTP 服务。
- 云托管服务拥有独立访问域名，支持版本、实例、自动伸缩、灰度发布和回滚。
- HTTP 网关可把统一域名和路径路由到云托管服务，便于小程序侧统一访问。
- 云存储支持图片、文档、音频、视频等非结构化文件，适合作为小程序上传视频与 AGU 输出结果的存储层。
- 云托管存在平台限制：请求超时时间 `60s`，请求包体大小 `20M`，环境内服务/实例/QPS 也有配额限制。

官方文档入口：

- CloudBase 云托管：https://docs.cloudbase.net/run/introduction
- CloudBase 云托管限制：https://docs.cloudbase.net/run/limitation
- CloudBase HTTP 网关：https://docs.cloudbase.net/service/introduce
- CloudBase 云存储：https://docs.cloudbase.net/storage/introduce

#### 当前 AGU 直接部署 CloudBase 适配度

可直接验证：

- `/health`、`/ready` 健康检查。
- 小样本、短视频、`vlm_mode=off` 的异步任务提交。
- `max_frames` 较小的分析任务。
- 通过轮询接口查询当前实例内任务状态。

不建议直接生产使用：

- 大视频直接通过 HTTP 上传或传请求体，受 `20M` 包体限制影响。
- 同步等待完整推理结果，受 `60s` 请求超时影响。
- 多实例自动伸缩生产部署，因为当前 `TaskManager` 是内存态，实例重启或请求打到另一实例时可能查不到任务。
- 输出只写本地 `analysis_outputs/` 和 `output_videos/`，实例重启或扩缩容后不适合作为长期结果存储。
- checkpoint 若直接打进镜像，会增加镜像体积和冷启动时间；若运行时下载，需要设计启动缓存和失败重试。

#### CloudBase 推荐协同形态

第一阶段：CloudBase 与 Docker AGU 联调

- AGU 以 Docker 运行在本地/自有服务器。
- `visual_coach` Rust BFF 作为小程序和管理端统一入口。
- basketball 数据库继续作为展示数据源。
- BFF 或 worker 从 AGU 查询结果后写入 basketball 数据库。

第二阶段：小程序集成验证

- 小程序上传视频到 CloudBase 云存储。
- `visual_coach` Rust BFF 接收 fileID 或临时访问 URL。
- Rust BFF 或 worker 将视频下载/转存到 AGU Docker 可访问的输入目录。
- AGU 只接收任务请求，返回 `task_id`，由前端轮询状态。
- 输出 JSON 和标注视频上传回 CloudBase 云存储，再把 fileID/访问 URL 写入 basketball 数据库。

第三阶段：生产化改造

- 将任务状态从 AGU 内存迁移到 basketball 数据库或 Redis 兼容服务。
- 将任务执行从 FastAPI `BackgroundTasks` 迁移到可恢复任务队列。
- 将输入/输出文件统一迁移到 CloudBase 云存储或 COS。
- 将模型 checkpoint 作为镜像层、对象存储启动下载或专用模型卷管理，禁止依赖临时本地路径。
- 按 CPU/内存峰值配置 Docker 宿主机资源，并根据实际推理耗时决定是否需要 GPU 服务。

#### CloudBase 环境变量建议（仅用于云托管验证）

CloudBase 云托管中通过平台环境变量注入：

```text
BASKETBALL_MODEL_PATH=/app/model_checkpoints/r2plus1d_v3/
BASKETBALL_BASE_MODEL_NAME=best
BASKETBALL_OUTPUT_DIR=/tmp/analysis_outputs
BASKETBALL_VIDEO_OUTPUT_DIR=/tmp/output_videos
BASKETBALL_HOST=0.0.0.0
BASKETBALL_PORT=8765
BASKETBALL_VLM_MODE=off
```

说明：

- `/tmp` 只能用于临时输出，生产结果必须上传到 CloudBase 云存储。
- 线上不提交 `.env`，全部由 CloudBase 控制台或部署配置注入。
- 若启用 VLM，`BASKETBALL_OLLAMA_HOST` 需要指向 CloudBase 环境可访问的模型服务地址。

#### CloudBase 部署后 API 验证

假设 CloudBase 云托管或 HTTP 网关分配的访问地址为：

```bash
export AGU_BASE_URL=https://<cloudbase-agu-domain>
```

健康检查：

```bash
curl -sS "$AGU_BASE_URL/health"
curl -sS "$AGU_BASE_URL/ready"
```

提交短任务：

```bash
curl -sS -X POST "$AGU_BASE_URL/api/v1/analysis/run" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "examples/lebron_shoots.mp4",
    "vlm_mode": "off",
    "max_frames": 60,
    "generate_video": false
  }'
```

查询任务：

```bash
curl -sS "$AGU_BASE_URL/api/v1/analysis/status/<task_id>"
```

CloudBase 验收标准：

- `/health` 与 `/ready` 均返回 `200`。
- 任务提交接口在 `60s` 内返回 `task_id`。
- 轮询接口可查到 `processing` 或 `completed`。
- 若失败，`error` 必须明确指出文件路径、模型、依赖或资源问题。
- CloudBase 日志中无模型加载循环失败、内存不足或实例频繁重启。

## 线上发布步骤

1. 发布前检查

```bash
python scripts/verify_harness.py
```

检查项：

- Python 文件可编译。
- `.env.example` 与 `app/config.py` 中 `BASKETBALL_` 配置一致。
- harness 文档结构完整。

2. 构建 Docker 镜像

```bash
export IMAGE_TAG=agu:<commit_sha>
docker build -t "$IMAGE_TAG" .
```

检查项：

- 镜像构建成功。
- 镜像不包含 `.env`、`dataset/`、`analysis_outputs/`、`output_videos/`、`model_checkpoints/` 大目录。
- checkpoint 通过宿主机 volume 挂载，不直接混入应用镜像。

3. 部署到 staging 宿主机

```bash
docker run -d --name agu-staging \
  --restart unless-stopped \
  -p 8765:8765 \
  -e BASKETBALL_HOST=0.0.0.0 \
  -e BASKETBALL_PORT=8765 \
  -e BASKETBALL_MODEL_PATH=/app/model_checkpoints/r2plus1d_v3/ \
  -e BASKETBALL_OUTPUT_DIR=/app/analysis_outputs \
  -e BASKETBALL_VIDEO_OUTPUT_DIR=/app/output_videos \
  -v /srv/agu-staging/model_checkpoints:/app/model_checkpoints \
  -v /srv/agu-staging/analysis_outputs:/app/analysis_outputs \
  -v /srv/agu-staging/output_videos:/app/output_videos \
  -v /srv/agu-staging/service_inputs:/app/service_inputs \
  "$IMAGE_TAG"
```

检查项：

- 容器持续运行。
- 模型加载完成。
- 输出目录可写。
- checkpoint 路径正确。

4. staging API 验证

```bash
export AGU_BASE_URL=http://<staging-host>:8765

curl -sS "$AGU_BASE_URL/health"
curl -sS "$AGU_BASE_URL/ready"
```

提交任务：

```bash
curl -sS -X POST "$AGU_BASE_URL/api/v1/analysis/run" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "examples/lebron_shoots.mp4",
    "vlm_mode": "off",
    "max_frames": 120,
    "generate_video": false
  }'
```

查询任务：

```bash
curl -sS "$AGU_BASE_URL/api/v1/analysis/status/<task_id>"
```

5. basketball 数据库写入验证

推荐由 `visual_coach` Rust BFF 或 result-sync worker 完成写入，AGU 只提供 API 返回。

验证步骤：

- BFF/worker 调用 `GET {AGU_BASE_URL}/api/v1/analysis/tasks/{task_id}/result`。
- BFF/worker 将任务状态 upsert 到 basketball 数据库 `analysis_tasks`。
- BFF/worker 将 `summary`、`records`、输出文件 URL/fileID 写入 `analysis_results`。
- basketball 小程序读取同一数据库记录，展示任务状态和分析摘要。

验收标准：

- 同一 `task_id` 重复同步不会产生重复记录。
- 失败任务会写入 `status=failed` 和可读 `error`。
- 成功任务会写入 `status=completed`、`summary`、`records`。
- 小程序端不直接访问 AGU 的本地文件路径，只读取数据库中的 URL/fileID。

6. 发布到 production

- 优先灰度或单实例滚动发布。
- 发布后先只执行健康检查，再执行小样本分析任务。
- 观察任务耗时、失败率、模型加载日志和输出目录写入。
- 观察 BFF/worker 写入 basketball 数据库的成功率、幂等冲突和失败重试。

## 部署后线上 API 效果验收

### 必测接口

- `GET /health`
- `GET /ready`
- `POST /api/v1/analysis/run`
- `GET /api/v1/analysis/status/{task_id}`
- `POST /api/v1/analysis/tasks`
- `GET /api/v1/analysis/tasks/{task_id}`
- `GET /api/v1/analysis/tasks/{task_id}/result`

### 验收样本

最小样本：

```json
{
  "video_path": "examples/lebron_shoots.mp4",
  "vlm_mode": "off",
  "max_frames": 120,
  "generate_video": false
}
```

完整样本：

```json
{
  "video_path": "examples/lebron_shoots.mp4",
  "vlm_mode": "low-confidence",
  "max_frames": 180,
  "generate_video": true,
  "tracker_conf_thres": 0.3,
  "tracker_iou_thres": 0.6,
  "tracker_min_appear_ratio": 0.02,
  "tracker_min_appear_abs": 5
}
```

### 成功标准

- 健康检查 `HTTP 200`。
- 任务提交返回 `task_id`。
- 查询接口最终返回 `completed`，或在异常时返回可诊断的 `failed` + `error`。
- 成功任务包含 `result.video`、`result.records`、`result.summary`。
- `generate_video=true` 时，可通过 `/static/videos/...` 访问生成视频。
- JSON 输出可通过 `/static/outputs/...` 访问。
- BFF/worker 可把 `task_id`、`summary`、`records`、输出 URL/fileID 写入 basketball 数据库。
- basketball 小程序可基于数据库记录展示分析状态与摘要。

### 指标门槛

- 健康检查成功率：`100%`。
- 小样本任务提交成功率：`>= 99%`。
- 小样本任务完成成功率：`>= 95%`。
- 单任务失败必须包含可读 `error`。
- 连续 3 次 checkpoint/model 加载失败即停止发布。
- basketball 数据库写入成功率：`>= 99%`。
- `task_id` 幂等写入重复记录数：`0`。

## 回滚条件

- `/health` 或 `/ready` 连续失败。
- 新任务无法提交或持续返回 `5xx`。
- 任务失败率超过 `5%` 且无法归因于输入视频。
- 输出目录不可写导致结果或视频无法生成。
- 模型加载时间或推理耗时显著高于上一版本。
- BFF/worker 无法把结果写入 basketball 数据库。
- 小程序无法读取已完成任务的数据库记录。

回滚后保留：

- 服务启动日志。
- 失败任务请求体。
- `task_id` 与查询响应。
- checkpoint 版本和环境变量快照。
- basketball 数据库写入失败记录和重试日志。

## 与 visual_coach Rust BFF 的联调边界

AGU 线上验证通过后，`visual_coach` Rust BFF 只应调用 AGU 的分析接口：

- `POST {AGU_BASE_URL}/api/v1/analysis/tasks`
- `GET {AGU_BASE_URL}/api/v1/analysis/tasks/{task_id}`
- `GET {AGU_BASE_URL}/api/v1/analysis/tasks/{task_id}/result`

AGU 不接收或处理：

- 用户登录、JWT、RBAC。
- BFF 限流、幂等、统一错误码。
- player_grouping、basketball、ops 聚合。

这些能力由 `visual_coach` Rust BFF 负责验收。
