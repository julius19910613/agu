# AGU

- [中文版本](#中文版本)
- [English Summary](#english-summary)

## 中文版本

### 项目简介

AGU 是一个开源篮球视频动作理解引擎，组合了球员跟踪、片段级动作分类、简单运动特征、可选本地 VLM 复核和标注视频输出。

AGU 的目标不是成为小程序后端或完整篮球 SaaS，而是作为可被其他系统调用的分析引擎：

```text
basketball video -> player tracks -> action clips -> structured JSON + optional annotated video
```

当前仓库提供：

- 基于 FastAPI 的异步视频分析服务。
- 轻量 CLI 客户端：`python -m app.cli`。
- Docker 自托管部署示例。
- 面向 Mac/CPU/MPS 的训练入口 `train_mac.py`。
- 兼容旧流程的脚本：`train.py`、`hybrid_analysis.py`、`hybrid_service.py`。

当前部署目标是 `model_checkpoints/r2plus1d_v3/best.pt`。推理预处理固定匹配 v3 训练分布：OpenCV BGR、resize 到 `112x112`、值域保持 `[0,255]`，不做 RGB 转换、`/255` 或 Kinetics normalize。

### 分析流程

1. 默认把视频切成带 overlap 的 segment，避免动作在切片边界被截断。
2. 在每个 segment 内读取视频并跟踪球员，默认使用 `YOLO` + ByteTrack，也支持 OpenCV legacy trackers。
3. YOLO 默认走 CPU、低帧率和较低 `imgsz` 跟踪；这些加速参数均从 `BASKETBALL_` 配置或请求体读取。
4. 按 `seq_length` 和 `vid_stride` 将球员轨迹切成重叠窗口；分段分析默认可使用更大的 `action_vid_stride` 减少重复推理。
5. 对每个球员窗口运行 R(2+1)D 动作分类，默认优先使用 MPS（可用时）并回退 CPU/CUDA。
6. 对低置信度片段可选调用本地 Ollama VLM 复核，也可对 segment contact sheet 做 VLM audit。
7. 融合模型、VLM 和时序证据，并做 temporal smoothing。
8. 输出 JSON 结果，可选生成标注视频。

### 动作标签

```text
0 block
1 pass
2 run
3 dribble
4 shoot
5 ball in hand
6 defense
7 pick
8 no_action
9 walk
```

### 主要功能

- 异步分析 API：
  - `POST /api/v1/analysis/run` 启动分析任务。
  - `GET /api/v1/analysis/status/{task_id}` 查询任务状态和结果。
- 多种跟踪方式：
  - 默认 `YOLO`，使用 `bytetrack.yaml`。
  - 可配置开源 tracker adapter：`BASKETBALL_TRACKER_BACKEND=bytetrack|botsort|custom`。
  - BoT-SORT/ReID 可通过 `BASKETBALL_YOLO_REID_ENABLED`、`BASKETBALL_YOLO_REID_MODEL` 和自定义 tracker YAML 接入。
  - OpenCV legacy trackers：`CSRT`、`MOSSE`、`KCF` 等。
- 球员身份 embedding：
  - 默认使用 `torchvision_mobilenet_v3_small` 生成 576 维 appearance embedding。
  - `sidecar_hsv_hist` 仅作为轻量 fallback/测试后端保留。
- 模型推理：
  - 从 checkpoint 加载 `R(2+1)D-18`。
- 可选 VLM 复核：
  - `off` / `low-confidence` / `always`。
- 默认长视频分段：
  - `segmented_analysis=true`，通过 `segment_duration_sec` 和 `segment_overlap_sec` 控制 segment。
  - `result.long_video.segments[]` 提供分段统计与 VLM audit 状态。
  - `result.player_identity_features[]` 提供局部轨迹模型 appearance embedding 和 continuity 特征。
  - 可选 `BASKETBALL_JERSEY_NUMBER_VLM_ENABLED=true` 后，`player_identity_features[].jersey_number_candidates[]` 会输出 VLM 读取到的球衣号码候选。
  - `result.long_video.players[]` 提供 segment-local 球员动作汇总和 `appearance_continuity_stitch_v2` 的轻量 `global_player_id` 身份候选。
  - `result.long_video.identity_duplicate_candidates[]` 提供疑似重复 `global_player_id` 的合并审核候选，不自动改写统计。
  - duplicate candidate 会使用采样 frame-level bbox 判断同屏硬冲突和重复框重叠。
  - `result.long_video.event_candidates[]` 提供 `block_candidate`、`rebound_candidate`、`steal_candidate` 事件线索。
- 推理加速：
  - `BASKETBALL_TRACKING_FPS=8.0` 对 YOLO 跟踪做低帧率采样。
  - `BASKETBALL_YOLO_IMGSZ=320` 降低 YOLO 输入尺寸。
  - `BASKETBALL_YOLO_DEVICE=cpu` 避免当前 Mac MPS 跑 YOLO 变慢。
  - `BASKETBALL_R2PLUS1D_DEVICE=mps_if_available` 让 R(2+1)D 在支持时走 MPS。
  - `BASKETBALL_ACTION_VID_STRIDE=24` 用于默认分段分析，减少重复动作窗口。
  - `BASKETBALL_MAX_PLAYERS_PER_SEGMENT=12` 保留出现最稳定的球员轨迹，避免噪声 track 放大推理量。
- 球员技术统计估算：
  - `statistics.points`、`assists`、`rebounds`、`blocks`、`steals`。
  - 当前为 `action_proxy_v1`，不是正式技术统计；points 来自 `shoot`，assists 来自 `pass`。
  - `block` 不再直接计入正式 `statistics.blocks`；会先输出 `block_candidate`，等待球/篮筐/投篮或 VLM 确认。
  - `rebound_candidate` 和 `steal_candidate` 由简化球权状态线索生成，仍需球检测、篮筐检测或 VLM/人工确认。
- 输出文件：
  - JSON：`analysis_outputs/*.json`
  - 视频：`output_videos/*.mp4`
- 静态文件路径：
- `/static/outputs`
- `/static/videos`

### 对外调用边界

AGU 仓库只负责篮球视频分析服务，不承载统一 BFF、鉴权、限流、分组聚合或运营后台 API。统一 BFF/API 网关归属 `visual_coach` 的 Rust 重构实现，由 `visual_coach` 调用 AGU 获取分析任务状态与结果。

AGU 当前提供以下分析接口供外部 BFF 调用：

- `POST /api/v1/analysis/tasks`
- `GET /api/v1/analysis/tasks/{id}`
- `GET /api/v1/analysis/tasks/{id}/result`

这些接口复用 `POST /api/v1/analysis/run` 与 `GET /api/v1/analysis/status/{task_id}`，用于兼容外部统一 API 契约。

### 技术栈

- Python / FastAPI / Pydantic-Settings
- PyTorch / TorchVision
- OpenCV / YOLOv8
- NumPy / scikit-learn
- urllib，用于 Ollama HTTP client
- pytest

### 快速开始

安装服务依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-service.txt
cp .env.example .env
```

启动 API：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

提交示例请求：

```bash
curl -sS -X POST http://127.0.0.1:8765/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "examples/lebron_shoots.mp4",
    "vlm_mode": "off",
    "generate_video": false,
    "segmented_analysis": false,
    "max_frames": 60
  }'
```

使用 CLI：

```bash
python -m app.cli analyze \
  --video examples/lebron_shoots.mp4 \
  --vlm-mode off \
  --max-frames 120 \
  --no-generate-video
```

查询任务：

```bash
python -m app.cli status <task_id>
```

### 目录结构

```text
.
├── app/
│   ├── main.py               FastAPI app 与生命周期
│   ├── config.py             从 .env 读取配置
│   ├── dependencies.py       全局 model/service 依赖
│   ├── analysis/
│   │   ├── router.py         /api/v1/analysis 路由
│   │   ├── schemas.py        请求/返回 schema
│   │   ├── service.py        分析流程编排
│   │   ├── tracking.py       跟踪与窗口裁剪
│   │   ├── inference.py      R(2+1)D 推理，v3 预处理入口
│   │   ├── motion.py         运动特征
│   │   ├── vlm.py            Ollama VLM 验证
│   │   ├── fusion.py         模型 + VLM 融合 + 平滑
│   │   └── task_manager.py   内存任务状态
│   ├── video/
│   │   └── writer.py         标注视频输出
│   └── models/
│       ├── r2plus1d.py       best.pt 加载器
│       └── preprocessing.py  v3 对齐预处理
├── dataset.py                SpaceJam 数据集包装
├── train_mac.py              推荐训练脚本
├── train.py                  历史训练脚本
├── hybrid_analysis.py        旧的一体化脚本
├── hybrid_service.py         旧的简易 HTTP 服务
├── scripts/                  辅助脚本
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

以下目录是运行时数据，默认不入库：

```text
dataset/
model_checkpoints/
analysis_outputs/
output_videos/
```

### 环境与配置

推荐使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements-service.txt
```

依赖文件分工：

| 文件 | 用途 |
| --- | --- |
| `requirements-service.txt` | FastAPI 服务、推理和 Docker 部署 |
| `requirements-training.txt` | 训练、数据处理和实验 |
| `requirements-dev.txt` | 测试、类型检查和开发工具 |
| `requirements.txt` | 历史快照，保留用于兼容旧环境 |

### 配置项

配置通过前缀 `BASKETBALL_` 从环境变量或 `.env` 读取。示例：

```text
BASKETBALL_MODEL_PATH=model_checkpoints/r2plus1d_v3/
BASKETBALL_BASE_MODEL_NAME=best
BASKETBALL_START_EPOCH=0
BASKETBALL_LR=0.0001
BASKETBALL_NUM_CLASSES=10
BASKETBALL_SEQ_LENGTH=16
BASKETBALL_VID_STRIDE=8
BASKETBALL_ACTION_VID_STRIDE=24
BASKETBALL_BATCH_SIZE=8
BASKETBALL_R2PLUS1D_DEVICE=mps_if_available
BASKETBALL_YOLO_DEVICE=cpu
BASKETBALL_TRACKING_FPS=8.0
BASKETBALL_YOLO_IMGSZ=320
BASKETBALL_MAX_PLAYERS_PER_SEGMENT=12
BASKETBALL_TORCH_NUM_THREADS=10
BASKETBALL_PROGRESS_LOG=true
BASKETBALL_TRACKER_TYPE=YOLO
BASKETBALL_TRACKER_BACKEND=bytetrack
BASKETBALL_YOLO_TRACKER_CONFIG=
BASKETBALL_YOLO_REID_ENABLED=false
BASKETBALL_YOLO_REID_MODEL=auto
BASKETBALL_IDENTITY_EMBEDDING_BACKEND=torchvision_mobilenet_v3_small
BASKETBALL_IDENTITY_EMBEDDING_WEIGHTS=default
BASKETBALL_IDENTITY_EMBEDDING_DEVICE=mps_if_available
BASKETBALL_IDENTITY_EMBEDDING_BATCH_SIZE=16
BASKETBALL_IDENTITY_EMBEDDING_ALLOW_FALLBACK=true
BASKETBALL_VLM_MODE=low-confidence
BASKETBALL_OLLAMA_MODEL=qwen3-vl:4b
BASKETBALL_OLLAMA_HOST=http://127.0.0.1:11434
BASKETBALL_OLLAMA_TIMEOUT=45.0
BASKETBALL_JERSEY_NUMBER_VLM_ENABLED=false
BASKETBALL_JERSEY_NUMBER_VLM_FRAMES=2
BASKETBALL_LOW_CONFIDENCE=0.45
BASKETBALL_HIGH_CONFIDENCE=0.70
BASKETBALL_SMOOTHING_CONFIDENCE=0.60
BASKETBALL_OUTPUT_DIR=analysis_outputs
BASKETBALL_VIDEO_OUTPUT_DIR=output_videos
BASKETBALL_HOST=127.0.0.1
BASKETBALL_PORT=8765
```

### 启动 API

```bash
cp .env.example .env
./venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

健康检查：

```bash
curl http://127.0.0.1:8765/health
```

本地与线上部署验收方案见 `docs/deploy-and-verify.md`。

API 契约见 `docs/api.md`，模型说明见 `docs/model-card.md`。
权重说明见 `docs/checkpoints.md`，扩展接口见 `docs/extensions.md`。
公开发布来源、许可证、数据集与权重策略见 `docs/release-notes.md` 和 `docs/datasets.md`。

### 提交分析任务

```bash
curl -X POST http://127.0.0.1:8765/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "examples/lebron_shoots.mp4",
    "vlm_mode": "low-confidence",
    "max_frames": 180,
    "generate_video": true,
    "tracker_conf_thres": 0.3,
    "tracker_iou_thres": 0.6,
    "tracker_min_appear_ratio": 0.02,
    "tracker_min_appear_abs": 5,
    "segmented_analysis": true,
    "segment_duration_sec": 15.0,
    "segment_overlap_sec": 2.0,
    "vlm_audit": true
  }'
```

返回示例：

```text
{"task_id":"...","status":"pending","message":"Analysis started asynchronously..."}
```

查询任务状态：

```bash
curl http://127.0.0.1:8765/api/v1/analysis/status/<task_id>
```

### 请求参数

```text
video_path                必填，视频路径
vlm_mode                  low-confidence | off | always
boxes_file                可选，初始 boxes JSON
max_frames                可选，限制读取帧数
generate_video            是否生成标注视频
tracker_conf_thres        默认 0.3
tracker_iou_thres         默认 0.6
tracker_min_appear_ratio  默认 0.02
tracker_min_appear_abs    默认 5
segmented_analysis        默认 true，统一走带 overlap 的分段分析
long_video_mode           兼容字段，true 时也启用分段分析
segment_duration_sec      默认 15.0
segment_overlap_sec       默认 2.0
segment_start_sec         默认 0.0，可用于局部 smoke
segment_end_sec           可选，局部分析结束时间
max_segments              可选，限制分段数量
vlm_audit                 默认 true，对 segment contact sheet 做 VLM audit
vlm_audit_frames          默认 6
vid_stride                可选，覆盖默认窗口步长
low_confidence            可选，覆盖低置信度阈值
high_confidence           可选，覆盖高置信度阈值
```

### 训练

推荐训练命令：

```bash
./venv/bin/python train_mac.py \
  --device mps \
  --batch-size 2 \
  --epochs 20 \
  --lr 1e-4 \
  --annotation-path dataset/annotation_dict.json \
  --augmented-path dataset/augmented_annotation_dict.json \
  --video-dir dataset/examples/ \
  --augmented-dir dataset/augmented-examples/ \
  --model-dir model_checkpoints/r2plus1d_v3/ \
  --history-path histories/history_r2plus1d_v3.txt
```

续训：

```bash
./venv/bin/python train_mac.py \
  --resume model_checkpoints/r2plus1d_v3/best.pt \
  --epochs 30
```

常用训练参数：

```text
--accum-steps
--best-metric
--early-stop-patience
--weight-decay
--label-smoothing
--fc-dropout
--no-freeze-bn
--no-class-weights
--no-sampler
--use-augmentation
--force-resplit
```

### 辅助脚本

```bash
./venv/bin/python scripts/check_training.py --history-path histories/history_r2plus1d_v3.txt
./venv/bin/python scripts/gen_augmented.py --minority-only --multiplier 3
./venv/bin/python scripts/gen_splits.py --annotation-path dataset/annotation_dict.json
./venv/bin/python scripts/manual_test_run.py
./venv/bin/python scripts/build_identity_duplicate_report.py \
  --analysis-json analysis_outputs/<analysis-id>.json \
  --output-json analysis_outputs/perf_runs/identity-duplicate-report.json \
  --screenshot-dir analysis_outputs/player_stat_screenshots_20260622 \
  --contact-sheet analysis_outputs/perf_runs/identity-duplicate-review.jpg
```

### Legacy 入口状态

| 文件 | 状态 | 建议 |
| --- | --- | --- |
| `app/main.py` | 主服务入口 | 推荐 |
| `python -m app.cli` | API CLI 客户端 | 推荐 |
| `train_mac.py` | 当前训练入口 | 推荐 |
| `train.py` | 历史训练入口 | 暂保留，后续合并 |
| `hybrid_analysis.py` | 历史一体化脚本 | 后续转正式 CLI 或归档 |
| `hybrid_service.py` | 历史简易 HTTP 服务 | FastAPI 已覆盖，后续归档 |

### 测试

```bash
./venv/bin/python scripts/smoke_open_source.py
./venv/bin/python -m pytest tests -q
./venv/bin/python -m compileall app train_mac.py scripts
./venv/bin/python scripts/validate_open_source_baseline.py
```

### 贡献与许可证

- 贡献指南：`CONTRIBUTING.md`
- 许可证：`LICENSE`
- 公开发布说明：`docs/release-notes.md`
- 数据集获取说明：`docs/datasets.md`

当前验证状态以 `docs/harness/TASK-BOARD.md` 的最新任务记录为准。

### 安全与运行时约束

- `POST /api/v1/analysis/run` 会校验 `video_path`，防止简单路径穿越。
- 任务状态和结果保存在内存 `TaskManager` 中，进程重启后会丢失。
- 数据集、模型权重和输出目录默认不入库。
- `app/models/preprocessing.py` 已按 v3 部署口径实现：BGR、`112x112`、`[0,255]`，不 `/255`，无 Kinetics normalize。

### 开源状态与待补充项

- 仓库包含 `LICENSE`，公开发布前仍需复核第三方依赖、数据集和模型权重的许可证边界。
- 数据集与权重不随仓库分发，需要自行准备 SpaceJam 标注和视频。
- `requirements.txt` 不是严格 lockfile。
- 若准备公开发布，需要补充来源声明、许可证、数据集获取说明和权重分发策略。

## English Summary

AGU is an open-source basketball video understanding engine. It focuses on video analysis only: player tracking, action classification, segmented long-video analysis, optional local VLM review, identity evidence, duplicate-identity review candidates, event candidates, and optional annotated outputs. It is not the BFF, authentication layer, rate limiter, product backend, or operations console.

Current API entry points:

- `POST /api/v1/analysis/run` starts an async analysis task.
- `GET /api/v1/analysis/status/{task_id}` returns task progress and result.
- `/api/v1/analysis/tasks*` aliases are kept for external gateway compatibility.

Current identity and statistics behavior:

- Long videos are analyzed through overlapped segments by default.
- `player_identity_features[]` exposes model/fallback appearance embeddings and continuity evidence.
- `long_video.players[]` exposes lightweight `global_player_id` candidates.
- `long_video.identity_duplicate_candidates[]` exposes review-only duplicate-ID merge candidates and does not rewrite statistics automatically.
- `statistics.points`, `assists`, `rebounds`, `blocks`, and `steals` are action-proxy estimates, not official box-score truth. Block, rebound, and steal evidence should be confirmed through event candidates, ball/rim/possession evidence, VLM, or human review.

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-service.txt
cp .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Useful docs:

- API contract: `docs/api.md`
- Model card: `docs/model-card.md`
- Checkpoints: `docs/checkpoints.md`
- Extensions: `docs/extensions.md`
- Open-source release notes: `docs/release-notes.md`
- Harness workflow and task board: `docs/harness/WORKFLOW.md`, `docs/harness/TASK-BOARD.md`
