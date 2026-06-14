# AGU

- [中文版本](#中文版本)
- [English Version](#english-version)

## 中文版本

### 项目简介

AGU 是一个面向篮球视频的动作与比赛理解项目，组合了球员跟踪、片段级动作分类、简单运动特征和可选的本地 VLM 复核。

当前仓库提供：

- 基于 FastAPI 的异步视频分析服务。
- 面向 Mac/CPU/MPS 的训练入口 `train_mac.py`。
- 兼容旧流程的脚本：`train.py`、`hybrid_analysis.py`、`hybrid_service.py`。

当前部署目标是 `model_checkpoints/r2plus1d_v3/best.pt`。推理预处理固定匹配 v3 训练分布：OpenCV BGR、resize 到 `112x112`、值域保持 `[0,255]`，不做 RGB 转换、`/255` 或 Kinetics normalize。

### 分析流程

1. 读取视频并跟踪球员，默认使用 `YOLO` + ByteTrack，也支持 OpenCV legacy trackers。
2. 按 `seq_length` 和 `vid_stride` 将球员轨迹切成重叠窗口。
3. 对每个球员窗口运行 R(2+1)D 动作分类。
4. 对低置信度片段可选调用本地 Ollama VLM 复核。
5. 融合模型、VLM 和时序证据，并做 temporal smoothing。
6. 输出 JSON 结果，可选生成标注视频。

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
  - OpenCV legacy trackers：`CSRT`、`MOSSE`、`KCF` 等。
- 模型推理：
  - 从 checkpoint 加载 `R(2+1)D-18`。
- 可选 VLM 复核：
  - `off` / `low-confidence` / `always`。
- 输出文件：
  - JSON：`analysis_outputs/*.json`
  - 视频：`output_videos/*.mp4`
- 静态文件路径：
  - `/static/outputs`
  - `/static/videos`

### 技术栈

- Python / FastAPI / Pydantic-Settings
- PyTorch / TorchVision
- OpenCV / YOLOv8
- NumPy / scikit-learn
- urllib，用于 Ollama HTTP client
- pytest

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

也可以直接使用仓库已有环境：

```bash
source ./venv/bin/activate
```

安装依赖：

```bash
./venv/bin/pip install -r requirements.txt

# 如果不使用旧 requirements 快照，可按核心依赖安装
./venv/bin/pip install fastapi uvicorn torch torchvision opencv-contrib-python ultralytics scikit-learn pydantic-settings tqdm easydict numpy
```

`requirements.txt` 是历史快照，不是严格可复现 lockfile。当前 FastAPI、YOLO、Pydantic-Settings 等服务依赖建议按平台补齐版本。

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
BASKETBALL_BATCH_SIZE=8
BASKETBALL_TRACKER_TYPE=YOLO
BASKETBALL_VLM_MODE=low-confidence
BASKETBALL_OLLAMA_MODEL=qwen3-vl:4b
BASKETBALL_OLLAMA_HOST=http://127.0.0.1:11434
BASKETBALL_OLLAMA_TIMEOUT=45.0
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
    "tracker_min_appear_abs": 5
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
```

### 测试

```bash
./venv/bin/python -m pytest tests -q
./venv/bin/python -m compileall app train_mac.py scripts
```

当前已验证状态：

```text
38 passed
```

### 安全与运行时约束

- `POST /api/v1/analysis/run` 会校验 `video_path`，防止简单路径穿越。
- 任务状态和结果保存在内存 `TaskManager` 中，进程重启后会丢失。
- 数据集、模型权重和输出目录默认不入库。
- `app/models/preprocessing.py` 已按 v3 部署口径实现：BGR、`112x112`、`[0,255]`，不 `/255`，无 Kinetics normalize。

### 开源状态与待补充项

- 当前仓库未包含 `LICENSE` 文件。
- 数据集与权重不随仓库分发，需要自行准备 SpaceJam 标注和视频。
- `requirements.txt` 不是严格 lockfile。
- 若准备公开发布，需要补充来源声明、许可证、数据集获取说明和权重分发策略。

## English Version

### Overview

AGU is a basketball action and game understanding project that combines player tracking, clip-level action classification, lightweight motion features, and optional local VLM review.

This repository provides:

- A FastAPI service for asynchronous video analysis.
- A Mac/CPU/MPS-friendly training entrypoint, `train_mac.py`.
- Legacy compatibility scripts: `train.py`, `hybrid_analysis.py`, and `hybrid_service.py`.

The current deployment target is `model_checkpoints/r2plus1d_v3/best.pt`. Inference preprocessing is fixed to the v3 training distribution: OpenCV BGR, resize to `112x112`, values remain in `[0,255]`, with no RGB conversion, no `/255`, and no Kinetics normalization.

### Analysis Pipeline

1. Load the video and track players. The default tracker is `YOLO` + ByteTrack, with OpenCV legacy trackers available as fallbacks.
2. Split player trajectories into overlapping windows using `seq_length` and `vid_stride`.
3. Run R(2+1)D action classification for each player window.
4. Optionally call a local Ollama VLM for low-confidence clips.
5. Fuse model, VLM, and temporal evidence, then apply temporal smoothing.
6. Emit JSON output and optionally write an annotated video.

### Action Labels

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

### Features

- Async analysis API:
  - `POST /api/v1/analysis/run` starts an analysis task.
  - `GET /api/v1/analysis/status/{task_id}` polls task status and results.
- Multiple trackers:
  - Default `YOLO`, using `bytetrack.yaml`.
  - OpenCV legacy trackers such as `CSRT`, `MOSSE`, and `KCF`.
- Model inference:
  - Loads `R(2+1)D-18` from a checkpoint.
- Optional VLM review:
  - `off` / `low-confidence` / `always`.
- Output files:
  - JSON: `analysis_outputs/*.json`
  - Video: `output_videos/*.mp4`
- Static routes:
  - `/static/outputs`
  - `/static/videos`

### Tech Stack

- Python / FastAPI / Pydantic-Settings
- PyTorch / TorchVision
- OpenCV / YOLOv8
- NumPy / scikit-learn
- urllib for the Ollama HTTP client
- pytest

### Repository Layout

```text
.
├── app/
│   ├── main.py               FastAPI app and lifespan
│   ├── config.py             Settings loaded from .env
│   ├── dependencies.py       Global model/service dependencies
│   ├── analysis/
│   │   ├── router.py         /api/v1/analysis routes
│   │   ├── schemas.py        Request/response schemas
│   │   ├── service.py        Analysis orchestration
│   │   ├── tracking.py       Tracking and window cropping
│   │   ├── inference.py      R(2+1)D inference and v3 preprocessing
│   │   ├── motion.py         Motion features
│   │   ├── vlm.py            Ollama VLM verifier
│   │   ├── fusion.py         Model + VLM fusion and smoothing
│   │   └── task_manager.py   In-memory task state
│   ├── video/
│   │   └── writer.py         Annotated video output
│   └── models/
│       ├── r2plus1d.py       best.pt loader
│       └── preprocessing.py  v3-aligned preprocessing
├── dataset.py                SpaceJam dataset wrapper
├── train_mac.py              Recommended training script
├── train.py                  Historical training script
├── hybrid_analysis.py        Legacy all-in-one script
├── hybrid_service.py         Legacy lightweight HTTP service
├── scripts/                  Helper scripts
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

Runtime data directories are ignored by default:

```text
dataset/
model_checkpoints/
analysis_outputs/
output_videos/
```

### Environment And Setup

Recommended virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

You can also use the existing repository environment:

```bash
source ./venv/bin/activate
```

Install dependencies:

```bash
./venv/bin/pip install -r requirements.txt

# If you do not want to use the historical requirements snapshot:
./venv/bin/pip install fastapi uvicorn torch torchvision opencv-contrib-python ultralytics scikit-learn pydantic-settings tqdm easydict numpy
```

`requirements.txt` is a historical snapshot, not a strict reproducible lockfile. Service dependencies such as FastAPI, YOLO, and Pydantic-Settings should be pinned per target platform before production use.

### Configuration

Settings are read from environment variables or `.env` using the `BASKETBALL_` prefix. Example:

```text
BASKETBALL_MODEL_PATH=model_checkpoints/r2plus1d_v3/
BASKETBALL_BASE_MODEL_NAME=best
BASKETBALL_START_EPOCH=0
BASKETBALL_LR=0.0001
BASKETBALL_NUM_CLASSES=10
BASKETBALL_SEQ_LENGTH=16
BASKETBALL_VID_STRIDE=8
BASKETBALL_BATCH_SIZE=8
BASKETBALL_TRACKER_TYPE=YOLO
BASKETBALL_VLM_MODE=low-confidence
BASKETBALL_OLLAMA_MODEL=qwen3-vl:4b
BASKETBALL_OLLAMA_HOST=http://127.0.0.1:11434
BASKETBALL_OLLAMA_TIMEOUT=45.0
BASKETBALL_LOW_CONFIDENCE=0.45
BASKETBALL_HIGH_CONFIDENCE=0.70
BASKETBALL_SMOOTHING_CONFIDENCE=0.60
BASKETBALL_OUTPUT_DIR=analysis_outputs
BASKETBALL_VIDEO_OUTPUT_DIR=output_videos
BASKETBALL_HOST=127.0.0.1
BASKETBALL_PORT=8765
```

### Start The API

```bash
cp .env.example .env
./venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```

### Submit An Analysis Task

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
    "tracker_min_appear_abs": 5
  }'
```

Example response:

```text
{"task_id":"...","status":"pending","message":"Analysis started asynchronously..."}
```

Poll task status:

```bash
curl http://127.0.0.1:8765/api/v1/analysis/status/<task_id>
```

### Request Parameters

```text
video_path                Required video path
vlm_mode                  low-confidence | off | always
boxes_file                Optional initial boxes JSON
max_frames                Optional frame limit
generate_video            Whether to generate an annotated video
tracker_conf_thres        Default 0.3
tracker_iou_thres         Default 0.6
tracker_min_appear_ratio  Default 0.02
tracker_min_appear_abs    Default 5
vid_stride                Optional window stride override
low_confidence            Optional low-confidence threshold override
high_confidence           Optional high-confidence threshold override
```

### Training

Recommended training command:

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

Resume training:

```bash
./venv/bin/python train_mac.py \
  --resume model_checkpoints/r2plus1d_v3/best.pt \
  --epochs 30
```

Common training flags:

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

### Helper Scripts

```bash
./venv/bin/python scripts/check_training.py --history-path histories/history_r2plus1d_v3.txt
./venv/bin/python scripts/gen_augmented.py --minority-only --multiplier 3
./venv/bin/python scripts/gen_splits.py --annotation-path dataset/annotation_dict.json
./venv/bin/python scripts/manual_test_run.py
```

### Testing

```bash
./venv/bin/python -m pytest tests -q
./venv/bin/python -m compileall app train_mac.py scripts
```

Current verified status:

```text
38 passed
```

### Safety And Runtime Notes

- `POST /api/v1/analysis/run` validates `video_path` to prevent simple path traversal.
- Task state and results are stored in the in-memory `TaskManager`; they are lost after process restart.
- Datasets, model weights, and output directories are ignored by default.
- `app/models/preprocessing.py` follows the deployed v3 preprocessing contract: BGR, `112x112`, `[0,255]`, no `/255`, and no Kinetics normalization.

### Open Source Readiness

- This repository currently has no `LICENSE` file.
- Datasets and model weights are not distributed with the repository. SpaceJam annotations and videos must be prepared separately.
- `requirements.txt` is not a strict lockfile.
- Before public release, add provenance notes, licensing, dataset acquisition instructions, and a model-weight distribution policy.
