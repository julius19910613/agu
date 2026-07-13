# AGU 开源价值评估与项目剥离建议

> 2026-07-13 P0/P1 update: AGU remains positioned as an extensible basketball
> video analysis engine/reference pipeline. The repository now has installable
> package metadata, versioned CLI, typed pipeline hooks, entry-point plugin
> discovery and diagnostics, response provenance manifests, optional dependency
> groups, a public contract benchmark, CI/release workflows, SBOM/license
> governance, task cancellation/retry/deadlines, configuration profiles, and
> community/security artifacts. This does not broaden AGU into a general CV
> framework or product BFF.

## 结论

AGU 适合作为一个开源的篮球视频分析与动作理解项目继续发展，但开源定位应收敛为：

```text
AGU = basketball video action understanding engine
```

也就是保留视频分析、球员跟踪、动作识别、时序片段、模型推理、训练复现和标准化输出能力。其他与私有业务系统强绑定的内容，例如 `basketball` 数据库写入、`visual_coach` BFF、微信小程序展示、CloudBase 业务环境、私有部署流程，都应从 AGU 剥离。

最适合开源和长期发展的核心是：

- 篮球视频动作识别 pipeline。
- R(2+1)D 推理与预处理契约。
- YOLO/ByteTrack 球员跟踪与片段切分。
- 模型、VLM、运动特征融合。
- 可复现实验训练脚本和数据处理工具。
- 标准 JSON 输出和可选标注视频生成。
- FastAPI 形式的轻量分析服务。

不适合留在 AGU 开源仓库的是：

- 面向 `basketball` 小程序的数据库写入。
- `visual_coach` Rust BFF、鉴权、限流、聚合接口。
- 私有部署、内网地址、CloudBase 环境配置。
- 具体业务库 schema、租户、用户、角色、运营后台逻辑。
- 大型模型权重、数据集、生成结果、私有视频样本。

## AGU 当前价值评估

### 1. 技术价值

AGU 当前已经具备完整的“视频输入 -> 球员跟踪 -> clip 切分 -> 动作分类 -> 结果融合 -> JSON/视频输出”链路。这比单个模型 demo 更有开源价值，因为它覆盖真实视频分析中最难串起来的工程路径。

高价值模块：

| 模块 | 当前价值 | 开源保留建议 |
| --- | --- | --- |
| `app/analysis/service.py` | 串联跟踪、推理、融合、输出 | 保留，作为核心 pipeline |
| `app/analysis/tracking.py` | 球员检测/跟踪与窗口生成 | 保留，重点完善文档和可替换 tracker |
| `app/analysis/inference.py` | R(2+1)D 推理入口 | 保留，稳定 v3 预处理契约 |
| `app/models/preprocessing.py` | 推理预处理一致性 | 保留，作为模型可复现关键 |
| `app/models/r2plus1d.py` | 模型加载与 checkpoint 适配 | 保留，补充权重下载/校验说明 |
| `app/analysis/fusion.py` | 多信号融合与平滑 | 保留，是项目差异化亮点 |
| `app/analysis/vlm.py` | 本地 VLM 复核 | 保留为可选插件，默认关闭 |
| `app/video/writer.py` | 标注视频输出 | 保留，便于 demo 和调试 |
| `app/analysis/router.py` | 轻量 API 服务 | 保留，但只做分析 API |

### 2. 产品价值

AGU 不应定位成完整篮球 SaaS 或小程序后端。它更适合成为一个“可被其他系统调用的分析引擎”。

适合开源展示的用户价值：

- 上传或指定一个篮球视频，得到结构化动作分析。
- 生成可视化标注视频，帮助理解模型输出。
- 复现训练和推理流程，方便研究者改模型。
- 在自己的系统里通过 API 调用 AGU。

不适合 AGU 自身承担的用户价值：

- 球员档案管理。
- 比赛管理。
- 小程序登录和权限。
- 教练后台人工复核工作流。
- 分组推荐、队伍分析、运营统计。

### 3. 社区传播价值

AGU 开源后最容易吸引贡献的方向：

- 替换更强的视频模型，例如 VideoMAE、Timesformer、SlowFast。
- 增加更多 tracker 或检测器。
- 改进动作标签和评估指标。
- 支持更多篮球数据集。
- 改进推理速度、Docker 部署和示例数据。
- 增加 notebook 或 CLI demo。

社区不太会参与、且容易污染开源项目的方向：

- 私有小程序数据库 schema。
- 业务后台 API。
- 某个组织的部署环境和内网拓扑。
- CloudBase/微信账号权限配置。

## 建议保留在 AGU 的功能

### P0：必须保留

- `app/analysis/` 核心分析链路。
- `app/models/` 模型定义、加载、预处理。
- `app/video/` 标注视频输出。
- `app/main.py` FastAPI 服务入口。
- `app/config.py` 基础配置。
- `tests/` 中覆盖推理、API、任务状态、训练韧性的测试。
- `train_mac.py`、`dataset.py`、`utils/metrics.py`、`utils/checkpoints.py`。
- `.env.example`、`README.md`、`docs/training-plan.md`、`docs/optimization-plan.md`。
- Docker 打包能力，但只作为通用部署示例。

### P1：建议保留但需要清理

- `train.py`：保留为 legacy 或统一到 `train_mac.py`，避免两个训练入口长期分叉。
- `hybrid_analysis.py`：保留为 legacy CLI，或迁移为正式 CLI 后删除旧脚本。
- `hybrid_service.py`：若 FastAPI 已覆盖，应标记废弃或剥离。
- `analysis/data_visualization.py`、`analysis/video_analysis.py`：如果是可复用可视化工具，迁移到 `scripts/` 或 `examples/`；如果依赖私有数据，剥离。
- `augment_videos.py`、`make_smoke_data.py`：保留为数据准备工具，但补文档。
- `scratch/create_dummy_weights.py`：迁移到 `scripts/` 或测试 fixture 工具。

### P2：只保留文档级说明

- 与 `visual_coach` 的调用边界。
- 与 `basketball` 小程序的集成边界。
- Docker 部署示例。

这些内容只需要说明“如何被外部系统调用”，不应保留外部系统的实现细节。

## 建议剥离出 AGU 的内容

### 1. 业务系统集成

剥离目标：

- `basketball` 数据库写入。
- `visual_coach` Rust BFF。
- `player_grouping` 分组分析。
- 微信小程序 API、登录、用户态。
- CloudBase 环境、数据库集合、文件存储规则。

归属建议：

| 内容 | 归属项目 |
| --- | --- |
| Rust BFF / API 网关 | `visual_coach` |
| 鉴权、RBAC、限流、错误码统一 | `visual_coach` |
| basketball 数据库写入和字段映射 | `visual_coach` 或 `result-sync` worker |
| 小程序展示与交互 | `basketball` |
| 分组算法和分析 | `player_grouping` |
| CloudBase 环境和集合配置 | `basketball` / `visual_coach` |

### 2. 私有部署策略

剥离目标：

- 具体线上域名。
- 具体 CloudBase 环境 ID。
- 具体数据库集合名，如果不是开源 demo。
- 真实视频、真实输出、真实 checkpoint。
- 运维回滚脚本中的私有路径。

保留方式：

- 只保留通用 Docker 示例。
- 只保留 `.env.example`。
- 只保留 `docs/deploy-and-verify.md` 中“自托管示例”和“外部系统集成边界”。

### 3. 过时或重复实验代码

建议剥离或归档：

- `C3D.py`：如果不再作为主模型，迁移到 `archive/` 或删除。
- `REVIEW_OUTPUT.md`、`CODE_REVIEW.md`：若是一次性评审材料，迁移到 `docs/archive/`。
- `claude.md`、`gemini.md`：若是临时协作记录，剥离出开源仓库。
- `gateway-rs/`：已不属于 AGU，应完全移出。
- `__pycache__/`、`.pytest_cache/`：不应进入开源仓库。

## 目标开源项目边界

### AGU 应该是什么

AGU 是一个可独立运行的篮球视频分析引擎：

- 输入：本地视频路径、可选 boxes 文件、推理参数。
- 输出：结构化 JSON、任务状态、可选标注视频。
- 部署：Python/FastAPI、Docker、自托管。
- 训练：提供可复现训练入口和数据准备说明。

### AGU 不应该是什么

AGU 不是：

- 小程序后端。
- 业务数据库写入服务。
- 统一 API 网关。
- 教练运营后台。
- 球员分组推荐系统。
- 私有模型权重分发仓库。

## 建议目标目录结构

```text
.
├── app/
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── analysis/
│   ├── models/
│   └── video/
├── scripts/
│   ├── check_training.py
│   ├── gen_splits.py
│   ├── gen_augmented.py
│   ├── make_smoke_data.py
│   └── create_dummy_weights.py
├── examples/
│   ├── sample_request.json
│   └── README.md
├── tests/
├── docs/
│   ├── api.md
│   ├── training-plan.md
│   ├── optimization-plan.md
│   ├── deploy-and-verify.md
│   └── open-source-scope-assessment.md
├── Dockerfile
├── requirements-service.txt
├── requirements-dev.txt
├── .env.example
├── README.md
└── LICENSE
```

## 剥离路线图

### Phase 1：文档和边界收敛

- 更新 README，把 AGU 定位为 open-source analysis engine。
- 在 README 中移除 `basketball`、`visual_coach`、CloudBase 私有部署的强绑定叙述。
- 保留“外部系统如何调用 AGU”的中性接口说明。
- 在 `docs/open-source-scope-assessment.md` 记录保留/剥离边界。

### Phase 2：文件清理

- 删除或迁出 `gateway-rs/`。
- 删除 `__pycache__/`、`.pytest_cache/`。
- 将一次性评审文档迁入 `docs/archive/` 或剥离。
- 将 scratch 工具迁入 `scripts/`。
- 将 legacy 脚本标注废弃状态。

### Phase 3：开源体验补齐

- 增加 `LICENSE`。已完成：`LICENSE`。
- 增加 `CONTRIBUTING.md`。已完成：`CONTRIBUTING.md`。
- 增加 `examples/sample_request.json`。已完成：`examples/sample_request.json`。
- 增加模型权重下载说明和 checksum 约定。已完成：`docs/checkpoints.md`。
- 增加一键 smoke test。已完成：`scripts/smoke_open_source.py`。
- 将 `requirements.txt` 拆分为 service/dev/training 三类依赖。已完成：`requirements-service.txt`、`requirements-training.txt`、`requirements-dev.txt`。

### Phase 4：能力增强

- 提供 CLI：`agu analyze --video ...`。已完成第一版：`python -m app.cli analyze --video ...`。
- 提供模型注册机制，支持替换 R(2+1)D。已完成第一版：`app/models/registry.py`。
- 提供 tracker 插件接口。已完成第一版：`app/analysis/tracker_registry.py`。
- 支持将输出写到本地、S3/COS/CloudBase Storage 等可选 backend。已完成本地 backend：`app/storage/backends.py`。
- 支持可恢复任务队列，但保持业务数据库无关。

## 当前保留/剥离清单

### 保留

- `app/`
- `tests/`
- `utils/`
- `scripts/verify_harness.py`
- `scripts/check_training.py`
- `scripts/gen_splits.py`
- `scripts/gen_augmented.py`
- `train_mac.py`
- `dataset.py`
- `Dockerfile`
- `requirements-service.txt`
- `.env.example`
- `docs/training-plan.md`
- `docs/optimization-plan.md`
- `docs/deploy-and-verify.md`

### 评估后处理

- `train.py`：合并或标记 legacy。
- `hybrid_analysis.py`：转 CLI 或标记 legacy。
- `hybrid_service.py`：FastAPI 已覆盖，建议剥离。
- `analysis/` 根目录：检查是否仍有复用价值，必要时迁入 `scripts/`。
- `C3D.py`：迁入 archive 或删除。
- `CODE_REVIEW.md`、`REVIEW_OUTPUT.md`：迁入 `docs/archive/` 或剥离。
- `claude.md`、`gemini.md`：剥离。

### 剥离

- `gateway-rs/`
- 私有业务库写入逻辑。
- `visual_coach` 相关 BFF 实现。
- `basketball` 小程序业务字段映射。
- 真实 dataset、checkpoint、输出视频、输出 JSON。
- 缓存目录和生成目录。

## 风险

- 如果过度剥离训练脚本，会削弱开源项目的可复现价值。
- 如果保留太多业务集成，会让 AGU 失去清晰定位，社区难以参与。
- 如果不提供示例 checkpoint 或下载方式，用户很难快速体验。
- 如果 README 继续强调内部系统融合，开源用户会误以为这是一个私有业务平台。

## 建议最终定位文案

AGU is an open-source basketball video action understanding engine. It combines player tracking, clip-level action recognition, motion features, optional local VLM review, temporal fusion, and annotated video output behind a lightweight FastAPI service.

中文定位：

AGU 是一个开源篮球视频动作理解引擎，提供球员跟踪、片段级动作识别、运动特征、可选本地 VLM 复核、时序融合和标注视频输出能力，可作为研究项目、训练基线或外部业务系统的分析服务使用。

## GitHub 相近项目对比

更新时间：2026-06-17。Star 数只作为社区热度参考，不作为技术质量判断。

### 代表项目

| 项目 | Stars | 定位 | AGU 对比 |
| --- | ---: | --- | --- |
| `open-mmlab/mmaction2` | 5066 | 通用视频理解 toolbox 和 benchmark | AGU 不应与其竞争框架完整度，应借鉴其模型注册、配置、数据集和评估设计 |
| `facebookresearch/pytorchvideo` | 3565 | 视频理解研究库 | AGU 可复用其模型/transform 思路，但应保持篮球分析应用层定位 |
| `SoccerNet/sn-spotting` | 100 | 足球 action spotting challenge 基线 | AGU 可借鉴 action spotting 的任务定义、评估指标和 benchmark 组织方式 |
| `lRomul/ball-action-spotting` | 137 | SoccerNet ball action spotting 比赛方案 | AGU 可学习其事件检测范式，但需要落到篮球动作和球员级结果 |
| `abdullahtarek/basketball_analysis` | 170 | 篮球球员/球/队伍检测、跟踪和战术分析 | AGU 更偏动作识别和行为理解；该类项目更偏 tracking、team assignment、court mapping |
| `HanaFEKI/AI_BasketBall_Analysis_v1` | 59 | YOLO 检测、队伍分类、控球、俯视映射、速度距离 | AGU 不应重复做完整战术可视化，应保留 tracking 作为动作识别前处理 |
| `gabarlacchi/MASK-CNN-for-actions-recognition-` | 20 | 多运动动作识别 demo | AGU 比它更工程化，有 API、任务状态、融合和输出视频 |
| `ericdjm/basketball-action-recognition` | 2 | 篮球动作识别工具 | AGU 当前链路更完整，但需要补开源体验、样例和 benchmark |

### 竞品格局判断

通用视频理解框架已经很强，AGU 没必要做第二个 `MMAction2` 或 `PyTorchVideo`。这些项目的优势是模型多、配置体系成熟、论文复现能力强；AGU 如果往这个方向扩张，会很快被框架复杂度吞掉。

篮球 tracking 类项目在 GitHub 上更常见，通常围绕 YOLO、ByteTrack、球/人检测、队伍分类、homography、速度距离和控球可视化展开。这类项目适合做战术数据，但多数并不深入解决“球员动作语义识别”。AGU 的差异化空间在这里：把 tracking 之后的球员片段变成动作、置信度、复核状态和结构化输出。

足球 action spotting 项目给 AGU 的启发最大：它们通常有明确数据集、任务定义、评估指标和挑战赛入口。AGU 如果想成为有影响力的开源项目，也需要从“能跑的分析服务”升级为“可复现、可评测、可比较的篮球动作理解基线”。

## AGU 的差异化定位

### 不建议的发展方向

- 不做通用视频理解大框架。
- 不做完整篮球 SaaS。
- 不做小程序后端或数据库服务。
- 不做战术白板、球队管理、运营后台。
- 不把 player grouping、coach dashboard、CloudBase 业务环境混进 AGU。

### 建议的发展方向

AGU 应该成为一个窄而深的开源项目：

```text
player-centric basketball action understanding engine
```

核心差异化：

- 比通用 action recognition demo 更贴近真实篮球视频。
- 比 tracking 项目多一层动作语义。
- 比私有 SaaS 更轻量、可部署、可替换模型。
- 比论文代码更工程化，有 API、Docker、输出格式和可视化结果。

## 当前短板

### 开源体验短板

- 缺少清晰的 `quickstart` 和最小 demo 数据。
- 缺少公开 checkpoint 下载方式、checksum 和 model card。
- `requirements.txt` 历史包过旧，服务、训练、开发依赖没有拆开。
- legacy 脚本较多，容易让新用户不知道该跑哪个入口。
- 缺少 CLI，例如 `agu analyze --video ...`。

### 技术竞争力短板

- 主模型仍是 R(2+1)D，技术路线偏传统。
- 当前 benchmark 不够明确，缺少公开评估指标对比。
- 任务状态仍是内存态，不适合作为生产任务系统。
- 输入输出仍偏本地路径，云存储/对象存储只是部署建议。
- tracking、action、fusion 的模块边界还可以更插件化。

### 社区传播短板

- 缺少 README 首屏效果图或 GIF。
- 缺少示例输出 JSON。
- 缺少 “why AGU instead of MMAction2 / basketball tracking repo” 的一句话解释。
- 缺少贡献指南、路线图、issue template。

## 发展路线建议

### 0-1 个月：开源可用

目标：让陌生用户 10 分钟内理解并跑通。

- 清理 README 定位：AGU 只做 basketball action understanding engine。
- 增加 `examples/sample_request.json`、`examples/sample_output.json`。
- 增加一段 GIF 或截图，展示输入视频、标注视频、JSON 摘要。
- 增加公开 checkpoint 下载说明；如果不能公开权重，提供 dummy model 和训练指引。
- 增加 CLI：`agu analyze --video examples/lebron_shoots.mp4 --max-frames 120`。
- 拆分依赖：`requirements-service.txt`、`requirements-training.txt`、`requirements-dev.txt`。
- 明确 legacy 脚本状态，减少入口混乱。

### 1-3 个月：可复现基线

目标：让 AGU 从工程 demo 变成可比较项目。

- 建立固定 smoke dataset 和小型 benchmark。
- 固定 evaluation 输出：accuracy、macro-F1、balanced accuracy、per-class recall。
- 输出 model card：训练数据、标签、预处理、限制、失败案例。
- 提供 `pytest` + smoke inference gate。
- 将训练/推理预处理完全统一。
- 标准化输出 schema，稳定 `records`、`summary`、`video`、`metadata`。

当前 Phase 2 已落地的低风险基线：

- `docs/api.md`：稳定 API 和输出字段说明。
- `docs/model-card.md`：记录 v3 模型、标签、预处理契约和评估指标。
- `examples/sample_request.json`、`examples/sample_output.json`：可被 Pydantic schema 校验的公开样例。
- `scripts/validate_open_source_baseline.py`：无 checkpoint、无视频依赖的开源基线校验。
- `scripts/verify_harness.py`：默认调用开源基线校验。

仍待后续完成：

- 真正的小型 smoke video dataset。
- 公开 checkpoint 或 dummy checkpoint 下载说明。
- 训练/推理预处理的重训级统一。
- benchmark 指标表。

### 3-6 个月：模型和插件化

目标：让 AGU 可以承接社区贡献。

- 增加 model adapter：R(2+1)D、VideoMAE、SlowFast 或 MMAction2 backend。
- 增加 tracker adapter：YOLO/ByteTrack、DeepSORT、可选检测器。
- 增加 storage adapter：local、S3/COS、CloudBase Storage，但只作为通用插件。
- 将 VLM review 做成可选 plugin，默认关闭。
- 引入 action spotting 思路，支持从长视频中发现候选动作片段。

### 6-12 个月：成为篮球动作理解基线

目标：形成项目护城河。

- 发布小型公开篮球动作 benchmark 或复现 SpaceJam 子集协议。
- 提供 HuggingFace model/release 权重。
- 增加可视化报告：动作时间线、球员片段、置信度曲线。
- 支持导出 COCO/CSV/JSONL 等研究友好格式。
- 与 `visual_coach`、`basketball` 保持外部调用关系，但不把业务逻辑放回 AGU。

## 最推荐的战略

AGU 最适合走 “domain engine + reproducible baseline” 路线。

不要试图覆盖所有篮球分析功能。tracking、court mapping、team assignment 这类能力可以保留为辅助输入或插件，但项目主线应始终围绕：

```text
从篮球视频中理解球员动作，并输出可复用、可评估、可解释的结构化结果。
```

这会让 AGU 与 GitHub 上两类常见项目错位竞争：

- 面对 `MMAction2` / `PyTorchVideo`：AGU 更具体、更容易直接用于篮球场景。
- 面对篮球 tracking 项目：AGU 更关注动作语义和结果可调用。

最终 AGU 应该成为其他系统的一块清晰积木，而不是一个混合了模型、后台、小程序和数据库的私有平台。
