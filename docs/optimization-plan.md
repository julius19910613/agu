# Basketball Defense Analysis 训练优化方案

更新时间：2026-06-11

## 1. 执行摘要

当前 `R(2+1)D-18` 已经在第 3 个 epoch 达到 `77.5%` 的最佳验证准确率，但这并不代表模型“学完了”，而是说明它在当前设置下很快开始记忆多数类，并且验证指标过于偏向整体 accuracy，掩盖了 `shoot`、`pick` 这类少数类的失败。

最关键的结论有 4 个：

1. **epoch 3 之后的过拟合是真实存在的**，主因不是模型太大本身，而是 `batch_size=2` 下 BN 统计不稳定、没有任何增强/正则、没有 LR 调度、没有 early stopping、且只看 `val accuracy`。
2. **少数类差不是单点问题，而是“长尾分布 + 无重加权 + 无重采样 + 无定向增强 + 指标选择错误”叠加的结果**。`shoot` 仅 426 条，较 `walk` 少 `27.6x`；`pick` 较 `walk` 少 `16.5x`。
3. **当前训练/推理预处理虽然已经在“BGR + [0,255]”上对齐，但仍然不是合理终态**。训练端仍在原始分辨率上学习，而推理端固定 `112x112`；更重要的是，当前训练并没有遵循 torchvision Kinetics 预训练权重的标准输入分布。
4. **优化顺序应该分两条线**：
   - 短线：基于当前 `best.pt` 快速续训，优先解决过拟合和评估偏置。
   - 中线：做一次干净的重训，把预处理、采样、增强、指标体系一次性拉正。

## 2. 当前结果诊断与根因分析

### 2.1 过拟合在 epoch 3 后出现的根因

现象：

- 训练准确率从 `61.3% -> 93.5%` 持续上升。
- 验证准确率在 epoch 3 达到 `77.5%` 后不再稳定提升。
- `train_mac.py` 当前没有 scheduler、没有 early stopping、没有 weight decay、没有 label smoothing，损失函数仍是裸 `CrossEntropyLoss()`。

根因拆解：

1. **BN 统计在超小 batch 下容易漂移**
   - 当前仅解冻 `layer3 + layer4 + fc`，而这些层里的 `BatchNorm3d` 在 `batch_size=2` 下更新 running stats，方差极大。
   - 这会导致模型在训练集上迅速贴合局部统计，但验证集泛化变差。
   - 对应代码：`[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:210)` 到 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:250)`。

2. **几乎没有正则化**
   - 当前 `model.fc` 直接替换为线性层，没有 dropout。
   - 优化器是 `Adam`，没有 `weight_decay`。
   - 没有 `label_smoothing`。
   - 对应代码：`[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:378)` 到 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:393)`。

3. **没有数据增强**
   - 当前 `train_mac.py` 训练并未对输入 clip 做任何 spatial / temporal augmentation。
   - `dataset.py` 的 `transform` 参数存在但没有被真正使用。
   - 对应代码：`[dataset.py](/Users/ppt/projects/basketball-defense-analysis/dataset.py:14)` 到 `[dataset.py](/Users/ppt/projects/basketball-defense-analysis/dataset.py:47)`。

4. **学习率策略过于静态**
   - 当前从头到尾固定 `1e-4`，没有 warmup、没有 decay。
   - 对少数类而言，后期通常需要更低 LR 才能微调 decision boundary，而不是继续强化多数类。

5. **best checkpoint 的选择标准有偏**
   - 当前以 `val accuracy` 保存 `best.pt`，这在长尾数据上天然偏向 `walk/no_action/run`。
   - 少数类 recall 下降时，总体 accuracy 仍可能不变甚至上升。
   - 对应代码：`[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:277)` 到 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:294)`。

### 2.2 `shoot` / `pick` 表现差的根因

现象：

- `shoot`: train `32/319`，val `1/91`，几乎失效。
- `pick`: train `4/285`，val `52/200`，整体仍很差，且 train/val 波动异常。

根因拆解：

1. **长尾极端严重**
   - `shoot` 仅占 `1.1%`，`pick` 仅占 `1.9%`。
   - 以当前无权重 CE 训练，模型最优策略会自然偏向多数类先验。

2. **少数类时间模式更难，仅靠空间外观不够**
   - `shoot` 与 `ball_in_hand`、`dribble` 的差异往往体现在出手阶段的短时动态。
   - `pick` 与 `defense`、`no_action` 也容易在单帧外观上混淆，必须依赖更稳定的时序表征。

3. **缺少定向增强**
   - 虽然仓库里有 `[augment_videos.py](/Users/ppt/projects/basketball-defense-analysis/augment_videos.py:1)`，但当前训练上下文说明 `augmented_annotation_dict.json` 为空，等于没有少数类扩增。

4. **指标和 checkpoint 机制不会奖励少数类改善**
   - 当前不会记录 per-class recall / macro-F1。
   - 即使 `shoot` 从 `1%` recall 提升到 `15%`，如果 `walk` 略降，也可能不会被保留成 best。

5. **预训练分布未充分利用**
   - 当前训练输入来自 `[dataset.py](/Users/ppt/projects/basketball-defense-analysis/dataset.py:54)` 到 `[dataset.py](/Users/ppt/projects/basketball-defense-analysis/dataset.py:75)`，是 `BGR + [0,255] + 原始分辨率`。
   - 这与 Kinetics 预训练权重的常规 RGB/归一化输入分布不一致，迁移学习收益被打折，通常最先伤害的就是样本少、边界窄的类别。

### 2.3 类别不平衡偏置的根因

现象：

- 混淆矩阵显示模型明显偏向 `walk`。
- 实际视频推理中也以 `ball_in_hand/dribble/defense` 等高频近邻类为主。

根因拆解：

1. **损失函数没有任何 class reweighting**
   - 当前是 `nn.CrossEntropyLoss()`。

2. **采样器没有任何 rebalance**
   - 当前 `DataLoader(..., shuffle=True)`，没有 `WeightedRandomSampler`。
   - 对应代码：`[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:419)` 到 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:447)`。

3. **验证与保存标准仍以 overall accuracy 为中心**
   - 这是“训练目标”和“业务目标”不一致。
   - 如果目标是更可靠的多动作识别，至少应该把 `macro-F1` 和 `balanced accuracy` 提到与 accuracy 同级。

4. **随机切分不是分层切分**
   - 当前使用 `random_split`，不是 stratified split。
   - 在少数类非常少时，这会增大验证集方差，导致结论不稳定。

## 3. 是否必须修复当前预处理不一致

结论：**需要修，而且优先级很高；但不建议把“彻底修复预处理”与“当前 best.pt 快速续训”混成一次实验。**

### 3.1 现在到底哪里不一致

当前训练端：

- `[dataset.py](/Users/ppt/projects/basketball-defense-analysis/dataset.py:54)` 到 `[dataset.py](/Users/ppt/projects/basketball-defense-analysis/dataset.py:75)`
- 读取 `BGR`
- 像素范围 `[0,255]`
- 不 resize
- 不 normalize

当前推理端：

- `[app/analysis/inference.py](/Users/ppt/projects/basketball-defense-analysis/app/analysis/inference.py:27)` 到 `[app/analysis/inference.py](/Users/ppt/projects/basketball-defense-analysis/app/analysis/inference.py:64)`
- 保持 `BGR`
- 保持 `[0,255]`
- **会 resize 到 `112x112`**

所以现在至少存在两个问题：

1. **训练与推理的空间尺度不一致**：训练看原始分辨率，推理看 `112x112`。
2. **训练/推理与 torchvision 预训练权重的标准输入分布不一致**：这会削弱预训练迁移效果。

### 3.2 应该怎么处理

1. **Phase 1 快速续训时**
   - 不建议一边续 `best.pt`，一边把输入分布彻底改成 RGB+normalize。
   - 这样会把“loss/scheduler/BN/采样”的收益和“输入分布跳变”的影响混在一起，实验不可解释。

2. **Phase 2 干净重训时**
   - 建议统一为：
     - `BGR -> RGB`
     - resize / crop 到 `112x112`
     - `/255`
     - 使用 `R2Plus1D_18_Weights.DEFAULT.transforms()` 对应的 mean/std 归一化
   - 训练、验证、推理三端共用同一份预处理函数。

一句话总结：

- **短期**：为了快速验证优化手段，可先维持现有输入分布。
- **中期**：为了把预训练权重真正用对，必须重训并统一预处理。

## 4. 优先级路线图总览

| 阶段 | 目标 | 推荐起点 | 预期主要收益 |
|---|---|---|---|
| Phase 1 `<2h` | 低风险快速止损，基于当前 `best.pt` 继续训练 | 直接续训 | 过拟合变缓，少数类 recall 有小幅可见提升 |
| Phase 2 `4-8h` | 系统解决长尾、正则化、增强、评估 | 新实验，建议从 pretrained 重新起跑 | `macro-F1` 和少数类 recall 明显提升 |
| Phase 3 `8h+` | 架构/模态升级，追求更稳上限 | 在 Phase 2 基线稳定后开展 | 进一步拉高少数类和复杂动作表现 |

## 5. Phase 1：Quick Wins（<2h，可直接从当前 `best.pt` 重启）

目标：**不推翻当前模型，只修正训练动态与评估偏差。**

### 5.1 建议清单

| 优先级 | 方案 | 具体代码修改 | 预期影响 | 实现复杂度 | 依赖/注意事项 |
|---|---|---|---|---|---|
| P0 | 把 best 指标从 `val accuracy` 改成 `macro-F1` 或 `0.5*macro_F1 + 0.5*balanced_acc` | 修改 `[utils/metrics.py](/Users/ppt/projects/basketball-defense-analysis/utils/metrics.py:1)`，返回 `macro_f1/micro_f1/per_class_recall/balanced_acc`；修改 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:277)` 的 best 保存逻辑，用 `val_macro_f1` 选 best | 通常不一定提高 top-1，但会显著减少“best.pt 仍偏多数类”的情况；`macro-F1` 可提升 `+2 ~ +5` 点 | 低 | 必须同步更新 history 输出格式 |
| P0 | 冻结 BN running stats，避免 `batch_size=2` 继续污染统计 | 在 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:162)` 前新增 `freeze_bn_stats(model)`，每个 train epoch 开始后对 `nn.BatchNorm3d` 执行 `m.eval()`，并可选冻结 BN 的 `weight/bias` | 常见收益是验证曲线更稳，过拟合延后 `1~3` 个 epoch，`val acc`/`macro-F1` 提升 `+1 ~ +3` 点 | 低 | 如果后续做全量重训，可再评估是否解冻 BN |
| P0 | 引入温和 class weight + label smoothing | 在 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:393)` 将 `criterion = nn.CrossEntropyLoss()` 改为带 `weight` 和 `label_smoothing=0.05` 的 CE；起始权重建议按 **inverse-sqrt frequency**，并归一到均值 1：`[1.34, 1.29, 0.55, 0.72, 2.05, 0.87, 0.68, 1.59, 0.53, 0.39]`，顺序对应 `[block, pass, run, dribble, shoot, ball_in_hand, defense, pick, no_action, walk]` | 少数类 recall 往往可提升 `1.5x ~ 3x`；整体 accuracy 可能 `-1 ~ +1` 波动，但 macro-F1 更可能上升 `+3 ~ +6` 点 | 低 | 不建议一上来用纯 inverse-frequency，`shoot` 权重会过猛 |
| P1 | 把优化器改为 `AdamW`，加 `weight_decay` 和 `ReduceLROnPlateau` | `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:392)` 附近改为 `AdamW(lr=3e-5 ~ 5e-5, weight_decay=1e-4)`；新增 scheduler：`ReduceLROnPlateau(mode=\"max\", factor=0.5, patience=1, min_lr=1e-6)`，监控 `val_macro_f1` | 典型收益是防止 epoch 3 之后继续把多数类边界“推得更硬”；`val` 稳定性明显改善 | 低 | 从 `best.pt` 续训时，建议把 LR 降到 `3e-5` 再启动 |
| P1 | 梯度累积，提升有效 batch | 在 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:223)` 的训练循环中引入 `accum_steps=4`，`loss = loss / accum_steps`，每 4 个 step 再 `optimizer.step()` | 有效 batch 从 2 提到 8，梯度噪声更小；常见收益 `+0.5 ~ +2` 点 | 中 | BN 已冻结时更适合做 gradient accumulation |
| P1 | 加 early stopping | 在 `train_model()` 内新增 patience=3，监控 `val_macro_f1` | 防止无意义地把模型从 epoch 3 的泛化状态继续拖坏 | 低 | 与 scheduler 一起使用效果最好 |

### 5.2 Phase 1 推荐实验顺序

1. 先做 `P0` 三项：`macro-F1 checkpoint + BN freeze + weighted CE`。
2. 再加 `AdamW + scheduler + early stopping`。
3. 如果显存仍允许，最后加 `grad accumulation`。

### 5.3 Phase 1 建议启动参数

建议直接从当前最佳模型续训：

```bash
python train_mac.py \
  --resume model_checkpoints/r2plus1d_v3/best.pt \
  --epochs 10 \
  --lr 3e-5 \
  --batch-size 2 \
  --num-workers 0
```

说明：

- 这里的 `--epochs 10` 指总 epoch 终点；如果 `best.pt` 来自 epoch 3，则会继续跑到 epoch 10。
- Phase 1 的目标不是冲更高 train acc，而是看 `macro-F1`、`shoot recall`、`pick recall` 是否回升。

## 6. Phase 2：Core Improvements（4-8h，解决长尾 + 正则 + 增强）

目标：**建立一个真正可复现、可解释、与预训练兼容的训练基线。**

### 6.1 建议清单

| 优先级 | 方案 | 具体代码修改 | 预期影响 | 实现复杂度 | 依赖/注意事项 |
|---|---|---|---|---|---|
| P0 | 统一训练/验证/推理预处理，并做一次干净重训 | 新建公共预处理模块，例如 `app/models/preprocessing.py`；让 `[dataset.py](/Users/ppt/projects/basketball-defense-analysis/dataset.py:54)` 和 `[app/analysis/inference.py](/Users/ppt/projects/basketball-defense-analysis/app/analysis/inference.py:27)` 共用；建议流程：`BGR->RGB`、resize/crop 到 `112x112`、`/255`、Kinetics mean/std normalize | 这是中期最重要的单项改动；常见收益可达 `macro-F1 +4 ~ +10` 点，少数类通常更受益 | 中 | **建议重训，不建议在当前 best.pt 上直接切换输入分布** |
| P0 | 使用 `WeightedRandomSampler` 或 class-aware sampler | 在 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:419)` 附近，基于训练子集标签构造 `sample_weights`，把 `shuffle=True` 改为 `sampler=train_sampler` | `shoot/pick/pass/block` 的曝光频次显著增加；少数类 recall 常见 `+5 ~ +15` 点 | 中 | 与 weighted loss 同时使用时，建议先减轻 loss 权重强度，避免过补偿 |
| P0 | 训练/验证切分改为分层切分，并持久化 split | 替换 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:410)` 附近的 `random_split`，用 `StratifiedShuffleSplit` 生成固定 `train/val/test` id 列表并写入 `dataset/splits/*.json` | 降低验证集波动，尤其是 `shoot/pick` 这类稀有类；实验结论更可靠 | 中 | 需要先从 annotation 中取出 label 列表 |
| P1 | 引入真正的视频增强管线 | `dataset.py` 中启用 `transform`；建议 train augmentation：`RandomResizedCrop(112, scale=(0.8,1.0))`、`RandomHorizontalFlip(0.5)`、轻度 `ColorJitter`、`GaussianBlur`、时间抖动或随机丢 1-2 帧；val 只做 deterministic resize/crop | 对过拟合抑制明显，常见 `val acc +1 ~ +3`，`macro-F1 +3 ~ +8` | 中 | 横向翻转对篮球动作一般可用，但若后续融合场地方向语义，需要重新确认 |
| P1 | 复用并扩展现有少数类增强脚本 | 扩展 `[augment_videos.py](/Users/ppt/projects/basketball-defense-analysis/augment_videos.py:1)`，不要只做 rotate/translate；新增 flip、scale jitter、brightness/contrast、轻微 temporal shift；优先对 `shoot/pick/pass/block` 生成增强样本 | 如果增强质量可控，`shoot/pick` recall 往往是最直接受益项，可能带来 `+5 ~ +12` 点 | 中 | 旋转过大可能破坏动作几何；建议从 `±10°`、轻微平移开始，而不是固定 `30°` |
| P1 | 调整 fine-tuning 策略为“渐进解冻 + 区分学习率” | 初始训练 `fc + layer4`，2-3 个 epoch 后再解冻 `layer3`，必要时最后再解冻 `layer2`；参数组示例：`fc=1e-4`、`layer4=3e-5`、`layer3=1e-5` | 比当前“一开始就一起训”更稳，通常能减少前期过拟合并更好利用预训练 | 中 | 仍不建议全 backbone 一次性解冻，MPS 上风险高 |
| P2 | 把 `model.fc` 改成带 dropout 的 head | 当前 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:378)` 是裸 `Linear`；建议改为 `nn.Sequential(nn.Dropout(0.3~0.5), nn.Linear(...))` | 低成本抑制 overfit，收益通常 `+0.5 ~ +2` 点 | 低 | 与 label smoothing 可叠加 |
| P2 | 缓存统一尺寸 tensor，减少每 epoch 视频解码开销 | 基于 `[dataset.py](/Users/ppt/projects/basketball-defense-analysis/dataset.py:137)` 的思路，预先导出 `112x112`、RGB、float32、已归一化的 `.pt` clip；训练时用 tensor dataset | 对 MPS/CPU 训练速度帮助很大，常见吞吐提升 `1.5x ~ 3x` | 中 | 需要额外磁盘空间；是 Phase 2 最值的效率优化 |

### 6.2 Phase 2 推荐实现细节

#### A. sampler 与 weighted loss 不要同时“拉满”

推荐起点：

- `WeightedRandomSampler` 负责提升少数类曝光。
- `CrossEntropyLoss(weight=...)` 只保留温和权重，或直接退到 `label_smoothing + 无权重 CE`。

原因：

- 如果同时使用强采样和强权重，`shoot/pick` 可能会被过拟合，导致 precision 急剧下滑。

#### B. 指标体系必须升级

建议每个 epoch 至少输出：

- `accuracy`
- `macro-F1`
- `micro-F1`
- `balanced accuracy`
- per-class `precision/recall/F1`
- confusion matrix

并将以下两个指标作为主判断标准：

1. `macro-F1`
2. `shoot/pick` 的 recall

#### C. Phase 2 的成功标准

相较当前 baseline，合理目标不是单看 overall accuracy，而是：

- `val accuracy`: `77.5% -> 78~81%`
- `macro-F1`: 提升 `+8 ~ +15` 点
- `shoot recall`: 从约 `1%` 提升到 `10~25%`
- `pick recall`: 再提升 `+10 ~ +20` 点

如果 overall accuracy 不涨，但 macro-F1 与少数类 recall 明显提升，这仍然是有效优化。

## 7. Phase 3：Advanced（8h+，上限优化）

目标：**在 Phase 2 已经稳定的前提下，进一步提高复杂动作和少数类表现。**

### 7.1 建议清单

| 优先级 | 方案 | 具体代码修改 | 预期影响 | 实现复杂度 | 依赖/注意事项 |
|---|---|---|---|---|---|
| P0 | 长尾专用损失：`Balanced Softmax` / `LDAM-DRW` / `Class-Balanced Focal` | 在 `train_mac.py` 外新增 `losses.py`，替换当前 CE；推荐尝试顺序：`Balanced Softmax` -> `LDAM-DRW` -> `CB-Focal` | 对极端长尾常比 plain weighted CE 更稳；少数类 recall / macro-F1 仍有 `+2 ~ +6` 点空间 | 中-高 | 先有稳定 Phase 2 基线，否则很难判断损失函数收益 |
| P1 | 双流或多模态融合：RGB + pose / motion | 仓库已有 `poseData` 分支和 `app/analysis/motion.py`；可新增双分支模型，一个分支吃 RGB clip，一个分支吃 skeleton 或 motion feature，最后 late fusion | 对 `shoot/pick/block` 这类强时序动作最有潜力，少数类 recall 可再上一个台阶 | 高 | 需要新增数据管线和训练脚本 |
| P1 | 架构升级：X3D / SlowFast / Temporal Attention Head | 新建模型构建模块，不直接替换线上推理前先离线对比；在 MPS 约束下优先尝试更轻量的 X3D-S 或在 R(2+1)D head 上加 temporal attention | 若训练稳定，可能带来 `+2 ~ +5` 点 | 高 | 训练成本和调参成本明显上升 |
| P2 | Hard example mining / curriculum learning | 基于 confusion matrix，把长期易错样本导出为 hard set，后续 epoch 提高采样比重 | 更适合修 `shoot` vs `ball_in_hand`、`pick` vs `defense` 的边界 | 中-高 | 需要先有 per-sample logging |
| P2 | 推理期校准与 TTA | 对验证集做 temperature scaling；推理时使用 temporal multi-crop / horizontal flip TTA | 准确率增益有限，但概率更可信，线上行为更稳 | 中 | 推理耗时会上升 |

## 8. MPS 训练效率与稳定性建议

这些建议不一定直接提升精度，但会提升实验迭代速度和稳定性。

### 8.1 优先做的

1. **不要每个 batch 都 `gc.collect()`**
   - 当前 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:274)` 和 `[train_mac.py](/Users/ppt/projects/basketball-defense-analysis/train_mac.py:347)` 每 batch 都回收，吞吐会明显受伤。
   - 建议改成每 `N=10` 或 `N=20` 个 batch 回收一次，或者只在 OOM / epoch 结束时回收。

2. **优先做 tensor cache，而不是盲目加 `num_workers`**
   - MPS 训练瓶颈通常不在矩阵乘本身，而在 CPU 端 `cv2.VideoCapture` 解码。
   - 在 raw mp4 读取模式下，`num_workers=0` 往往比 `2~4` 更稳。
   - 真正有效的是 Phase 2 的“预缓存 112x112 tensor”。

3. **使用 gradient accumulation 替代更大的真实 batch**
   - 对 16GB unified memory，这是最实用的做法。

4. **保持 FP32，先不要默认启用 AMP**
   - MPS 的 mixed precision 在 3D CNN 上并不总是稳定，收益也未必显著。
   - 没有基准验证前，不建议把 AMP 当作默认优化。

### 8.2 可选优化

1. **`optimizer.zero_grad(set_to_none=True)`**
   - 可降低一点内存压力，也更快。

2. **只在必要时保留 optimizer state**
   - Adam/AdamW 的 state 在 MPS 上也占内存。
   - 如果后续发现优化器 state 明显挤占内存，可测试 `SGD + momentum` 作为对照，但通常收敛速度会慢。

3. **避免同时解冻太多层**
   - 这既是泛化建议，也是 MPS 内存建议。

4. **先别默认上 `torch.compile`**
   - 在 MPS 上收益不稳定，且调试成本高。
   - 除非已有实测收益，否则不建议作为主线优化。

## 9. 推荐的落地顺序

### 第一步：今天就能做

1. `macro-F1` 作为 best 指标
2. 冻结 BN
3. `AdamW + weight_decay + ReduceLROnPlateau`
4. 温和 weighted CE + label smoothing
5. 从 `best.pt` 继续跑到 epoch 8~10

### 第二步：下一轮正式实验

1. 统一预处理到 `RGB + 112x112 + normalize`
2. 分层切分并固定 split
3. `WeightedRandomSampler`
4. train-time augmentation
5. dropout head
6. 渐进解冻

### 第三步：如果还要继续追上限

1. Balanced Softmax / LDAM-DRW
2. RGB + pose/motion 融合
3. 轻量时序注意力或更适合长尾的视频架构

## 10. 最终判断

如果只能做最少改动，我的建议是：

- **先不要急着继续追更高 accuracy。**
- 先把 `best` 标准、BN、loss、scheduler 改对，再用当前 `best.pt` 续训一轮看 `macro-F1` 和少数类 recall。

如果可以接受一次正式重训，我的建议是：

- **优先修预处理与采样，而不是优先换模型。**
- 当前瓶颈主要是训练策略和数据分布，不是 R(2+1)D 本身已经到上限。

一句话版本：

> 当前模型的问题首先是“训练目标与数据分布不匹配”，其次才是“模型容量与结构”。先把长尾学习、评估标准和预处理统一起来，收益会大于直接换更复杂模型。
