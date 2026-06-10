# 训练 Pipeline 重建方案

## 问题诊断

现有 checkpoint `r2plus1d_multiclass_19_0.0001.pt` 是空训练的：
- 所有 37 层 BatchNorm 的 `running_mean = 0`, `running_var = 1`（初始值）
- Optimizer state 为空（step = 0）
- 但训练历史显示 epoch 19 达到了 ~85% val accuracy

**结论**：模型训练成功过，但 checkpoint 保存/传输过程损坏了权重。

## 方案：在 Mac mini 上重建训练

### 环境限制
- Mac mini M1, 16GB RAM, 无 CUDA
- MPS (Metal) 可用于 PyTorch 加速，但 R(2+1)D 在 MPS 上有已知兼容性问题
- 建议：CPU 训练，调低 batch_size + 用更少 epoch

### 训练数据
- SpaceJam 数据集：~49,901 个短视频（16帧 × 128×176）
- 源：https://github.com/simonefrancia/SpaceJam
- 本地无数据，需要重新下载 (~5GB)

### Phase 1: 数据准备

```bash
# 1. 克隆 SpaceJam 仓库获取标注
git clone https://github.com/simonefrancia/SpaceJam.git dataset/spacejam

# 2. 下载视频片段（需要从原始源下载）
# SpaceJam 的 clips 大约 32,560 个 mp4
# 增强后 49,901 个

# 3. 验证标注文件
python3 -c "import json; d=json.load(open('dataset/annotation_dict.json')); print(len(d))"
```

### Phase 2: 修复训练脚本

`train.py` 需要的修改：

1. **移除 CUDA 硬编码**（L243-248, L279-283）
2. **支持 MPS/CPU 训练**：
   ```python
   device = torch.device("mps" if torch.backends.mps.is_available() 
                         else "cuda" if torch.cuda.is_available() 
                         else "cpu")
   ```
3. **降低内存占用**：
   - `batch_size`: 8 → 2（CPU/MPS）
   - `num_workers`: 调低避免内存压力
   - 添加 `pin_memory=False`（非 CUDA）
4. **使用新版 torchvision API**：
   - `pretrained=True` → `weights=R2Plus1D_18_Weights.DEFAULT`
5. **修复已知 bug**：推理预处理缺少 BGR→RGB 和 /255 归一化

### Phase 3: 训练策略

#### 方案 A：完整 fine-tune（推荐）
```python
# 初始化：Kinetics-400 预训练
model = models.video.r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
model.fc = nn.Linear(512, 10)

# 解冻策略：layer3 + layer4 + fc
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if any(layer in name for layer in ['layer3', 'layer4', 'fc']):
        param.requires_grad = True

# 训练 15-20 epoch，lr=1e-4
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
```

#### 方案 B：全量 fine-tune（更高精度但更慢）
- 解冻所有层
- 差异学习率：backbone 1e-5, fc 1e-3
- 需要 25+ epoch

#### 方案 C：仅训练 fc 层（最快但精度低）
- 冻结所有卷积层
- 只训练 fc
- 5-10 epoch 即可
- 预期 ~70% accuracy

### Phase 4: 验证新 checkpoint

```python
# 验证 BN stats 非零
ckpt = torch.load('model_checkpoints/r2plus1d_v2/best.pt')
sd = ckpt['state_dict']
for k in sd:
    if 'running_mean' in k:
        assert not torch.allclose(sd[k], torch.zeros_like(sd[k])), f"{k} is still zero!"
        print(f"✓ {k}: mean={sd[k].mean():.4f}")
```

### 预估时间

| 方案 | 每 Epoch 时间 (CPU) | 总 Epochs | 总时间 |
|------|---------------------|-----------|--------|
| C: fc only | ~2h | 10 | ~20h |
| A: layer3+4+fc | ~4h | 20 | ~80h |
| B: full fine-tune | ~6h | 25 | ~150h |

> ⚠️ CPU 训练非常慢。如果有云端 GPU（Colab Pro, Lambda Labs 等），方案 A 只需 1-2 小时。

### 替代方案：Google Colab

```python
# Colab 上可直接运行 train.py（有免费 T4 GPU）
# Batch size = 32, 每 epoch ~10min
# 方案 A 总计 ~3h
```

## 临时解决方案（已完成）

在重新训练之前，已使用 Kinetics-400 预训练 backbone + 随机 fc 验证：

| 指标 | 空 checkpoint | 预训练 backbone |
|------|--------------|-----------------|
| walk 占比 | 100% | 1.6% |
| pick 占比 | 0% | 28.9% |
| ball_in_hand | 0% | 25.2% |
| pass | 0% | 18.1% |
| dribble | 0% | 17.1% |

预训练 backbone 已经能有效区分动作，但 fc 层是随机的所以分类不准（pick 偏多）。正式训练后应能达到 85%+ 准确率。
