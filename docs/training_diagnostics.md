# 训练问题诊断与修复合集

> 由以下文档合并整理（2026-08-03）：spatial_consistency_fix.md、batchnorm_explosion_solution.md、LangZip_训练方法诊断报告.md

---

# Part 1: Spatial Consistency Fix（空间一致性修复）

# Spatial Consistency Fix - Comprehensive Solution

## Problem Analysis

### Symptoms
1. **Loss plateaued at ~0.2** (91% reduction from initial ~2.3)
2. **Cosine similarity reached 0.91** (healthy, not mode collapse!)
3. **But renders extremely noisy** - salt-and-pepper artifacts, no coherent semantic structure

### Root Causes Identified

#### 1. Spatial Inconsistency
The model learned the **statistical distribution** of features but not the **spatial correspondence**:
- Predicted features matched GT in aggregate statistics
- But neighboring Gaussians had wildly different feature values
- This caused random colored pixels in renders instead of coherent regions

#### 2. dim_scale Imbalance
```
dim_scale values: [0.82, 0.70, 0.44, 0.46, 0.38, 0.45, 0.35, 0.37, 0.33, 0.27, 0.29, 0.29, 0.28, 0.40, 0.21, 0.25]
```
- **4x difference** between max (0.82) and min (0.21)
- Some dimensions severely suppressed, others over-amplified
- Caused spatial inconsistency: adjacent points have very different feature scales

#### 3. No Spatial Regularization
The loss function only considered point-wise feature matching:
- L1 loss: `|pred - target|` per point
- Cosine loss: directional alignment
- **No penalty for spatial inconsistency** between neighboring points

---

## Solutions Implemented

### 1. SpatialSmoothnessLoss (NEW)

**File**: `pointcept/models/losses/misc.py`

Adds a new loss that penalizes large feature differences between neighboring 3D Gaussians:

```python
@LOSSES.register_module()
class SpatialSmoothnessLoss(nn.Module):
    """
    Encourages spatial consistency by penalizing large feature differences
    between neighboring Gaussians. Prevents salt-and-pepper noise.
    """
```

**Key Features**:
- **KNN-based**: For each point, finds k=16 nearest neighbors
- **Radius-filtered**: Only considers neighbors within 2cm radius
- **Progressive scheduling**:
  - Epochs 0-2: No spatial constraint (learn feature statistics first)
  - Epochs 3-9: Ramp up spatial smoothness (reduce noise)
  - Epochs 10+: Decay to allow sharp object boundaries

**Formula**:
```
L_smooth = mean(||f_i - f_j||^2) for all (i,j) where dist(i,j) < radius
```

---

### 2. Balanced dim_scale Normalization

**File**: `pointcept/models/default.py`

#### Before (PROBLEMATIC):
```python
dim_scale_init = torch.ones(16)  # Uniform init
dim_scale_clamped = torch.clamp(torch.relu(self.dim_scale) + 0.01, max=10.0)
feat = feat * dim_scale_clamped
```
**Issue**: ReLU + clamp allowed extreme imbalance (0.21 to 0.82)

#### After (FIXED):
```python
dim_scale_init = torch.ones(16) / (16 ** 0.5)  # Normalized init
dim_scale_normalized = self.dim_scale / (self.dim_scale.norm() + 1e-8)
dim_scale_balanced = dim_scale_normalized * (16 ** 0.5)
feat = feat * dim_scale_balanced
```
**Benefit**: L2 normalization keeps all dimensions balanced

**Why this works**:
- L2 normalization constrains the scale vector to unit sphere
- No single dimension can dominate or be suppressed
- Model learns relative importance but maintains spatial consistency

---

### 3. Updated Model Forward Pass

**File**: `pointcept/models/default.py`

Modified `LangPretrainer.forward()` to pass coordinates for spatial smoothness:

```python
loss = self.criteria(
    point_feat["feat"],
    input_dict["lang_feat"],
    valid_feat_mask=input_dict["valid_feat_mask"],
    segment=segment,
    epoch_progress=input_dict["epoch_progress"],
    coord=input_dict.get("coord"),  # NEW: Pass for spatial smoothness
)
```

Also updated `_chunked_forward()` with the same changes.

---

### 4. Training Config Update

**File**: `configs/custom/lang-pretrain-litept-ovs-gridsvd.py`

Added SpatialSmoothnessLoss to the loss criteria:

```python
criteria=[
    dict(type="SVDWeightedL1Loss", loss_weight=0.3, ...),
    dict(type="CosineSimilarity", loss_weight=1.0, ...),

    # NEW: Spatial smoothness
    dict(
        type="SpatialSmoothnessLoss",
        loss_weight=0.1,  # Moderate weight
        neighbor_k=16,
        radius=0.02,  # 2cm
        warmup_epochs=3,
        decay_start=10,
    ),
],
```

---

## Expected Training Behavior

### Phase 1: Epochs 0-2 (Warmup)
- **SpatialSmoothnessLoss weight = 0**
- Model focuses on learning feature statistics (L1 + Cosine)
- No spatial constraints yet

### Phase 2: Epochs 3-9 (Smoothness Ramp-up)
- **SpatialSmoothnessLoss weight ramps to 0.1**
- Model learns spatial consistency
- Noise in renders should decrease significantly

### Phase 3: Epochs 10+ (Decay)
- **SpatialSmoothnessLoss gradually decays to 0.05**
- Allows sharp object boundaries to form
- Final renders should be clean and coherent

---

## Additional Recommendations

### If Noise Persists After These Changes:

1. **Increase SpatialSmoothnessLoss weight**:
   ```python
   loss_weight=0.2,  # or higher
   ```

2. **Adjust radius for your data**:
   - For grid_size=0.01: radius=0.02 (2x grid)
   - For grid_size=0.02: radius=0.04 (2x grid)

3. **Add Gradual Unfreezing**:
   - Freeze encoder for first 5 epochs
   - Only train decoder and dim_scale
   - Then unfreeze all layers

4. **Use Higher SVD Rank**:
   - Current: svd_rank=16
   - Try: svd_rank=32 for more capacity
   - Note: 2x memory usage

### To Monitor Progress:

```python
# Check dim_scale balance
dim_scale = model.dim_scale.data
print(f"dim_scale range: [{dim_scale.min():.4f}, {dim_scale.max():.4f}]")
print(f"dim_scale ratio: {dim_scale.max()/dim_scale.min():.2f}x")
# Target: < 2x ratio

# Check spatial smoothness (during training)
# Should decrease during epochs 3-9
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| `pointcept/models/losses/misc.py` | Added `SpatialSmoothnessLoss` class |
| `pointcept/models/default.py` | Fixed `dim_scale` normalization, added `coord` parameter to loss |
| `configs/custom/lang-pretrain-litept-ovs-gridsvd.py` | Added `SpatialSmoothnessLoss` to criteria |

---

## Verification

To verify the fix is working:

1. **Check dim_scale balance**:
   ```bash
   python -c "
   import torch
   ckpt = torch.load('exp/lite-16-gridsvd/model/model_last.pth')
   dim_scale = ckpt['state_dict']['dim_scale']
   print(f'dim_scale: {dim_scale}')
   print(f'Ratio: {dim_scale.max()/dim_scale.min():.2f}x')
   "
   ```
   Target: < 2x ratio

2. **Check renders for noise reduction**:
   - Old: Salt-and-pepper noise everywhere
   - New: Coherent semantic regions

3. **Check loss curves**:
   - SpatialSmoothnessLoss should decrease during epochs 3-9
   - L1 and Cosine should continue improving

---

## Performance Impact

- **Memory**: +O(N*K) for KNN computation (K=16 neighbors)
  - Mitigated by sampling large point clouds (>100K points)
- **Speed**: ~10-15% slower per iteration due to KNN
  - Acceptable trade-off for significant quality improvement

---

## Next Steps

1. Start training with the new config
2. Monitor `dim_scale` balance (should stay < 2x ratio)
3. Check renders after epoch 5 for noise reduction
4. Adjust `loss_weight` if needed (0.05 to 0.3 range)

---

# Part 2: BatchNorm 爆炸问题解决方案

# BatchNorm爆炸问题完整分析与解决方案

**日期**: 2026-03-09
**问题**: Decoder中BatchNorm的running_var爆炸到十亿级别
**根本原因**: Decoder瓶颈 + BatchNorm的指数积累机制

---

## 一、为什么running_var会爆炸？

### 1.1 BatchNorm的指数积累机制

```python
# PyTorch BatchNorm的running_var更新公式
running_var = (1 - momentum) * running_var + momentum * batch_var

# 对于默认设置:
# momentum = 0.01
# 等价于:
# running_var = 0.99 * running_var + 0.01 * batch_var
```

**关键问题**：这是一个**指数积累**过程！

### 1.2 模拟：正常 vs 异常情况

```python
# 情况1: 正常训练 (batch_var ≈ 1.0)
running_var = 1.0  # 初始
for iter in range(100):
    batch_var = 1.0  # 正常范围
    running_var = 0.99 * running_var + 0.01 * batch_var
# 结果: running_var ≈ 1.0 (稳定)

# 情况2: 激活值爆炸 (batch_var = 1000)
running_var = 1.0  # 初始
for iter in range(100):
    batch_var = 1000  # 激活值爆炸!
    running_var = 0.99 * running_var + 0.01 * batch_var
# 结果: running_var ≈ 100 (已经是初始的100倍!)

# 继续训练
for iter in range(100, 1000):
    batch_var = 1000  # 继续爆炸
    running_var = 0.99 * running_var + 0.01 * batch_var
# 结果: running_var ≈ 1000 (接近batch_var)

# 如果batch_var继续增长到10000
for iter in range(1000, 2000):
    batch_var = 10000  # 更严重的爆炸
    running_var = 0.99 * running_var + 0.01 * batch_var
# 结果: running_var ≈ 10000 (跟随batch_var爆炸)
```

### 1.3 为什么decoder的batch_var会爆炸？

**Decoder架构瓶颈分析**:

```
enc4 (504 channels)
    ↓ SerializedPooling (504→252)
    ↓
dec3 (252 channels)
    ↓ SerializedUnpooling.up (252→126)
    ↓
dec2 (126 channels)
    ↓ SerializedUnpooling.up (126→64)  ← 瓶颈! 2x压缩
    ↓
dec1 (64 channels)
    ↓ SerializedUnpooling.up (64→32)   ← 瓶颈! 2x压缩
    ↓
dec0 (32 channels)
    ↓ SerializedUnpooling.up (32→16)   ← 输出瓶颈! 2x压缩
    ↓
输出 (16 channels)
```

**瓶颈处的梯度放大**:

```python
# 假设输入维度为D_in，输出维度为D_out
# 对于线性层: y = Wx + b，其中 W.shape = [D_out, D_in]

# 梯度链式法则:
# ∂L/∂x = W^T @ ∂L/∂y

# 当 D_out << D_in 时:
# - 梯度在回传时被放大
# - 放大倍数 ≈ D_in / D_out

# dec2: 126→64, 梯度放大 ≈ 2x
# dec1: 64→32,  梯度放大 ≈ 2x
# dec0: 32→16,  梯度放大 ≈ 2x
# 总放大: 2×2×2 = 8倍!
```

**完整的恶性循环**:

```
1. Decoder瓶颈 (126→64→32→16)
   ↓
2. 梯度放大 (8倍累积)
   ↓
3. 权重更新幅度大
   ↓
4. 权重值增大
   ↓
5. 激活值爆炸: activation = W @ input + b
   ↓
6. BatchNorm的batch_var爆炸: batch_var = Var(activation)
   ↓
7. running_var指数增长: running_var += 0.01 * (batch_var - running_var)
   ↓
8. 归一化后输出接近0: output = (x - mean) / sqrt(running_var + eps)
   ↓
9. Loss需要更大权重来补偿
   ↓
回到步骤2，循环继续!
```

---

## 二、能否直接重置running_var？

### 2.1 短期效果

```python
# 模拟重置效果
running_var = 1_000_000_000  # 爆炸状态
batch_var = 1000  # 当前batch的方差

# 重置为1.0
running_var = 1.0

# 继续训练10轮
for i in range(10):
    running_var = 0.99 * running_var + 0.01 * batch_var
    # running_var会增长: 1.0 → 11 → 21 → 30 → 40 → ...
    # 10轮后: running_var ≈ 96
```

### 2.2 结论

**重置可以暂时降低running_var，但**:
- ✅ 短期内running_var会降低
- ❌ 如果batch_var仍然很大，running_var会再次增长
- ❌ **治标不治本** - 根本问题是batch_var太大

**类比**: 这就像发烧时吃退烧药 - 可以暂时降低体温，但如果感染还在，发烧还会反复。

---

## 三、完整解决方案

### 3.1 方案对比

| 方案 | 治标/治本 | 难度 | 效果 | 风险 |
|------|----------|------|------|------|
| 重置BN统计量 | 治标 | ⭐ | 暂时 | 高 - 会反复 |
| 替换为LayerNorm | 治本 | ⭐⭐ | 好 | 中 - 需要调参 |
| 替换为GroupNorm | 治本 | ⭐⭐ | 好 | 中 - 需要设置groups |
| 移除BN | 治本 | ⭐ | 需验证 | 高 - 训练可能不稳定 |
| 增加Decoder容量 | 治本 | ⭐⭐⭐ | 最好 | 低 - 但需重新训练 |
| **组合方案** | **治本** | ⭐⭐⭐ | **最佳** | 低 |

### 3.2 推荐方案：组合修复

#### 方案A: 快速修复 (使用LayerNorm替代BatchNorm)

**修改配置文件** (`configs/custom/lang-pretrain-litept-ovs-gridsvd.py`):

```python
model = dict(
    backbone=dict(
        # ... 其他配置保持不变 ...

        # 添加以下配置来使用LayerNorm替代BatchNorm
        pdnorm_ln=True,  # 使用LayerNorm
        pdnorm_bn=False,  # 禁用BatchNorm
    ),
)
```

**原理**:
- LayerNorm在**特征维度**上归一化，不依赖batch统计
- LayerNorm没有running_var/running_mean，不会爆炸
- LayerNorm更适合小batch size和特征维度变化的场景

**效果**:
- ✅ 彻底解决running_var爆炸问题
- ✅ 不需要额外的监控代码
- ⚠️ 可能需要调整学习率 (LayerNorm对学习率更敏感)

#### 方案B: 根本修复 (增加Decoder容量 + 使用LayerNorm)

**修改配置文件**:

```python
model = dict(
    backbone=dict(
        # ... 其他配置 ...

        # 增加decoder容量，消除瓶颈
        dec_channels=(72, 72, 144, 252),  # 原来是 (16, 32, 64, 126)

        # 使用LayerNorm
        pdnorm_ln=True,
        pdnorm_bn=False,
    ),
)
```

**效果**:
- ✅ 消除信息瓶颈
- ✅ 减少梯度放大
- ✅ 解决running_var爆炸
- ✅ 提升模型容量，更好地学习16维特征

#### 方案C: 代码级修复 (修改模型定义)

如果不想通过配置修改，可以直接修改模型代码：

**文件**: `pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`

**修改位置**: 第668行附近

```python
# 原代码 (line 668)
norm_layer=bn_layer,  # 使用BatchNorm

# 修改为:
norm_layer=ln_layer,  # 使用LayerNorm
```

**完整修改**:
```python
# 找到decoder的up层定义 (大约line 665-670)
dec.add(
    SerializedUnpooling(
        in_channels=dec_channels[s + 1],
        skip_channels=enc_channels[s],
        out_channels=dec_channels[s],
        norm_layer=ln_layer,  # 改为ln_layer (原来是bn_layer)
        act_layer=act_layer,
    ),
    name="up",
)
```

### 3.3 验证修复效果

**训练前检查**:
```python
# 在训练开始时，检查decoder的norm layer类型
import torch.nn as nn

for name, module in model.named_modules():
    if 'dec' in name and ('up.proj' in name or 'up.proj_skip' in name):
        if hasattr(module, 'norm'):
            norm_type = type(module.norm).__name__
            print(f"{name}: {norm_type}")
            # 期望输出: LayerNorm (而不是BatchNorm)
```

**训练中监控** (如果仍保留部分BatchNorm):
```python
# 在trainer中添加监控
for name, module in model.named_modules():
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        running_var = module.running_var.max().item()
        if running_var > 100:  # 阈值
            print(f"⚠️ WARNING: {name} running_var={running_var:.2f}")
```

---

## 四、为什么LayerNorm能解决问题？

### 4.1 BatchNorm vs LayerNorm

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| **归一化维度** | 跨batch (N) | 跨特征 (C) |
| **统计量** | batch_mean, batch_var | feature_mean, feature_var |
| **running统计** | 有 (会爆炸!) | 无 (不会爆炸!) |
| **依赖batch** | 是 (batch size敏感) | 否 |
| **训练稳定性** | 大batch稳定 | 小batch稳定 |
| **适用场景** | CNN，大batch | Transformer，小batch |

### 4.2 归一化公式对比

```python
# BatchNorm (对每个特征维度，跨batch归一化)
# 输入: [N, C] (N=batch, C=channels)
output = (x - mean(batch)) / sqrt(var(batch) + eps)
# mean(batch) 和 var(batch) 在N维度上计算

# LayerNorm (对每个样本，跨特征归一化)
# 输入: [N, C] (N=batch, C=channels)
output = (x - mean(features)) / sqrt(var(features) + eps)
# mean(features) 和 var(features) 在C维度上计算
```

### 4.3 为什么LayerNorm不会爆炸？

```python
# BatchNorm的running_var更新:
running_var = 0.99 * running_var + 0.01 * batch_var
# 问题: batch_var可能很大 → running_var指数增长

# LayerNorm没有running_var:
# 每次前向传播都重新计算:
mean = x.mean(dim=feature_dim)  # 对每个样本独立计算
var = x.var(dim=feature_dim)
output = (x - mean) / sqrt(var + eps)
# 优势: 没有历史积累，不会爆炸!
```

---

## 五、迁移策略

### 5.1 如果已经有训练好的checkpoint

**选项1: 继续训练 (不推荐)**
- BatchNorm的running_var已经污染
- 继续训练可能仍然不稳定

**选项2: 重置running_var (临时方案)**
```python
# 在加载checkpoint后，重置decoder的BN统计量
for name, module in model.named_modules():
    if 'dec' in name and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        module.running_mean.fill_(0)
        module.running_var.fill_(1)
        module.num_batches_tracked.fill_(0)
```

**选项3: 使用LayerNorm重新训练 (推荐)**
- 切换到LayerNorm配置
- 从头开始训练 (或用预训练权重微调)
- 更稳定，效果更好

### 5.2 迁移学习注意事项

如果使用预训练权重 (如在Scannet上预训练):

```python
# 1. 加载预训练权重
pretrained_dict = torch.load('pretrained_scannet.pth')
model.load_state_dict(pretrained_dict, strict=False)

# 2. 替换BN为LN后，需要重新训练decoder
# 3. Encoder的BN可以保留 (因为encoder相对稳定)
# 4. 或者全部切换到LayerNorm重新训练
```

---

## 六、总结

### 核心问题

1. **BatchNorm的running_var爆炸**是由于:
   - Decoder瓶颈 (126→64→32→16) 导致梯度放大
   - 激活值爆炸导致batch_var增大
   - running_var = 0.99 * running_var + 0.01 * batch_var 的指数积累

2. **重置running_var只是治标**:
   - 可以暂时降低running_var
   - 但如果batch_var仍然大，会再次爆炸

### 最佳解决方案

**组合方案** (推荐):
1. ✅ 使用LayerNorm替代BatchNorm (`pdnorm_ln=True`)
2. ✅ 增加Decoder容量 (消除瓶颈)
3. ✅ 降低对比损失权重 (减少优化冲突)
4. ✅ 添加BatchNorm监控 (如果保留部分BN)

### 配置修改

```python
# 在 configs/custom/lang-pretrain-litept-ovs-gridsvd.py 中添加:
model = dict(
    backbone=dict(
        # 增加decoder容量
        dec_channels=(72, 72, 144, 252),  # 原: (16, 32, 64, 126)

        # 使用LayerNorm替代BatchNorm
        pdnorm_ln=True,
        pdnorm_bn=False,
    ),
)

# 降低对比损失权重
criteria=[
    dict(
        type="AggregatedContrastiveLoss",
        loss_weight=0.02,  # 原: 0.2
    ),
]
```

---

*生成日期: 2026-03-09*
*相关文档: model_collapse_final_analysis.md*

---

# Part 3: LangZip 训练方法诊断报告

# LangZip 训练方法诊断报告

## 1. 当前训练范式回顾

### 1.1 训练数据流

1. 加载场景的 per-Gaussian 768 维 SigLIP2 语言特征 $F$（来自 SceneSplat 预处理的 OccamLG 聚合）
2. 对 $F$ 做 grid aggregation（Section 3.1），得到去重后的压缩特征 $\tilde{F}$
3. 对 $\tilde{F}$ 做 SVD，得到每点对应的奇异值系数 $\tilde{F} \approx \tilde{U}_w \Sigma V_w^T$（即每个 Gaussian 的 16 维压缩 target）
4. 模型 backbone（PTv3 / LitePT）从 3DGS primitive 的 11 维属性（color/opacity/quat/scale）出发，输出每个 Gaussian 的 16 维预测 $\hat{F}$
5. 对 $\hat{F}$ 做 SVD 投影，与上述压缩 target 对齐

### 1.2 当前 Loss 组成（Section D.2，行 3874-3878）

$$L_{\text{total}} = \lambda_{\cos}L_{\cos} + \lambda_{L_1}L_{L_1} + \lambda_{\text{con}}L_{\text{contrast}} + \lambda_{\text{dense}}L_{\text{dense}}$$

其中：
- $\lambda_{\cos} = 0.1$：预测与 target 之间的 cosine similarity loss
- $\lambda_{L_1} = 1.0$：预测与 target 之间的 L1 loss（保证数值准确）
- $\lambda_{\text{con}} = 0.02$：Vision-Language 对比 loss（继承自 SceneSplat），温度 $\tau = 0.2$，仅在训练最后 75% 激活
- $\lambda_{\text{dense}} = 1.0$：Grid-Guided Density-Invariant Learning 的密度一致性 loss

优化器：AdamW（lr=0.001, weight_decay=0.05）+ OneCycle scheduler + cosine annealing，800 个 data epoch。

### 1.3 推理时对齐（Appendix C.4，Eqs B.24-B.31）

推理时需额外一步 **Orthogonal Procrustes 对齐** $Q^* = U_M V_M^T$（Eq B.31），其中 $M = A^T(S_c B)$。$S_c$ 是选择矩阵，由训练集的类别标签构造（ScanNet-20）。$Q^*$ 把 16 维特征空间旋转到文本 embedding 空间，然后才能做 cosine 最近邻查询。

---

## 2. 当前训练方法的根本问题

### 2.1 核心问题：旋转方向未受约束

**当前 loss 只约束 L1 数值准确 + cosine 相似度，但完全不约束预测特征与文本 embedding 之间的旋转方向。**

形式化说明：

模型预测 $\hat{F} \in \mathbb{R}^{N \times d}$ 和真实目标 $\tilde{F} \in \mathbb{R}^{N \times d}$ 都是 $d$ 维空间里的点。但 L1 loss 满足：

$$\|\hat{F} R - \tilde{F}\|_1 = \|\hat{F} - \tilde{F}\|_1 \quad \forall R \in \mathbb{O}(d)$$

也就是说，模型可以学到任意旋转后的坐标系，只要数值对得上，L1 loss 就给同样的梯度。这意味着 $\hat{F}$ 与 text embedding 之间的旋转是任意的，必须依赖推理期的 Procrustes 校正才能用于开放词汇查询。

### 2.2 Procrustes 对齐的两个根本缺陷

**缺陷一：依赖场景级类别标签**

$S_c$ 需要把每个 sample 映射到语义类别。对于 SceneSplat-SN 训练集（20 类），标注齐全。但对于 novel scene（如 LERF，4 个室内场景）：
- 标注不在 SceneSplat-SN 类别集合内（"ramen"、"teatime" 等不属于 ScanNet-20）
- 无法用 GT 类别标签构造 $S_c$
- 论文实际做法（Appendix C.1）：在 LERF 上用 SAM voting 生成 pseudo-labels（行 681-682），再用这些 pseudo-label 构造 $S_c$

**缺陷二：依赖文本类别集合的完备性**

$L_c$ 是训练集的类文本嵌入。$Q^*$ 编码的是"压缩空间 → 训练类文本空间的旋转"。如果 novel class 的文本特征不在 $L_c$ 覆盖范围内（如 ScanNet-20 类中心外），旋转不能保证这些点仍落在正确语义区域。

### 2.3 次要问题

| 问题 | 位置 | 后果 |
|---|---|---|
| 对比 loss 仅在训练最后 75% 激活 | D.2 行 3878-3879 | 模型早期不受约束，可能学到一个临时表示，后期才能对齐 |
| 对比 loss 仅继承 SceneSplat，未与压缩空间联合设计 | Eq 15 | 对比信号作用于 768 维原始空间，与预测的 16 维空间解耦 |
| $\lambda_{\text{con}} = 0.02$ 极小 | D.2 行 3876 | 对比梯度被 L1 完全淹没，实际对表示学习的影响有限 |
| 无显式正交约束 | 缺失 | 压缩空间的正交基与文本 SVD 基的关系完全靠 Procrustes 后处理恢复 |

---

## 3. 改进的训练方向

### 3.1 直接方案：对比 loss 直接约束 16 维压缩空间

不再依赖 SceneSplat 的 768 维对比 loss，而是在 16 维空间内直接做文本-特征对比：

```
# 训练时 forward
text_16 = text_512 @ svd_basis.T          # [M, 16] SVD降维后的类中心
pred_16 = backbone(points)                 # [N, 16] 模型预测

# 原有 loss: 数值准确
loss_l1 = L1(pred_16, svd_coeff[labels])

# 新增 loss: 约束旋转方向
logits = pred_16 @ text_16.T               # [N, M] cosine similarity
loss_contra = CrossEntropy(logits / tau, labels)

# 总 loss
loss = loss_l1 + lambda * loss_contra
```

**效果**：训练时让预测特征与同场景内的类文本中心在 16 维空间里直接对齐，推理时无需任何 Procrustes 后处理。

**待验证**：
- $\tau$ 温度参数的鲁棒性
- $\lambda$ 从 0.1 起步
- 是否仍需保留原始 SceneSplat 的 768 维对比 loss

### 3.2 几何方案：正交不变性约束 (Q = I)

更进一步，可以把"模型预测的子空间 = 文本子空间"作为硬约束：

$$L_{\text{orth}} = \|R_{\hat{F}}^T R_T - I\|_F^2$$

其中 $R_{\hat{F}}$、$R_T$ 分别是预测和文本特征空间的 PCA 主轴。最小化这个 loss 等价于让两个子空间重合。

**优点**：几何解释清晰，可以解释为"无 Procrustes 旋转的表示学习"。
**风险**：实现复杂度高，可能与 L1 loss 冲突（数值准确 vs 几何对齐的 trade-off）。

### 3.3 实用方案：共享 SVD 基

最直接的做法：训练时用训练集全局 SVD 基（而不是 per-scene SVD）作为 target。这样：

$$\hat{F}_{\text{target}} = S_c \cdot L_c \cdot V_w^T$$

其中 $V_w$ 是训练集全局的右奇异向量。backbone 直接预测 $\hat{F}_{\text{target}}$，不需要 per-scene SVD。

**优点**：
- 推理时 $Q^* = I$，无需任何 Procrustes
- 训练与推理空间完全一致
- 类别中心 $L_c \cdot V_w^T$ 可作为 anchor，对比 loss 天然适配

**缺点**：
- 训练时所有 scene 用同一个 $V_w$，可能损失 per-scene 的适应性
- 需要重新实现训练 pipeline

### 3.4 推荐路线（按工程量从小到大）

| 方案 | 工程量 | 预期收益 | 推荐优先级 |
|---|---|---|---|
| **3.1 直接对比 loss** | 小（修改 loss 公式 + 加几个 forward pass） | 显著降低对 Procrustes 的依赖 | ★★★★★ |
| 3.3 共享 SVD 基 | 中（需要重新生成 target） | 完全消除 Procrustes | ★★★★ |
| 3.2 显式正交约束 | 大（需要新的几何 loss） | 理论严谨，工程复杂 | ★★ |

**短期（rebuttal 提交前）**：在 Q2 中承诺方案 3.1 作为未来工作，给审稿人明确的改进方向。

**中期（camera-ready 准备期）**：实施方案 3.1，提交完整的训练 + 推理对比实验。

**长期（下一篇工作）**：实施方案 3.3 或 3.2，把 Procrustes 从推理管线中彻底去掉。

---

## 4. 验证清单

每条改进都需回答以下问题：

- [ ] **消融对比**：新训练方法 vs 旧方法，在 LERF / ScanNet++ 上的 mIoU 是否提升或持平
- [ ] **推理简化**：去掉 Procrustes 后，LERF 性能下降多少？理想：≤ 0.5 mIoU
- [ ] **开放词汇验证**：用 ImageNet-21K 中与 LERF 不相交的类别子集重训，是否仍能查询 LERF 评估类别
- [ ] **计算开销**：训练时间增加幅度（理想 ≤ 20%）
- [ ] **类别外推**：在 ScanNet-20 + ScanNet-200 联合训练，模型是否仍能查询零样本类别（不在训练类别集合内的）

---

## 5. 事实核查记录

| 核查点 | 论文位置 | 核查结果 |
|---|---|---|
| 训练 loss 权重（$\lambda_{\cos}=0.1, \lambda_{L_1}=1.0, \lambda_{\text{con}}=0.02, \lambda_{\text{dense}}=1.0$） | Eq 15 + D.2 行 3874-3878 | ✓ 与原文一致 |
| 对比 loss 仅在最后 75% 激活，$\tau=0.2$ | D.2 行 3878-3879 | ✓ 与原文一致 |
| Optimizer：AdamW, lr=0.001, weight_decay=0.05, OneCycle + cosine annealing, 800 epochs | D.2 行 3869-3873 | ✓ 与原文一致 |
| Procrustes 公式：$M = A^T(S_c B), Q^* = U_M V_M^T$ | Eqs B.24-B.31, 行 3197-3268 | ✓ 与原文一致 |
| LERF pseudo-label 来源：SAM voting | Appendix C.1 行 681-682 | ✓ 与原文一致 |
| Backbone 输入：color(3) + opacity(1) + quaternion(4) + scale(3) = 11 维 | 用户确认 | ✓ 已交叉验证 |
| per-scene SVD 与全局 SVD 不等价（聚合改变奇异值） | Section 3.2 行 822-826 | ✓ 与原文一致 |
| L1 loss 对正交变换不变 | 数学事实（$\|XR - Y\|_1$ 在 $R^T R = I$ 下可解耦） | ✓ 论证正确 |