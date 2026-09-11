# 3D 开放词汇查询流程分析与改进方案

## 1. 整体流程

### 1.1 数据格式

每个 3DGS 场景是一组 `.npy` 文件：

| 文件 | 维度 | 说明 |
|------|------|------|
| `coord.npy` | (N, 3) | 点坐标 |
| `color.npy` | (N, 3) | RGB |
| `scale.npy` | (N, 3) | Gaussian 尺度 |
| `quat.npy` | (N, 4) | Gaussian 旋转 |
| `opacity.npy` | (N, 1) | 不透明度 |
| `lang_feat.npy` | (N, 768) | 语言特征（SigLIP2 提取） |

### 1.2 2D → 3D 特征投影

- `scripts/preprocess_siglip2_sam2.py` + `tools/data/gaussian_feature_extractor.py`
- SigLIP2 (ViT-SO400M-14, 768-dim) 提取每帧 2D 语言特征
- SAM2 提供逐像素分割 mask
- 通过可微渲染/投影，将 2D 特征投射到 3D Gaussian → `lang_feat.npy`

### 1.3 3D 骨干网络

- `pointcept/models/point_transformer_v3/` — SparseTransformerV3 / LitePT
- 输入：771 维 (768 lang_feat + 3 color)
- 体积化 → spconv 稀疏卷积 → SerializedAttention → Linear decoder
- 输出：每个点的预测特征

### 1.4 SVD 压缩

- `tools/compression/compress_grid_svd.py`
- 按空间网格对 lang_feat 做 RPCA + SVD，存 rank=8/16/32 的 U 矩阵
- 运行时解压：`U @ SVD_coefficients` → 近似恢复 768-dim 特征
- 训练时模型直接预测 16-dim SVD 系数，而非完整 768-dim

### 1.5 查询推理

两个阶段：
1. 模型 forward → 输出16-dim 特征
2. 与 CLIP 文本 embedding 做余弦相似度 → 匹配

---

## 2. 核心问题：推理时的维度对齐

### 2.1 问题描述

- 模型输出：**16-dim** SVD 系数
- 文本查询：**512-dim** CLIP embedding（或768-dim SigLIP2）
- 维度不匹配，无法直接算 cosine similarity

### 2.2 当前做法（有缺陷）

```
训练时：L1 loss on 16-dim SVD 系数（无旋转约束）
推理时：
  1. 对 text 做 SVD 降维 → 16-dim
  2. 用 GT labels 算 Procrustes Q 矩阵 → 对齐
  3. cosine similarity
```

**缺陷**：Procrustes 需要 GT 语义标签来算 Q 矩阵，推理时没有 GT labels，Procrustes 被跳过，直接用未对齐的特征做 cosine（效果差）。

### 2.3 为什么需要 Procrustes

训练时 L1 loss 只约束数值大小，不约束旋转方向。模型输出的 16-dim 和 SVD-reduced text embedding 虽然在同一空间，但坐标系不同，差一个正交旋转矩阵 Q。

---

## 3. 解决方案

### 方案 A：训练时约束 Q=I（推荐）

**原理**：训练时加对比学习 loss，直接约束模型输出和 text embedding 在同一旋转空间，推理时无需 Procrustes。

```python
# 训练时
text_16 = text_512 @ svd_basis.T          # [M, 16] SVD降维类中心
pred_16 = backbone(points)                 # [N, 16]

# 原有 loss
loss_l1 = L1(pred_16, svd_coeff[labels])

# 新增：对比 loss，约束旋转方向
logits = pred_16 @ text_16.T / tau         # [N, M]
loss_contra = CrossEntropy(logits, labels)

loss = loss_l1 + lambda * loss_contra
```

**效果**：

| | 训练时 | 推理时 |
|---|---|---|
| 当前 | L1 only | 需要 Procrustes（依赖 GT labels） |
| 方案 A | L1 + 对比 loss | 直接 cosine，无需对齐 |

**优点**：最彻底，推理零开销，不依赖 GT labels。

**注意事项**：
- `text_16` 用 SVD 降维后的类中心（M 个），不是全部点
- `tau` 温度参数，控制分布锐度，建议从 0.1 开始调
- `lambda` 平衡系数，建议从 0.1 开始调
- 对比 loss 不替换 L1，L1 保证数值准确，对比 loss 保证旋转对齐

### 方案 B：学投影头代替 SVD

```python
proj = nn.Linear(512, 16)                  # 可训练
text_16 = proj(text_512)                    # [M, 16]
pred_16 = backbone(points)                  # [N, 16]
loss = L1(pred_16, text_16[labels])
```

投影矩阵和模型端到端训练，自然对齐。但改变了模型结构。

### 方案 C：推理时直接 cosine（不改训练，最快验证）

```python
# 对text做SVD降维
Vt = np.linalg.svd(text_emb, full_matrices=False)[2]
text_16 = text_emb @ Vt.T

# 直接cosine，跳过Procrustes
logits = pred_16 @ text_16.T
```

训练时 L1 loss 已让模型大致在 SVD 系数空间内，如果旋转不大，效果可能还行。作为 baseline 验证。

### 方案 D：一轮伪标签对齐（不改训练，效果更好）

```python
# 无对齐，直接算相似度
text_16 = text_emb @ Vt.T
sim0 = pred_16 @ text_16.T
pseudo_labels = sim0.argmax(dim=1)

# 用伪标签算一次Procrustes
Q = compute_procrustes(pred_16, text_16[pseudo_labels])
pred_aligned = pred_16 @ Q
logits = pred_aligned @ text_16.T
```

只做一轮，不迭代。伪标签初始准确率 > 50% 时一轮够用。

---

## 4. 推荐路径

1. **先试方案 C**（一行代码改动），看 baseline 效果
2. 如果不够好，**加方案 D**（一轮伪标签对齐）
3. **长期用方案 A**（改训练），彻底解决问题

---

## 5. 关键文件索引

| 文件 | 作用 |
|------|------|
| `pointcept/models/default.py:78-147` | LangPretrainer 模型定义 |
| `pointcept/engines/test.py:409-452` | 推理时 Procrustes 对齐逻辑 |
| `pointcept/engines/hooks/evaluator.py:816-869` | 训练时验证的 Procrustes 计算 |
| `tools/projection/compute_procrustes_alignment_simple.py:135-216` | Procrustes Q 计算（无标签版） |
| `tools/projection/compute_procrustes_alignment_simple.py:219-336` | Procrustes Q 计算（有标签版） |
| `tools/visualization/query_open_vocabulary.py:130-145` | 可视化查询（L2 fallback） |
| `configs/custom/lang-pretrain-litept-ovs-gridsvd.py` | SVD 压缩训练配置 |
| `tools/compression/compress_grid_svd.py` | 网格 SVD 压缩脚本 |

---

## 6. 文献参考

- **Wasserstein-Procrustes** (Grave et al., ICML 2019): 无监督对齐两个特征空间，联合估计正交矩阵和置换矩阵，交替优化
- **OpenScene** (Peng et al., CVPR 2023): 训练时直接用 CLIP 2D 特征做监督，让 3D 特征直接落在 CLIP 空间，推理时无需对齐
- **Manifold Alignment using Procrustes Analysis** (Wang et al., ICML 2008): 用 landmark points 做 Procrustes 对齐低维流形
