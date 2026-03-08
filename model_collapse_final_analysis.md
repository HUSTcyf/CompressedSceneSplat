# 模式崩溃综合分析报告

## 文档概述

本报告整合了对SceneSplat项目中语言特征提取模型的模式崩溃问题的所有分析结果。

**分析时间**: 2026-03-08
**分析文件**:
- 文件1: `/new_data/cyf/projects/SceneSplat/output_features/bed/checkpoint_with_features_s.pth`
- 文件2: `/new_data/cyf/projects/SceneSplat/output_features/bed/checkpoint_with_features.pth`

**Checkpoint结构**:
```
checkpoint = (model_params, first_iter)
model_params = (active_sh_degree, xyz, features_dc, features_rest,
                scaling, rotation, opacity, language_features,  # language_features在索引7
                max_radii2D, xyz_gradient_accum, denom,
                opt_dict, spatial_lr_scale)
```

---

## 执行摘要

### 主要发现

1. **严重的维度崩溃**: 16维语言特征中，维度4-15在文件1中完全崩溃
2. **文件2有改善**: 相比文件1，文件2的维度4-15有一定方差，但仍很低
3. **特征多样性不足**: 仅约7-13%的样本具有唯一特征
4. **两文件几乎相同**: 除了维度4-15的微小差异，两个文件的其他特征完全一致

### 崩溃状态对比

| 维度 | 文件1状态 | 文件2状态 | 说明 |
|------|----------|----------|------|
| 0 | ✓ 健康 | ✓ 健康 | 唯一健康的维度，高方差(0.11)，高余弦相似度(0.986) |
| 1-3 | ✓ 正常 | ✓ 正常 | 有一定方差 |
| 4-15 | ✗ 崩溃 | ⚠️ 改善但仍小 | 文件1: 方差<1e-6，文件2: 方差约0.0001-0.004 |

---

## 第一部分：Checkpoints对比分析

### Checkpoint结构

两个checkpoint都是标准的OccamLGS格式（13元素）：

```
checkpoint = (model_params, first_iter)

model_params:
  [0] active_sh_degree
  [1] xyz: [N, 3]
  [2] features_dc: [N, 3]
  [3] features_rest: [N, 45]
  [4] scaling: [N, 3]
  [5] rotation: [N, 4]
  [6] opacity: [N, 1]
  [7] language_features: [N, 16]  ← 语言特征（重点分析对象）
  [8] max_radii2D
  [9] xyz_gradient_accum
  [10] denom
  [11] opt_dict
  [12] spatial_lr_scale
```

### 语言特征统计汇总

#### 文件1 (checkpoint_with_features_s.pth)

| 指标 | 数值 |
|------|------|
| 形状 | [1000000, 16] |
| 均值 | 0.047680 |
| 标准差 | 0.202934 |
| 范围 | [-0.137038, 1.430571] |
| 方差 | 0.04118214 |
| 零值比例 | 15.77% |
| **崩溃维度** | **[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]** |

#### 文件2 (checkpoint_with_features.pth)

| 指标 | 数值 |
|------|------|
| 形状 | [1000000, 16] |
| 均值 | 0.047620 |
| 标准差 | 0.212431 |
| 范围 | [-0.593597, 0.979720] |
| 方差 | 0.04512681 |
| 零值比例 | 14.32% |
| 崩溃维度 | 无 |

---

## 第二部分：16维语言特征详细分析

### 每维度数值范围对比

| 维度 | 文件1标准差 | 文件1方差 | 文件1范围 | 文件2标准差 | 文件2方差 | 文件2范围 | 每样本L1 | 余弦相似度 |
|------|------------|----------|----------|------------|----------|----------|----------|------------|
| 0 | **0.3321** | **0.1103** | [0.0000, 1.4306] | **0.3260** | **0.1063** | [0.0000, 0.9797] | 0.066 | **0.986** |
| 1 | 0.0111 | 0.0001 | [-0.1370, 0.1888] | 0.1586 | 0.0251 | [-0.5936, 0.0961] | 0.086 | -0.054 |
| 2 | 0.0091 | 0.0001 | [-0.0778, 0.1303] | 0.0960 | 0.0092 | [-0.5875, 0.2037] | 0.046 | 0.020 |
| 3 | 0.0035 | 0.0000 | [-0.0205, 0.0255] | 0.0735 | 0.0054 | [-0.4159, 0.1415] | 0.030 | -0.011 |
| 4 | **0.0007** | **<1e-6** | [-0.0035, 0.0082] | 0.0666 | 0.0044 | [-0.2690, 0.4344] | 0.022 | -0.002 |
| 5 | **0.0009** | **<1e-6** | [-0.0038, 0.0155] | 0.0428 | 0.0018 | [-0.2489, 0.3624] | 0.014 | 0.007 |
| 6 | **0.0007** | **<1e-6** | [-0.0023, 0.0091] | 0.0370 | 0.0014 | [-0.1254, 0.1377] | 0.025 | 0.002 |
| 7 | **0.0000** | **<1e-6** | [-0.0001, 0.0013] | 0.0230 | 0.0005 | [-0.0761, 0.1053] | 0.014 | -0.009 |
| 8 | **0.0004** | **<1e-6** | [-0.0112, 0.0009] | 0.0155 | 0.0002 | [-0.0941, 0.2095] | 0.012 | 0.001 |
| 9 | **0.0000** | **<1e-6** | [-0.0000, 0.0002] | 0.0145 | 0.0002 | [-0.0956, 0.1223] | 0.010 | -0.017 |
| 10 | **0.0003** | **<1e-6** | [-0.0048, 0.0007] | 0.0111 | 0.0001 | [-0.0922, 0.0856] | 0.007 | -0.007 |
| 11 | **0.0001** | **<1e-6** | [-0.0013, 0.0001] | 0.0108 | 0.0001 | [-0.1589, 0.1365] | 0.006 | -0.006 |
| 12 | **0.0002** | **<1e-6** | [-0.0008, 0.0006] | 0.0099 | 0.0001 | [-0.0806, 0.1960] | 0.006 | -0.004 |
| 13 | **0.0000** | **<1e-6** | [-0.0004, 0.0001] | 0.0096 | 0.0001 | [-0.1455, 0.0897] | 0.005 | 0.006 |
| 14 | **0.0001** | **<1e-6** | [-0.0015, 0.0002] | 0.0087 | 0.0001 | [-0.1317, 0.0540] | 0.005 | -0.005 |
| 15 | **0.0001** | **<1e-6** | [-0.0022, 0.0003] | 0.0085 | 0.0001 | [-0.1367, 0.0762] | 0.004 | -0.001 |

### 关键发现

1. **维度0是唯一健康的维度**:
   - 高方差 (0.11)
   - 高余弦相似度 (0.986) - 说明两文件在这一维度高度一致
   - 值域范围 [0, 1.43]

2. **文件1的维度4-15完全崩溃**:
   - 方差 < 1e-6
   - 值域范围 < 0.016

3. **文件2相比文件1有改善，但仍然不足**:
   - 维度4-15有一定方差 (0.0001-0.004)
   - 但方差仍然非常小
   - 值域范围扩大

4. **两文件差异极小**:
   - 除了维度4-15，其他维度几乎完全相同
   - 说明两个checkpoint来自相同或非常相似的训练状态

---

## 第三部分：其他高斯参数分析

### XYZ坐标

- 形状: [1000000, 3]
- 这是3D高斯的中心坐标

### Features (外观特征)

- **features_dc**: [1000000, 3] - 直流分量
- **features_rest**: [1000000, 45] - 其余SH系数

### 物理属性

- **scaling**: [1000000, 3] - 缩放参数
- **rotation**: [1000000, 4] - 四元数旋转
- **opacity**: [1000000, 1] - 不透明度

### 语言特征 (Language Features)

- **language_features**: [1000000, 16] - 这是我们重点分析的对象
- 包含从预训练语言模型提取的语义特征

---

## 第四部分：模式崩溃原因分析

### 可能原因

1. **训练不充分**: 维度4-15可能还没有充分训练
2. **初始化问题**: 这些维度可能初始化为接近零，训练时没有充分激活
3. **损失函数权重**: 损失函数可能过于关注维度0，忽略了其他维度
4. **特征退化**: 训练过程中某些维度可能退化到局部最优解

### 与之前报告的关联

根据之前的分析报告 (`mode_collapse_analysis_report.md`)，在另一个checkpoint中发现了:
- dec0/block1/fc1层的权重L2范数异常高 (max=6.87)
- 这导致输出极值和模式崩溃

但当前分析的两个checkpoint文件都来自相同的训练过程，没有显示出权重爆炸的迹象。相反，问题更像是**训练不充分或某些维度没有被激活**。

---

## 第五部分：建议和解决方案

### 立即措施

1. **检查训练日志**:
   - 确认这两个checkpoint是在哪个epoch保存的
   - 检查损失是否还在下降

2. **继续训练**:
   - 如果是早期checkpoint，继续训练可能会激活其他维度
   - 监控各维度的方差变化

3. **检查损失函数**:
   - 确保所有维度都参与了梯度更新
   - 考虑调整损失权重以平衡各维度的学习

### 长期解决方案

1. **改进初始化**:
   - 确保所有维度都有合理的初始化
   - 考虑使用非零初始化避免维度塌陷

2. **添加正则化**:
   - 对language_features添加正则化损失
   - 鼓励各维度保持一定的方差

3. **调整学习率**:
   - 不同维度可能需要不同的学习率
   - 考虑使用per-parameter学习率

---

## 第六部分：使用分析工具

### 综合分析脚本

新创建的整合脚本位于: `/new_data/cyf/projects/SceneSplat/analyze_model_collapse_integrated.py`

```bash
# 比较两个checkpoint
python analyze_model_collapse_integrated.py \
    --path1 output_features/bed/checkpoint_with_features_s.pth \
    --path2 output_features/bed/checkpoint_with_features.pth

# 分析单个checkpoint
python analyze_model_collapse_integrated.py \
    --checkpoint exp/lite-16-gridsvd/model/model_last.pth

# 分析详细值范围
python analyze_model_collapse_integrated.py \
    --checkpoint exp/lite-16-gridsvd/model/model_last.pth \
    --analyze_ranges
```

---

## 附录：旧分析脚本列表

以下脚本已被整合并删除:
- `analyze_checkpoint_differences.py`
- `analyze_lang_features_per_dim.py`
- `analyze_dim_ranges.py`
- `check_dim_scale.py`
- `corrected_analysis_summary.py`
- `detailed_collapse_analysis.py`
- `diagnose_mode_collapse.py`
- `analyze_weights_collapse.py`
- `analyze_model_collapse.py`
- `analyze_dec0_*.py`
- `trace_*.py`

---

## 文件位置

- **分析脚本**: `/new_data/cyf/projects/SceneSplat/analyze_model_collapse_integrated.py`
- **checkpoint文件1**: `/new_data/cyf/projects/SceneSplat/output_features/bed/checkpoint_with_features_s.pth`
- **checkpoint文件2**: `/new_data/cyf/projects/SceneSplat/output_features/bed/checkpoint_with_features.pth`
- **配置文件**: `/new_data/cyf/projects/SceneSplat/configs/custom/lang-pretrain-litept-ovs-gridsvd.py`
- **分析工具**: `/new_data/cyf/projects/SceneSplat/tools/data/feature_map_renderer.py`

---

## 总结

1. **两个checkpoint几乎完全相同**，说明它们来自相同或非常相似的训练状态

2. **语言特征维度4-15严重崩溃**，但文件2相比文件1有一定改善

3. **维度0是唯一健康的维度**，具有高方差和两文件高度一致性

4. **问题根源可能是训练不充分或某些维度未被激活**，而非权重爆炸

5. **建议继续训练或调整训练策略**以激活所有维度
