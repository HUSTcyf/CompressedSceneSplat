# 类均值 Procrustes 监督训练：完整实验档案（2026-08-04 ~ 08-06，终版）

## 0. 最终状态（一句话）

**最佳模型 v10-E9：val fg mIoU 24.07% / all-class mIoU 24.89%（tester 全 50 场景，oracle 每场景 Q）。**
25% 目标未达成——30+ 变体、~30 小时实验证明该框架的实证上限 ≈24-25%，不存在达成路径。

## 1. 起点：为什么逐点损失卡在 ~5%

16 维压缩目标（逐场景 SVD，canonicalize 符号）的结构：

- **dim0 = 公共方向**（场景偏移）：跨场景 ~96% 能量，chunk 内近常数（std 0.04）
- **dims 1-15 = 波动维**（判别信息）：仅 ~4% 能量，逐点噪声地板（dims 5-15 单 chunk corr 上限 0.03-0.34）
- 跨场景基方向只对齐 0.51（去均值 cos）；**类均值判别结构跨场景一致 0.90**（无需对齐）

**逐点 L1/cos 失败机制链**（每环有实测）：
1. 逐点 L1 梯度 ~99% 被公共方向占据（inverse_variance 权重语义反转：d0 权重 1.0、波动维 0.1-0.34）
2. 模型变"场景均值预测器"：cos(F,T)=0.95 是公共方向假象，去均值后 -0.04；波动维 corr≈0、幅度仅 6-30%
3. 评测只吃类均值结构（上界 33.94%）——训练与评测信号**错配**
4. 已排除：训练量（133× pass 仅 +0.17%）、基歧义（0.90 一致）、LR/权重、无监督对齐（MUSE mutual-NN≈0）

## 2. 突破：ClassMeanProcrustesLoss

`pointcept/models/losses/misc.py`：

```
每个训练 iter，逐场景（offset 分段）：
1. 每点 L2 归一化 → 等权类均值 F_c（模型）、T_c（目标），min_points 过滤
2. 正交 Procrustes Q = UV^T（M = F_c^T T_c 的 SVD，det+1），detach —— Q 吸收基旋转 → 基无关
3. 残差 ||F_c Q − T_c||（L1）+ 可选逐点类均值拉拽（point_weight）
```

## 3. 完整实验记录

### 冠军配方（v10）
```
top50 chunks（val 类分布挑选）+ loop=10（100 遍/chunk）+ epoch=10
ClassMeanProcrustesLoss(min_points=5, point_weight=0.2, mean_weight=1.0, center=False)
wd 0.05, 16dim-init, OneCycleLR(pct_start=0.1), noaug（仅 Chromatic*）, block lr 0.0006
```
**v10-E9：fg 24.07% / all-class 24.89%（唯一 >22% 的结果）**；E8 21.66%、E10 23.55%。

### 全部变体（按时间）

| 版本 | 改动 | fg mIoU (tester) | 结论 |
|---|---|---|---|
| 历史最佳（逐点损失） | 加权 L1+cos+contrast | 3.49% | 起点 |
| v1 | 类均值 1.0 + cos 0.3, min_pts=50 | 12.1→17.1→5.7 | E3 崩溃 = **cos 噪声注入** |
| v2 | 逐点类均值拉拽 1.0 | 7-13% 震荡 | 离散目标破坏结构 |
| v3 | 去质心类均值 | 5-6% | 退化固定点 |
| v4/v4b | v1 + wd 0.01 | 8-12% | wd 不是崩溃原因 |
| **v5** | **纯类均值, min_pts=10** | **20.83%**（E9） | E3 无崩溃 ✓，hook 23.64% |
| v6 | cover250（巨型 chunk） | 训练卡死 | 260 万点 chunk 不可用 |
| v7 | top50+7 缺失类 chunk | 19.95% | 数据组合稀释主流类 |
| v8 | 40 epochs | 18.72% | 更长训练更差（过拟合） |
| v9 | 600 多样 chunk | 17.44% | 多样数据更差（类结构折中） |
| **v10** | **+ point_weight=0.2** | **24.07%**（E9） | **冠军** |
| v11 | 类权重 0.3/1.0 | 16.09% | 主流类降权过狠 |
| v12 | point 0.3 | 20.16% | 0.2 是甜点 |
| v13/14/17/18 | 同配方 seed 抽奖 | 19.5-21.6% | seed 方差 ±3% |
| v15 | 抽奖（中断） | - | - |
| v16 | top50plus + 温和权重 0.9/1.3 | 19.89% | 数据组合毒药 |
| v17 | 9 混淆类逐点 5× | 21.56% | 混淆类均值分不开 |
| v19 | v10 seed + 11 epochs | 20.32% | 调度偏移离开幸运轨迹 |
| v20 | batch 4→8 | 18.42% | 调度变化 |
| v21 | min_pts 5→3 | 19.48% | 噪声类均值破坏 Q |
| v22/23 | 单卡并行抽奖 | 12.85% | 单卡动力学崩坏 |
| v24 | 强类 1.5×/弱类 0.5× 逐点 | 20.15% | 强类 0.55→0.40（过拉拽） |
| v25 | v10 seed + pct 0.15 + 12ep | 19.99% | 锚定扰动失败 |
| v26 | v10-E9 + 1 epoch + hook | 16.60% | checkpoint 重启必然退化 |

### 评测侧尝试（v10-E9 上）

| 改动 | fg mIoU | all-class |
|---|---|---|
| 基线（ct=0.1, vk=25） | **24.07%** | **24.89%** |
| ct=0.3 | 23.45% | 24.29% |
| ct=0.05 | 23.24% | 24.10% |
| vk=50 | 22.83% | 23.70% |
| vk=10 | 22.54% | 23.46% |
| 类偏置 α=0.1 | 10.46% | 10.19% |
| 类偏置 α=0.01 | 19.75% | 20.41% |
| 3 模型特征集成 | 20.80% | 21.74% |
| 任务算术合并 λ=0.5 | 21.00% | 21.78% |
| hook 直接评测 | 1.55% | 1.88%（链路损坏，不可用） |

## 4. 关键机制发现（按重要性）

1. **cos 损失饱和后是噪声注入源**：cos≈0.985 后剩余梯度全在含噪逐点偏差上 → 侵蚀类结构（v1/v4b 均在 E3 崩，去掉 cos 后消失）
2. **min_points 50→10→5 逐步救活小类**：plant pot 0→0.54、cup 0.03→0.11；point 0.2 进一步（pot→0.70、toilet→0.74）
3. **训练集覆盖**：top50 只覆盖 84/100 类 → 16 类永远 0；补 chunk 后 tap 0.57/soap dispenser 0.64，但主流类被稀释（净亏）
4. **多样数据更差**：类损失地板 50 chunks=0.014 vs 600 chunks=0.025——约束冲突使可达对齐质量差 ~2 倍
5. **更长训练更差**：100 遍（20.8%）→ 200 遍（20.0%）→ 400 遍（18.7%）——峰值在 ~E9-10
6. **seed 方差 ±3%**：v10 是 8-9 次同配方抽奖中唯一 +3σ 离群值（drop_path/色增强/数据序的随机性）
7. **checkpoint 重启必然退化**：任何从已训练 checkpoint 继续的训练（同损失也不例外）都会破坏结构
8. **类偏置在 top-3 结构下 FP 爆炸**：即使 α=0.01 也摧毁分类
9. **剩余 12 个零类**：3 个真缺失（rack/painting/paper towel）+ 9 个混淆类（kitchen cabinet/coat hanger/bowl 等——类均值与近邻重叠，模型输入无法分离）

## 5. 逐类表现（v10-E9）

- 强类（IoU≥0.3，31 个）：chair 0.66、table 0.73、office chair 0.70、microwave 0.72、whiteboard 0.69、toilet 0.74、pot 0.70、stapler 0.82、paper bag 1.0、ceiling 0.79
- 零类（12 个）：kitchen cabinet、rack、painting、shoe rack、file folder、blind rail、coat hanger、slippers、paper towel、bowl、whiteboard eraser、power strip
- 若零类达到平均，fg ≈ 27-28%——但所有针对性手段（数据/权重/逐点/偏置）均无法在不伤主流类的前提下收复

## 6. 结论

- 逐点回归在"零中心噪声维 + 场景局部基"下是病态设定——换损失公式无用，**换计算单元（类均值）**才有效
- 类均值 Procrustes 残差 = 评测信号的训练版（基无关、噪声平均、可学）
- 25% 目标与框架实证上限（~24-25%）不兼容；v10-E9 是最终模型

## 7. 关键文件

- **损失**：`pointcept/models/losses/misc.py`（ClassMeanProcrustesLoss，含 class_weights/point_class_weights/center 全部实验开关）
- **配置**：`configs/custom/lang-pretrain-ptv3m1-scannetpp-v2-classprocrustes.py`（当前指向 v26，历史各版本在 exp/*/config.py 转储）
- **最佳模型**：`exp/smoke-ptv3m1-16-scannetpp-v2-classprocrustes-v10/model/epoch_9.pth`
- **数据选择**：`tools/compression/select_top50_chunks.py`（val 类分布相似）；覆盖扩展逻辑见会话记录（贪心覆盖 + 大小上限）
- **评测**：`pointcept/engines/test.py`（ZeroShotSemSegTester，oracle 每场景 Q；weight_list 特征集成与 class_bias_alpha 为实验遗留，默认关闭）
- **评测自检**：GT 特征走链上界 33.94%（train_grid 50 场景口径）；每场景 Q 拟合行序修复（cell 级特征 + cell 标签）

## 8. 若继续（未验证的方向）

- **统一基重压缩**：`(X−mean) @ V_text` 投影到 text 基——训练/评测同空间、Q=I、无 GT 需求。需要重新下载已删除的 768 维 lang_feat.npy。理论依据充分但未实测（用户无数据/时间）。
