# Rebuttal 实验表格（2026-08-04，待填）

> ⚠️ 状态注记（2026-08-06）：本表主结果行停留在 final-l1only 时代，已过期。
> 当前主结果：v10-E9（ClassMeanProcrustesLoss）fg 24.07% / all-class 24.89%，
> 详见 `docs/classmean_procrustes_training.md` 与 `docs/investigation_summary.md`。
> 下表实验设计仍有效，数字待填。

> 实现：test.py 已加 `procrustes_mode`（oracle/pred/none）+ `procrustes_perturb`（none/flip/rotate）。
> 评测：ScanNet++ v2 val（50 场景，top100，excluded wall/floor/ceiling，vote_k=25）。

## Table 1 — Reviewer Gvzn Q1：Backbone vs Compression 贡献解耦（zero-shot）

> "A controlled experiment varying only the backbone (or only the compression)"

| 实验 | 变化量 | 固定量 | Val mIoU | fg mIoU |
|---|---|---|---|---|
| ① 基线 | — | PT-v3m1 + SVD-r16（主结果） | | |
| ② 只变 backbone | LitePT | 同 r16 压缩、同训练/评测 | | |
| ③ 只变 compression | SVD-**r8** / SVD-**r32** | 同 PT-v3m1、同评测链 | | |
| ④ 分析性：MLP 信息上界 | MLP（诊断） | SVD-r16 | raw L1=0.019（corr 0.886） | — |

注：768 全精度对比跳过（2026-08-04 用户确认，特征已删）；r8/r32 压缩特征已存在（lang_feat_grid_svd_r{8,32}.npz），可直接跑。

## Table 2 — Reviewer Hugw Q1-follow：SVD 基歧义鲁棒性（训练目标扰动）

> "A controlled experiment with sign-flipped or randomly rotated **compressed targets**"——训练目标扰动后重训，评测对齐后看语义一致性。

| 实验 | 训练目标 | Val mIoU | fg mIoU |
|---|---|---|---|
| ① 基线 | 原始 canonicalize 目标 | | |
| ② 符号翻转重训 | 目标每维 ±1 翻转（seed 固定，等价基） | | |
| ③ 正交旋转重训 | 目标乘固定随机正交矩阵（等价基） | | |

判读：三者 mIoU 一致 → 模型学基不变语义结构（对齐机制鲁棒）；不一致 → 依赖训练基。
（另：评测侧 flip/rotate 扰动为链自检辅助，数学上必不变。）

## Table 3 — Reviewer Hugw Q2-follow：Procrustes 对齐对对应的依赖

> "compare the current alignment strategy with an oracle version where ground-truth class assignments are used"

**注意**：当前 tester 的 Q 拟合用 GT 标签（oracle，有 GT 泄露）；**pred（预测对应）才是真正 zero-shot 主结果**。

| 对齐策略 | Val mIoU | fg mIoU | fg mAcc | 说明 |
|---|---|---|---|---|
| **Prediction-based（预测对应，无监督）** | | | | 真正 zero-shot 主结果（procrustes_mode=pred） |
| Oracle（GT 对应，当前 tester 默认） | | | | 上界（GT 泄露，procrustes_mode=oracle） |
| No alignment（raw logits） | | | | 对齐贡献下界（procrustes_mode=none） |
| **GT 上界（train_grid 50 场景，GT 特征走链）** | **33.94%** | **32.46%** | **57.15%** | 2026-08-04 实测；41.8% 为独立类平均 Q 的完美上限 |

判读：pred vs oracle 差小 → 对齐无需 GT（论文主张成立）；差大 → 对齐质量是瓶颈。

## 对比基线（SceneSplat 论文主表，ScanNet++ benchmark，2026-08-04 记录）

| 方法 | 训练数据 | Params | ScanNet++ f-mIoU | ScanNet++ f-mAcc |
|---|---|---|---|---|
| **SceneSplat（单数据集对照）** | SN++ | 91.72M | **26.8** | **45.3** |
| Mosaic3D | SN++ 等 | 683.72M | 16.2 | 27.1 |

（多数据集训练的行已移除——SN,SN++,MP3D 的 28.4/50.0 与我们的单数据集实验不可比。）
→ 我们的 16 维压缩实验对照：SceneSplat 单数据集 26.8/45.3，GT 上界 33.94%（train_grid 走链）/41.8%（完美上限）。

## 主结果速查

| 训练配置 | Val mIoU | fg mIoU |
|---|---|---|
| v1（noaug+v2 注册表+随机 init，1 epoch） | 4.62% | 3.79% |
| noaug+v3+16dim-init（1 epoch） | 3.55% | 2.82% |
| 增强+v3+16dim-init（1 epoch） | 3.15% | 2.65% |
| 增强+v3+16dim-init+normal（1 epoch） | 3.03% | 2.32% |
| **final-l1only（noaug+v3+16dim-init+L1-only+block 满速，10 epochs）** | **待训练完成** | |
| GT 上界 | 41.8% | |
