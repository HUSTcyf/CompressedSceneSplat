# 16 维 SVD 压缩特征探究过程总述（2026-08-03 ~ 08-06）

> 起点：Reviewer Q1 实验（PT-v3m1 vs LitePT，16 维 SVD 压缩目标 zero-shot 语义分割 mIoU ≈ 0.6%）。
> 终点：最佳 fg mIoU 24.07%（v10-E9），实证上限 ≈24-25%，25% 目标不可达。
> 最佳模型：`exp/smoke-ptv3m1-16-scannetpp-v2-classprocrustes-v10/model/epoch_9.pth`（服务器）。

## 1. 时间线

| 日期 | 事件 | 结论 |
|---|---|---|
| 08-03 | 评测链 Q 拟合行序错位修复（test.py cell 级） | GT 走链 0%→34.9%，真实模型 0.6%→5.4%（第一主因） |
| 08-03 | 训练 bug 三连修：inverse_variance 死分支、trainer 双源真理（方案 X 显式对齐）、删 trainer 手写 tanh | commit 8de19bf |
| 08-03 | 方案 X 逐点 Q 对齐 → 模式坍缩（列符号随机）→ 放弃；canonicalize max-abs 锚无效 | 原始目标可学，逐点对齐负收益 |
| 08-03 | 判别实验：类均值判别结构跨场景一致 0.90 | "基歧义"叙事推翻 |
| 08-03 | 增强扰动定位：去几何增强后 minor corr 0.09→0.26、幅度比→1.0 | 波动压制的第一直接原因 |
| 08-03 | Plan D：trainer 统一走模型 forward（双源真理架构级修复） | commit fb50e49 |
| 08-04 | ClassMeanProcrustesLoss + v10 配方 | **fg 24.07% / all 24.89%（冠军）** |
| 08-05~06 | v11~v26 + 20+ tester 变体 sweep | 全部更差；v26（ckpt 重启）16.60% |
| 08-06 | 终版档案 `docs/classmean_procrustes_training.md` 定稿 | 上限 ≈24-25%，12 零类不可收复 |
| 08-06 | train_lite（LitePT+gridsvd）精细 code review | output_bias 死参数、dec0 低 lr、consistency 零梯度、NaN 静默返回 |

## 2. 三层根因（叠加导致 mIoU≈0.6%）

1. **评测链 bug（第一主因，已修复）**：tester Q 拟合用点级 mask 配 cell 编号序特征行 → Q 被错位标签污染。
2. **训练架构缺陷（第二主因，已修复）**：DensityInvariantTrainer 绕过模型 forward，手写 backbone/tanh/criteria——只改 forward 的训练修改静默失效。
3. **损失与评测错配（第三层，方向性修复）**：逐点 L1/cos 梯度被 96% 公共方向 + 逐点噪声主导；评测只吃类均值结构 → 换计算单元（类均值 Procrustes 残差）才有效。

## 3. 关键机制发现（都有实测支撑）

- 目标 = 公共方向 dim0（~96% 能量，chunk 内近常数）+ 波动 dims1-15（判别信息，~4%）。
- `cos(F,T)=0.95` 是公共方向假象，去均值后 -0.04。
- cos 损失饱和后是噪声注入源（E3 崩溃）；min_points 50→10→5 救活小类。
- 更长训练/多样数据/类权重/集成/评测旋钮全部更差；seed 方差 ±3%（v10 是 +3σ 离群值）。
- checkpoint 重启必然退化；类偏置在 top-3 结构下 FP 爆炸。

## 4. 已排除的方向

训练量、统一符号/基（v3 注册表反而 2.53%）、去均值压缩（上界 44→31%）、无监督对齐（MUSE mutual-NN≈0）、768 全精度对比（特征已删）、r8/r32（特征仍在，可跑但未跑）。

## 5. 文档关系

- `docs/classmean_procrustes_training.md`：最终实验档案（准）。
- `docs/svd_compression_diagnosis.md`：诊断过程 v5（含评测链 bug 清单、增强扰动、Plan D）。
- `docs/training_launch_procedure.md`：启动清单；**§3.2 增强结论已过期**（见 §6 勘误）。
- `docs/training_diagnostics.md`：历史合集（空间一致性/BN 爆炸/LangZip）。
- `docs/rebuttal_tables.md`：8/4 待填表，主结果行已过期（见 §6 勘误）。
- `OPTIMIZATION_SUMMARY.md`：采样器性能优化（3.5x 采样加速）+ 验证方法附录。
- `CLAUDE.md`：SVD 经验与最终结论（§训练瓶颈与突破）。

## 6. 已知过期内容勘误（本次清理已标注，未删原文）

1. `pointcept/models/losses/misc.py` 2026-08-03 注释称"原方差加权给公共方向权重 1.0"——与实测语义相反：canonicalize 后 dim0 方差最小，`variance` 策略下 dim0 得最小权重、波动维得高权重。注释已修正。
2. `training_launch_procedure.md §3.2`（正式配置保留几何增强）——被冠军配方（noaug + 仅 Chromatic*）推翻，已加注。
3. `rebuttal_tables.md` 主结果速查表停留在 final-l1only 时代——已加注指向 v10-E9。

## 7. 本次清理删除的中间产物

根目录诊断脚本（`analyze_*.py` ×7、`trace_collapse_simple.py`、`verify_inference_gt.py`、
`visualize_per_dimension_weights.py` 等）、已合并文档（`VERIFICATION.md`→OPTIMIZATION 附录、
`spatial_consistency_fix.md`/`batchnorm_explosion_solution.md`→training_diagnostics、
`model_collapse_final_analysis.md`、`checkpoint_parameter_analysis_report.md`、
`projection_matrix_example.py`、`test_output_bias_gradient.py`、`scripts/TROUBLESHOOTING.md`）。
诊断方法沉淀为文档 §8（上界实验/评测链自检/2×2 交叉/cos 分解/每维幅度比/KNN 地板/行序检查），脚本已清。

## 8. 未验证方向（若继续）

统一基重压缩 `(X−mean) @ V_text`——需重新下载已删的 768 维 lang_feat.npy，理论充分但未实测。
