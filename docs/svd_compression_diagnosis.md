# 16 维 SVD 压缩特征评测诊断报告（最终版）

日期：2026-08-03（第五版，增强扰动定位后）
背景：Reviewer Q1 实验（PT-v3m1 vs LitePT，16 维 SVD 压缩目标）中 zero-shot 语义分割 mIoU ≈ 0.6%，远低于预期。本报告系统性定位根因。

## 0. 重要勘误

**第一版（作废）**：基于错位的 segment 标签（诊断脚本 `reshape(-1)` 展平 [N,3] 的 segment.npy，正确做法 `segment[:, 0]`），结论全部错误。

**第二版（不完整）**：修正标签后上界 41.8%（真实），但**未发现 tester 评测链自身的 Q 拟合错位 bug**——误把 mIoU≈0 全部归因于模型训练。

**第三版**：**评测链 bug 修复（§4 #9、§8.12）后：GT 特征走完整链 34.9%（≈上界），真实模型 0.6% → 5.4%（9 倍）。** mIoU≈0 的第一大原因是评测链 Q 拟合错位；模型波动维没学到（§2 数据仍然成立）是第二大原因。

**第四版（训练侧结论重大修正）**：
1. **"基歧义"叙事被判别实验推翻**：跨场景 chunk 的判别结构（类均值去质心方向）**绝对一致性 0.91**（无需任何对齐）——SVD 基在类均值层面跨场景相当稳定，模型输入含场景信息、基可从输入推断——**基歧义不是波动学不到的主因**
2. **方案 X（训练目标在线对齐 text16）第一版未生效**：`DensityInvariantTrainer` 绕过 `LangPretrainer.forward` 直接 backbone + criteria（**双源真理**）——只改模型 forward 的训练修改静默失效，损失曲线照常无报错
3. **方案 X 生效后引入更严重的 Q 列符号歧义**：Q 拟合（torch.linalg.svd 的 Procrustes 解）列符号跨 chunk 随机（实测 16 列平均一致率 0.456，15/16 列翻转）→ 对齐后目标 T' 波动列符号随机 → 模型学平均符号 → **模式坍缩**（PerDimMonitor 报警 Trivial solution，Dim0 corr -0.23）——逐点对齐是**负收益**，方案 X 放弃
4. **canonicalize（max-abs 锚）修复无效**：波动列幅度小，max-abs 锚定在噪声点上（M2 反而 0.96→0.87 < 原始目标 0.90）
5. **收敛结论**：原始目标（canonicalize 后列符号一致、判别结构 0.90 一致）**可学**——真正的瓶颈 = **损失设计与评测目标的错配**（逐点 L1/cos 被 83% 逐点噪声和 96% 公共方向主导；评测只用类均值结构）→ **方向：类级监督**（contrast 强化/类均值结构损失——符号/基无关，直接对齐评测）
6. **训练代码正确性未完整验证**（本版进行中）——见 §9 code review 清单

**第五版（本版，增强扰动定位——波动学不到的第一直接原因）**：
1. **单 chunk 目标结构澄清**：单 chunk 内 dim0（公共方向）是**常数**（|mean|=0.92、std=0.04，能量占比仅 1.4%）——dim0 corr 低是常数序列的 corr 无意义，不是没学好；**判别信息全在 dim1-15 波动**（std 0.04-0.21）
2. **单 chunk 目标波动空间平滑**（KNN-16 残差 = 幅度的 19.9%，随机邻居 103%；每维不可预测仅 12-29%）——**"83% 噪声地板"叙事对单 chunk 不成立**——目标可学、邻域聚合架构可表达（decoder 为逐点 Block + 线性 unpool，非纯插值）
3. **增强扰动 = 波动学习的主要压制因素（决定性实验）**：RandomRotate/Scale 等几何增强改变 GridSample 的 cell 哈希 → **同一物理点的目标特征随 iter 变化**（目标扰动）→ 模型只能学平均 → 波动坍缩。确定性训练（去全部随机增强）300 epoch 后：**minor corr 0.07-0.09 → 0.26（3 倍）、波动幅度比 0.06-0.30 → ≈1.0（86% 恢复）、dim1/dim4 corr 0.73/0.71**——**无增强时波动显著可学**
4. **剩余瓶颈**（确定性下 0.26 仍不够）：小幅度维度（dim5-15 std 0.03-0.08）噪声相对大 → corr 低（0.03-0.34）；方向只对齐 51%（去均值 cos 0.513）；300 epoch 可能不足（L1 仍缓慢下降）
5. **方案 D 完成**：trainer 统一走模型 forward（双源真理架构级修复）——确定性实验即验证（重构后代码行为正确）
6. **对正式训练的含义**：训练必须**去掉几何增强**（旋转/缩放/弹性/抖动——都改 cell 哈希）；颜色增强（Chromatic*）不影响坐标可保留；历史所有带增强的训练（含 baseline 5.4%）均在目标扰动下训练——波动被压制是结构性原因之一

## 1. 最终结论

**三层问题叠加导致 mIoU≈0.6%（第四版）**：
1. **评测链 bug（第一主因，已修复）**：tester 的 Procrustes Q 拟合行序错位——GT 特征走链 0% → 修复后 34.9%，真实模型 0.6% → 5.4%。
2. **训练代码架构缺陷（第二主因，第四版发现）**：`DensityInvariantTrainer` 绕过模型 forward（双源真理）——训练修改静默失效；方案 X 生效后引入 Q 拟合列符号歧义 → 模式坍缩（Trivial solution）——训练侧实验（方案 X 及任何只改 forward 的修改）在此架构下不可信。
3. **损失设计与评测目标错配（第三层，待修复）**：原始目标（canonicalize 后）判别结构跨 chunk 一致 0.90、列符号一致——**可学**；baseline 5.4% 的瓶颈是逐点 L1/cos 监督（83% 逐点噪声 + 96% 公共方向主导梯度）vs 评测只用的类均值结构——**方向：类级监督**。

**注意**：第四版推翻了"基歧义"作为主要根因的叙事——判别实验（§8.13）证明类均值判别结构跨场景绝对一致 0.90。

## 2. 核心机制（已全部实验证实）

```
目标 T = 公共方向 μ（场景偏移，占 96% 能量）+ 波动 S（判别信息，占 4%）
模型 F = 公共方向（cos 0.998 对齐）+ 波动（幅度 1/10、corr ≈ 0、方向无关）

cos(F, T) = 0.95 是公共方向的假象：去均值后 cos = -0.04
判别信息 100% 在波动维 → 波动没学到 → 分类 0.6%
```

关键数字：
- F ≈ D·T（线性关系残差 0.079），D 为极端各向异性对角缩放：dim0 ≈ 0.84（学好），dim1-15 ≈ 0.06-0.30
- 每维 corr（F vs T）：mean ≈ 0（-0.02）
- 目标波动维 83% 空间不可预测（KNN 残差 0.048/幅度 0.056）——L1 的硬地板
- 模型 L1 = 0.22 已优于 KNN 基线（推算 0.72）——波动维数值层面在学

## 3. 上界与评测链路（第三版，评测链修复后）

**上界来源确认**：41.8% 来自独立脚本 `diag_fixed2.py`（历史对话找回）——cell 级加载（`np.unique(indices, return_index=True)` 反查每 cell 首点标签）+ 归一化类求和/类平均 Q + 逐 cell 分类。已精确复刻：4 个 chunk 场景 18.4-52.3%（均值 ≈ 41.8%），**上界真实**。

| 实验（2026-08-03 实测，4 chunk 场景） | mIoU |
|---|---|
| 独立上界脚本（diag_fixed2 逻辑，cell 级） | 35-52% |
| **GT 特征走完整 tester 链（修复前）** | **0%** |
| **GT 特征走完整 tester 链（修复后）** | **34.9%**（allAcc 78%）✓ 与上界一致 |
| 真实模型 val（修复前，0.6% 时代） | 0.6% |
| **真实模型 val（修复后 tester）** | **5.4%**（allAcc 25.2%） |

**评测链自检方法（重要）**：任何评测链改动后，先把 GT 目标特征当作模型输出走完整链——若 ≈ 独立上界则链正确，否则链有 bug。

## 4. 评测链路已修复的 bug（按发现顺序）

| # | Bug | 影响 | 修复 |
|---|---|---|---|
| 1 | 符号翻转（逐场景 SVD 基符号任意） | cos spike 22%（三峰 0.05/1.00/1.94） | 加载时 canonicalize（`svd_sign.py`），spike → 0% |
| 2 | 评测 hook 单 Q 全局应用 | 只对齐第一个场景 | 改为每场景 Q（与 tester 一致） |
| 3 | Procrustes 拟合类求和（未归一化） | 大类（wall/floor）主导 Q | 归一化 + 类平均（1/n） |
| 4 | tanh 不一致（训练有、评测无） | 方向空间错位（影响小——输出在 tanh 线性区） | 评测路径补 tanh |
| 5 | model_best 缺失崩溃 | 评测中断 | fallback 到当前权重 |
| 6 | torch.load weights_only | PyTorch 2.6+ 默认拒绝 | weights_only=False |
| 7 | AggregatedContrastiveLoss 从未激活 | epoch_progress = 0/max 永远跳过 | 迭代级进度 |
| 8 | 诊断脚本 segment 错位 | 第一版结论全错 | segment[:, 0] + valid 空间 |
| **9** | **tester Q 拟合行序错位（本版新增，最关键）** | **Q 被点级标签配 cell 特征行污染 → 完美 GT 特征走链 0%，真实模型 0.6%** | **Q 拟合改 cell 级特征 + `np.unique(inverse)` 反查 cell 代表点标签（见 §8.12）→ GT 34.9%、真实模型 5.4%** |

## 5. 训练侧修复（进行中）

- **SVDWeightedL1Loss 权重反转**：`variance` → `inverse_variance`（波动维高权重，公共方向 0.1）
  - 原方差加权是反的：公共方向权重 1.0、波动维 0.1 → 模型只学公共方向
- **AggregatedContrastiveLoss 激活**（权重 0.1→0.3）：直接监督输出空间类间判别（不依赖目标坐标，不受公共方向主导）
- 当前状态：训练 smoke 验证中（L1 卡 0.22 ≈ 数据地板；contrast 0.5-0.8 激活）

## 6. 未解决的根因与方向（第四版）

**第四版修正**："逐场景 SVD 基旋转歧义"不再是主要根因（判别实验：类均值判别结构跨场景绝对一致 0.90，基可从输入推断）。剩余问题分三层：
1. **训练代码正确性**（待完整验证——§9 code review 清单）：双源真理架构（trainer 绕过 forward）、损失实现细节、梯度流
2. **损失与评测错配**：逐点监督（噪声/公共方向主导）vs 评测类均值结构——**类级监督**（contrast 强化、类均值结构损失）
3. **逐点噪声地板**（83% 空间不可预测，旋转不变）：逐点目标的硬上限——类级监督平滑掉它

方向（按优先级）：
1. **完整 code review 训练代码**（进行中）——未验证前不跑全量实验
2. **类级监督**：AggregatedContrastiveLoss 全程激活 + 权重提高；可选类均值 Procrustes 残差损失（评测 Q 拟合逻辑的训练版，符号/基无关）
3. ~~**统一基**（远期）~~ **已排除**：统一基的前提（基不一致是瓶颈）已被 §8.13 推翻（类均值层面 0.90 一致），统一符号实测反降（v3 2.53%），去均值压缩上界 44%→31%。不再作为方向。

## 7. 关键文件

- `pointcept/utils/svd_sign.py`：canonicalize_svd_sign（符号）+ remove_scene_mean（去均值，方案 1 已否决保留开关 svd_center）
- `tools/projection/compute_procrustes_alignment_simple.py`：Procrustes 拟合（归一化 + 类平均）
- `pointcept/models/losses/misc.py`：SVDWeightedL1Loss（inverse_variance 策略）
- `pointcept/engines/hooks/evaluator.py`：评测 hook（每场景 Q）
- `pointcept/engines/test.py`：tester（每场景 Q + tanh + pred 缓存兼容）

## 8. 问题日志：诊断过程与方法（按时间顺序）

> 记录每个问题：症状 → 诊断方法 → 结论 → 解决/状态。诊断脚本已清理，方法可复现。

### 8.1 符号翻转（cos spike 三峰）
- **症状**：cos_loss 从 0.05 跳到 1.0/1.94，直方图三峰 {0.05: 78%, 1.00: 19.5%, 1.94: 1%}
- **诊断**：全量直方图 → 三峰与 M=2 二项分布完美拟合（dense/single 两 scenario 独立翻转，p=11.7%）→ 逐场景 SVD 基符号任意（2^16），模型无法从输入推断符号约定
- **解决**：加载时 `canonicalize_svd_sign`（每列最大绝对值取正）→ spike 22%→0%

### 8.2 cos 卡 0.05
- **症状**：cos_loss 收敛到 0.05 后降不下去
- **诊断**：纯空间邻居基线（3D 近邻特征均值预测目标）cos_loss ≈ 0.037 → 不是数据天花板 → 但去均值后 cos = -0.04 暴露真相
- **结论**：0.05 是"公共方向对齐"的假象（见 8.8）

### 8.3 评测 mIoU ≈ 0（第一轮：上界实验 1.6% 误导）
- **症状**：tester mIoU 0.006，仅 wall/floor/ceiling 有 IoU
- **诊断**：上界实验（压缩坐标直接分类）显示 1.6% → 一度误判"压缩特征无语义"；随后发现类均值间余弦 0.991（异常）→ 用户质疑 → **数据格式检查发现 segment.npy 是 [N,3]，诊断脚本用 reshape(-1) 错位**
- **解决**：修正标签（`segment[:, 0]` + valid 空间）→ 上界 41.8%——**第一版诊断结论全部作废**（教训：先验证数据对齐再下结论）

### 8.4 评测 hook 单 Q 全局应用
- **症状**：hook 评测 mIoU 0.0048，与 tester 不一致
- **诊断**：code review 发现 `if self.use_procrustes and not procrustes_computed` 只在第一个场景拟合一次
- **解决**：改为每场景 Q（与 tester 一致）

### 8.5 Procrustes 拟合类求和 bug
- **症状**：修复 8.4 后 mIoU 仍 0.6%（上界 41.8%）
- **诊断**：2×2 交叉实验（目标/模型输出 × 各自 Q）分离变量；代码审查发现 `sum_j.index_add_` 未归一化 + 类求和——大类（wall/floor 数百万点）主导 Q
- **解决**：归一化 + 类平均（1/n）→ 上界 44.3%

### 8.6 tanh 不一致
- **症状**：模型输出 vs text16 余弦（对齐后 0.54-0.63）远低于目标级（0.93）
- **诊断**：code review 发现训练输出 `tanh(backbone)`、评测输出 backbone 原始
- **解决**：评测路径补 tanh——**实测无效果**（backbone 输出在 tanh 线性区，tanh ≈ 恒等）→ 非主因

### 8.7 L1 卡 0.2（inverse_variance 训练中）
- **症状**：L1 从 2.06 快速降到 0.22 后平台
- **诊断**：目标波动维 KNN 预测残差 0.048/幅度 0.056 = **83% 空间不可预测**（CLIP 多视角噪声）→ L1 硬地板；模型残差 0.014 已优于 KNN 基线
- **结论**：数据上限，非训练问题

### 8.8 模型波动维学不到（核心根因）
- **症状**：每维 corr（F vs T）≈ 0；分类失败
- **诊断**：cos 分解（公共方向 cos 0.998 vs 波动 cos -0.04）→ 每维幅度比（dim0 0.84，dim1-15 0.06-0.30）→ 线性关系检验 F ≈ D·T（残差 0.079，D 为极端各向异性对角缩放）
- **结论**：目标 96% 能量在公共方向 → 损失梯度被公共方向主导 → 波动（判别）学不到；叠加逐场景基旋转/置换歧义（canonicalize 只修符号）
- **解决（进行中）**：inverse_variance L1 + contrast 损失（8.9）

### 8.9 AggregatedContrastiveLoss 从未激活
- **症状**：contrast_loss 全程 0.0000
- **诊断**：`epoch_progress = self.epoch / self.max_epoch`，第一个 epoch self.epoch=0 → schedule="last_75" 永远跳过
- **解决**：迭代级进度 `(epoch + iter/total) / max_epoch` → contrast 激活（0.5-0.8）

### 8.10 去均值方案否决
- **症状**：去均值上界 44%→31%，f-mIoU 24.6%→14.7%
- **诊断**：per-class 分解——掉分集中在结构类（ceiling/wall/floor -0.4~-0.5）及大面积物体类（blanket/curtain），公共方向携带结构类判别信息
- **结论**：去均值代价 > 收益，否决（svd_center=False）

### 8.11 其他修复（小问题）
- model_best.pth 缺失崩溃（Best mIoU=-inf 从未保存）→ fallback 当前权重
- torch.load weights_only（PyTorch 2.6+ 默认拒绝 numpy.dtype）→ weights_only=False
- pred 缓存（result_ScanNetPPGSDataset/*_pred.npy）——改动评测逻辑后必须删除，否则加载旧预测跳过前向
- spconv "Can't find algo" 警告——进程内缓存自限，无需处理

### 8.12 tester Q 拟合行序错位（2026-08-03 第三版，最关键的评测链 bug）
- **症状**：真实模型 tester mIoU 0.6%，独立上界脚本 41.8%——用 GT 特征走完整 tester 链验证：**0%**！评测链自身有 bug。
- **诊断过程**：
  1. 历史对话找回上界脚本 `diag_fixed2.py`，精确复刻 → 18.4-52.3% ✓ 上界真实
  2. GT 特征（stub 输出 dataset 的 lang_feat）走 tester 链 → 0%，Q 拟合 M_matrix 奇异值谱 rank-1 坍缩（300-500 倍）
  3. 打印真实数据：`frag["index"]` = GridSample(mode=train) 输出行号（= cell 编号，704242 个），`inverse` = 每点→cell 编号，scatter 按行号 → **accumulated_features 行 = cell 编号序**
  4. **bug 定位**：`X_c = accumulated_features[valid_mask]`——valid_mask 是**点级**（segment != -1，777632 长），取出的行是 **cell 编号序**特征，配的标签是**点级 segment**——行序不对应 → Q 被错位标签污染（随机标签配随机特征 → 类均值坍缩成公共方向）
  5. 修复：Q 拟合改用 cell 级（`feature_counts > 0` 行，全非零）+ cell 标签 = `np.unique(inverse, return_index=True)` 反查每 cell 首个原始点的标签——与上界脚本 `np.unique(idx, return_index=True)` 同语义
- **结果**：GT 特征走链 0% → **34.9%**（≈上界 ✓ 评测链正确）；真实模型 0.6% → **5.4%**
- **Q 拟合 cell 级 vs 逐点级等价性**：本架构下同 cell 点特征相同（GridSample 降采样输入 + `compressed[indices]` 共享）→ 归一化类均值方向严格等价。选 cell 级：无零行污染、计算量小、与上界脚本一致
- **训练评测 hook 检查**（LangPretrainZeroShotSemSegEval / Multi）：直接对整个 val batch 前向（无 scatter/inverse 环节），point_feat 行序 = segment 行序（同一 GridSample 输出）→ **自洽，无需修复**。注意：hook metric 在 cell 级算（每 cell 一票），与 tester 逐点近似等价
- **教训**：评测链必须用 GT 特征走完整链自检（不能只信独立脚本）；scatter 类评测链的高危点 = "行序对齐"（cell 编号 vs 原始点序）
- **修复后必做**：删除旧 pred 缓存（`result_ScanNetPPGSDataset/*_pred.npy`，错位 Q 时代产物），否则 tester 加载旧预测跳过前向

## 9. 可复用的诊断方法

1. **上界实验**：目标特征（修正标签）直接每场景 Q 分类——任何改动前先测上界，判断潜力
2. **评测链自检（最重要，8.12 的教训）**：把 GT 目标特征当作模型输出走**完整 tester 链**——若 ≈ 独立上界则链正确，否则链有 bug。任何评测链改动后必做
3. **2×2 交叉**：目标/模型输出 × 各自 Q——分离"特征差异"与"拟合差异"
4. **cos 分解**：全部 / 去均值后 / 均值方向——识别"公共方向假象"
5. **每维幅度比 + corr**：F ≈ D·T 线性检验——判断"信息缺失"vs"坐标变换"
6. **KNN 残差地板**：目标的 3D 近邻预测残差——量化"数据不可预测部分"
7. **数据对齐检查**：先验证 segment/indices/coord 长度与语义（[N,3] 取第 0 列）再下结论
8. **行序对齐检查（scatter 类链路的通病）**：打印 `frag["index"]`/`inverse`/buffer 的语义——cell 编号 ≠ 原始点索引，Q 拟合/标签必须同一行序

### 8.13 判别实验：基歧义不是主因（第四版核心证据）
- **方法**：`tools/diagnosis/diag_align16_consistency.py`（带 6 道自检：S1 测量函数 / S2 双脚本互证 / S3 Q 正交 / S4 复刻 vs 官方 0.00e+00 / S5 对齐生效 / S6 数学必然）+ 跨场景抽样（按场景分组取首个 chunk，修正同场景块偏置）
- **结果**（10 个不同场景）：
  - M2 跨场景判别结构（出现类类均值去质心方向）**绝对一致性 T=0.90**（无需任何对齐）——基在类均值层面跨场景稳定
  - 方案 X 对齐后 T'=0.96（提升小）；canonicalize 后 0.87（max-abs 锚不可靠）
  - M3 评测对齐潜力（出现类 vs text16 Procrustes）0.55-0.58——评测上界的结构上限
  - 自检 S2 修正：原"0.73"是转述错误，原始 check_wave_consistency 实测 0.8539（同场景块）/0.545（跨场景全行）
- **结论**：per-chunk SVD 基的旋转歧义对"类均值判别结构"影响 ~0.10（0.90 一致性）；**模型输入含场景信息，输出场景特定特征是可学的**（dim0 学好 0.84 是证据）

### 8.14 双源真理：trainer 绕过模型 forward（第四版架构发现）
- **症状**：方案 X（对齐在 LangPretrainer.forward）训练损失曲线与 baseline 几乎一样
- **定位**：`DensityInvariantTrainer` 手动重写前向+损失——直接 `backbone(point)`（trainer:1256-1293）+ 手动 `torch.tanh`（"match LangPretrainer.forward()"注释与现实不符——模型 forward 无 tanh）+ 直接 `criteria`（trainer:1415-1468）——**任何只改模型 forward 的训练修改静默失效**
- **修复**：方案 X 对齐显式加入 trainer 的 criteria 调用前；tanh 已删（8.6 实测线性区无效果）
- **原始项目对照**（/home/cyf/SceneSplat）：训练 `self.model(input_dict)`（train.py:204）与测试同一 forward（test.py:336/982），行为一致——我们项目的 trainer 是唯一偏离点
- **教训**：改训练逻辑必须同时检查 trainer 和模型 forward 两处（或统一架构）

### 8.15 Q 拟合列符号歧义 → 模式坍缩（第四版最严重发现）
- **症状**：方案 X 生效后 PerDimMonitor 报警 `TRIVIAL SOLUTION DETECTED! Dim 0 corr: -0.2259, Minor corr: -0.0130`——模型预测常数（坍缩）
- **机制**：Q 拟合（torch.linalg.svd 的 Procrustes 解）的列符号跨 chunk 任意（det 修正只保证 det(Q)=+1，180° 翻转也是 proper rotation）→ 对齐后目标 T' 的**波动列（1-15）符号随机翻转**——实测 16 列平均一致率 **0.456**（随机水平），15/16 列翻转；列 0（公共方向）恰好一致（全负）
- **为什么判别实验 M2=0.955 没抓到**：去质心类均值里列 0（符号一致）主导 cos，列 1-15 的随机符号被掩盖；6-10 个 chunk 样本太小
- **canonicalize（max-abs 锚）修复无效**：波动列幅度小，max-abs 点即噪声，锚不可靠（M2 0.96→0.87 < 原始目标 0.90）
- **结论**：方案 X 的逐点 Q 对齐是**负收益**（把符号已 canonicalize 的可学目标变成符号随机的坍缩目标）——放弃逐点对齐
- **对照**：baseline 目标（compressed canonicalize 后）列符号一致 ✓、判别结构 0.90 一致 ✓——**原始目标可学**

### 8.16 收敛结论（第四版）
- 原始目标（canonicalize 后）**可学**：列符号一致 + 判别结构跨 chunk 0.90 一致
- baseline 5.4% 的真实瓶颈 = **损失设计与评测目标错配**：逐点 L1/cos 的梯度被 83% 逐点噪声（KNN 残差，旋转不变）和 96% 公共方向主导；评测只用类均值结构（Q 拟合 → text16 → argmax，平滑掉逐点噪声）
- **方向**：类级监督（AggregatedContrastiveLoss 强化——类内聚合/类间分离，符号/基无关；可选类均值 Procrustes 残差损失——把评测 Q 拟合逻辑变成训练损失）
- **前提**：训练代码正确性必须完整验证（§9 清单）——未验证前不再跑全量实验

## 9. 可复用的诊断方法

### 8.17 增强扰动：波动学不到的第一直接原因（第五版核心发现）
- **症状**：单 chunk 过拟合（300 epoch）L1 平台 0.40、minor corr 0.09——"完美拟合"未实现（用户指正）
- **诊断链**：
  1. 单 chunk 目标结构：dim0 常数（std 0.04、能量 1.4%）、dim1-15 波动（判别信息）——dim0 corr 低是常数序列无意义
  2. 单 chunk 目标波动空间平滑（KNN-16 残差 19.9% vs 随机 103%）——目标可学——排除"噪声地板"（此前 83% 是多场景混合口径）
  3. decoder 逐点 Block（非纯插值）——架构可表达——排除架构限制
  4. 确定性实验（去全部随机增强）：**minor corr 0.09→0.26、波动幅度比→1.0、dim1/dim4 corr 0.73/0.71**——增强扰动成立
- **机制**：几何增强（RandomRotate/Scale/Flip/Jitter/ElasticDistortion）改变坐标 → GridSample 的 cell 哈希随 iter 变化 → 同一物理点聚合到不同 cell → 目标特征随 iter 扰动 → 模型只能学平均（波动坍缩）；确定性输入下目标固定 → 波动可学
- **修复**：训练 transform 去掉全部几何增强（颜色增强 Chromatic* 不影响坐标可保留）；`lang-pretrain-ptv3m1-scannetpp-v2-overfit-det.py` 为确定性参考
- **影响**：历史所有带增强的训练（含 baseline 5.4%）均在目标扰动下训练——波动被压制是结构性原因之一（与监督错配并列）

### 8.18 单 chunk 过拟合实验（第五版验证手段）
- **配置**：overfit config（split="" 单 chunk 00777c41d4_0、epoch=300、batch_size=1——drop_last=True 时 batch>1 会使 loader 长度为 0 导致 OneCycleLR total_steps=0 崩溃）
- **增强版结果**：L1 21→0.40 平台、cos 0.65→0.08、minor corr 0.09——波动没学
- **确定性版结果**：L1 10→0.37-0.43 平台、cos 0.98→0.07、minor corr 0.12（训练中）——离线分析（analyze_det_ckpt.py）：per-dim corr dim1=0.73/dim4=0.71、波动幅度比 ≈1.0、去均值 cos 0.51——**波动显著可学**
- **TRIVIAL 报警噪音**：PerDimMonitor 的 warmup 用 comm_info['iter']（每 epoch 重置为 0）→ warmup_iters=200 永远不满足 → 每 epoch 报警（小 bug，非训练问题）
- **方案 D 验证**：确定性实验运行在统一 forward 重构（trainer 调 self.model）后——行为正确（L1/cos 正常下降、波动可学）——双源真理架构级修复通过验证

### 8.19 方案 D：双源真理架构级修复（完成）
- **问题**：DensityInvariantTrainer 手动 backbone + 输出处理 + criteria（绕过 LangPretrainer.forward）——模型 forward 的修改静默失效（方案 X 第一版即此）
- **修复**：trainer 组装 full_input（coord/feat/batch/grid_size/epoch_progress/valid_feat_mask/lang_feat/segment/offset/opacity/quat/scale）→ `self.model(full_input)`——前向/输出处理（训练分支不 normalize）/目标对齐（方案 X）/损失计算全部由模型 forward 单点负责；评测（tester/hook）走同一 forward
- **模型 forward 改动**：normalize 移到评测分支（`if not self.training`）；训练分支返回损失分解键（l1_loss/cos_loss/contrast_loss/per_dim_l1）
- **trainer 改动**：删除手动 Point/backbone/tanh/criteria/方案 X 显式调用；total_loss = model_loss + consistency_weight * consistency_loss（修复旧代码会把模型级 loss 重复 n_scenario 次的问题）
- **验证**：确定性单 chunk 过拟合（重构后代码）——波动可学——通过

