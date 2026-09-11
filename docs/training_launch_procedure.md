# 训练启动固定化流程（2026-08-03 定稿）

> 用途：SVD 压缩特征（16 维）PT-v3m1 训练的全部启动操作。防遗忘清单——
> 环境、配置铁律、启动命令、验证步骤、已知坑。每条规则都有事故记录支撑
> （详见 docs/svd_compression_diagnosis.md）。

## 1. 环境

| 项 | 值 |
|---|---|
| 服务器 | `isom@172.22.1.12`，repo `/home/isom/cyf/CompressedSceneSplat` |
| conda | `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate scene_splat` |
| 本地 repo | `/home/cyf/CompressedSceneSplat`（改代码 → scp 同步 → 服务器跑） |
| GPU | 143GB 卡 × 8；先 `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` 查空闲 |

## 2. 启动命令模板

### 多 chunk 正式训练（2 GPU）
```bash
ssh isom@172.22.1.12
cd /home/isom/cyf/CompressedSceneSplat
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate scene_splat
mkdir -p exp/<NAME>
CUDA_VISIBLE_DEVICES=<G0,G1> nohup python tools/train_lite.py \
  --config-file configs/custom/<CONFIG>.py \
  --num-gpus 2 --density-invariant \
  < /dev/null > exp/<NAME>/launch.stdout.log 2>&1 &
```
**`< /dev/null` 必须**（否则 ssh 会话挂住不返回）。

### 单 chunk 过拟合（1 GPU，诊断用）
```bash
CUDA_VISIBLE_DEVICES=<G> nohup python tools/train_lite.py \
  --config-file configs/custom/lang-pretrain-ptv3m1-scannetpp-v2-overfit-*.py \
  --options batch_size=1 batch_size_val=1 batch_size_test=1 \
  --num-gpus 1 --density-invariant \
  < /dev/null > exp/<NAME>/launch.stdout.log 2>&1 &
```
两个硬性要求：`batch_size=1` + `--num-gpus 1`（`batch_size % world_size == 0` 断言，defaults.py:136；且单样本 loader 配 batch_size>1 → drop_last 清空 → OneCycleLR total_steps=0 崩溃）。

## 3. 配置铁律（每条都有事故）

0. **`loop` 陷阱（2026-08-04 事故，最重要）**：`defaults.py:123` 会强制注入
   `cfg.data.train.loop = cfg.epoch // cfg.eval_epoch`——eval_epoch=1、epoch=20 时
   loop=20 → 每 epoch 数据过 20 遍 → 20 epochs 数据共过 400 遍 ≈ **115h**（预期 5.6h 的 20 倍）！
   已修复：defaults.py 改为 `cfg.data.train.get("loop", epoch // eval_epoch)`（尊重显式值），
   **正式 config 必须显式 `loop = 1`**（每 epoch 数据 1 遍；checkpoint 由 CheckpointSaver(save_freq=1)
   每 epoch 保存，不依赖 loop）。症状识别：日志 `[1/20][iter/20115]`（20115 ≈ 1006×20）或
   Remain 显示 100+ 小时。
1. **transform 顺序**：`FilterValidPoints` 必须在 `FilterCoordOutliers` **之前**（SVD 特征只覆盖 valid 点，长度 ≠ 全点长）。
2. **数据增强（⚠️ 本条已过期，2026-08-06）**：去增强实验（noaug 1.61% vs 增强版 3.29%）表明去增强没用——**正式配置保留标准几何增强 + Chromatic***。注意：单 chunk 诊断实验（overfit-det 系列）仍用确定性配置（去全部 random）以隔离变量。
   **勘误**：该结论在逐点损失时代成立；后续冠军配方（v10，fg 24.07%）使用 **noaug（仅 Chromatic*）**——几何增强改变 GridSample cell 哈希、扰动同一物理点目标（见 svd_compression_diagnosis.md §8.17），正式训练应去几何增强。
3. **`align_text16=False`**（方案 X 已放弃：Q 列符号跨 chunk 随机 → 模式坍缩，dim0 corr −0.23）。
4. **`weight_strategy` 语义（实测，与直觉相反！）**：
   - canonicalize 后 d0（公共方向）被压成全正区间 → **方差最小** → `inverse_variance` 给 d0 权重 **1.0**（意图是 0.1！）
   - 单 chunk：用 `variance`（d0 0.1、大波维 1.0）——已验证 corr 0.73→0.90（overfit-det-v2）
   - 多 chunk：`variance` 尾部不稳定（globalsign-v2 3.59% < v1 4.62%）；**当前推荐 `inverse_variance`（v1 设置）**，等待更好方案
5. **对比损失 schedule 语义**：`schedule="last_75"` = **训练后 75%**（25% 进度即激活，非后 25%！）——激活后总 loss 从 ~0.4 跳到 1.1-1.9 是正常现象（对比值大 + 类均值尚不判别），**不是训练崩溃**。
6. **初始化权重（必须！此前全部遗漏）**：
   ```
   weight=/home/isom/cyf/CompressedSceneSplat/checkpoints/lang-pretrain-pt-v3m1-16dim-init.pth
   ```
   768→16 维转换的预训练 backbone（转换自 lang-pretrain-concat-scan-ppv2-matt，best mIoU 0.208，2026-08-02 生成，配套 smoke）。**所有 2026-08-03 的实验（noaug/globalsign/overfit 系列）都是随机初始化——后续训练必须加**。
7. **符号对齐注册表（多 chunk 必须）——当前用 v3 版**：
   ```
   global_sign_path=/home/isom/cyf/SceneSplat/scannetpp_v2/lang_feat_grid_svd_r16_global_signs_v3.npz
   ```
   v3（多参考投票，2026-08-03 完成）：3821/3821 chunk 全覆盖（v2 只有 3799）、d0 零翻转（v2 有 577 个噪声翻转）。生成脚本 `tools/compression/build_global_sign_registry_v3.py`（只读，输出 `_v3.npz`——**注意 config 必须指向 `_v3` 后缀文件**）。原始 npz **零写入**原则：注册表是 sidecar，数据集加载时翻转（scannetppgs.py 的 global_sign_path 参数，默认关）。
8. **评测链路不可动**：tester 的 Q 拟合行序修复在 pointcept/engines/test.py（cell 级特征 + cell 代表点标签）——改动任何评测逻辑后必须做自检（GT 特征走链 ≈ 上界 41.8%）。

## 4. 验证清单（启动后）

- [ ] ~4 分钟内出现 `Train: [1/1][N/1006]`（首 batch 数据加载耗时）
- [ ] loss 前 100 iter 从 ~10 快速下降（随机初始化时）
- [ ] 训练结束读 `exp/<NAME>/per_dim_correlation_history.json`：
  - 多 chunk：d1 corr 应 >0.3（符号对齐生效）；全维 ≈0 说明符号没生效或目标不一致
  - 单 chunk：d1 corr 应 >0.8（variance 加权 + 足够 iter）
- [ ] tester 输出 `Val result: mIoU/mAcc/allAcc`，与基线对比（下表）
- [ ] 离线全 chunk 分析（诊断用）：`analyze_v2_ckpt.py` 同口径方法测 per-dim corr / 真实 L1

## 5. 已知坑清单

| 坑 | 症状 | 解决 |
|---|---|---|
| ssh + nohup 挂住 | 命令 120s 不返回 | 必须 `< /dev/null` |
| batch_size % world_size | AssertionError | 单卡 batch_size=1 + --num-gpus 1 |
| OneCycleLR total_steps=0 | 启动即崩 | 单样本时必须 batch_size=1 |
| 日志文件名 | `train_YYYYMMDD_HHMMSS.log`（带时间戳） | `ls -t exp/<NAME>/train_*.log` 取最新 |
| 评测缓存 | 修改评测逻辑后 tester 加载旧预测 | 删除 `exp/<NAME>/result_ScanNetPPGSDataset/` |
| "L1 平台 0.4" 误读 | 训练日志 L1 不降 | 该值是 `0.5×Σ w_d·|e_d|` 合成数；真实逐点 L1 看离线分析（已到 0.068） |
| loss 尾段反弹 | ~iter 253 起 0.4→1.1-1.9 | 对比损失 last_75 激活，正常 |
| flash 不支持 RPE | 断言 | RPE 需 `enable_flash=False`（慢 2-3×，验证中） |

## 6. 实验结果速查（截至 2026-08-03）

| 配置 | 关键差异 | Val mIoU |
|---|---|---|
| noaug | 无增强 + 随机初始化 + 无注册表 | 1.61%（→结论：去增强无用） |
| **globalsign** | + v2 符号注册表 | **4.62%** |
| globalsign-v2 | + variance 加权 | 3.59%（尾部不稳定） |
| globalsign-v3 | + v3 注册表 + loop=1 + 16dim-init，**2026-08-04 重启中** | — |
| 增强版 baseline（历史） | 带几何增强 | 3.29% |
| GT 上界（完美模型） | GT 特征走链 | 41.8% |

2026-08-04 修正：v3 正式 config 改名 `lang-pretrain-ptv3m1-scannetpp-v2-globalsign-v3.py`
（去掉 noaug，恢复标准增强——用户定论增强无用论只针对"去增强"，正式配置用默认增强）；
save_path = `exp/smoke-ptv3m1-16-scannetpp-v2-globalsign-v3`。

单 chunk 诊断结论：variance 加权 + 1200 iter → dim1 corr 0.90、跨 chunk 迁移 0.90（_1）、真实 L1 0.068（MLP 上界 0.019）。

## 7. 下一步待办（2026-08-03 状态）

- [x] v3 注册表构建（3821/3821 覆盖，d0 零翻转）——config 已建 `lang-pretrain-ptv3m1-scannetpp-v2-noaug-globalsign-v3.py`
- [ ] **跑 globalsign-v3 正式训练**（v3 注册表 + inverse_variance；用 16dim-init 初始化）
- [ ] **用 16dim-init 初始化重跑多 chunk 正式训练**（weight 遗漏的修正）
- [ ] RPE 单 chunk 验证结果分析（exp/overfit_det_rpe，非 flash 路径 patch_size 前导 0 bug 已修）
- [ ] 多 chunk 加 epoch（当前 epoch=1 = 每 chunk 平均只见 1 次，欠拟合）
- [ ] 伪标签 Q 消融（reviewer Q2：对齐对对应关系质量的敏感性）
- [ ] 统一基（text16 投影压缩）——需重新下载原始 768 维特征
