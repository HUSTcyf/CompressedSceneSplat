# CLAUDE.md

> ⛔ **DEEPSEEK MODELS ARE PERMANENTLY BANNED from touching any code or running any command in this project (and all projects).** See `~/.claude/CLAUDE.md` for the full global ban and the 2026-08-07 incident that caused it. If you are running on a DeepSeek model, STOP and tell the user to switch models before doing any work.

> ⛔ **NO AI AGENT MAY RUN ANY COMMAND ON THE GPU SERVER (`isom`) ON ITS OWN.** All `ssh isom` / `scp ... isom:` / `rsync ... isom:` commands must be handed to the user to run manually (copy-pasteable block), then wait for their output. Local laptop commands are fine. This is non-negotiable — see `~/.claude/CLAUDE.md` "SERVER COMMANDS MUST BE RUN BY THE USER MANUALLY".

> ⛔ **ONLY GPU 6 AND 7 ON `isom` — ABSOLUTE RULE, NEVER BREAK.** Every GPU-touching command on the server MUST have `CUDA_VISIBLE_DEVICES=6,7`. GPUs 0–5 are other users' — touching them deadlocks the NVIDIA driver (the 8/7 incident). This is the single most important rule on this server. See `~/.claude/CLAUDE.md` "ONLY GPU 6 AND 7 ON THE SERVER".

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SceneSplat is a research project implementing Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining (ICCV 2025 Oral). The project provides a generalizable, open-vocabulary 3D Gaussian Splatting (3DGS) encoder that operates natively on 3DGS using vision-language pretraining and self-supervised training schemes.

## Recent Enhancements

The project has been extended with several new capabilities:
- **LitePT Integration**: Lightweight Point Transformer backbone (3.6× fewer parameters, 2× faster)
- **LoRA Fine-tuning**: Parameter-efficient adaptation for new datasets
- **SVD Compression**: Memory optimization through feature compression
- **SAM2 + SigLIP2 Fusion**: Enhanced segmentation with three-view fusion strategy
- **Open-Vocabulary Scenes (OVS)**: Zero-shot generalization to custom 3DGS data

## Environment Setup

```bash
conda env create -f env.yaml
conda activate scene_splat
```

The environment requires:
- Python 3.10
- PyTorch 2.7.0 with CUDA 12.8
- Custom compiled libraries in `libs/` (pointops, pointgroup_ops)

**Conda Environment Path**: `/home/isom/.conda/envs/scene_splat`
- Python executable: `/home/isom/.conda/envs/scene_splat/bin/python`
- Use this path for running scripts: `/home/isom/.conda/envs/scene_splat/bin/python script.py`

## Common Development Commands

### Training Commands

**Single-GPU Training:**
```bash
python tools/train.py \
  --config-file configs/scannet/lang-pretrain-scannet-mcmc-wo-normal-contrastive.py \
  --options save_path=exp_runs/experiment_name \
  --num-gpus 1
```

**Multi-GPU Training (Single Node):**
```bash
python tools/train.py \
  --config-file configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
  --options save_path=exp_runs/experiment_name \
    batch_size=8 batch_size_val=4 batch_size_test=4 \
    num_worker=32 gpu_nums=4 \
  --num-gpus 4
```

**Multi-Node Training (SLURM):**
```bash
srun python tools/train.py \
  --config-file configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
  --options save_path=exp_runs/experiment_name \
  --multi_node
```

### Self-Supervised Pretraining
```bash
python tools/ssl_pretrain.py \
  --config-file configs/concat_dataset/ssl-pretrain-concat-scan-ppv2-matt-3rscan-arkit-hyper-mcmc-base.py \
  --options save_path=exp_runs/ssl_pretrain/experiment_name
```

### Testing/Evaluation
```bash
python tools/train.py \
  --config-file configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
  --options save_path=exp_runs/experiment_name \
    weight=model_best.pth test_only=True
```

### Data Preprocessing
```bash
# Convert 3DGS .ply files to .npy format
python scripts/preprocess_gs.py \
  --input_root /path/to/ply_files \
  --output_root /path/to/npy_files

# Chunk scenes for training
python -u pointcept/datasets/preprocessing/sampling_chunking_data_gs.py \
  --dataset_root /path/to/preprocessed/data \
  --output_dir /path/to/chunked/output \
  --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3
```

### Checkpoint Conversion for OVS Data

When adapting pre-trained Scannet checkpoints to Open-Vocabulary Scene (OVS) data:

```bash
# Convert Scannet checkpoint to OVS checkpoint (11 channels, no coord)
python tools/modify_scannet_to_ovs.py
```

**Channel Mapping:**
- Original (Scannet): `in_channels=6` (xyz + rgb)
- Target (OVS): `in_channels=11` (color + opacity + quat + scale, **without coord**)

The conversion script modifies the stem layer weights:
- Copies RGB channels from original to new position
- Initializes opacity, quaternion, and scale channels with small random values
- Excludes coord from the input features

## Project Architecture

### Core Framework Structure (pointcept/)

- **datasets/**: Dataset loaders and preprocessing
  - `generic_gs.py`: Base dataset class for 3DGS data
  - Dataset-specific loaders (scannetgs.py, scannetppgs.py, matterport3dgs.py)
  - `transform.py`: Data augmentations and transformations

- **models/**: Model architectures
  - `point_transformer_v3/`: Main PT-v3m1 backbone
  - `point_transformer_v3_ssl/`: SSL variant with SimDINO
  - `default.py`: LangPretrainer model implementation
  - `losses/`: Custom loss functions

- **engines/**: Training and evaluation logic
  - `train.py`: Training engine with distributed support
  - `test.py`: Testing and evaluation engine
  - `pretrain.py`: Self-supervised pretraining engine

### Data Format

The project uses standardized `.npy` files for 3DGS data:
```
scene_folder/
├── coord.npy           # 3D coordinates [N, 3]
├── color.npy          # RGB colors [N, 3]
├── opacity.npy        # Opacity values [N, 1]
├── quat.npy           # Quaternion rotation [N, 4]
├── scale.npy          # Scale parameters [N, 3]
├── lang_feat.npy      # Language features [N, D] (optional)
├── valid_feat_mask.npy # Valid feature mask [N] (optional)
└── segment.npy        # Semantic labels [N] (for evaluation)
```

### Configuration System

Configurations follow a hierarchical pattern:
- Base configs in `configs/_base_/` define runtime and dataset settings
- Dataset-specific configs extend base configs
- Use `_base_` list to inherit configurations
- Override settings directly in config files or via `--options` flag

### Key Patterns

**Registry Pattern**: All components (models, datasets, losses) registered via decorators:
```python
@MODELS.register_module()
class MyModel(nn.Module):
    pass
```

**Distributed Training**: Built-in multi-GPU and multi-node support via NCCL
- Single node: Automatic via torch.distributed.launch
- Multi-node: SLURM integration with srun

**Evaluation Pipeline**:
- Training: Fast evaluation with grid sampling
- Testing: Full-scene evaluation with chunking for memory efficiency

## Working with Custom Data

### For Inference Only
1. Preprocess 3DGS scenes to .npy format using `scripts/preprocess_gs.py`
2. Use `GenericGSDataset` in test configuration
3. Set `test_only=True`, `skip_eval=True`, `save_feat=True`

### For Evaluation with Labels
1. Add `segment.npy` with per-gaussian semantic labels
2. Encode class names using `scripts/encode_labels.py`
3. Configure `class_names`, `text_embeddings`, `excluded_classes` in tester

## Important Notes

- GPU memory requirements: Vision-language pretraining requires ≥48GB GPU memory
- Multi-node training requires NCCL configuration (see SLURM scripts for examples)
- Batch sizes scale with GPU count: `batch_size = 2 * gpu_nums`
- Use `enable_amp=True` for mixed-precision training
- Evaluation uses neighbor voting (k=25) to improve segmentation quality
- Structural classes (wall, floor, ceiling) are excluded from foreground mIoU calculations

## Keep opencode Temporary Output Off the Server

opencode 的临时产物/日志一律**不要写到远端服务器**（`isom@172.22.1.12`），
本地 WSL 机器的 `/tmp/opencode` 不受此限。

背景（2026-08-02 事故）：在服务器上把需要终端输入的交互命令
（如 `zip -FF broken.npz --out fixed.npz`）重定向到 `/tmp/opencode/fix.log`，
该进程在无输入时死循环输出，**写满 504G 磁盘**，导致远端训练中断、rsync 上传失败。

规则：
- 不要在服务器上把交互式命令（zip -FF、修复工具等）重定向到日志文件；确需运行就用
  `timeout 30` 之类限制时长，或输出到本地后再传回
- 临时下载/调试产物优先放本地（WSL），需要时再 scp 到服务器
- 服务器 `/tmp` 只放短期、用完即删的文件；正式数据走 `/home/isom/cyf/` 目录
- 服务器磁盘紧张时先检查 `/tmp` 下的大文件（`du -sh /tmp/* | sort -rh | head`）

## Critical Configuration Issues

### Transform Pipeline Order for SVD-Compressed Features

When using SVD-compressed language features (`load_compressed_lang_feat=True`), the transform pipeline **must** apply `FilterValidPoints` **before** `FilterCoordOutliers`.

**Why This Order Matters:**

1. **SVD Loading Creates Size Mismatch**: When SVD compression is enabled:
   - `coord.npy` contains all points (e.g., 1,000,000 points)
   - `lang_feat_grid_svd_r{rank}.npz` contains features only for valid points (e.g., 856,838 points)
   - The SVD indices map valid points to compressed grid features

2. **Wrong Order Causes Misalignment**:
   ```
   FilterCoordOutliers FIRST → coord: 1M → 970K
   FilterValidPoints SECOND → coord: 970K, lang_feat: 856K (MISMATCH!)
   Result: No correspondence between coord and lang_feat
   ```

3. **Correct Order Ensures Alignment**:
   ```
   FilterValidPoints FIRST → coord: 1M → 856K (matches lang_feat)
   FilterCoordOutliers SECOND → coord: 856K → 836K, lang_feat: 856K → 836K
   Result: coord and lang_feat stay aligned throughout
   ```

**How FilterValidPoints Works:**

The `FilterValidPoints` transform in `pointcept/datasets/transform.py` (lines 2188-2248) only filters arrays where `len(value) == len(valid_mask)`. Since SVD-loaded `lang_feat` has fewer elements than `valid_feat_mask`, it gets automatically skipped, preserving the alignment with `coord` after filtering.

**Correct Transform Pipeline Configuration:**

```python
transform=[
    dict(type="CenterShift", apply_z=True),
    # Step 1: Filter valid points FIRST to align coord with lang_feat (CRITICAL for SVD)
    dict(type="FilterValidPoints", key="valid_feat_mask"),
    # Step 2: Filter outliers on the aligned data
    dict(type="FilterCoordOutliers", percentile_low=0.5, percentile_high=99.5),
    dict(type="CenterShift", apply_z=True),
    # ... rest of pipeline
]
```

**Files with Correct Order:**
- `configs/custom/lang-pretrain-litept-ovs-gridsvd.py` (lines 389-391)
- `configs/custom/lang-pretrain-litept-scannet.py` (only uses FilterValidPoints)
- `configs/custom/lang-pretrain-litept-matt.py` (only uses FilterValidPoints)

**Verification:**

To verify coord-lang_feat alignment during training:
```bash
python tools/train.py --config-file configs/custom/lang-pretrain-litept-ovs-gridsvd.py
```

Expected output should show:
- `coord shape: (N, 3)` and `lang_feat shape: (N, 16)` with matching N
- Adjacent points have smaller feature differences than random points (spatial coherence preserved)

## SVD 压缩评测与训练关键经验（2026-08）

### segment.npy 读取（易错点）
- scannetpp 的 `segment.npy` 是 **[N, 3] 多维**，语义标签在第 0 列——必须 `segment[:, 0]`（见 scannetppgs.py:183）。
- 诊断脚本用 `reshape(-1)` 展平会**错位**（第一版评测诊断结论因此全部作废，上界 1.6% 是错位标签的假象；修正后为 41.8%）。

### 16 维压缩目标的结构（评测失败根因）
- 目标 = 公共方向（场景偏移，~96% 能量）+ 波动维（判别信息，~4%）。
- 模型只学到公共方向：`cos(F,T)=0.95` 是公共方向的假象（去均值后 cos ≈ -0.04）；波动维每维 corr ≈ 0、幅度仅为目标的 6-30% → 分类失败（f-mIoU 0.25%）。
- 根因链：逐场景 SVD 基的旋转/置换歧义（`canonicalize_svd_sign` 只修符号 2^16→1）+ 原方差加权 L1 **反向**（公共方向权重 1.0、波动维 0.1）。

### 训练配置（当前推荐）
- `SVDWeightedL1Loss`: `weight_strategy="inverse_variance"`（波动维高权重，公共方向 0.1）。
- `AggregatedContrastiveLoss`: `loss_weight=0.3`。epoch_progress bug 已修（迭代级进度 `(epoch + iter/total)/max_epoch`）——1-epoch 训练下后 25% 激活，此前 contrast 全程 0.0000。
- `svd_center=False`（去均值方案已否决：上界 44%→31%，且 f-mIoU 同样下降——大面积物体类也依赖公共方向）。

### 评测链路（已修复的坑）
- **【最关键，2026-08-03】tester 的 Q 拟合行序错位**：`accumulated_features[valid_mask]` 用点级 valid_mask 取 cell 编号序的特征行 + 点级 segment 标签——行序不对应，Q 被错位标签污染（rank-1 坍缩）→ 完美 GT 特征走链 0%（上界 41.8%）。**修复**：Q 拟合改用 cell 级特征（`feature_counts > 0` 行）+ cell 标签 = `np.unique(inverse, return_index=True)` 反查每 cell 首个原始点标签（pointcept/engines/test.py）。修复后：GT 走链 34.9%（✓≈上界）、真实模型 0.6%→5.4%。
- 行序语义：`frag["index"]` = GridSample(mode=train) 输出行号（= cell 编号，非原始点索引）；`inverse[i]` = 点 i 的 cell 编号；`pred[inverse]` 是逐点映射（正确）。**Q 拟合/标签必须同一行序（cell 级），不能用点级 mask 配 cell 行**。
- 评测链自检：把 GT 目标特征（dataset 的 lang_feat）当模型输出走完整 tester 链——若 ≈ 独立上界（35-52%）则链正确。任何评测链改动后必做。
- 训练评测 hook（LangPretrainZeroShotSemSegEval）直接前向 val batch（无 scatter），point_feat 与 segment 行序一致——自洽无需修复；metric 在 cell 级算（与 tester 逐点近似等价）。
- Procrustes 拟合必须**归一化 + 类平均 1/n**（tools/projection/compute_procrustes_alignment_simple.py）——类求和会让大类（wall/floor/ceiling）主导 Q。
- 评测 hook 已改为**每场景 Q**（与 tester 一致），不再是第一个场景单 Q 全局应用。
- **pred 缓存**：`save_path/result_ScanNetPPGSDataset/*_pred.npy`——修改评测逻辑后必须删除，否则 tester 直接加载旧预测跳过前向（错位 Q 时代的缓存更是必须清）。

### 上界验证方法
- 目标特征（`segment[:,0]` + valid 空间 + canonicalize）直接每场景 Q 分类 = 模型完美的 mIoU 上界。来源：`diag_fixed2.py` 逻辑（cell 级加载 + `np.unique(idx, return_index=True)` 反查 cell 标签 + 归一化类平均 Q）——精确复刻 18.4-52.3%（均值 ≈ 41.8%）。改目标/评测前先跑上界，可快速判断改动是否有潜力。

### 训练瓶颈与突破（2026-08-06 最终定稿）
- **逐点损失（加权 L1/cos）在 16 维压缩目标上失败**：96% 能量在公共方向（dim0），逐点 L1 梯度 ~99% 被其主导 → 模型变场景均值预测器（3.49% vs 33.94% 上界）。评测只吃类均值结构（跨场景一致 0.90、可学）——训练与评测信号错配。
- **突破：ClassMeanProcrustesLoss**（`pointcept/models/losses/misc.py`）——逐场景类均值 + detach 正交 Procrustes Q（吸收基旋转，基无关）残差。3.49% → **24.07% fg / 24.89% all-class**（tester 全 50 场景，oracle 每场景 Q）。
- **冠军配方（v10）**：top50 chunks（val 类分布挑选）+ loop=10 + 10 epochs + `ClassMeanProcrustesLoss(min_points=5, point_weight=0.2)` + wd 0.05 + 16dim-init + OneCycleLR(pct_start=0.1) + noaug。最佳 checkpoint：`exp/smoke-ptv3m1-16-scannetpp-v2-classprocrustes-v10/model/epoch_9.pth`。
- **关键坑**：cos 损失饱和后是噪声注入源（E3 崩溃，去掉后消失）；min_points 50→10→5 救活小类；top50 只覆盖 84/100 类；多样数据/更长训练/类权重/逐点增强/集成/评测旋钮全部更差；seed 方差 ±3%（v10 是唯一 +3σ 离群值）；从 checkpoint 重启必然退化。
- **剩余瓶颈**：12 个零类（3 个真缺失 + 9 个混淆类）不可收复；25% 目标与框架实证上限不兼容。详见 `docs/classmean_procrustes_training.md`（完整实验档案）。
- 已排除方向：训练量（133× pass +0.17%）、统一符号/基（注册表 v3 2.53% 反降）、去均值压缩（上界 44→31%）、无监督对齐（MUSE mutual-NN≈0）。