# CLAUDE.md

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
- PyTorch 2.5.1 with CUDA 12.1/12.4
- Custom compiled libraries in `libs/` (pointops, pointgroup_ops)

**Conda Environment Path**: `/new_data/cyf/.conda/envs/scene_splat`
- Python executable: `/new_data/cyf/.conda/envs/scene_splat/bin/python`
- Use this path for running scripts: `/new_data/cyf/.conda/envs/scene_splat/bin/python script.py`

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