"""
Config file for Vision-Language Pretraining with LitePT on OVS (Open-Vocabulary Scenes) 3DGS data.
Using SVD-compressed grid features with density-invariant training.

This config implements representation learning with compressed language features (rank=16),
enabling open-vocabulary 3D scene understanding with memory-efficient training.

Key features:
- Uses SVD-compressed language features (rank=16) instead of full 768-dim features
- Density-invariant training with multi-scenario sampling (dense, half, single)
- Decoder output dimension matches SVD rank (16 dims)
- Grid-based feature alignment for consistency across different densities

Combines datasets from:
- /new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs
- /new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS

SVD files should be located at: /new_data/cyf/projects/SceneSplat/grid_svd_output/

Usage:
    # Using train_lite.py with density-invariant training (default for this config)
    python tools/train_lite.py --config-file configs/custom/lang-pretrain-litept-ovs-gridsvd.py --num-gpus 4

    # Override SVD rank
    python tools/train_lite.py --config-file configs/custom/lang-pretrain-litept-ovs-gridsvd.py \\
        --options density_invariant.svd_rank=8 --num-gpus 4
"""

_base_ = [
    "../_base_/default_runtime.py",
]

# ============================================================================
# Misc custom settings
# ============================================================================
debug = 0
gpu_nums = 1  # if debug else 4

# Save path for LitePT-16 training (compressed features)
save_path = "exp/lite-16-gridsvd"
batch_size = 1 * gpu_nums
batch_size_val = 1 * gpu_nums
batch_size_test = 1 * gpu_nums
num_worker = 0  # Set to 0 for single GPU to avoid multiprocessing issues
mix_prob = 0.0  # no mixup for language pretraining
empty_cache = False
enable_amp = True
evaluate = False
find_unused_parameters = True

# ============================================================================
# Model settings - LitePT with SVD-compressed Vision-Language Pretraining
# ============================================================================
# Trainer type: DensityInvariantTrainer for multi-scenario density-invariant training
train = dict(type="DensityInvariantTrainer")

# SVD-compressed language feature dimension
# When using svd_rank=16, we predict 16-dimensional compressed features
# This is much more memory-efficient than predicting full 768-dim features
svd_rank = 16  # SVD compression rank (8, 16, 32 are common choices)
lang_feat_dim = svd_rank  # Decoder output matches SVD rank
FD = lang_feat_dim

model = dict(
    type="LangPretrainer",  # Language Pretrainer for VL learning
    backbone=dict(
        type="LitePT",
        in_channels=11,  # 3DGS features: color(3) + opacity(1) + quat(4) + scale(3) [coord removed]
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # Encoder (scaled to support 16 decoder output)
        # Reduced encoder channels since we only need to output 16 dims
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),  # Last encoder stage matches decoder input
        enc_num_head=(2, 4, 8, 14, 28),  # Adjust heads based on FD
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        # Decoder (output 16 dimensions to match SVD rank)
        # Simplified decoder for compressed feature prediction
        dec_depths=(2, 2, 2, 2),
        dec_channels=(FD, FD*2, FD*4, 144),  # Gradually reduce from 16 to 2
        dec_num_head=(32, 16, 8, 4),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(True, True, True, False),
        dec_attn=(False, False, False, True),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        # Common settings
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False,
    ),
    # Language pretraining losses for compressed features
    criteria=[
        dict(type="L2Loss", loss_weight=1.0),  # MSE loss for compressed features
        dict(type="CosineSimilarity", loss_weight=1.0),  # Cosine similarity for alignment
        # dict(type="ValidNonValidContrastiveLoss", loss_weight=0.5, temperature=0.1, margin=0.5),
    ],
)

# ============================================================================
# Density-Invariant Training Configuration
# ============================================================================
density_invariant = dict(
    # SVD compression rank to load (must match lang_feat_dim)
    svd_rank=16,  # 8, 16, 32 are common choices
    # Note: SVD files are expected to be in the same directory as the scene data
    # (e.g., /path/to/dataset/train/{scene_name}/lang_feat_grid_svd_r16.npz)

    # Sampling ratios for half-density scenario
    min_sample_ratio=0.3,  # Minimum sampling ratio (30%)
    max_sample_ratio=0.7,  # Maximum sampling ratio (70%)

    # Consistency loss settings
    consistency_weight=10,  # Weight for density consistency loss
    consistency_type="mse",  # Options: "mse", "cosine", "kl"

    # Training scenarios to use
    scenarios=["dense", "half", "single"],  # All three scenarios
    # Weight for each scenario's loss
    scenario_weights=dict(
        dense=1.0,    # Dense input (all valid points)
        half=1.0,    # Half density (30-70% sampling)
        single=1.0,  # Single point per grid
    ),

    # Whether to use compressed features for grid alignment loss
    use_compressed_features=True,
)

# ============================================================================
# Scheduler settings
# ============================================================================
eval_epoch = 10  # total eval & checkpoint epoch
epoch = eval_epoch * 10  # total data loops (100 epochs for pretraining)
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],  # [backbone_lr, block_lr]
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# ============================================================================
# Dataset settings
# ============================================================================
dataset_type = "GenericGSDataset"

# OVS data roots (point to parent directory, split specifies the subdirectory)
data_root_ovs_1 = "/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs"
data_root_ovs_2 = "/new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS"

data = dict(
    num_classes=100,  # Placeholder for pretraining (not used)
    ignore_index=-1,
    train=dict(
        type="ConcatDataset",  # Use ConcatDataset to combine multiple data sources
        datasets=[
            dict(
                type=dataset_type,
                split="train",  # Use 'train' subdirectory (matches train/* pattern)
                data_root=data_root_ovs_1,  # First data source: lerf_ovs
                sample_tail_classes=False,
                load_compressed_lang_feat=True,  # Load SVD-compressed lang_feat (16-dim instead of 768-dim)
                svd_rank=16,  # SVD rank to load (must match density_invariant.svd_rank)
                transform=[
                    # Initial preprocessing
                    dict(type="CenterShift", apply_z=True),
                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="NormalizeCoord"),  # Normalize after GridSample
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    # Color augmentation
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    dict(type="FilterValidPoints", key="valid_feat_mask"),
                    dict(
                        type="GridSample",
                        grid_size=0.01,  # Smaller grid_size to preserve more valid points
                        hash_type="fnv",
                        mode="train",
                        keys=(
                            "coord",
                            "color",
                            "opacity",
                            "quat",
                            "scale",
                            "lang_feat",
                            "valid_feat_mask",
                            "point_to_grid",  # Sampled along with other data
                        ),
                        return_grid_coord=True,  # Required by LitePT's GridPooling
                    ),
                    # Optional: SphereCrop for additional point reduction if needed
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    # Collect features - grid_coord is required by LitePT's GridPooling
                    # Also collect name and scene_path for SVD file loading
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask", "name", "scene_path", "point_to_grid"),
                        feat_keys=("color", "opacity", "quat", "scale"),  # 11 channels
                    ),
                ],
                test_mode=False,
            ),
            dict(
                type=dataset_type,
                split="train",  # Use 'train' subdirectory (matches train/* pattern)
                data_root=data_root_ovs_2,  # Second data source: 3DOVS
                sample_tail_classes=False,
                load_compressed_lang_feat=True,  # Load SVD-compressed lang_feat (16-dim instead of 768-dim)
                svd_rank=16,  # SVD rank to load (must match density_invariant.svd_rank)
                transform=[
                    # Initial preprocessing
                    dict(type="CenterShift", apply_z=True),
                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="NormalizeCoord"),  # Normalize after GridSample
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    dict(type="FilterValidPoints", key="valid_feat_mask"),
                    dict(
                        type="GridSample",
                        grid_size=0.01,  # Smaller grid_size to preserve more valid points
                        hash_type="fnv",
                        mode="train",
                        keys=(
                            "coord",
                            "color",
                            "opacity",
                            "quat",
                            "scale",
                            "lang_feat",
                            "valid_feat_mask",
                            "point_to_grid",  # Sampled along with other data
                        ),
                        return_grid_coord=True,  # Required by LitePT's GridPooling
                    ),
                    # Optional: SphereCrop for additional point reduction if needed
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    # Collect features - grid_coord is required by LitePT's GridPooling
                    # Also collect name and scene_path for SVD file loading
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask", "name", "scene_path", "point_to_grid"),
                        feat_keys=("color", "opacity", "quat", "scale"),  # 11 channels
                    ),
                ],
                test_mode=False,
            ),
        ],
        loop=1,  # ConcatDataset loop parameter
    ),
    val=dict(
        type=dataset_type,
        split="val",  # Use 'val' subdirectory
        data_root=data_root_ovs_1,  # Use lerf_ovs parent directory
        transform=[
            dict(type="CenterShift", apply_z=True),
            # Step 1: Filter to only valid points
            dict(type="FilterValidPoints", key="valid_feat_mask"),
            # Step 2: Grid sample (before NormalizeCoord) - required for GridPooling
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "valid_feat_mask", "point_to_grid"),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            # grid_coord is required by LitePT's GridPooling
            # Also collect name and scene_path for SVD file loading
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask", "name", "scene_path"),
                feat_keys=("color", "opacity", "quat", "scale"),  # 11 channels
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",  # Use 'val' subdirectory
        data_root=data_root_ovs_1,  # Use lerf_ovs parent directory
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            # Step 1: Filter to only valid points
            dict(type="FilterValidPoints", key="valid_feat_mask"),
            # Step 2: Grid sample for initial point reduction (test_cfg.voxelize does further processing)
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "valid_feat_mask", "point_to_grid"),
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "point_to_grid"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "lang_feat", "valid_feat_mask", "name", "scene_path", "point_to_grid"),
                    feat_keys=("color", "opacity", "quat", "scale"),  # 11 channels
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)

# ============================================================================
# Hooks
# ============================================================================
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]
