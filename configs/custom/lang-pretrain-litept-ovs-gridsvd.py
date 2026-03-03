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
empty_cache = True  # ENABLED to free memory between iterations and reduce OOM
enable_amp = True  # Re-enabled for memory efficiency, using gradient clipping to prevent NaN
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
    verbose_losses=True,  # Enable verbose loss printing (L2 and Cos per iteration)
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
        dec_depths=(2, 2, 2, 2),
        dec_channels=(FD, FD*2, FD*4, 126),  # (16, 64, 128, 252) - 4x intermediate capacity
        dec_num_head=(1, 2, 4, 7),  # Updated last head for 252 channels (14*18=252)
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(True, True, True, False),
        dec_attn=(False, False, False, True),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        # Common settings
        # REDUCED mlp_ratio from 4 to 2 for stability
        # - Prevents gradient explosion in low-dimensional decoder stages (e.g., dec0 with 16-dim)
        # - Reduces parameter count by ~50% in MLP layers
        # - Maintains expressiveness while improving training stability
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.1,  # REDUCED from 0.3 - 16-dim task needs less dropout
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False,
    ),
    # Language pretraining losses for compressed features
    #
    # PER-DIMENSION WEIGHTED REGRESSION LOSSES for 16-dimensional SVD features
    # ------------------------------------------------------------------
    # FIX: Use data-driven variance-based weighting to handle skewed SVD distribution
    #
    # Two weighting strategies available:
    #   1. "static": Exponential decay based on dimension index (d[0] > d[1] > ... > d[15])
    #   2. "variance": Data-driven - higher variance = higher weight (DEFAULT)
    #
    # Variance-based weighting (current config):
    #   - Automatically computes per-dimension variance from GT data
    #   - Uses EMA (exponential moving average) for stable estimates
    #   - Higher variance = more information = higher weight
    #   - Adapts to different datasets automatically
    #
    # Weight configuration:
    #   - base_weight=1.0: Maximum weight for highest-variance dimension
    #   - min_weight=0.1: Minimum weight (prevents zero gradients)
    #   - variance_momentum=0.99: EMA smoothing for variance estimation
    #
    # Expected behavior:
    #   - Early SVD dimensions (d[0]-d[4]) typically have high variance → high weight
    #   - Later SVD dimensions (d[10]-d[15]) typically have low variance → low weight
    #   - But the actual weights are determined by DATA, not by assumption
    #
    criteria=[
        # SVDWeightedL1Loss: Variance-based dimension weighting
        dict(
            type="SVDWeightedL1Loss",
            loss_weight=1.0,  # Overall loss weight
            reduction="mean",
            base_weight=1.0,  # Maximum weight
            min_weight=0.1,  # Minimum weight (prevents zero gradients)
            weight_strategy="variance",  # "variance" (data-driven) or "static" (exponential decay)
            variance_momentum=0.99,  # EMA momentum for stable variance estimation
        ),

        # CosineSimilarity: Focus on directional alignment (not weighted)
        dict(
            type="CosineSimilarity",
            loss_weight=1.0,
            reduction="mean",
        ),

        # Rendered2DLoss: Spatial consistency via Gaussian splatting rendering
        # Enforces spatially consistent predictions by comparing rendered 2D features
        # Uses gsplat rasterization with pre-rendered GT feature maps
        # Weight schedule: ramps up from 0 to 1.0 by 50% training progress
        # INCREASED WEIGHT for better spatial smoothness and feature consistency
        # dict(
        #     type="Rendered2DLoss",
        #     loss_weight=1.0,
        #     gaussian_train_root="/new_data/cyf/projects/SceneSplat/gaussian_train",
        #     datasets_root="/new_data/cyf/projects/SceneSplat/datasets",
        #     warmup_progress=0.0,  # Start immediately (no warmup)
        #     target_progress=0.5,  # Reach max weight at 50% training progress
        #     max_num_views=10,  # Use up to 10 views per scene to save memory
        # ),

        # AggregatedContrastiveLoss: DISABLED for 16-dim training
        # Re-enable only when using SVD rank 32 or higher
        # dict(
        #     type="AggregatedContrastiveLoss",
        #     temperature=0.2,
        #     reduction="mean",
        #     loss_weight=0.1,
        #     schedule="all",
        # ),
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
    consistency_weight=0.5,  # Density consistency loss weight
    consistency_type="mse",  # Options: "mse", "cosine", "kl"

    # Training scenarios to use
    # Temporarily removed "half" scenario to reduce memory pressure and improve speed
    scenarios=["dense", "single"],  # Two scenarios (half removed for optimization)
    # Weight for each scenario's loss
    scenario_weights=dict(
        dense=1.0,    # Dense input (all valid points)
        # half=1.0,   # Half density (30-70% sampling) - TEMPORARILY DISABLED
        single=1.0,  # Single point per grid
    ),

    # Whether to use compressed features for grid alignment loss
    use_compressed_features=True,

    # Forward pass mode: batched vs separate
    # False: Each scenario forwarded independently (slower but no cross-contamination)
    # True: All scenarios batched together (faster but may interfere via sparse conv)
    batched_forward=True,  # CHANGED to False for cleaner training (was True)
)

# ============================================================================
# Scheduler settings
# ============================================================================
eval_epoch = 10  # total eval & checkpoint epoch
epoch = eval_epoch * 10  # total data loops (200 epochs for pretraining)

# ============================================================================
# Optimizer settings with mode-collapse prevention
# ============================================================================
# Base optimizer configuration
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)

# Gradient clipping (configured separately, used in trainer)
clip_grad = 1.0

# Scheduler configuration
scheduler = dict(
    type="OneCycleLR",
    # max_lr 对应所有参数组: [默认组, enc.block, dec.block, dim_scale, dec0, dec0.mlp, dec0.fc]
    max_lr=[0.001, 0.001, 0.0001, 0.001, 0.00005, 0.00002, 0.00001],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# ============================================================================
# Layer-specific parameter groups for mode-collapse prevention
# ============================================================================
# Based on mode collapse analysis: dec0/block1/fc1 has the highest weight norms
# (L2 norm max=6.87) causing the collapse chain.
#
# Strategy: Apply higher weight decay and lower learning rate to the problematic
# decoder MLP layers to prevent weight explosion while maintaining normal
# training for other parts of the model.
#
# Parameter groups are matched by keyword in parameter names (prefix matching):
# - "block": Matches all encoder/decoder transformer blocks (default group)
# - "dim_scale": Learnable dimension-wise scaling factor
# - "dec0.block1.mlp": The problematic decoder stage 0, block 1 MLP layers
# - "dec0.block1.mlp.0.fc": Specifically targets fc1/fc2 in the MLP
#
# Layer naming in LitePT:
#   backbone.dec.dec{s}.block{i}.mlp.{j}.fc{k}.{weight|bias}
#   where: s=stage (3,2,1,0), i=block_index (0,1), j=mlp_index (0), k=fc_layer (1,2)
#   Problematic layers: dec0.block1.mlp.0.fc1, dec0.block1.mlp.0.fc2
# ============================================================================
param_dicts = [
    # Group 1: Encoder transformer blocks
    # Keep original learning rate for encoder (features extraction is stable)
    dict(keyword="enc.block", lr=0.001, weight_decay=0.05),

    # Group 2: Decoder transformer blocks (higher stages: dec3, dec2, dec1)
    # Reduced learning rate for decoder blocks to prevent collapse
    dict(keyword="dec.block", lr=0.0001, weight_decay=0.05),

    # Group 3: Dimension-wise scaling (for SVD feature magnitude compensation)
    dict(keyword="dim_scale", lr=0.001, weight_decay=0.05),

    # Group 4: Final decoder stage (dec4/s=0 in code, outputs 16-dim features)
    # This is the bottleneck stage with extreme gradient amplification
    # Channels: 126→64→32→16, causing 8× gradient amplification
    dict(
        keyword="dec0",  # Matches all layers in final decoder stage (s=0)
        lr=0.00005,  # 95% lower than encoder block lr
        weight_decay=0.15,  # 3× higher weight decay for regularization
    ),

    # Group 5: dec0.block1 MLP (specific problematic layer)
    # The MLP with mlp_ratio=2 on 16-dim features: 16→32→16 (reduced from 16→64→16)
    # Lower expansion ratio reduces gradient amplification while maintaining capacity
    dict(
        keyword="dec0.block1.mlp",
        lr=0.00002,  # 98% lower than encoder block lr
        weight_decay=0.2,  # 4× higher weight decay
    ),

    # Group 6: Specifically target fc1/fc2 linear layers in dec0.block1.mlp
    # These are the layers where weight explosion was observed (L2 norm max=6.87)
    # fc1: [32,16], fc2: [16,32] - gradient amplification reduced due to lower mlp_ratio
    dict(
        keyword="dec0.block1.mlp.0.fc",
        lr=0.00001,  # 99% lower than encoder block lr (minimum viable)
        weight_decay=0.3,  # 6× higher weight decay for strongest regularization
    ),
]

# ============================================================================
# Dataset settings
# ============================================================================
dataset_type = "GenericGSDataset"

# OVS data roots (point to parent directory, split specifies the subdirectory)
data_root_ovs_1 = "/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs"
data_root_ovs_2 = "/new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS"

# Single scene for debugging
single_scene_train_root = "/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train/figurines"
single_scene_val_root = "/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/val/figurines"

data = dict(
    num_classes=100,
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
                    # CRITICAL: Filter valid points FIRST to match SVD lang_feat size
                    dict(type="FilterValidPoints", key="valid_feat_mask"),
                    # Initial preprocessing
                    dict(type="CenterShift", apply_z=True),
                    # Step 1: Filter outliers (removes long-tail boundary points, keeps dense 98% region)
                    dict(type="FilterCoordOutliers", percentile_low=0.5, percentile_high=99.5),
                    # Step 3: Re-center coordinates to the filtered dense region
                    dict(type="CenterShift", apply_z=True),
                    # Step 4: GridSampleAveraged for representative sampling (with feature averaging)
                    dict(
                        type="GridSampleAveraged",
                        grid_size=0.01,
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
                            "point_to_grid",
                            "segment",
                        ),
                        return_grid_coord=True,
                        average_keys=("color", "opacity", "quat", "scale"),
                        first_keys=("coord",),
                    ),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask", "name", "scene_path", "point_to_grid", "segment"),
                        feat_keys=("color", "opacity", "quat", "scale"),
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
                    # Step 1: Filter outliers (same as train pipeline)
                    dict(type="FilterCoordOutliers", percentile_low=0.5, percentile_high=99.5),
                    # Step 2: Filter valid points
                    dict(type="FilterValidPoints", key="valid_feat_mask"),
                    # Step 3: Re-center coordinates to the filtered dense region
                    dict(type="CenterShift", apply_z=True),
                    # Step 4: GridSampleAveraged for representative sampling (with feature averaging)
                    dict(
                        type="GridSampleAveraged",
                        grid_size=0.01,
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
                            "point_to_grid",
                            "segment",
                        ),
                        return_grid_coord=True,
                        average_keys=("color", "opacity", "quat", "scale"),
                        first_keys=("coord",),
                    ),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask", "name", "scene_path", "point_to_grid", "segment"),
                        feat_keys=("color", "opacity", "quat", "scale"),
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
            # CRITICAL: Filter valid points FIRST to match SVD lang_feat size
            dict(type="FilterValidPoints", key="valid_feat_mask"),
            dict(type="CenterShift", apply_z=True),
            # Step 2: GridSampleAveraged (before NormalizeCoord) - required for GridPooling
            dict(
                type="GridSampleAveraged",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "valid_feat_mask", "point_to_grid", "segment"),
                return_grid_coord=True,
                average_keys=("color", "opacity", "quat", "scale"),
                first_keys=("coord",),
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask", "name", "scene_path"),
                feat_keys=("color", "opacity", "quat", "scale"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",  # Use 'val' subdirectory
        data_root=data_root_ovs_1,  # Use lerf_ovs parent directory
        transform=[
            # CRITICAL: Filter valid points FIRST to match SVD lang_feat size
            dict(type="FilterValidPoints", key="valid_feat_mask"),
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            dict(type="Copy", keys_dict={"segment": "origin_segment", "coord": "origin_coord", "valid_feat_mask": "origin_feat_mask"}),
            # Step 2: GridSampleAveraged for initial point reduction (test_cfg.voxelize does further processing)
            dict(
                type="GridSampleAveraged",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "valid_feat_mask"),
                return_inverse=True,
                average_keys=("color", "opacity", "quat", "scale"),
                first_keys=("coord",),
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
                    keys=("coord", "grid_coord", "index", "lang_feat", "valid_feat_mask", "name", "scene_path", "point_to_grid", "segment"),
                    feat_keys=("color", "opacity", "quat", "scale"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)

# SINGLE SCENE CONFIGURATION (commented out, kept for debugging only)
# data = dict(
#     num_classes=100,
#     ignore_index=-1,
#     train=dict(
#         type=dataset_type,
#         split="",
#         data_root=single_scene_train_root,
#         sample_tail_classes=False,
#         load_compressed_lang_feat=True,
#         svd_rank=16,
#         transform=[
#             dict(type="CenterShift", apply_z=True),
#             dict(type="FilterCoordOutliers", percentile_low=0.5, percentile_high=99.5),
#             dict(type="FilterValidPoints", key="valid_feat_mask"),
#             dict(type="CenterShift", apply_z=True),
#             dict(type="GridSampleAveraged", grid_size=0.01, hash_type="fnv", mode="train", keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "valid_feat_mask", "point_to_grid", "segment"), return_grid_coord=True, average_keys=("color", "opacity", "quat", "scale"), first_keys=("coord",)),
#             dict(type="NormalizeColor"),
#             dict(type="ToTensor"),
#             dict(type="Collect", keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask", "name", "scene_path", "point_to_grid", "segment"), feat_keys=("color", "opacity", "quat", "scale")),
#         ],
#         test_mode=False,
#     ),
#     val=dict(
#         type=dataset_type,
#         split="",
#         data_root=single_scene_val_root,
#         sample_tail_classes=False,
#         load_compressed_lang_feat=True,
#         svd_rank=16,
#         transform=[
#             dict(type="CenterShift", apply_z=True),
#             dict(type="FilterValidPoints", key="valid_feat_mask"),
#             dict(type="GridSampleAveraged", grid_size=0.01, hash_type="fnv", mode="train", keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "valid_feat_mask", "point_to_grid", "segment"), return_grid_coord=True, average_keys=("color", "opacity", "quat", "scale"), first_keys=("coord",)),
#             dict(type="CenterShift", apply_z=False),
#             dict(type="NormalizeColor"),
#             dict(type="ToTensor"),
#             dict(type="Collect", keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask", "name", "scene_path"), feat_keys=("color", "opacity", "quat", "scale")),
#         ],
#         test_mode=False,
#     ),
#     test=dict(
#         type=dataset_type,
#         split="",
#         data_root=single_scene_val_root,
#         sample_tail_classes=False,
#         load_compressed_lang_feat=True,
#         svd_rank=16,
#         transform=[
#             dict(type="CenterShift", apply_z=True),
#             dict(type="NormalizeColor"),
#             dict(type="Copy", keys_dict={"segment": "origin_segment", "coord": "origin_coord", "valid_feat_mask": "origin_feat_mask"}),
#             dict(type="GridSampleAveraged", grid_size=0.01, hash_type="fnv", mode="train", keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "valid_feat_mask"), return_inverse=True, average_keys=("color", "opacity", "quat", "scale"), first_keys=("coord",)),
#         ],
#         test_mode=True,
#         test_cfg=dict(
#             voxelize=dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="test", keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "point_to_grid"), return_grid_coord=True),
#             crop=None,
#             post_transform=[
#                 dict(type="CenterShift", apply_z=False),
#                 dict(type="ToTensor"),
#                 dict(type="Collect", keys=("coord", "grid_coord", "index", "lang_feat", "valid_feat_mask", "name", "scene_path", "point_to_grid", "segment"), feat_keys=("color", "opacity", "quat", "scale")),
#             ],
#             aug_transform=[
#                 [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
#                 [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1)],
#                 [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
#                 [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1)],
#                 [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[0.95, 0.95])],
#                 [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[0.95, 0.95])],
#                 [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[0.95, 0.95])],
#                 [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[0.95, 0.95])],
#                 [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[1.05, 1.05])],
#                 [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[1.05, 1.05])],
#                 [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[1.05, 1.05])],
#                 [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[1.05, 1.05])],
#                 [dict(type="RandomFlip", p=1)],
#             ],
#         ),
#     ),
# )
# data = dict(
#     num_classes=100,
#     ignore_index=-1,
#     train=dict(
#         type="ConcatDataset",  # Use ConcatDataset to combine multiple data sources
#         datasets=[
#             dict(
#                 type=dataset_type,
#                 split="train",  # Use 'train' subdirectory (matches train/* pattern)
#                 data_root=data_root_ovs_1,  # First data source: lerf_ovs
#                 sample_tail_classes=False,
#                 load_compressed_lang_feat=True,  # Load SVD-compressed lang_feat (16-dim instead of 768-dim)
#                 svd_rank=16,  # SVD rank to load (must match density_invariant.svd_rank)
#                 transform=[
#                     # Initial preprocessing
#                     dict(type="CenterShift", apply_z=True),
#                     # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
#                     # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
#                     # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
#                     # dict(type="RandomScale", scale=[0.9, 1.1]),
#                     # dict(type="RandomFlip", p=0.5),
#                     # dict(type="RandomJitter", sigma=0.005, clip=0.02),
#                     # dict(type="NormalizeCoord"),  # Normalize after GridSample
#                     # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
#                     # Color augmentation
#                     # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
#                     # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
#                     # dict(type="ChromaticJitter", p=0.95, std=0.05),
#                     # Step 1: Filter outliers (removes long-tail boundary points, keeps dense 98% region)
#                     dict(type="FilterCoordOutliers", percentile_low=0.5, percentile_high=99.5),
#                     # Step 2: Filter valid points (those with valid language features)
#                     dict(type="FilterValidPoints", key="valid_feat_mask"),
#                     # Step 3: Re-center coordinates to the filtered dense region
#                     # This shifts coordinates so the dense region is centered around origin
#                     # GridSample then operates on these centered coordinates for better spatial distribution
#                     dict(type="CenterShift", apply_z=True),
#                     # Step 4: GridSample for representative sampling
#                     dict(
#                         type="GridSample",
#                         grid_size=0.01,  # Smaller grid_size to preserve more valid points
#                         hash_type="fnv",
#                         mode="train",
#                         keys=(
#                             "coord",
#                             "color",
#                             "opacity",
#                             "quat",
#                             "scale",
#                             "lang_feat",
#                             "valid_feat_mask",
#                             "point_to_grid",  # Sampled along with other data
#                             "segment",  # Required for AggregatedContrastiveLoss
#                         ),
#                         return_grid_coord=True,  # Required by LitePT's GridPooling
#                     ),
#                     # dict(type="SphereCrop", point_max=204800, mode="random"),
#                     dict(type="NormalizeColor"),
#                     dict(type="ToTensor"),
#                     # Collect features - grid_coord is required by LitePT's GridPooling
#                     # Gaussian params (opacity, quat, scale) are extracted from feat in trainer
#                     dict(
#                         type="Collect",
#                         keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask", "name", "scene_path", "point_to_grid", "segment"),
#                         feat_keys=("color", "opacity", "quat", "scale"),  # 11 channels for model input
#                     ),
#                 ],
#                 test_mode=False,
#             ),
#             dict(
#                 type=dataset_type,
#                 split="train",  # Use 'train' subdirectory (matches train/* pattern)
#                 data_root=data_root_ovs_2,  # Second data source: 3DOVS
#                 sample_tail_classes=False,
#                 load_compressed_lang_feat=True,  # Load SVD-compressed lang_feat (16-dim instead of 768-dim)
#                 svd_rank=16,  # SVD rank to load (must match density_invariant.svd_rank)
#                 transform=[
#                     # Initial preprocessing
#                     dict(type="CenterShift", apply_z=True),
#                     # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
#                     # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
#                     # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
#                     # dict(type="RandomScale", scale=[0.9, 1.1]),
#                     # dict(type="RandomFlip", p=0.5),
#                     # dict(type="RandomJitter", sigma=0.005, clip=0.02),
#                     # dict(type="NormalizeCoord"),  # Normalize after GridSample
#                     # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
#                     # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
#                     # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
#                     # dict(type="ChromaticJitter", p=0.95, std=0.05),
#                     # Step 1: Filter outliers (same as train pipeline)
#                     dict(type="FilterCoordOutliers", percentile_low=0.5, percentile_high=99.5),
#                     # Step 2: Filter valid points
#                     dict(type="FilterValidPoints", key="valid_feat_mask"),
#                     # Step 3: Re-center coordinates to the filtered dense region
#                     dict(type="CenterShift", apply_z=True),
#                     # Step 4: GridSample for representative sampling
#                     dict(
#                         type="GridSample",
#                         grid_size=0.01,  # Smaller grid_size to preserve more valid points
#                         hash_type="fnv",
#                         mode="train",
#                         keys=(
#                             "coord",
#                             "color",
#                             "opacity",
#                             "quat",
#                             "scale",
#                             "lang_feat",
#                             "valid_feat_mask",
#                             "point_to_grid",  # Sampled along with other data
#                             "segment",  # Required for AggregatedContrastiveLoss
#                         ),
#                         return_grid_coord=True,  # Required by LitePT's GridPooling
#                     ),
#                     # Optional: SphereCrop for additional point reduction if needed
#                     # dict(type="SphereCrop", point_max=204800, mode="random"),
#                     dict(type="NormalizeColor"),
#                     dict(type="ToTensor"),
#                     # Collect features - grid_coord is required by LitePT's GridPooling
#                     # Gaussian params (opacity, quat, scale) are extracted from feat in trainer
#                     dict(
#                         type="Collect",
#                         keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask", "name", "scene_path", "point_to_grid", "segment"),
#                         feat_keys=("color", "opacity", "quat", "scale"),  # 11 channels
#                     ),
#                 ],
#                 test_mode=False,
#             ),
#         ],
#         loop=1,  # ConcatDataset loop parameter
#     ),
#     val=dict(
#         type=dataset_type,
#         split="val",  # Use 'val' subdirectory
#         data_root=data_root_ovs_1,  # Use lerf_ovs parent directory
#         transform=[
#             dict(type="CenterShift", apply_z=True),
#             # Step 1: Filter to only valid points
#             dict(type="FilterValidPoints", key="valid_feat_mask"),
#             # Step 2: Grid sample (before NormalizeCoord) - required for GridPooling
#             dict(
#                 type="GridSample",
#                 grid_size=0.01,
#                 hash_type="fnv",
#                 mode="train",
#                 keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "valid_feat_mask", "point_to_grid", "segment"),
#                 return_grid_coord=True,
#             ),
#             dict(type="CenterShift", apply_z=False),
#             dict(type="NormalizeColor"),
#             dict(type="ToTensor"),
#             # grid_coord is required by LitePT's GridPooling
#             # Gaussian params (opacity, quat, scale) are extracted from feat in trainer
#             dict(
#                 type="Collect",
#                 keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask", "name", "scene_path"),
#                 feat_keys=("color", "opacity", "quat", "scale"),  # 11 channels
#             ),
#         ],
#         test_mode=False,
#     ),
#     test=dict(
#         type=dataset_type,
#         split="val",  # Use 'val' subdirectory
#         data_root=data_root_ovs_1,  # Use lerf_ovs parent directory
#         transform=[
#             dict(type="CenterShift", apply_z=True),
#             dict(type="NormalizeColor"),
#             # Step 1: Filter to only valid points
#             dict(type="FilterValidPoints", key="valid_feat_mask"),
#             # Step 2: Grid sample for initial point reduction (test_cfg.voxelize does further processing)
#             dict(
#                 type="GridSample",
#                 grid_size=0.01,
#                 hash_type="fnv",
#                 mode="train",
#                 keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "valid_feat_mask", "point_to_grid", "segment"),
#                 return_inverse=True,
#             ),
#         ],
#         test_mode=True,
#         test_cfg=dict(
#             voxelize=dict(
#                 type="GridSample",
#                 grid_size=0.02,
#                 hash_type="fnv",
#                 mode="test",
#                 keys=("coord", "color", "opacity", "quat", "scale", "lang_feat", "point_to_grid"),
#                 return_grid_coord=True,
#             ),
#             crop=None,
#             post_transform=[
#                 dict(type="CenterShift", apply_z=False),
#                 dict(type="ToTensor"),
#                 dict(
#                     type="Collect",
#                     keys=("coord", "grid_coord", "index", "lang_feat", "valid_feat_mask", "name", "scene_path", "point_to_grid", "segment"),
#                     feat_keys=("color", "opacity", "quat", "scale"),  # 11 channels
#                 ),
#             ],
#             aug_transform=[
#                 [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
#                 [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1)],
#                 [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
#                 [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1)],
#                 [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[0.95, 0.95])],
#                 [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[0.95, 0.95])],
#                 [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[0.95, 0.95])],
#                 [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[0.95, 0.95])],
#                 [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[1.05, 1.05])],
#                 [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[1.05, 1.05])],
#                 [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[1.05, 1.05])],
#                 [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1), dict(type="RandomScale", scale=[1.05, 1.05])],
#                 [dict(type="RandomFlip", p=1)],
#             ],
#         ),
#     ),
# )

# ============================================================================
# Hooks
# ============================================================================
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PerSceneLossVisualizer",
         enabled=True,
         save_every_n_epochs=1,  # Save loss curves every epoch (set to None to only save at end)
         save_at_end=True),  # Also save final curves at end of training
]
