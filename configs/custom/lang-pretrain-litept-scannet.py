"""
Config file for Vision-Language Pretraining with LitePT on ScanNet 3DGS data.

This config implements representation learning with full 768-dim language features,
using the lightweight LitePT backbone for efficient training on ScanNet dataset.

Key features:
- Uses LitePT backbone (3.6× fewer parameters, 2× faster than PT-v3m1)
- Full 768-dim language features (no SVD compression)
- Compatible with ScanNet200GSDataset
- Same training pipeline as original PT-v3m1 Scannet config

Usage:
    python tools/train.py --config-file configs/custom/lang-pretrain-litept-scannet.py --num-gpus 4
"""

_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# ============================================================================
# Misc custom settings
# ============================================================================
N_GPU = 4
debug = 0
gpu_nums = 1 if debug else N_GPU
batch_size = 3 * gpu_nums
batch_size_val = 3 * gpu_nums
batch_size_test = 1 * gpu_nums
num_worker = 24 * gpu_nums if not debug else 0
mix_prob = 0.8
empty_cache = False
enable_amp = False

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
            loss_weight=0.5,  # Overall loss weight
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
        # dict(
        #     type="Rendered2DLoss",
        #     loss_weight=1.0,  # Maximum weight
        #     gaussian_train_root="/new_data/cyf/projects/SceneSplat/gaussian_train",
        #     datasets_root="/new_data/cyf/projects/SceneSplat/datasets",
        #     warmup_progress=0.0,  # Start immediately (no warmup)
        #     target_progress=0.5,  # Reach max weight at 50% training progress
        #     max_num_views=10,  # Use up to 10 views per scene to save memory
        # ),

        # AggregatedContrastiveLoss: ENABLED for contrastive learning
        dict(
            type="AggregatedContrastiveLoss",
            temperature=0.2,
            reduction="mean",
            loss_weight=0.1,
            schedule="last_75",
        ),
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
# Scheduler settings (same as original Scannet config)
# ============================================================================
epoch = 80
eval_epoch = 10  # evaluate and save every 10 epochs
max_grad_threshold = 4.0
decoder_grad_warn_threshold = 3.0  # only warn when decoder grad > 3.0
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
# scheduler = dict(
#     type="OneCycleLR",
#     max_lr=[0.006, 0.0006],
#     pct_start=0.05,
#     anneal_strategy="cos",
#     div_factor=10.0,
#     final_div_factor=1000.0,
# )
# param_dicts = [dict(keyword="block", lr=0.0006)]
scheduler = dict(
    type="OneCycleLR",
    # max_lr 对应所有参数组: [默认组, enc.block, dec.block, dim_scale, dec0, dec0.mlp, dec0.fc]
    max_lr=[0.006, 0.006, 0.0006, 0.006, 0.0003, 0.00012, 0.00006],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [
    # Group 1: Encoder transformer blocks
    # Keep original learning rate for encoder (features extraction is stable)
    dict(keyword="enc.block", lr=0.006, weight_decay=0.05),

    # Group 2: Decoder transformer blocks (higher stages: dec3, dec2, dec1)
    # Reduced learning rate for decoder blocks to prevent collapse
    dict(keyword="dec.block", lr=0.0006, weight_decay=0.05),

    # Group 3: Dimension-wise scaling (for SVD feature magnitude compensation)
    dict(keyword="dim_scale", lr=0.006, weight_decay=0.05),

    # Group 4: Final decoder stage (dec4/s=0 in code, outputs 16-dim features)
    # This is the bottleneck stage with extreme gradient amplification
    # Channels: 126→64→32→16, causing 8× gradient amplification
    dict(
        keyword="dec0",  # Matches all layers in final decoder stage (s=0)
        lr=0.0003,  # 95% lower than encoder block lr
        weight_decay=0.15,  # 3× higher weight decay for regularization
    ),

    # Group 5: dec0.block1 MLP (specific problematic layer)
    # The MLP with mlp_ratio=2 on 16-dim features: 16→32→16 (reduced from 16→64→16)
    # Lower expansion ratio reduces gradient amplification while maintaining capacity
    dict(
        keyword="dec0.block1.mlp",
        lr=0.00012,  # 98% lower than encoder block lr
        weight_decay=0.2,  # 4× higher weight decay
    ),

    # Group 6: Specifically target fc1/fc2 linear layers in dec0.block1.mlp
    # These are the layers where weight explosion was observed (L2 norm max=6.87)
    # fc1: [32,16], fc2: [16,32] - gradient amplification reduced due to lower mlp_ratio
    dict(
        keyword="dec0.block1.mlp.0.fc",
        lr=0.00006,  # 99% lower than encoder block lr (minimum viable)
        weight_decay=0.3,  # 6× higher weight decay for strongest regularization
    ),
]
# Save path for LitePT-16 training (compressed features)
save_path = "exp/lite-16-scannet-gridsvd"

# ============================================================================
# Dataset settings (same as original Scannet config)
# ============================================================================
dataset_type = "ScanNet200GSDataset"
data_root = "/new_data/cyf/Datasets/SceneSplat7k/scannet"
repo_root = "/new_data/cyf/projects/SceneSplat"

class_names_path = f"{repo_root}/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_labels.txt"
text_embeddings_path = f"{repo_root}/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_text_embeddings_siglip2.pt"

# ============================================================================
# Hooks (same as original Scannet config)
# ============================================================================
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="LangPretrainZeroShotSemSegEval",
        class_names=class_names_path,
        text_embeddings=text_embeddings_path,
        excluded_classes=["wall", "floor", "ceiling"],
        ignore_index=-1,
        vote_k=25,
        enable_voting=True,
        confidence_threshold=0.1,
        svd_rank=16,  # SVD rank for text embeddings (must match model output)
    ),
    dict(type="CheckpointSaver", save_freq=10),  # save checkpoint every 10 epochs
    dict(type="PreciseEvaluator", test_last=False),
]

# ============================================================================
# Tester (same as original Scannet config)
# ============================================================================
test = dict(
    type="ZeroShotSemSegTester",
    class_names=class_names_path,
    text_embeddings=text_embeddings_path,
    excluded_classes=["wall", "floor", "ceiling"],
    enable_voting=True,
    vote_k=25,
    confidence_threshold=0.1,
)

# ============================================================================
# Data pipeline (same as original Scannet config)
# ============================================================================
data = dict(
    num_classes=200,
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split=("train_grid1.0cm_chunk6x6_stride3x3", "test_grid1.0cm_chunk6x6_stride3x3"),
        data_root=data_root,
        sample_tail_classes=False,
        load_compressed_lang_feat=True,  # Load SVD-compressed lang_feat (16-dim instead of 768-dim)
        svd_rank=16,  # SVD rank to load (must match density_invariant.svd_rank)
        transform=[
            # CRITICAL: Filter to valid points BEFORE GridSample to match SVD lang_feat size
            # SVD files contain features only for valid points (where valid_feat_mask==True)
            # Without this, coord has all points but lang_feat has only valid points → IndexError
            dict(type="FilterValidPoints", key="valid_feat_mask"),
            dict(type="CenterShift", apply_z=True),
            # dict(
            #     type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            # ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.01),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=(
                    "coord",
                    "color",
                    "opacity",
                    "quat",
                    "scale",
                    "normal",
                    "segment",
                    "lang_feat",
                    "valid_feat_mask",
                    "point_to_grid", 
                ),
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=192000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "lang_feat", "valid_feat_mask", "name", "scene_path", "point_to_grid"),
                feat_keys=("color", "opacity", "quat", "scale"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        load_compressed_lang_feat=False,  # No SVD loading for validation - direct inference
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=(
                    "coord",
                    "color",
                    "opacity",
                    "quat",
                    "scale",
                    "normal",
                    "segment",
                    "lang_feat",
                    "instance",
                ),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "lang_feat",
                    "instance",
                    "name",
                    "scene_path",
                ),
                feat_keys=("color", "opacity", "quat", "scale"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        load_compressed_lang_feat=True,  # Load SVD-compressed lang_feat (16-dim instead of 768-dim)
        svd_rank=16,  # SVD rank to load (must match density_invariant.svd_rank)
        transform=[
            # CRITICAL: Filter to valid points BEFORE GridSample to match SVD lang_feat size
            dict(type="FilterValidPoints", key="valid_feat_mask"),
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            dict(
                type="Copy",
                keys_dict={
                    "segment": "origin_segment",
                    "coord": "origin_coord",
                    "valid_feat_mask": "origin_feat_mask",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=(
                    "coord",
                    "color",
                    "opacity",
                    "quat",
                    "scale",
                    "normal",
                    "lang_feat",
                    "valid_feat_mask",
                    "point_to_grid", 
                ),
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="test",
                keys=(
                    "coord",
                    "color",
                    "opacity",
                    "quat",
                    "scale",
                    "normal",
                    "lang_feat",
                    "valid_feat_mask",
                    "point_to_grid", 
                ),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=(
                        "coord",
                        "grid_coord",
                        "index",
                        "lang_feat",
                        "valid_feat_mask",
                        "name",
                        "scene_path",
                        "point_to_grid",
                    ),
                    feat_keys=("color", "opacity", "quat", "scale"),
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
                ]
            ],
        ),
    ),
)
