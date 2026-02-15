"""
Config file for Vision-Language Pretraining with LitePT on OVS (Open-Vocabulary Scenes) 3DGS data.

This config implements representation learning with language features from CLIP/SigLIP models,
enabling open-vocabulary 3D scene understanding.

Combines datasets from:
- /new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs
- /new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS

Usage:
    python tools/train_lite.py --config-file configs/custom/lang-pretrain-litept-ovs.py
"""

_base_ = [
    "../_base_/default_runtime.py",
]

# ============================================================================
# Misc custom settings
# ============================================================================
debug = 0
gpu_nums = 1 # if debug else 4

# Save path for LitePT-768 training (all outputs saved here)
save_path = "exp/lite-768"
batch_size = 3 * gpu_nums
batch_size_val = 1 * gpu_nums
batch_size_test = 1 * gpu_nums
num_worker = 0  # Set to 0 for single GPU to avoid multiprocessing issues
mix_prob = 0.0  # no mixup for language pretraining
empty_cache = False
enable_amp = True
evaluate = False
find_unused_parameters = True

# ============================================================================
# Model settings - LitePT with Vision-Language Pretraining
# ============================================================================
# Trainer type for language pretraining
train = dict(type="DefaultTrainer")

# Language feature dimension from vision encoder (SigLIP/ViT)
# Common dimensions: 512 (OpenCLIP ViT-L/14), 768 (OpenCLIP ViT-H/14), 1152 (SigLIP SO400M)
lang_feat_dim = 768  # Must match LitePT decoder output channels
FD = lang_feat_dim

model = dict(
    type="LangPretrainer",  # Language Pretrainer for VL learning
    backbone=dict(
        type="LitePT",
        in_channels=11,  # 3DGS features: color(3) + opacity(1) + quat(4) + scale(3) [coord removed]
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # Encoder (scaled to support 768 decoder output)
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28) ,
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        # Decoder (output 768 dimensions, channels_per_head=24 for PointROPE)
        dec_depths=(2, 2, 2, 2),
        dec_channels=(FD, FD//2, FD//4, FD//8),
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
    # Language pretraining losses
    criteria=[
        dict(type="L2Loss", loss_weight=1.0),
        dict(type="CosineSimilarity", loss_weight=1.0),
        # dict(type="ValidNonValidContrastiveLoss", loss_weight=0.5, temperature=0.1, margin=0.5),
    ],
)

# ============================================================================
# Scheduler settings
# ============================================================================
eval_epoch = 20  # total eval & checkpoint epoch
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
                transform=[
                    # Initial preprocessing
                    dict(type="CenterShift", apply_z=True),
                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(type="NormalizeCoord"),  # Normalize world coordinates
                    dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    # Color augmentation
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # Grid sampling with all features including language features
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=(
                            "coord",
                            "color",
                            "opacity",
                            "quat",
                            "scale",
                            "lang_feat",  # Language features from CLIP/SigLIP
                            "valid_feat_mask",  # Mask for valid language features
                        ),
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    # Collect features (coord is NOT used as input feature)
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask"),
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
                transform=[
                    # Initial preprocessing
                    dict(type="CenterShift", apply_z=True),
                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(type="NormalizeCoord"),
                    dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
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
                        ),
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask"),
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
            dict(
                type="GridSample",
                grid_size=0.02,
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
                ),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "lang_feat", "valid_feat_mask"),
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
                    "lang_feat",
                    "valid_feat_mask",
                ),
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
                keys=("coord", "color", "opacity", "quat", "scale", "lang_feat"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "lang_feat", "valid_feat_mask"),
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
