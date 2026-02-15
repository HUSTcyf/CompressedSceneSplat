"""
Config file for training LitePT on OVS (Open-Vocabulary Scenes) 3DGS data.

Combines datasets from:
- /new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs
- /new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS

Usage:
    python tools/train_lite.py --config-file configs/custom/semseg-litept-ovs.py
"""

_base_ = [
    "../_base_/default_runtime.py",
]

# ============================================================================
# Misc custom settings
# ============================================================================
gpu_nums = 1
batch_size = 1
batch_size_val = 1
batch_size_test = 1
num_worker = 4
mix_prob = 0.8
empty_cache = False
enable_amp = True

# ============================================================================
# Model settings - LitePT with language features
# ============================================================================
# Note: OVS datasets have no segment labels, this config is for:
# 1. Representation learning / pretraining
# 2. Open-vocabulary feature extraction
# 3. Fine-tuning on downstream tasks later

# Language feature dimension for open-vocabulary classification
# Common dimensions: 512 (OpenCLIP ViT-L/14), 768 (OpenCLIP ViT-H/14)
lang_feat_dim = 72  # LitePT decoder output channels (must match last dec_channels value)

model = dict(
    type="DefaultSegmentorV2",
    num_classes=20,  # Placeholder for downstream tasks
    backbone_out_channels=lang_feat_dim,  # Must match the last value in dec_channels
    backbone=dict(
        type="LitePT",
        in_channels=14,  # 3DGS features: color(3) + opacity(1) + quat(4) + scale(3) + coord(3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        # Decoder settings
        dec_depths=(2, 2, 2, 2),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
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
    # L2 and Cosine losses for language feature reconstruction
    criteria=[
        dict(type="L2Loss", loss_weight=1.0),
        dict(type="CosineSimilarity", loss_weight=1.0),
    ],
)

# ============================================================================
# Scheduler settings
# ============================================================================
eval_epoch = 100  # total eval & checkpoint epoch
epoch = 8 * eval_epoch  # total data loops
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# ============================================================================
# Dataset settings
# ============================================================================
dataset_type = "GenericGSDataset"

# Combine both OVS directories
data_root_ovs_1 = "/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs"
data_root_ovs_2 = "/new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS"

data = dict(
    num_classes=20,  # Placeholder
    ignore_index=-1,
    names=["class_{}".format(i) for i in range(20)],  # Placeholder
    train=dict(
        type=dataset_type,
        split="*",  # Match all scene directories under data_root
        data_root=data_root_ovs_1,  # Primary data root
        # Additional scenes can be added via data_list or symlink approach
        sample_tail_classes=False,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout",
                dropout_ratio=0.2,
                dropout_application_ratio=0.2
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # Normalize coordinates before ElasticDistortion (OVS data uses world coordinates)
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
                keys=("coord", "grid_coord"),
                feat_keys=("color", "opacity", "quat", "scale", "coord"),  # Include coord as feature
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="*",  # Match all scene directories under data_root
        data_root=data_root_ovs_1,
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
                keys=("coord", "grid_coord"),
                feat_keys=("color", "opacity", "quat", "scale", "coord"),  # Include coord as feature
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="*",  # Match all scene directories under data_root
        data_root=data_root_ovs_1,
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
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "opacity", "quat", "scale", "coord", "lang_feat"),
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
