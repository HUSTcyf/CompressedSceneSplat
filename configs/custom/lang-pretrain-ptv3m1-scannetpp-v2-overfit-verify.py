"""
Experiment B config: PT-v3m1 (original SceneSplat backbone) on ScanNet++ v2 with
SVD-r16 compressed grid language features.

Purpose: Entangle backbone vs compression in the zero-shot setting.
- LitePT  config: configs/custom/lang-pretrain-litept-scannetpp-v2-smoke.py
- PT-v3m1 config: this file

Both share the SAME compressed data targets (lang_feat_grid_svd_r16.npz),
same training recipe, same ZeroShotSemSegTester + Procrustes evaluation.
The only difference is the backbone, so any zero-shot mIoU difference is
attributable to the architecture change (reviewer point Q1, branch B).

Backbone notes:
- PT-v3m1 decoder output dim == dec_channels[0] (e.g. (768,512,256)->768-dim).
- Here dec_channels=(256,128,16) so that dec0 = final output = 16 (== SVD rank).
- num_stages == len(enc_depths);  len(dec_channels) == num_stages - 1.

Usage:
    python tools/train_lite.py --config-file configs/custom/lang-pretrain-ptv3m1-scannetpp-v2-smoke.py --num-gpus 2
"""

_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# ============================================================================
# Misc custom settings (same as litept smoke)
# ============================================================================
N_GPU = 2
debug = 0
gpu_nums = 1 if debug else N_GPU
batch_size = 2 * gpu_nums
batch_size_val = 1 * gpu_nums
batch_size_test = 1 * gpu_nums
num_worker = 12 * gpu_nums if not debug else 0
mix_prob = 0.8
empty_cache = False
enable_amp = False

# ============================================================================
# Model settings - PT-v3m1 with SVD-compressed Vision-Language Pretraining
# ============================================================================
train = dict(type="DensityInvariantTrainer")

svd_rank = 16  # SVD compression rank (same as litept baseline)
lang_feat_dim = svd_rank  # Decoder output matches SVD rank
FD = lang_feat_dim

# 方案 X 锚文本（必须与 tester 用同一文件/同一 SVD 投影，保证训练/评测同空间）
repo_root = "/home/isom/cyf/CompressedSceneSplat"
text_embeddings_path = f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt"

model = dict(
    type="LangPretrainer",
    verbose_losses=True,
    # 方案 X：训练目标逐 chunk Procrustes 对齐到 text16（与评测 Q 拟合同构）
    backbone=dict(
        type="PT-v3m1",
        in_channels=11,  # 3DGS features: color(3) + opacity(1) + quat(4) + scale(3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 6),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2),
        # dec_channels[0] == final output dim -> 16 (matches SVD rank)
        # Progression: dec2->dec1->dec0,  dec0 is the LAST stage and outputs dec_channels[0]
        dec_channels=(FD, 128, 256),  # (16, 128, 256), output = 16
        dec_num_head=(16, 16, 16),
        dec_patch_size=(1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.1,  # reduced from 0.3 (16-dim task, mirror of litept smoke)
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,  # match original PT-v3m1 concat config
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        # Match original PT-v3m1 (concat config): plain BN/LN, no PDNorm condition required
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNetPP",),
    ),
    # Same criteria as litept smoke (SVD-weighted L1 + Cosine + Agg. Contrastive)
    criteria=[
        dict(
            type="SVDWeightedL1Loss",
            loss_weight=0.5,
            reduction="mean",
            base_weight=1.0,
            min_weight=0.1,
            weight_strategy="variance",
            variance_momentum=0.99,
        ),
        dict(
            type="CosineSimilarity",
            loss_weight=1.0,
            reduction="mean",
        ),
        dict(
            type="AggregatedContrastiveLoss",
            temperature=0.2,
            reduction="mean",
            # 2026-08-04 消融：禁用对比损失（验证 contrast 与 L1 幅度对抗假设——
            # contrast push 类间分离（大幅）vs L1 目标幅度（std 0.064），
            # 对抗平衡在输出 2.2 倍 → raw L1 高 MLP 3.6 倍。若禁用后幅度比→1、
            # L1→0.019 则确认）
            loss_weight=0.3,
            schedule="skip",
        ),
    ],
)

# ============================================================================
# Density-Invariant Training Configuration (same as litept smoke)
# ============================================================================
density_invariant = dict(
    svd_rank=16,
    svd_center=False,
    min_sample_ratio=0.3,
    max_sample_ratio=0.7,
    consistency_weight=0.5,
    consistency_type="mse",
    scenarios=["dense", "single"],
    scenario_weights=dict(
        dense=1.0,
        single=1.0,
    ),
    use_compressed_features=True,
    batched_forward=True,
)

# ============================================================================
# Scheduler settings (same as litept smoke)
# ============================================================================
epoch = 1200
eval_epoch = 1200
max_grad_threshold = 4.0
decoder_grad_warn_threshold = 3.0
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.006,  # 单组（param_dicts=None）
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = None  # block 满速（验证：完整训练 raw L1 终值）

# Save path for PT-v3m1-16 training (compressed features)
save_path = "exp/overfit_verify1200"

# ============================================================================
# Dataset settings (identical to litept smoke)
# ============================================================================
dataset_type = "ScanNetPPGSDataset"
data_root = "/home/isom/cyf/SceneSplat/scannetpp_v2"
evaluate = False
# scannetpp_v2 GT 为 top100 benchmark 编码（preprocess_scannetpp_gs.py 用 top100.txt 映射）
# （repo_root / text_embeddings_path 已在文件顶部定义，训练端对齐与评测端必须同一份）
class_names_path = f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt"

# ============================================================================
# Hooks (same as litept smoke)
# ============================================================================
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=1200),
]

# ============================================================================
# Tester (same as litept smoke)
# ============================================================================
test = dict(
    type="ZeroShotSemSegTester",
    class_names=class_names_path,
    text_embeddings=text_embeddings_path,
    excluded_classes=["wall", "floor", "ceiling"],
    enable_voting=True,
    vote_k=25,
    confidence_threshold=0.1,
    svd_rank=16,
    use_procrustes=True,
    svd_center=False,
)

# ============================================================================
# Data pipeline (identical to litept smoke)
# ============================================================================
data = dict(
    num_classes=100,  # scannetpp top100 benchmark
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split="",
        data_root="/home/isom/cyf/SceneSplat/scannetpp_v2/train_grid1.0cm_chunk6x6_stride3x3/00777c41d4_0",
        sample_tail_classes=False,
        load_compressed_lang_feat=True,
        svd_rank=16,
        svd_center=False,
        transform=[
            dict(type="FilterValidPoints", key="valid_feat_mask"),
            dict(type="CenterShift", apply_z=True),
            # 2026-08-03 确定性过拟合：全部 random 增强已删除（RandomRotate/
            # RandomScale/RandomFlip/RandomJitter/ElasticDistortion/Chromatic*）——
            # 增强改变 GridSample 的 cell 哈希，同一物理点的目标特征随 iter 变化
            # （目标扰动），模型只能学平均 → L1 平台。此实验判别"增强扰动"是否瓶颈。
            # SphereCrop 保留：随机裁剪 = 子集采样，不改变可见点的目标，且控制显存。
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
        load_compressed_lang_feat=False,
        svd_center=False,
        transform=[
            dict(type="FilterValidPoints", key="valid_feat_mask"),
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
        load_compressed_lang_feat=False,
        svd_center=False,  # val 仅测试用，无 SVD 文件也不需要 lang_feat
        transform=[
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