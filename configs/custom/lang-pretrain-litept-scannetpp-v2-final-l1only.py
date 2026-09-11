"""
Vision-Language Pretraining with LitePT on ScanNet++ v2 3DGS data (SVD-r16).

2026-08-04 REBUTTAL T1-2: 只变 backbone（PT-v3m1 -> LitePT），其余对齐
final-l1only 基线配方：
  - noaug transform（几何增强移除，只留 Chromatic*——基线实测波动坍缩机制）
  - SVDWeightedL1Loss weight_strategy="inverse_variance" + CosineSimilarity，
    无 AggregatedContrastiveLoss（基线实测破坏）
  - epoch=10、loop=1、v3 符号注册表
  - weight = litept-16dim-init（scannet 训练，结构与本配置一致——启动前验证
    state_dict 兼容性；若 CheckpointLoader 报大量 shape mismatch 则需随机初始化）

保留 LitePT 自身训练配方（用户确认 2026-08-04）：param_dicts 多学习率组
（dec0 低 lr 防梯度爆炸）+ smoke 的 scheduler max_lr 列表。

Usage:
    python tools/train_lite.py --config-file configs/custom/lang-pretrain-litept-scannetpp-v2-final-l1only.py --num-gpus 2 --density-invariant
"""

_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# ============================================================================
# Misc custom settings (same as smoke)
# ============================================================================
N_GPU = 2
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
svd_rank = 16  # SVD compression rank (8, 16, 32 are common choices)
lang_feat_dim = svd_rank  # Decoder output matches SVD rank
FD = lang_feat_dim

repo_root = "/home/isom/cyf/CompressedSceneSplat"
text_embeddings_path = f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt"

model = dict(
    type="LangPretrainer",  # Language Pretrainer for VL learning
    verbose_losses=True,  # Enable verbose loss printing (L2 and Cos per iteration)
    backbone=dict(
        type="LitePT",
        in_channels=11,  # 3DGS features: color(3) + opacity(1) + quat(4) + scale(3) [coord removed]
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # Encoder (scaled to support 16 decoder output)
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
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.1,  # REDUCED from 0.3 - 16-dim task needs less dropout
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False,
        # Normalization layer for decoder upsampling layers
        pdnorm_ln=True,
        pdnorm_bn=False,
    ),
    # 2026-08-04 对齐 baseline final-l1only 配方（REBUTTAL T1-2 只变 backbone）：
    # - inverse_variance（波动维高权重）
    # - 无 AggregatedContrastiveLoss（基线实测破坏）
    criteria=[
        dict(
            type="SVDWeightedL1Loss",
            loss_weight=0.5,
            reduction="mean",
            base_weight=1.0,
            min_weight=0.1,
            weight_strategy="inverse_variance",
            variance_momentum=0.99,
        ),
        dict(
            type="CosineSimilarity",
            loss_weight=1.0,
            reduction="mean",
        ),
    ],
)

# ============================================================================
# Density-Invariant Training Configuration (same as smoke)
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
# Scheduler settings (LitePT 自身配方：param_dicts 多学习率组，保留)
# ============================================================================
epoch = 10  # 2026-08-04：对齐 baseline（final-l1only）
eval_epoch = 1
max_grad_threshold = 4.0
decoder_grad_warn_threshold = 3.0
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    # max_lr 对应所有参数组: [默认组, enc.block, dec.block, dec0.mlp, dec0.fc]
    max_lr=[0.006, 0.006, 0.0006, 0.0003, 0.0003],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [
    # Group 1: Encoder transformer blocks
    dict(keyword="enc.block", lr=0.006, weight_decay=0.05),

    # Group 2: Decoder transformer blocks (all stages: dec3, dec2, dec1, dec0)
    dict(keyword="dec.block", lr=0.0006, weight_decay=0.05),

    # Group 3: dec0.block1 MLP (specific problematic layer)
    dict(
        keyword="dec0.block1.mlp",
        lr=0.0003,  # Lower than dec.block for stability
        weight_decay=0.2,  # 4x higher weight decay
    ),

    # Group 4: Specifically target fc1/fc2 linear layers in dec0.block1.mlp
    dict(
        keyword="dec0.block1.mlp.0.fc",
        lr=0.0003,
        weight_decay=0.3,  # 6x higher weight decay for strongest regularization
    ),
]
# Save path for LitePT-16 training (compressed features)
save_path = "exp/smoke-lite-16-scannetpp-v2-final-l1only"

# 2026-08-04 REBUTTAL T1-2：LitePT 16dim-init（scannet 训练，结构一致）。
# 启动前必须验证 state_dict 兼容性（tools/compression 或 python 逐 key 对比）。
weight = "/home/isom/cyf/CompressedSceneSplat/checkpoints/litept/lang-pretrain-litept-16dim-init-scannet.pth"

# ============================================================================
# Dataset settings (same as original Scannet config)
# ============================================================================
dataset_type = "ScanNetPPGSDataset"
data_root = "/home/isom/cyf/SceneSplat/scannetpp_v2"
evaluate = False  # smoke test: skip epoch-end evaluation

# scannetpp_v2 GT 为 top100 benchmark 编码（preprocess_scannetpp_gs.py 用 top100.txt 映射）
class_names_path = f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt"

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
        use_procrustes=True,  # Enable on-the-fly Procrustes alignment
    ),
    dict(type="CheckpointSaver", save_freq=1),  # save checkpoint every epoch
    dict(type="PreciseEvaluator", test_last=True),
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
    svd_rank=16,  # SVD rank for text embeddings (must match model output)
    use_procrustes=True,  # Enable on-the-fly Procrustes alignment
)

# ============================================================================
# Data pipeline (same as original Scannet config)
# ============================================================================
data = dict(
    num_classes=100,  # scannetpp top100 benchmark
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split=("train_grid1.0cm_chunk6x6_stride3x3", "test_grid1.0cm_chunk6x6_stride3x3"),
        data_root=data_root,
        sample_tail_classes=False,
        load_compressed_lang_feat=True,  # Load SVD-compressed lang_feat (16-dim instead of 768-dim)
        svd_rank=16,  # SVD rank to load (must match density_invariant.svd_rank)
        # 2026-08-04 对齐 baseline：显式 loop=1（必须在此处，defaults.py 读 data.train.loop）
        loop=1,
        # 2026-08-04 对齐 baseline：v3 全局符号注册表（加载时翻转列符号）
        global_sign_path="/home/isom/cyf/SceneSplat/scannetpp_v2/lang_feat_grid_svd_r16_global_signs_v3.npz",
        transform=[
            # CRITICAL: Filter to valid points BEFORE GridSample to match SVD lang_feat size
            dict(type="FilterValidPoints", key="valid_feat_mask"),
            dict(type="CenterShift", apply_z=True),
            # 2026-08-04 对齐 baseline：几何增强全部移除（RandomRotate×3/RandomScale/
            # RandomFlip/RandomJitter/ElasticDistortion）——增强扰动 GridSample cell 哈希
            # → 波动维目标随 iter 变化 → 模型学平均 → 波动坍缩。保留 Chromatic*（只改
            # color 输入，不改坐标 → 不扰动目标）。
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
        load_compressed_lang_feat=False,  # val 仅测试用，无 SVD 文件也不需要 lang_feat
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
