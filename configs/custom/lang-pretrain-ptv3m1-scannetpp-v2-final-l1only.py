"""
Formal run (standard augmentation): PT-v3m1 on ScanNet++ v2 with SVD-r16
compressed grid language features. Derived from the noaug-globalsign-v3 config
(2026-08-04):
  1. align_text16=False（方案 X 已放弃——Q 列符号跨 chunk 随机 → 模式坍缩，见
     docs/svd_compression_diagnosis.md §8.16）
  2. 数据增强恢复为标准设置（与 smoke 一致：RandomRotate×3/RandomScale/
     RandomFlip/RandomJitter/ElasticDistortion + Chromatic*）。
     2026-08-03 用户定论：去增强实验（noaug 1.61% vs 增强 3.29%）表明
     去增强没用，正式配置保留标准几何增强。
  3. loop=1 显式（修复 defaults.py 的 loop=epoch//eval_epoch 陷阱：
     eval_epoch=1 时 loop=20 → 20 epochs 数据过 400 遍 ≈ 115h）
  4. weight=16dim-init（768→16 维转换的预训练 backbone，rule #6）
  3. 其余（epoch=1、losses、density_invariant、hooks、tester）与 smoke 完全一致，
     保证与 baseline（5.4% / 3.29%）的差异只来自"无几何增强"。

Usage:
    python tools/train_lite.py --config-file configs/custom/lang-pretrain-ptv3m1-scannetpp-v2-noaug.py --num-gpus 2
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

# 锚文本（评测端 tester 用同一文件；训练端 align_text16=False 时不用，保留以便回切）
repo_root = "/home/isom/cyf/CompressedSceneSplat"
text_embeddings_path = f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt"

model = dict(
    type="LangPretrainer",
    verbose_losses=True,
    # 方案 X 已放弃（Q 列符号歧义 → 模式坍缩，dim0 corr -0.23）。训练目标 = 原始
    # canonicalize 后的压缩特征（与 compress_grid_svd.py 一致），评测端再 Procrustes。
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
    # 2026-08-04 最终配置（诊断链结论）：
    # - SVDWeightedL1Loss（inverse_variance：波动维高权重）+ CosineSimilarity
    # - AggregatedContrastiveLoss loss_weight=0.1（2026-08-04 用户定：0.3 崩溃、
    #   0.02 原始值、0.1 折中；原 0.3 崩溃 = 15 倍权重 × 突变激活 × 未归一化聚合）
    # - 训练输出不 normalize（2026-08-04 用户定：目标压缩特征无显式 L2 normalize；
    #   实测范数 ≈0.98±0.017 在球面附近，恢复 normalize 的 L1 地板仅 ~0.03，可选）
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
# Scheduler settings (same as smoke)
# ============================================================================
epoch = 10  # 2026-08-04：用户时间限制，20→10
eval_epoch = 1
# 2026-08-04 修复：defaults.py 默认 loop=epoch//eval_epoch=20 → 每 epoch 数据
# 过 20 遍（20 epochs = 400 遍 ≈ 115h）。显式 loop=1：每 epoch 数据过 1 遍，
# 20 epochs ≈ 5.6h，每 epoch eval + 每 epoch checkpoint（CheckpointSaver save_freq=1）。
# 注意：loop 必须写在 data.train dict 内（defaults.py 读 cfg.data.train.get("loop")，
# 顶层 loop 变量不会被看到——2026-08-04 已踩坑）。
max_grad_threshold = 4.0
decoder_grad_warn_threshold = 3.0
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],  # 两组：main 0.006 + block 0.0006（恢复原始默认限速）
    pct_start=0.1,  # 2026-08-04 修正：0.5 无依据（单 chunk 问题错套多 chunk），多 chunk 20120 iters 下 0.1=2000 iters 高 LR 足够
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
# 2026-08-04 恢复原始默认 block 限速：epoch_1 实测 block 满速（0.006）1.14%
# vs 限速（0.0006）3.55%（noaug-16diminit）——多 chunk 1-epoch 下满速差 3 倍
# （contrast 在 epoch_1 未激活，唯一差异即 block lr）。单 chunk verify1200
# 的"满速快 2.7 倍"结论不迁移到多 chunk 全量训练。
param_dicts = [dict(keyword="block", lr=0.0006)]

# Save path (2026-08-04: 正式 run，noaug，block 限速)
save_path = "exp/smoke-ptv3m1-16-scannetpp-v2-final-l1only-blocklr"

# 2026-08-04 修复：16dim-init 初始化（768→16 维转换的预训练 backbone）。
# 此前所有 2026-08-03 实验都是随机初始化（文档 rule #6 遗漏）。
weight = "/home/isom/cyf/CompressedSceneSplat/checkpoints/lang-pretrain-pt-v3m1-16dim-init.pth"

# ============================================================================
# Dataset settings (identical to smoke)
# ============================================================================
dataset_type = "ScanNetPPGSDataset"
data_root = "/home/isom/cyf/SceneSplat/scannetpp_v2"
evaluate = False
# scannetpp_v2 GT 为 top100 benchmark 编码（preprocess_scannetpp_gs.py 用 top100.txt 映射）
class_names_path = f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt"

# ============================================================================
# Hooks (same as smoke)
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
        svd_rank=16,
        use_procrustes=True,
        svd_center=False,
    ),
    dict(type="CheckpointSaver", save_freq=1),
    dict(type="PreciseEvaluator", test_last=True),
]

# ============================================================================
# Tester (same as smoke)
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
# Data pipeline
# ============================================================================
data = dict(
    num_classes=100,  # scannetpp top100 benchmark
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split=("train_grid1.0cm_chunk6x6_stride3x3", "test_grid1.0cm_chunk6x6_stride3x3"),
        data_root=data_root,
        sample_tail_classes=False,
        load_compressed_lang_feat=True,
        svd_rank=16,
        svd_center=False,
        # 2026-08-04 修复：显式 loop=1（必须在此处，defaults.py 读 data.train.loop）
        loop=1,
        # 2026-08-03 全局符号对齐注册表（加载时翻转列符号，原始 npz 不动）。
        # 生成：tools/compression/build_global_sign_registry.py（只读）。
        global_sign_path="/home/isom/cyf/SceneSplat/scannetpp_v2/lang_feat_grid_svd_r16_global_signs_v3.npz",
        transform=[
            dict(type="FilterValidPoints", key="valid_feat_mask"),
            dict(type="CenterShift", apply_z=True),
            # 2026-08-04 noaug 对照（v1 最优条件 + v3 注册表 + 16dim-init）：
            # 几何增强全部移除（RandomRotate×3/RandomScale/RandomFlip/RandomJitter/
            # ElasticDistortion）。机制：增强扰动 GridSample cell 哈希 → 同一物理点
            # 的波动维目标随 iter 变化 → 模型学平均 → 波动坍缩（v3 多 epoch 实测：
            # epoch1 3.15% → epoch3 2.53%，minor corr 0.014→0.01）。
            # v1（noaug+注册表）4.62% 是最优基线；本配置验证 noaug 在
            # v3 注册表 + 16dim-init 下是否 ≥4.62%（唯一差异=增强）。
            # 保留 Chromatic* 颜色增强：只改 color 输入，不改坐标 → 不扰动目标。
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
