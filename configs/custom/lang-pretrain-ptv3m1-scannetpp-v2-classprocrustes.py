"""
ClassMeanProcrustesLoss v7 训练（2026-08-05，目标 val mIoU > 25%）。

实验史（2026-08-04/05）：
- v1（类均值 + cos，min_points=50）：12.1 → 17.1 → 5.7（E3 崩溃 = cos 噪声注入）
- v5（纯类均值，min_points=10，top50）：13.9 → ... → 20.6 → 23.6 → 22.4（无崩溃 ✓，
  但 top50 只覆盖 84/100 类 → 16 类 val 全 0，上限被锁）
- v6（cover250）：覆盖 100 类但含 260 万点巨型 chunk → 训练 3× 慢 + 评测 hook 卡死

v7 = v5 配方 + 小规模全覆盖集：
1. split = train_top50plus（50 top50 + 7 个含缺失 16 类的 chunk，≤80 万点，
   100/100 类覆盖，57 chunks）
2. loop=10（570 样本/epoch ≈ 142 iter），epoch=20
3. 损失同 v5：纯类均值 Procrustes 残差（min_points=10，center=False）
4. eval_during_train=False（hook 评测卡死未知原因；训练完用 tester 评测）
5. wd 0.05，weight=16dim-init，noaug（仅 Chromatic*），block lr 0.0006
"""

_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# ============================================================================
# Misc custom settings (same as 50chunk)
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

repo_root = "/home/isom/cyf/CompressedSceneSplat"
text_embeddings_path = f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt"

model = dict(
    type="LangPretrainer",
    verbose_losses=True,
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
        drop_path=0.1,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNetPP",),
    ),
    # 2026-08-05 v5（纯类均值残差，无逐点损失）：
    # - ClassMeanProcrustesLoss：类均值等权残差，min_points=10（小类进损失）
    # - 移除 CosineSimilarity（饱和后噪声注入 → E3 崩溃）
    criteria=[
        dict(
            type="ClassMeanProcrustesLoss",
            loss_weight=1.0,
            reduction="mean",
            min_points_per_class=3,  # v21: 3-4 点/块的小类进损失
            residual="l1",
            point_weight=0.2,   # v13: v10 精确配方重跑（E9=24.07%，赌 seed 方差）
            mean_weight=1.0,
            center=False,
        ),
    ],
)

# ============================================================================
# Density-Invariant Training Configuration (same as 50chunk)
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
# Scheduler settings (same as 50chunk)
# ============================================================================
epoch = 1
eval_epoch = 1
max_grad_threshold = 4.0
decoder_grad_warn_threshold = 3.0
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],  # 两组：main 0.006 + block 0.0006
    pct_start=0.15,  # v25: LR 尾部延长（v10 seed 锚定）
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

save_path = "exp/smoke-ptv3m1-16-scannetpp-v2-classprocrustes-v26"

weight = "/home/isom/cyf/CompressedSceneSplat/checkpoints/lang-pretrain-pt-v3m1-16dim-init.pth"

# ============================================================================
# Dataset settings (same as 50chunk)
# ============================================================================
dataset_type = "ScanNetPPGSDataset"
data_root = "/home/isom/cyf/SceneSplat/scannetpp_v2"
evaluate = False
class_names_path = f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt"

# ============================================================================
# Hooks (50chunk + eval_during_train)
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
        eval_during_train=True,  # v26: 用 hook（20 场景子集）评测 v10-E9
    ),
    dict(type="CheckpointSaver", save_freq=1),
    dict(type="PreciseEvaluator", test_last=True),
]

# ============================================================================
# Tester (same as 50chunk)
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
# Data pipeline (50chunk: noaug + Chromatic* + GridSample + SphereCrop)
# ============================================================================
data = dict(
    num_classes=100,  # scannetpp top100 benchmark
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split=("train_top50_grid1.0cm_chunk6x6_stride3x3",),
        data_root=data_root,
        sample_tail_classes=False,
        load_compressed_lang_feat=True,
        svd_rank=16,
        svd_center=False,
        # v7：57 chunks 全覆盖，每 epoch 10 遍（570 样本/epoch ≈ 142 iter）
        loop=10,
        global_sign_path="/home/isom/cyf/SceneSplat/scannetpp_v2/lang_feat_grid_svd_r16_global_signs_v3.npz",
        transform=[
            dict(type="FilterValidPoints", key="valid_feat_mask"),
            dict(type="CenterShift", apply_z=True),
            # noaug：几何增强会扰动 GridSample cell 哈希 → 目标随 iter 变化。
            # 保留 Chromatic*（只改输入颜色，不改坐标 → 不扰动目标）。
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
        svd_center=False,
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
