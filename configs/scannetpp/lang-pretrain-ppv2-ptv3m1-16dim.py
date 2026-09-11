# ============================================================================
# B experiment: PT-v3m1 16-dim on scannetpp_v2
# Aligned with official configs/scannetpp/lang-pretrain-ppv2-mcmc-wo-normal-contrastive.py
# Differences from official:
#   1. dec_channels=(16,128,256) -> 16-dim output (SVD rank), official is (768,512,256)
#   2. in_channels=11 (no normal; scannetpp_v2 data has no normal.npy), official 14
#   3. SVD compressed lang_feat pipeline (FilterValidPoints + load_compressed_lang_feat + point_to_grid)
#   4. data_root/scannetpp_v2 + chunk6x6 splits (official grid1mm dir no longer exists)
#   5. class_names/text_embeddings from scannetpp metadata (official used scannet200)
# Everything else (optimizer, scheduler, loss, hooks, num_classes) identical to official.
# ============================================================================

_base_ = ["../_base_/default_runtime.py", "../_base_/dataset/scannetpp.py"]

# misc custom setting
debug = 0
gpu_nums = 1 if debug else 2
batch_size = 2 * gpu_nums
batch_size_val = 2 * gpu_nums
batch_size_test = 1 * gpu_nums
num_worker = 16 * gpu_nums if not debug else 0
mix_prob = 0.8
empty_cache = False
enable_amp = True

# model settings
svd_rank = 16
model = dict(
    type="LangPretrainer",
    backbone=dict(
        type="PT-v3m1",
        in_channels=11,  # gaussian: color 3, quaternion 4, scale 3, opacity 1 (no normal)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 6),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2),
        dec_channels=(svd_rank, 128, 256),  # <- 16-dim output (B experiment)
        dec_num_head=(16, 16, 16),
        dec_patch_size=(1024, 1024, 1024),
        mlp_ratio=4,  # official
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,  # official
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
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CosineSimilarity", reduction="mean", loss_weight=1.0),
        dict(type="L2Loss", reduction="mean", loss_weight=1.0),
        dict(
            type="AggregatedContrastiveLoss",
            temperature=0.2,
            reduction="mean",
            loss_weight=0.020,
            schedule="last_75",
        ),
    ],
)

# scheduler settings
epoch = 800
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

# dataset settings
dataset_type = "ScanNetPPGSDataset"
data_root = "/home/isom/cyf/SceneSplat/scannetpp_v2"

repo_root = "/home/isom/cyf/CompressedSceneSplat"
class_names_path = f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt"
text_embeddings_path = f"{repo_root}/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt"

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
    ),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="BeginningEvaluator", test_last=True),
]

# Tester
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
)

data = dict(
    num_classes=100,
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split=(
            "train_grid1.0cm_chunk6x6_stride3x3",
            "test_grid1.0cm_chunk6x6_stride3x3",
        ),
        data_root=data_root,
        sample_tail_classes=False,
        load_compressed_lang_feat=True,
        svd_rank=16,
        transform=[
            dict(type="FilterValidPoints", key="valid_feat_mask"),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
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
                grid_size=0.02,
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
                    "segment",
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
                keys=("coord", "grid_coord", "segment", "lang_feat", "valid_feat_mask", "name", "scene_path"),
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
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            dict(
                type="Copy",
                keys_dict={
                    "segment": "origin_segment",
                    "coord": "origin_coord",
                    "valid_feat_mask": "origin_feat_mask",
                },
            ),
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
