# Inference config for LitePT LangPretrainer with OVS data and SVD compression

# SVD rank (must match training)
FD = 16  # Feature dimension
svd_rank = 16

# Model configuration (same as training)
model = dict(
    type="LangPretrainer",
    backbone=dict(
        type="LitePT",
        in_channels=11,  # color 3, opacity 1, quat 4, scale 3
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(FD, FD*2, FD*4, 144),  # (16, 32, 64, 144)
        dec_num_head=(32, 16, 8, 4),
        dec_patch_size=(1024, 1024, 1024, 1024),  # 4 elements to match dec_depths
        dec_conv=(True, True, True, False),
        dec_attn=(False, False, False, True),
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
    criteria=[
        dict(type="L2Loss", loss_weight=1.0),
        dict(type="CosineSimilarity", loss_weight=1.0),
    ],
)

# Feature keys for 3D Gaussian Splatting
feat_keys = ("color", "opacity", "quat", "scale")  # 11 channels total
grid_sample_keys = (
    "coord",
    "color",
    "opacity",
    "quat",
    "scale",
    "segment",
    "valid_feat_mask",
)
grid_sample_keys_test = (
    "coord",
    "color",
    "opacity",
    "quat",
    "scale",
    "segment",
    "valid_feat_mask",
)
collect_keys_test = (
    "coord",
    "grid_coord",
    "index",
    "segment",
    "valid_feat_mask",
)

inference = dict(
    transform=[
        dict(type="CenterShift", apply_z=True),
        dict(type="NormalizeColor"),
        dict(
            type="Copy",
            keys_dict=dict(
                segment="origin_segment",
                coord="origin_coord",
                valid_feat_mask="origin_feat_mask",
            ),
        ),
        dict(
            type="GridSample",
            grid_size=0.01,  # Smaller grid for more points
            hash_type="fnv",
            mode="train",
            keys=grid_sample_keys,
            apply_to_pc=False,  # Don't sample pc_coord/pc_segment
            return_inverse=True,  # Return mapping for full scene reconstruction
        ),
    ],
    test_cfg=dict(
        voxelize=dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="test",  # test mode creates fragments
            keys=grid_sample_keys_test,
            apply_to_pc=False,
            return_grid_coord=True,
        ),
        crop=None,  # No cropping, use full scene
        post_transform=[
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=collect_keys_test,
                feat_keys=feat_keys,
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
    chunk_size=600000,  # Large chunk size for full scene
    save_features=dict(
        output_dir=None,  # Will be overridden
        backbone=dict(enabled=False),  # We handle saving ourselves
    ),
    default_scene_name="scenesplat_scene",
    device="cuda",
    return_numpy=True,
)
