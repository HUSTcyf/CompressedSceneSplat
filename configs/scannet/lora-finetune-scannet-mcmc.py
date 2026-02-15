# LoRA Fine-tuning Configuration for SceneSplat
# This config extends the base SceneSplat configuration with LoRA settings

_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannet.py",
]

# Model configuration - use pretrained model as base
model = dict(
    type="LangPretrainer",
    backbone=dict(
        type="PointTransformerV3",
        in_channels=11,  # xyz(3) + color(3) + opacity(1) + quat(4) = 11
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
    ),
    # LoRA configuration
    lora=dict(
        enabled=True,
        r=8,  # Low-rank dimension
        lora_alpha=8,  # Scaling factor
        target_modules=["attn"],  # Apply to attention layers only
        enable_prompt=False,  # Disable prompt MLP for simplicity
        encoder_only=True,  # Only fine-tune encoder
        target_stages=None,  # All stages
        freeze_backbone=True,  # Freeze non-LoRA parameters
    ),
    # Other model settings
    embed_layer="last",
    embed_norm=True,
    loss="contrastive",
)

# Data configuration
data = dict(
    num_classes=807,  # ScanNet++ classes
    ignore_index=-1,
)

# Optimizer - only optimize LoRA parameters
optimizer = dict(
    type="AdamW",
    lr=0.0001,  # Lower learning rate for fine-tuning
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        # Only train LoRA parameters
        custom_keys={
            "lora_": dict(lr_mult=1.0),  # LoRA A and B matrices
            "prompt": dict(lr_mult=0.5),  # Prompt MLP (if enabled)
        },
    ),
)

# Scheduler
scheduler = dict(
    type="CosineAnnealingLR",
    warmup_lr=1e-5,
    warmup_epochs=10,
    min_lr=1e-6,
)

# Training configuration
train = dict(
    max_epochs=50,
    eval_epoch=1,
    save_epoch=5,
)

# LoRA-specific settings
lora_training = dict(
    save_lora_only=True,  # Only save LoRA parameters
    merge_before_eval=False,  # Don't merge weights during training
    # Resume from pretrained checkpoint
    pretrained_path="checkpoints/pretrained_model.pth",
    # Optional: merge and save full model after training
    merge_on_end=False,
)
