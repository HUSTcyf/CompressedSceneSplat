# PointLoRA Integration Guide for SceneSplat

This guide explains how to use LoRA (Low-Rank Adaptation) for efficient fine-tuning of SceneSplat models.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Best Practices](#best-practices)

---

## Overview

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that:
- Freezes pre-trained model weights
- Adds trainable rank decomposition matrices to each layer
- Reduces trainable parameters by ~1000x
- Maintains model performance while enabling efficient adaptation

### Integration Architecture

```
SceneSplat (PT-v3)
├── pointcept/models/utils/
│   ├── lora.py              # LoRA layer implementations
│   └── lora_injector.py     # PT-v3 specific injection logic
├── configs/
│   └── scannet/lora-finetune-*.py  # LoRA training configs
└── tools/
    └── train_lora.py        # LoRA fine-tuning script
```

---

## Quick Start

### 1. Basic Fine-tuning

```bash
python tools/train_lora.py \
    --config-file configs/scannet/lora-finetune-scannet-mcmc.py \
    --options save_path=exp_runs/lora_finetune_scannet \
                lora.r=8 \
                lora_training.pretrained_path=checkpoints/pretrained.pth
```

### 2. Using Presets

```bash
# Minimal preset (rank=4, deeper stages only)
python tools/train_lora.py \
    --config-file configs/scannet/lora-finetune-scannet-mcmc.py \
    --options save_path=exp_runs/lora_minimal \
                lora.preset=minimal

# Full preset (rank=16, attn+mlp, with prompt)
python tools/train_lora.py \
    --config-file configs/scannet/lora-finetune-scannet-mcmc.py \
    --options save_path=exp_runs/lora_full \
                lora.preset=full
```

### 3. Custom Configuration

```python
from pointcept.models.utils import (
    inject_lora_to_ptv3,
    mark_only_lora_as_trainable,
    print_lora_summary,
)

# Load pretrained model
model = build_model(cfg.model)
checkpoint = torch.load("checkpoints/pretrained.pth")
model.load_state_dict(checkpoint["state_dict"], strict=False)

# Inject LoRA (rank=8, attention only, encoder stages 3-4)
model = inject_lora_to_ptv3(
    model,
    r=8,
    lora_alpha=8,
    target_modules=["attn"],
    encoder_only=True,
    target_stages=[3, 4],
)

# Freeze non-LoRA parameters
mark_only_lora_as_trainable(model)

# Print summary
print_lora_summary(model)
```

---

## Configuration

### LoRA Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 8 | Low-rank dimension (higher = more capacity) |
| `lora_alpha` | int | 8 | Scaling factor (alpha/r) |
| `target_modules` | list | ["attn"] | Which layers to adapt |
| `enable_prompt` | bool | False | Enable prompt MLP |
| `encoder_only` | bool | True | Skip decoder stages |
| `target_stages` | list | None | Specific stages to target |

### Target Modules Options

- `["attn"]` - Attention QKV and projection layers (default, recommended)
- `["mlp"]` - MLP fc1 and fc2 layers
- `["attn", "mlp"]` - Both attention and MLP

### Stage Indices

For encoder (5 stages):
- Stage 0: 32 channels, 2 blocks
- Stage 1: 64 channels, 2 blocks
- Stage 2: 128 channels, 2 blocks
- Stage 3: 256 channels, 6 blocks ← **Most important**
- Stage 4: 512 channels, 2 blocks

Recommendation: Target stages [3, 4] for minimal fine-tuning.

---

## API Reference

### Core Functions

#### `inject_lora_to_ptv3(model, r, lora_alpha, ...)`

Inject LoRA adapters into PT-v3 model.

```python
model = inject_lora_to_ptv3(
    model=model,
    r=8,                      # Rank
    lora_alpha=8,             # Alpha
    target_modules=["attn"],   # Target modules
    enable_prompt=False,       # Enable prompt MLP
    encoder_only=True,         # Skip decoder
    target_stages=None,        # All stages (or [3, 4] for specific)
)
```

#### `inject_lora_with_preset(model, preset, **)`

Inject LoRA using preset configuration.

```python
# Available presets: "minimal", "standard", "full"
model = inject_lora_with_preset(model, preset="standard")
```

#### `mark_only_lora_as_trainable(model, bias)`

Freeze all parameters except LoRA.

```python
mark_only_lora_as_trainable(model, bias="none")
```

#### `get_lora_parameters(model)`

Get list of LoRA parameters for optimizer.

```python
lora_params = get_lora_parameters(model)
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
```

#### `print_lora_summary(model)`

Print parameter statistics.

```python
print_lora_summary(model)
# Output:
# ============================================================
# LoRA Parameter Summary
# ============================================================
# Total parameters: 31,456,789
# LoRA parameters: 245,760
# Trainable parameters: 245,760
# LoRA ratio: 0.78%
# ============================================================
```

### Checkpoint Functions

#### `save_lora_checkpoint(model, path, metadata)`

Save only LoRA parameters.

```python
save_lora_checkpoint(
    model,
    path="checkpoints/lora_adapter.pth",
    metadata={"r": 8, "lora_alpha": 8, "target_modules": ["attn"]},
)
```

#### `load_lora_checkpoint(model, path)`

Load LoRA parameters into model.

```python
load_lora_checkpoint(model, "checkpoints/lora_adapter.pth")
```

#### `merge_lora_weights(model)`

Merge LoRA into base weights (for deployment).

```python
merge_lora_weights(model)
# Now model can be saved without LoRA overhead
torch.save(model.state_dict(), "checkpoints/merged_model.pth")
```

---

## Examples

### Example 1: Fine-tune on New Dataset

```python
# Load pretrained SceneSplat model
from pointcept.models.builder import build_model

cfg = Config.fromfile("configs/scannet/lang-pretrain-scannet-mcmc.py")
model = build_model(cfg.model)

# Load pretrained weights
checkpoint = torch.load("checkpoints/pretrained_scene_splat.pth")
model.load_state_dict(checkpoint["state_dict"], strict=False)

# Inject LoRA for fine-tuning
from pointcept.models.utils import inject_lora_to_ptv3, mark_only_lora_as_trainable

model = inject_lora_to_ptv3(
    model,
    r=8,
    lora_alpha=8,
    target_modules=["attn"],
    target_stages=[3, 4],  # Only fine-tune deeper stages
)

# Freeze backbone
mark_only_lora_as_trainable(model)

# Setup optimizer (only LoRA params)
from pointcept.models.utils import get_lora_parameters

lora_params = get_lora_parameters(model)
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

# Train...
```

### Example 2: Domain Adaptation

```python
# Adapt from indoor (ScanNet) to outdoor (KITTI-360)
model = load_pretrained_model("checkpoints/scannet_pretrained.pth")

# Use higher rank for domain shift
model = inject_lora_to_ptv3(
    model,
    r=16,  # Higher capacity
    lora_alpha=16,
    target_modules=["attn", "mlp"],  # Adapt both
    enable_prompt=True,  # Enable prompt for extra adaptation
)

# Train on KITTI-360...
```

### Example 3: Multi-Task Learning

```python
# Base model for segmentation
base_model = load_pretrained_model("checkpoints/pretrained.pth")

# Create task-specific LoRA adapters
for task in ["segmentation", "detection", "classification"]:
    model = copy.deepcopy(base_model)

    # Task-specific adapter
    model = inject_lora_to_ptv3(
        model,
        r=8,
        lora_alpha=8,
        target_modules=["attn"],
    )

    # Train on task-specific data...
```

---

## Best Practices

### 1. Rank Selection

| Use Case | Recommended Rank | Reason |
|----------|------------------|--------|
| Minimal fine-tuning | 4-8 | Sufficient for small domain shifts |
| Standard fine-tuning | 8-16 | Good balance for most cases |
| Large domain shift | 16-32 | More capacity needed |
| Multi-task | 8-16 per task | Prevent interference |

### 2. Stage Selection

- **Deeper stages only** ([3, 4]): 10x fewer parameters, good for small datasets
- **All stages**: Better for large domain shifts
- **Early stages**: Rarely needed unless input characteristics change

### 3. Learning Rate

```python
# LoRA typically needs lower LR than full fine-tuning
optimizer = torch.optim.AdamW(
    lora_params,
    lr=1e-4,      # Standard: 1e-3 to 1e-4
    weight_decay=0.01,
)
```

### 4. Training Tips

1. **Start with frozen backbone**: Use `mark_only_lora_as_trainable(model)`
2. **Monitor overfitting**: LoRA has fewer parameters, less prone to overfitting
3. **Use cosine annealing**: Standard LR schedule works well
4. **Save LoRA checkpoints**: Much smaller than full model

### 5. Deployment

```python
# Option 1: Keep LoRA separate (recommended for flexibility)
save_lora_checkpoint(model, "adapter.pth")
# Load: base_model + load_lora_checkpoint(base_model, "adapter.pth")

# Option 2: Merge weights (for efficiency)
merge_lora_weights(model)
torch.save(model.state_dict(), "merged_model.pth")
```

---

## Preset Comparison

| Preset | Rank | Modules | Stages | Prompt | Params | Use Case |
|--------|------|---------|--------|--------|--------|----------|
| `minimal` | 4 | attn | [3,4] | No | ~50K | Quick adaptation |
| `standard` | 8 | attn | all | No | ~200K | General fine-tuning |
| `full` | 16 | attn+mlp | all | Yes | ~2M | Large domain shift |
| `decoder_only` | 8 | attn | dec | No | ~100K | Task-specific heads |

---

## Troubleshooting

### Issue: Poor convergence

**Solution**: Try increasing rank or adding more stages:
```python
model = inject_lora_to_ptv3(model, r=16, target_stages=[2, 3, 4])
```

### Issue: Overfitting

**Solution**: Reduce rank or use dropout:
```python
model = inject_lora_to_ptv3(model, r=4, target_stages=[4])
```

### Issue: CUDA OOM

**Solution**: Reduce batch size or use gradient checkpointing:
```python
# Already handled by chunking in SceneSplat
# Just reduce chunk_size if needed
```

---

## References

- LoRA Paper: https://arxiv.org/abs/2106.09685
- PointLoRA: https://github.com/Pointcept/PointLoRA
- SceneSplat: https://github.com/Pointcept/SceneSplat
