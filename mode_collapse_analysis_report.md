# Mode Collapse Analysis Report

## Summary

The output features from the LangPretrainer model exhibit **severe mode collapse** (cosine similarity ~0.98) when processing both ScanNet and OVS data. This report documents the root cause investigation findings.

## Problem Description

- **Checkpoint**: `checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth`
- **Model**: PT-v3m1 backbone with LangPretrainer head
- **Symptom**: Output features have cosine similarity of 0.98 (severe collapse) on all test data
- **Expected**: Cosine similarity < 0.7 for diverse features

## Investigation Results

### 1. Input Data is Healthy
- Input features (11-dim from color+opacity+quat+scale): cos_sim=0.67 ✓
- FilterCoordOutliers transform works correctly
- No coordinate overflow issues

### 2. Source Checkpoint Weights are Healthy
- Weight diversity: 99.99% unique values ✓
- Column cosine similarity: ~0 (orthogonal) ✓
- No obvious weight corruption

### 3. Collapse Happens During Forward Pass

| Stage | cos_sim | mean_proj | Status |
|-------|---------|-----------|--------|
| INPUT (11-dim) | 0.68 | 0.83 | ✓ Healthy |
| embedding | 0.82 | 0.90 | ⚠ Some collapse |
| enc0 | 0.65 | 0.81 | ✓ Healthy |
| enc1 | 0.50 | 0.70 | ✓ Healthy |
| enc2 | 0.42 | 0.64 | ✓ Healthy |
| enc3 | 0.29 | 0.55 | ✓ **Excellent** |
| dec2 | 0.29 | 0.53 | ✓ Healthy |
| dec1 | 0.46 | 0.67 | ✓ Healthy |
| **dec0/block1/fc1** | **0.89** | **0.94** | ⚠ **Moderate collapse** |
| **dec0/block1/fc2** | **0.98** | **0.99** | ✗ **SEVERE COLLAPSE** |
| FINAL (768-dim) | 0.98 | 0.99 | ✗ **SEVERE COLLAPSE** |

### 4. Root Cause: Weight Explosion in dec0/block1/fc1

#### Comparison of fc1 Weight Norms Across Decoder Blocks

| Block | fc1 L2 norm mean | fc1 L2 norm max |
|-------|------------------|-----------------|
| dec2/block0 | 1.26 | 2.04 |
| dec2/block1 | 1.20 | 4.79 |
| dec1/block0 | 1.63 | 6.26 |
| dec1/block1 | 1.69 | 4.60 |
| dec0/block0 | 2.04 | 2.97 |
| **dec0/block1** | **2.39** | **6.87** |

The **dec0/block1/fc1** layer has the largest weight norms, causing the collapse chain:

#### Chain of Collapse in dec0/block1 MLP

1. **fc1 INPUT**: cos_sim=0.52, max=12.8 (healthy but has extreme values)
2. **fc1 OUTPUT**: cos_sim=0.89, mean=-5.4, max=35.1 (moderate collapse + extreme values)
3. **GELU act**: cos_sim=0.76, max=35.1 (preserves positive extreme values)
4. **fc2 OUTPUT**: cos_sim=0.98, max=1147 (severe collapse, extreme values amplified)

#### fc1 Weight Statistics (dec0/block1)

```
fc1.weight: shape=(3072, 768)
  mean=0.019, std=0.086
  min=-1.10, max=1.14
  L2 norm per output: mean=2.39, std=0.52, max=6.87
  L2 norm per input: mean=4.88, std=0.37, max=7.54

fc1.bias: shape=(3072,)
  mean=-0.075, std=0.056
  min=-0.348, max=0.104
```

## Technical Details

### Model Path to Collapse
```
backbone.dec.dec0.block1.mlp.0
├── fc1 (Linear: 768 → 3072)  ← WEIGHT EXPLOSION HERE
├── act (GELU)
├── fc2 (Linear: 3072 → 768)  ← Amplifies collapse
└── drop (Dropout p=0.0)
```

### Feature Value Progression
```
fc1 INPUT:    mean=-0.18, std=1.20, min=-6.29, max=12.79
fc1 OUTPUT:   mean=-5.36, std=4.94, min=-61.75, max=35.13
act OUTPUT:   mean=0.34,  std=1.54, min=-0.17,  max=35.13
fc2 OUTPUT:   mean=1.00,  std=38.28, min=-231.27, max=1147.05
```

## Conclusion

The checkpoint `lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth` contains **excessively large weights** in the `dec0.block1.mlp.0.fc1` layer (L2 norm max=6.87, much higher than other blocks at ~2-5).

This weight explosion causes:
1. Extreme output values from fc1 (max=35)
2. GELU activation preserves positive extreme values
3. fc2 amplifies these extremes into a collapsed manifold (cos_sim=0.98)

## Recommendations

1. **Check training logs** for this checkpoint to understand why dec0/block1/fc1 weights grew so large
2. **Apply weight clipping** or stronger weight decay on fc1 layers in decoder blocks
3. **Consider gradient clipping** during training to prevent weight explosion
4. **Test with an earlier checkpoint** to see if this issue developed over training
5. **Apply weight normalization** to fc1/fc2 layers to prevent this specific failure mode

## File Locations

- **Problematic checkpoint**: `/new_data/cyf/projects/SceneSplat/checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth`
- **Problematic layer path**: `model.backbone.dec.dec0.block1.mlp.0.fc1`
- **Config**: `configs/inference/lang-pretrain-pt-v3m1-3dgs.py`

## Verification Commands

```bash
# Check weight norms for all decoder fc1 layers
python -c "
import torch
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.engines.hooks.misc import CheckpointLoader

cfg = Config.fromfile('configs/inference/lang-pretrain-pt-v3m1-3dgs.py')
model = build_model(cfg.model)
# ... load checkpoint ...

for dec_name in ['dec2', 'dec1', 'dec0']:
    dec = getattr(model.backbone.dec, dec_name)
    for j in [0, 1]:
        block = getattr(dec, f'block{j}')
        fc1 = block.mlp[0].fc1
        l2_norm = fc1.weight.float().norm(dim=1)
        print(f'{dec_name}/block{j}/fc1 L2: mean={l2_norm.mean():.2f}, max={l2_norm.max():.2f}')
"
```
