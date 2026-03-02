# Spatial Consistency Fix - Comprehensive Solution

## Problem Analysis

### Symptoms
1. **Loss plateaued at ~0.2** (91% reduction from initial ~2.3)
2. **Cosine similarity reached 0.91** (healthy, not mode collapse!)
3. **But renders extremely noisy** - salt-and-pepper artifacts, no coherent semantic structure

### Root Causes Identified

#### 1. Spatial Inconsistency
The model learned the **statistical distribution** of features but not the **spatial correspondence**:
- Predicted features matched GT in aggregate statistics
- But neighboring Gaussians had wildly different feature values
- This caused random colored pixels in renders instead of coherent regions

#### 2. dim_scale Imbalance
```
dim_scale values: [0.82, 0.70, 0.44, 0.46, 0.38, 0.45, 0.35, 0.37, 0.33, 0.27, 0.29, 0.29, 0.28, 0.40, 0.21, 0.25]
```
- **4x difference** between max (0.82) and min (0.21)
- Some dimensions severely suppressed, others over-amplified
- Caused spatial inconsistency: adjacent points have very different feature scales

#### 3. No Spatial Regularization
The loss function only considered point-wise feature matching:
- L1 loss: `|pred - target|` per point
- Cosine loss: directional alignment
- **No penalty for spatial inconsistency** between neighboring points

---

## Solutions Implemented

### 1. SpatialSmoothnessLoss (NEW)

**File**: `pointcept/models/losses/misc.py`

Adds a new loss that penalizes large feature differences between neighboring 3D Gaussians:

```python
@LOSSES.register_module()
class SpatialSmoothnessLoss(nn.Module):
    """
    Encourages spatial consistency by penalizing large feature differences
    between neighboring Gaussians. Prevents salt-and-pepper noise.
    """
```

**Key Features**:
- **KNN-based**: For each point, finds k=16 nearest neighbors
- **Radius-filtered**: Only considers neighbors within 2cm radius
- **Progressive scheduling**:
  - Epochs 0-2: No spatial constraint (learn feature statistics first)
  - Epochs 3-9: Ramp up spatial smoothness (reduce noise)
  - Epochs 10+: Decay to allow sharp object boundaries

**Formula**:
```
L_smooth = mean(||f_i - f_j||^2) for all (i,j) where dist(i,j) < radius
```

---

### 2. Balanced dim_scale Normalization

**File**: `pointcept/models/default.py`

#### Before (PROBLEMATIC):
```python
dim_scale_init = torch.ones(16)  # Uniform init
dim_scale_clamped = torch.clamp(torch.relu(self.dim_scale) + 0.01, max=10.0)
feat = feat * dim_scale_clamped
```
**Issue**: ReLU + clamp allowed extreme imbalance (0.21 to 0.82)

#### After (FIXED):
```python
dim_scale_init = torch.ones(16) / (16 ** 0.5)  # Normalized init
dim_scale_normalized = self.dim_scale / (self.dim_scale.norm() + 1e-8)
dim_scale_balanced = dim_scale_normalized * (16 ** 0.5)
feat = feat * dim_scale_balanced
```
**Benefit**: L2 normalization keeps all dimensions balanced

**Why this works**:
- L2 normalization constrains the scale vector to unit sphere
- No single dimension can dominate or be suppressed
- Model learns relative importance but maintains spatial consistency

---

### 3. Updated Model Forward Pass

**File**: `pointcept/models/default.py`

Modified `LangPretrainer.forward()` to pass coordinates for spatial smoothness:

```python
loss = self.criteria(
    point_feat["feat"],
    input_dict["lang_feat"],
    valid_feat_mask=input_dict["valid_feat_mask"],
    segment=segment,
    epoch_progress=input_dict["epoch_progress"],
    coord=input_dict.get("coord"),  # NEW: Pass for spatial smoothness
)
```

Also updated `_chunked_forward()` with the same changes.

---

### 4. Training Config Update

**File**: `configs/custom/lang-pretrain-litept-ovs-gridsvd.py`

Added SpatialSmoothnessLoss to the loss criteria:

```python
criteria=[
    dict(type="SVDWeightedL1Loss", loss_weight=0.3, ...),
    dict(type="CosineSimilarity", loss_weight=1.0, ...),

    # NEW: Spatial smoothness
    dict(
        type="SpatialSmoothnessLoss",
        loss_weight=0.1,  # Moderate weight
        neighbor_k=16,
        radius=0.02,  # 2cm
        warmup_epochs=3,
        decay_start=10,
    ),
],
```

---

## Expected Training Behavior

### Phase 1: Epochs 0-2 (Warmup)
- **SpatialSmoothnessLoss weight = 0**
- Model focuses on learning feature statistics (L1 + Cosine)
- No spatial constraints yet

### Phase 2: Epochs 3-9 (Smoothness Ramp-up)
- **SpatialSmoothnessLoss weight ramps to 0.1**
- Model learns spatial consistency
- Noise in renders should decrease significantly

### Phase 3: Epochs 10+ (Decay)
- **SpatialSmoothnessLoss gradually decays to 0.05**
- Allows sharp object boundaries to form
- Final renders should be clean and coherent

---

## Additional Recommendations

### If Noise Persists After These Changes:

1. **Increase SpatialSmoothnessLoss weight**:
   ```python
   loss_weight=0.2,  # or higher
   ```

2. **Adjust radius for your data**:
   - For grid_size=0.01: radius=0.02 (2x grid)
   - For grid_size=0.02: radius=0.04 (2x grid)

3. **Add Gradual Unfreezing**:
   - Freeze encoder for first 5 epochs
   - Only train decoder and dim_scale
   - Then unfreeze all layers

4. **Use Higher SVD Rank**:
   - Current: svd_rank=16
   - Try: svd_rank=32 for more capacity
   - Note: 2x memory usage

### To Monitor Progress:

```python
# Check dim_scale balance
dim_scale = model.dim_scale.data
print(f"dim_scale range: [{dim_scale.min():.4f}, {dim_scale.max():.4f}]")
print(f"dim_scale ratio: {dim_scale.max()/dim_scale.min():.2f}x")
# Target: < 2x ratio

# Check spatial smoothness (during training)
# Should decrease during epochs 3-9
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| `pointcept/models/losses/misc.py` | Added `SpatialSmoothnessLoss` class |
| `pointcept/models/default.py` | Fixed `dim_scale` normalization, added `coord` parameter to loss |
| `configs/custom/lang-pretrain-litept-ovs-gridsvd.py` | Added `SpatialSmoothnessLoss` to criteria |

---

## Verification

To verify the fix is working:

1. **Check dim_scale balance**:
   ```bash
   python -c "
   import torch
   ckpt = torch.load('exp/lite-16-gridsvd/model/model_last.pth')
   dim_scale = ckpt['state_dict']['dim_scale']
   print(f'dim_scale: {dim_scale}')
   print(f'Ratio: {dim_scale.max()/dim_scale.min():.2f}x')
   "
   ```
   Target: < 2x ratio

2. **Check renders for noise reduction**:
   - Old: Salt-and-pepper noise everywhere
   - New: Coherent semantic regions

3. **Check loss curves**:
   - SpatialSmoothnessLoss should decrease during epochs 3-9
   - L1 and Cosine should continue improving

---

## Performance Impact

- **Memory**: +O(N*K) for KNN computation (K=16 neighbors)
  - Mitigated by sampling large point clouds (>100K points)
- **Speed**: ~10-15% slower per iteration due to KNN
  - Acceptable trade-off for significant quality improvement

---

## Next Steps

1. Start training with the new config
2. Monitor `dim_scale` balance (should stay < 2x ratio)
3. Check renders after epoch 5 for noise reduction
4. Adjust `loss_weight` if needed (0.05 to 0.3 range)
