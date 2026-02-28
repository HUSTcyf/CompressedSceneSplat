# Mode Collapse Analysis and Solutions

## Problem Summary

### Current Situation
Based on the training logs at Epoch 19:
```
Loss dense: 0.236087
Loss single: 0.236599
Total Loss: 0.474388
```

**Key observations:**
1. Dense and single losses are nearly identical (diff = 0.0005)
2. This indicates the model is producing the same output regardless of input density
3. **No AggregatedContrastiveLoss output visible in logs** - this is the critical issue

### Root Cause Analysis

#### 1. AggregatedContrastiveLoss Not Activating

Looking at `pointcept/models/losses/misc.py`:

```python
# Line 367: Minimum samples requirement
if indices.numel() < 100:
    continue  # insufficient samples
```

**The AggregatedContrastiveLoss requires >=100 samples per class to activate.**

With GridSample (grid_size=0.01), the number of samples per semantic class may be:
- Too small for rare classes
- Below the 100 threshold for most classes
- Resulting in loss = 0 for most iterations

#### 2. Ground Truth Feature Similarity

From the cross-scene similarity analysis:
- Ground truth SVD features have 0.86-0.88 similarity across scenes
- This is relatively high (16-dimensional compression limits diversity)
- The model is learning to predict similar features because ground truth IS similar

#### 3. Loss Weight Too Low

```python
criteria=[
    dict(type="L2Loss", loss_weight=0.5),
    dict(type="CosineSimilarity", loss_weight=1.0),
    dict(type="AggregatedContrastiveLoss", loss_weight=0.05),
]
```

Total reconstruction loss weight = 1.5
Contrastive loss weight = 0.05 (only 3% of total loss)

Even when AggregatedContrastiveLoss activates, it's too weak to overcome the reconstruction losses.

#### 4. 16-Dimensional SVD Feature Space

16 dimensions provide limited capacity for semantic diversity:
- Full 768-dim features can represent ~48x more information
- Information bottleneck forces similar representations
- Mode collapse is more likely with compressed features

---

## Solutions

### Solution 1: Lower the Minimum Sample Threshold (Easiest)

**File:** `pointcept/models/losses/misc.py`

Change line 367:
```python
# Before:
if indices.numel() < 100:
    continue

# After:
if indices.numel() < 20:  # Lower threshold
    continue
```

**Rationale:** GridSample reduces sample count per class. Lower threshold allows more classes to participate.

---

### Solution 2: Increase Contrastive Loss Weight

**File:** `configs/custom/lang-pretrain-litept-ovs-gridsvd.py`

Change line 110:
```python
# Before:
loss_weight=0.05,

# After:
loss_weight=0.2,  # 4x stronger
```

Also consider reducing L2 loss weight:
```python
# Before:
dict(type="L2Loss", loss_weight=0.5),

# After:
dict(type="L2Loss", loss_weight=0.1),  # Allow more feature diversity
```

---

### Solution 3: Add Intra-Class Variance Loss

Create a new loss that explicitly penalizes high intra-class similarity:

```python
@LOSASSES.register_module()
class IntraClassVarianceLoss(nn.Module):
    """
    Encourages diversity within each semantic class.
    
    For each class, compute the variance of features and penalize low variance.
    This prevents all samples from collapsing to the same representation.
    """
    def __init__(self, loss_weight=1.0, min_variance=0.1):
        super().__init__()
        self.loss_weight = loss_weight
        self.min_variance = min_variance
    
    def forward(self, pred, segment, valid_feat_mask, **kwargs):
        valid_idx = (valid_feat_mask > 0) & (segment != -1)
        if valid_idx.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        features = pred[valid_idx]
        labels = segment[valid_idx]
        
        loss = 0
        count = 0
        for lab in torch.unique(labels):
            mask = labels == lab
            if mask.sum() < 10:
                continue
            class_features = features[mask]
            variance = class_features.var(dim=0).mean()
            # Penalize variance below minimum
            loss += F.relu(self.min_variance - variance)
            count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=pred.device)
        
        return self.loss_weight * loss / count
```

---

### Solution 4: Orthogonal Regularization

Add orthogonal regularization to the decoder output to encourage diverse feature dimensions:

```python
@LOSSES.register_module()
class OrthogonalRegularization(nn.Module):
    """
    Encourages feature dimensions to be orthogonal to each other.
    This maximizes information capacity in limited dimensions.
    """
    def forward(self, pred, valid_feat_mask, **kwargs):
        valid_idx = valid_feat_mask > 0
        if valid_idx.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        features = pred[valid_idx]  # [N, D]
        
        # Compute covariance matrix
        features_centered = features - features.mean(dim=0)
        cov = (features_centered.T @ features_centered) / features.size(0)
        
        # Penalize off-diagonal elements (covariance between dimensions)
        off_diagonal = cov - torch.diag(torch.diag(cov))
        loss = (off_diagonal ** 2).mean()
        
        return self.loss_weight * loss
```

---

### Solution 5: Use Higher SVD Rank

**Change from 16 to 32 or 64 dimensions:**

```python
svd_rank = 32  # or 64
lang_feat_dim = svd_rank
FD = lang_feat_dim
```

Trade-off: More memory and computation, but better semantic diversity.

---

### Solution 6: Feature Decorrelation Loss

Add a loss that decorrelates features from different semantic classes:

```python
@LOSSES.register_module()
class ClassDecorrelationLoss(nn.Module):
    """
    Encourages features from different classes to be uncorrelated.
    """
    def forward(self, pred, segment, valid_feat_mask, **kwargs):
        valid_idx = (valid_feat_mask > 0) & (segment != -1)
        if valid_idx.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        features = pred[valid_idx]
        labels = segment[valid_idx]
        
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Compute mean feature for each class
        class_means = []
        for lab in unique_labels:
            mask = labels == lab
            if mask.sum() >= 10:
                class_means.append(features[mask].mean(dim=0))
        
        if len(class_means) < 2:
            return torch.tensor(0.0, device=pred.device)
        
        class_means = torch.stack(class_means)
        
        # Normalize
        class_means = F.normalize(class_means, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = class_means @ class_means.T
        
        # Penalize high off-diagonal similarities
        mask = ~torch.eye(len(class_means), dtype=torch.bool, device=pred.device)
        off_diagonal_sim = sim_matrix[mask]
        
        loss = off_diagonal_sim.abs().mean()
        
        return self.loss_weight * loss
```

---

## Recommended Action Plan

### Immediate (Try First):
1. **Lower AggregatedContrastiveLoss threshold to 20** (Solution 1)
2. **Increase contrastive loss weight to 0.2** (Solution 2)
3. **Reduce L2 loss weight to 0.1**

### Secondary (If above doesn't work):
4. Add IntraClassVarianceLoss (Solution 3)
5. Add OrthogonalRegularization (Solution 4)

### Long-term (If still needed):
6. Increase SVD rank to 32 (Solution 5)

---

## Verification Steps

After implementing changes:

1. **Check that AggregatedContrastiveLoss is activating:**
   - Look for "contrastive loss:" or similar in logs
   - Should see non-zero values

2. **Monitor feature diversity:**
   ```bash
   python tools/analyze_feature_similarity.py --checkpoint exp/lite-16-gridsvd/model_best.pth
   ```

3. **Compare dense vs single loss:**
   - Should see meaningful difference (not 0.0005)
   - Difference should be > 0.01 at minimum

4. **Cross-scene similarity analysis:**
   - Run on trained checkpoint
   - Target: 0.85-0.90 range (not 0.95+)

---

## Files to Modify

1. `/new_data/cyf/projects/SceneSplat/pointcept/models/losses/misc.py` - Lower threshold
2. `/new_data/cyf/projects/SceneSplat/configs/custom/lang-pretrain-litept-ovs-gridsvd.py` - Loss weights

