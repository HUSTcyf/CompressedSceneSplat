#!/usr/bin/env python3
"""
Verify if the inference script is returning GT features instead of model predictions.

This will check if checkpoint_with_features_s.pth contains GT features.
"""

import torch
import numpy as np
from pathlib import Path

print("=" * 80)
print("Verifying: Are checkpoint features actually GT?")
print("=" * 80)

# Scene to check
scene = "figurines"

# Paths
ckpt_s_path = f"/new_data/cyf/projects/SceneSplat/output_features/{scene}/checkpoint_with_features_s.pth"
gt_path = f"/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train/{scene}/lang_feat.npy"

# Load checkpoint features
print(f"\nLoading: {ckpt_s_path}")
ckpt_data = torch.load(ckpt_s_path, map_location="cpu", weights_only=False)
model_params, _ = ckpt_data
ckpt_feat = model_params[7].numpy()  # [N, 768]
print(f"Checkpoint features: {ckpt_feat.shape}")

# Load GT features
print(f"\nLoading: {gt_path}")
gt_feat = np.load(gt_path)  # [M, 768] - only valid points
print(f"GT features: {gt_feat.shape}")

# The checkpoint should have 1M points, but GT only has valid points
# Need to check with valid_feat_mask

# Load valid_feat_mask
mask_path = f"/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train/{scene}/valid_feat_mask.npy"
print(f"\nLoading: {mask_path}")
valid_mask = np.load(mask_path)
print(f"Valid mask: {valid_mask.shape}, {valid_mask.sum()} valid points")

# The GT features only contain valid points
# Checkpoint should have zeros for invalid points

# Compare - only compare valid points
print("\n" + "=" * 60)
print("COMPARISON: Checkpoint vs GT")
print("=" * 60)

# Get valid points from checkpoint (using boolean mask)
bool_mask = valid_mask.astype(bool)
ckpt_valid = ckpt_feat[bool_mask]
gt_valid = gt_feat

print(f"Checkpoint valid features: {ckpt_valid.shape}")
print(f"GT valid features: {gt_valid.shape}")

# Check dimensions match
assert ckpt_valid.shape == gt_valid.shape, f"Shape mismatch: {ckpt_valid.shape} vs {gt_valid.shape}"

# Check if they are identical
are_same = np.allclose(ckpt_valid, gt_valid, rtol=1e-5, atol=1e-8)
print(f"\nAre they IDENTICAL (rtol=1e-5)? {are_same}")

# Check mean difference
mean_diff = np.abs(ckpt_valid - gt_valid).mean()
print(f"Mean absolute difference: {mean_diff:.10f}")

# Check correlation
ckpt_flat = ckpt_valid.flatten()
gt_flat = gt_valid.flatten()

# Compute cosine similarity for first 10000 points for efficiency
n_samples = min(10000, ckpt_valid.shape[0])
ckpt_norm = ckpt_valid[:n_samples]
gt_norm_samples = gt_valid[:n_samples]

# L2 normalize
ckpt_norm = ckpt_norm / np.linalg.norm(ckpt_norm, axis=1, keepdims=True)
gt_norm_samples = gt_norm_samples / np.linalg.norm(gt_norm_samples, axis=1, keepdims=True)

# Cosine similarity
cos_sim = (ckpt_norm * gt_norm_samples).sum(axis=1).mean()
print(f"Mean cosine similarity (first {n_samples} samples): {cos_sim:.10f}")

# Check if checkpoint is just GT with noise
if are_same or mean_diff < 0.001:
    print("\n⚠️  WARNING: Checkpoint features appear to be GT features!")
    print("   The inference script is NOT using the model predictions!")
elif cos_sim > 0.999:
    print("\n⚠️  WARNING: Checkpoint features are nearly identical to GT!")
    print(f"   Cosine similarity: {cos_sim:.10f}")
else:
    print(f"\n✓ Checkpoint features differ from GT (cosine sim: {cos_sim:.4f})")

# Additional check: sample some features and compare
print("\n" + "=" * 60)
print("Sample Comparison (first 10 features, first 10 dimensions)")
print("=" * 60)

np.set_printoptions(precision=6, suppress=True)

print("\nCheckpoint[0:5, 0:5]:")
print(ckpt_valid[0:5, 0:5])

print("\nGT[0:5, 0:5]:")
print(gt_valid[0:5, 0:5])

print("\nDifference:")
print(ckpt_valid[0:5, 0:5] - gt_valid[0:5, 0:5])

# Check pairwise similarities within checkpoint
print("\n" + "=" * 60)
print("Mode Collapse Check")
print("=" * 60)

# Sample 1000 features
n_check = min(1000, ckpt_valid.shape[0])
sample = ckpt_valid[np.random.choice(ckpt_valid.shape[0], n_check, replace=False)]
sample_norm = sample / np.linalg.norm(sample, axis=1, keepdims=True)

# Random pairs
n_pairs = 10000
idx_i = np.random.choice(n_check, n_pairs)
idx_j = np.random.choice(n_check, n_pairs)

sim = (sample_norm[idx_i] * sample_norm[idx_j]).sum(axis=1)
print(f"Checkpoint pairwise similarity (random {n_pairs} pairs):")
print(f"  Mean: {sim.mean():.6f}")
print(f"  Std: {sim.std():.6f}")
print(f"  Min: {sim.min():.6f}")
print(f"  Max: {sim.max():.6f}")

high_sim_pct = (sim >= 0.95).mean() * 100
print(f"  High similarity % (>=0.95): {high_sim_pct:.2f}%")

if high_sim_pct > 50:
    print("\n⚠️  SEVERE MODE COLLAPSE detected!")
else:
    print(f"\n✓ No severe mode collapse (high sim: {high_sim_pct:.1f}%)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if are_same or mean_diff < 0.001:
    print("The checkpoint contains GT features, NOT model predictions!")
    print("This is a BUG in the inference script.")
elif cos_sim > 0.99:
    print("The checkpoint is nearly identical to GT.")
    print("Either: 1) Model is perfectly predicting GT (unlikely)")
    print("Or:     2) Inference script has a bug")
else:
    print("The checkpoint contains different features from GT.")
    print("This suggests model predictions are being used.")
    if high_sim_pct > 50:
        print("However, there is SEVERE MODE COLLAPSE in the predictions.")
