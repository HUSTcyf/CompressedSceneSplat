#!/usr/bin/env python3
"""
Analyze language feature differences between two checkpoint files.

This script:
1. Loads both checkpoints
2. Extracts non-zero language features
3. Computes L1 distance per dimension
4. Identifies which dimension has the largest L1 distance
"""

import torch
import numpy as np
from pathlib import Path


def analyze_checkpoint_differences(
    path1: str,
    path2: str,
):
    """Analyze differences between two checkpoint files."""

    print("=" * 70)
    print("LANGUAGE FEATURE DIFFERENCE ANALYSIS")
    print("=" * 70)

    # Load checkpoints
    print(f"\nLoading checkpoint 1: {path1}")
    ckpt1 = torch.load(path1, map_location='cpu', weights_only=False)

    print(f"\nLoading checkpoint 2: {path2}")
    ckpt2 = torch.load(path2, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    # Format 1: dict with keys
    # Format 2: tuple/list (data, metadata)
    # Format 3: tensor directly

    def extract_features(ckpt):
        """Extract features from various checkpoint formats."""
        if isinstance(ckpt, dict):
            print(f"Checkpoint is dict with keys: {list(ckpt.keys())}")
            # Try to find features in dict
            for key in ckpt.keys():
                if 'lang_feat' in key.lower() or 'feature' in key.lower():
                    if isinstance(ckpt[key], torch.Tensor):
                        print(f"Found features: key='{key}', shape={ckpt[key].shape}")
                        return ckpt[key]
                elif key == 'output' and isinstance(ckpt[key], dict):
                    if 'feat' in ckpt[key]:
                        print(f"Found features: key='output/feat', shape={ckpt[key]['feat'].shape}")
                        return ckpt[key]['feat']
            # If no specific key found, try to find any tensor with right shape
            for key, val in ckpt.items():
                if isinstance(val, torch.Tensor):
                    print(f"Found tensor: key='{key}', shape={val.shape}")
                    return val
        elif isinstance(ckpt, (tuple, list)):
            print(f"Checkpoint is tuple/list with {len(ckpt)} elements")
            for i, item in enumerate(ckpt):
                print(f"  [{i}] type={type(item).__name__}", end="")
                if isinstance(item, torch.Tensor):
                    print(f", shape={item.shape}")
                elif isinstance(item, dict):
                    print(f", dict with keys={list(item.keys())}")
                    # Check for features in nested dict
                    for key, val in item.items():
                        if isinstance(val, torch.Tensor):
                            print(f"      [{i}][{key}]: shape={val.shape}")
                            if 'feat' in key.lower() or 'lang' in key.lower():
                                return val
                else:
                    print()
            # Assume first tensor is features
            for item in ckpt:
                if isinstance(item, torch.Tensor):
                    print(f"Using first tensor as features: shape={item.shape}")
                    return item
        elif isinstance(ckpt, torch.Tensor):
            print(f"Checkpoint is tensor: shape={ckpt.shape}")
            return ckpt
        return None

    feat1 = extract_features(ckpt1)
    feat2 = extract_features(ckpt2)

    if feat1 is None or feat2 is None:
        print("\n[ERROR] Could not extract features from both checkpoints!")
        return

    # Convert to numpy for easier analysis
    if isinstance(feat1, torch.Tensor):
        feat1 = feat1.detach().cpu().numpy()
    if isinstance(feat2, torch.Tensor):
        feat2 = feat2.detach().cpu().numpy()

    print(f"\nFeature shapes:")
    print(f"  Checkpoint 1: {feat1.shape}")
    print(f"  Checkpoint 2: {feat2.shape}")

    # Find non-zero rows (valid features)
    # A row is non-zero if any dimension has absolute value > threshold
    threshold = 1e-6

    # Check per-row norm
    norm1 = np.linalg.norm(feat1, axis=1) if feat1.ndim > 1 else np.abs(feat1)
    norm2 = np.linalg.norm(feat2, axis=1) if feat2.ndim > 1 else np.abs(feat2)

    valid1 = norm1 > threshold
    valid2 = norm2 > threshold

    print(f"\nNon-zero rows:")
    print(f"  Checkpoint 1: {valid1.sum()} / {len(valid1)} ({valid1.sum()/len(valid1)*100:.1f}%)")
    print(f"  Checkpoint 2: {valid2.sum()} / {len(valid2)} ({valid2.sum()/len(valid2)*100:.1f}%)")

    # Find common valid rows
    valid_common = valid1 & valid2
    print(f"  Common valid: {valid_common.sum()} / {len(valid_common)} ({valid_common.sum()/len(valid_common)*100:.1f}%)")

    if valid_common.sum() == 0:
        print("\n[WARNING] No common valid rows found! Using all rows for comparison.")
        valid_common = np.ones(len(feat1), dtype=bool)

    # Extract valid features
    feat1_valid = feat1[valid_common]
    feat2_valid = feat2[valid_common]

    print(f"\nValid feature shapes:")
    print(f"  Checkpoint 1: {feat1_valid.shape}")
    print(f"  Checkpoint 2: {feat2_valid.shape}")

    # Compute statistics
    print("\n" + "-" * 70)
    print("STATISTICS")
    print("-" * 70)

    print(f"\nCheckpoint 1 (valid rows):")
    print(f"  Mean: {feat1_valid.mean():.6f}")
    print(f"  Std:  {feat1_valid.std():.6f}")
    print(f"  Min:  {feat1_valid.min():.6f}")
    print(f"  Max:  {feat1_valid.max():.6f}")
    print(f"  L2 norm: {np.linalg.norm(feat1_valid):.6f}")

    print(f"\nCheckpoint 2 (valid rows):")
    print(f"  Mean: {feat2_valid.mean():.6f}")
    print(f"  Std:  {feat2_valid.std():.6f}")
    print(f"  Min:  {feat2_valid.min():.6f}")
    print(f"  Max:  {feat2_valid.max():.6f}")
    print(f"  L2 norm: {np.linalg.norm(feat2_valid):.6f}")

    # Compute difference
    diff = feat1_valid - feat2_valid

    print(f"\nDifference (ckpt1 - ckpt2):")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Std:  {diff.std():.6f}")
    print(f"  Min:  {diff.min():.6f}")
    print(f"  Max:  {diff.max():.6f}")
    print(f"  Absolute mean: {np.abs(diff).mean():.6f}")

    # Per-dimension L1 distance
    if feat1_valid.ndim > 1:
        per_dim_l1 = np.abs(feat1_valid - feat2_valid).mean(axis=0)  # [D]
        per_dim_l1_sum = np.abs(feat1_valid - feat2_valid).sum(axis=0)  # [D]

        print("\n" + "-" * 70)
        print("PER-DIMENSION L1 DISTANCE")
        print("-" * 70)

        print(f"\nMean L1 distance per dimension:")
        for d in range(len(per_dim_l1)):
            print(f"  Dim {d:2d}: {per_dim_l1[d]:.6f}")

        print(f"\nTotal L1 distance per dimension:")
        for d in range(len(per_dim_l1_sum)):
            print(f"  Dim {d:2d}: {per_dim_l1_sum[d]:.6f}")

        # Find dimension with maximum L1 distance
        max_dim = np.argmax(per_dim_l1)
        max_l1 = per_dim_l1[max_dim]

        print("\n" + "-" * 70)
        print("MAXIMUM L1 DISTANCE")
        print("-" * 70)
        print(f"\nDimension with LARGEST mean L1 distance: Dim {max_dim}")
        print(f"  Mean L1 distance: {max_l1:.6f}")
        print(f"  Total L1 distance: {per_dim_l1_sum[max_dim]:.6f}")

        # Show distribution for max dimension
        print(f"\nDistribution for Dim {max_dim}:")
        print(f"  Checkpoint 1: mean={feat1_valid[:, max_dim].mean():.6f}, std={feat1_valid[:, max_dim].std():.6f}")
        print(f"  Checkpoint 2: mean={feat2_valid[:, max_dim].mean():.6f}, std={feat2_valid[:, max_dim].std():.6f}")
        print(f"  Difference: mean={diff[:, max_dim].mean():.6f}, std={diff[:, max_dim].std():.6f}")

        # Overall L1 distance
        total_l1 = np.abs(feat1_valid - feat2_valid).sum()
        mean_l1 = np.abs(feat1_valid - feat2_valid).mean()

        print("\n" + "-" * 70)
        print("OVERALL L1 DISTANCE")
        print("-" * 70)
        print(f"\nTotal L1 distance: {total_l1:.6f}")
        print(f"Mean L1 distance: {mean_l1:.6f}")
        print(f"L2 distance: {np.linalg.norm(feat1_valid - feat2_valid):.6f}")

        # Cosine similarity
        feat1_flat = feat1_valid.flatten()
        feat2_flat = feat2_valid.flatten()
        cosine_sim = np.dot(feat1_flat, feat2_flat) / (np.linalg.norm(feat1_flat) * np.linalg.norm(feat2_flat))

        print(f"Cosine similarity: {cosine_sim:.6f}")

    else:
        print("\nFeatures are 1D, showing simple comparison:")
        print(f"  Absolute difference: {np.abs(diff):.6f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    path1 = "/new_data/cyf/projects/SceneSplat/output_features/bed/checkpoint_with_features_p.pth"
    path2 = "/new_data/cyf/projects/SceneSplat/output_features/bed/checkpoint_with_features.pth"

    analyze_checkpoint_differences(path1, path2)
