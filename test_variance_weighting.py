#!/usr/bin/env python3
"""
Test Variance-Based Weighting for SVD-Weighted L1 Loss

This script analyzes the actual variance distribution in GT SVD features
to verify the variance-based weighting approach.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def analyze_gt_variance(data_paths):
    """Analyze variance distribution in GT SVD features."""
    print("=" * 80)
    print("GT SVD Feature Variance Analysis")
    print("=" * 80)

    all_dim_stats = []

    for data_path in data_paths:
        npz_file = f"{data_path}/lang_feat_grid_svd_r16.npz"
        try:
            data = np.load(npz_file)
            compressed = data['compressed']
            all_dim_stats.append(compressed)
            print(f"Loaded: {npz_file} | shape: {compressed.shape}")
        except FileNotFoundError:
            print(f"Skip: {npz_file} not found")
            continue

    if not all_dim_stats:
        print("No valid data found!")
        return None, None

    # Concatenate all data
    all_data = np.concatenate(all_dim_stats, axis=0)
    print(f"\nTotal samples: {all_data.shape[0]:,}")
    print(f"Feature dimensions: {all_data.shape[1]}")

    # Per-dimension variance
    dim_variance = all_data.var(axis=0)
    dim_mean = all_data.mean(axis=0)
    dim_std = all_data.std(axis=0)
    dim_l1 = np.abs(all_data).mean(axis=0)

    print("\n" + "=" * 80)
    print("Per-Dimension Statistics")
    print("=" * 80)
    print(f"{'Dim':<6} {'Mean':>12} {'Std':>12} {'Variance':>12} {'L1':>12} {'RelVar':>10}")
    print("-" * 75)

    total_variance = dim_variance.sum()
    cumulative_variance = 0

    for i in range(16):
        cumulative_variance += dim_variance[i]
        rel_var = (cumulative_variance / total_variance) * 100
        print(f"d[{i:2d}]: {dim_mean[i]:12.6f} {dim_std[i]:12.6f} {dim_variance[i]:12.6f} {dim_l1[i]:12.6f} {rel_var:9.1f}%")

    print("\n" + "=" * 80)
    print("Variance Distribution")
    print("=" * 80)

    # Sort dimensions by variance (descending)
    sorted_indices = np.argsort(dim_variance)[::-1]
    print(f"\nDimensions sorted by variance (descending):")
    for i, idx in enumerate(sorted_indices):
        print(f"  Rank {i+1}: d[{idx}] with variance = {dim_variance[idx]:.6f}")

    # Compute variance-based weights
    variance_min = dim_variance.min()
    variance_max = dim_variance.max()
    base_weight = 1.0
    min_weight = 0.1

    if variance_max > variance_min:
        normalized = (dim_variance - variance_min) / (variance_max - variance_min)
        weights = min_weight + normalized * (base_weight - min_weight)
    else:
        weights = np.full_like(dim_variance, base_weight)

    print(f"\n" + "=" * 80)
    print("Computed Weights (Variance-Based)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  base_weight: {base_weight}")
    print(f"  min_weight: {min_weight}")
    print(f"  variance_range: [{variance_min:.6f}, {variance_max:.6f}]")

    print(f"\n" + "-" * 75)
    print(f"{'Dim':<6} {'Variance':>12} {'Weight':>12} {'CumWeight%':>12}")
    print("-" * 55)

    cumulative_weight = 0
    for i in range(16):
        cumulative_weight += weights[i]
        cum_pct = (cumulative_weight / weights.sum()) * 100
        print(f"d[{i:2d}]: {dim_variance[i]:12.6f} {weights[i]:12.6f} {cum_pct:11.1f}%")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Variance per dimension
    ax = axes[0, 0]
    dims = np.arange(16)
    ax.bar(dims, dim_variance, color='steelblue', alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Variance')
    ax.set_title('Per-Dimension Variance in GT Features')
    ax.set_xticks(dims)
    ax.set_xticklabels([f'd[{i}]' for i in dims], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Variance-based weights
    ax = axes[0, 1]
    ax.bar(dims, weights, color='coral', alpha=0.7)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Max weight')
    ax.axhline(y=0.1, color='orange', linestyle='--', label='Min weight')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Weight')
    ax.set_title('Variance-Based Weights')
    ax.set_xticks(dims)
    ax.set_xticklabels([f'd[{i}]' for i in dims], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Variance vs Weight correlation
    ax = axes[1, 0]
    ax.scatter(dim_variance, weights, c=dims, cmap='viridis', s=100, alpha=0.7)
    ax.set_xlabel('Variance')
    ax.set_ylabel('Weight')
    ax.set_title('Variance vs Weight Correlation')
    ax.grid(alpha=0.3)

    # Add dimension labels to scatter plot
    for i in range(16):
        ax.annotate(f'd[{i}]', (dim_variance[i], weights[i]), fontsize=8, alpha=0.7)

    # Plot 4: Comparison - Static vs Variance weights
    ax = axes[1, 1]
    # Static weights (exponential decay)
    static_weights = np.array([max(0.1, 1.0 * (0.85 ** i)) for i in range(16)])

    x = np.arange(16)
    width = 0.35
    ax.bar(x - width/2, static_weights, width, label='Static (exponential decay)', color='lightblue')
    ax.bar(x + width/2, weights, width, label='Variance-based (data-driven)', color='lightcoral')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Weight')
    ax.set_title('Static vs Variance-Based Weights')
    ax.set_xticks(x)
    ax.set_xticklabels([f'd[{i}]' for i in x], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/new_data/cyf/projects/SceneSplat/variance_based_weights.png', dpi=150)
    print(f"\nVisualization saved to: variance_based_weights.png")

    return dim_variance, weights


def main():
    # Analyze GT variance
    data_paths = [
        '/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train/figurines',
        '/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train/bed',
        '/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train/bench',
        '/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train/lawn',
        '/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train/ramen',
        '/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train/room',
        '/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train/sofa',
        '/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train/teatime',
        '/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train/waldo_kitchen',
    ]

    dim_variance, weights = analyze_gt_variance(data_paths)

    if dim_variance is not None:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"\nVariance statistics:")
        print(f"  Min variance:  {dim_variance.min():.6f} (d[{dim_variance.argmin()}])")
        print(f"  Max variance:  {dim_variance.max():.6f} (d[{dim_variance.argmax()}])")
        print(f"  Mean variance: {dim_variance.mean():.6f}")
        print(f"  Std variance:  {dim_variance.std():.6f}")
        print(f"  Variance ratio (max/min): {dim_variance.max() / dim_variance.min():.2f}x")

        print(f"\nWeight statistics:")
        print(f"  Min weight:    {weights.min():.6f}")
        print(f"  Max weight:    {weights.max():.6f}")
        print(f"  Mean weight:   {weights.mean():.6f}")
        print(f"  Std weight:    {weights.std():.6f}")

        # Count high-weight dimensions
        high_weight_dims = (weights > 0.5).sum()
        mid_weight_dims = ((weights > 0.3) & (weights <= 0.5)).sum()
        low_weight_dims = (weights <= 0.3).sum()

        print(f"\nWeight distribution:")
        print(f"  High (> 0.5):  {high_weight_dims} dimensions")
        print(f"  Mid (0.3-0.5): {mid_weight_dims} dimensions")
        print(f"  Low (< 0.3):   {low_weight_dims} dimensions")

        print("\n" + "=" * 80)
        print("Configuration Recommendations")
        print("=" * 80)
        print("\nTo use static exponential decay instead of variance-based:")
        print("  weight_strategy='static'")
        print("  decay_rate=0.85")
        print("  # Remove variance_momentum parameter")

        print("\nTo adjust variance-based weighting:")
        print("  # More aggressive (focus on high-variance dims):")
        print("  min_weight=0.05  # Lower minimum weight")
        print("  ")
        print("  # More uniform (include low-variance dims):")
        print("  min_weight=0.2   # Higher minimum weight")
        print("  ")
        print("  # Slower EMA adaptation:")
        print("  variance_momentum=0.999  # More stable variance estimates")

if __name__ == '__main__':
    main()
