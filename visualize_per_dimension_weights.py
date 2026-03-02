#!/usr/bin/env python3
"""
Per-Dimension Weight Visualization for SVD-Weighted L1 Loss

This script visualizes the weight distribution across dimensions
for the SVD-weighted L1 loss configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def compute_svd_weights(num_dims=16, base_weight=1.0, decay_rate=0.85, min_weight=0.1):
    """Compute SVD weights based on exponential decay."""
    weights = []
    for i in range(num_dims):
        w = base_weight * (decay_rate ** i)
        w = max(w, min_weight)  # Clamp to minimum weight
        weights.append(w)
    return np.array(weights)

def main():
    # Configuration (matching the config file)
    num_dims = 16
    base_weight = 1.0
    decay_rate = 0.85
    min_weight = 0.1

    # Compute weights
    weights = compute_svd_weights(num_dims, base_weight, decay_rate, min_weight)

    print("=" * 80)
    print("Per-Dimension Weight Configuration for SVD-Weighted L1 Loss")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  num_dims:    {num_dims}")
    print(f"  base_weight: {base_weight}")
    print(f"  decay_rate:  {decay_rate}")
    print(f"  min_weight:  {min_weight}")

    print(f"\nResulting weights:")
    print("-" * 80)
    print(f"{'Dim':<6} {'Weight':>10} {'Cumulative%':>12} {'Energy%':>10}")
    print("-" * 40)

    cumulative = 0
    for i in range(num_dims):
        cumulative += weights[i]
        cum_pct = (cumulative / weights.sum()) * 100
        # Typical SVD energy distribution (for reference)
        energy_pct = max(0, 85 * (0.5 ** i))  # Approximate SVD energy
        print(f'd[{i:2d}]: {weights[i]:10.4f} {cum_pct:11.1f}% {energy_pct:9.1f}%')

    print("-" * 40)
    print(f'Total: {weights.sum():10.4f}')

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Bar chart of weights
    ax = axes[0]
    dims = np.arange(num_dims)
    ax.bar(dims, weights, color='steelblue', alpha=0.7)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Base weight (d[0])')
    ax.axhline(y=0.1, color='orange', linestyle='--', label='Min weight')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Weight')
    ax.set_title('Per-Dimension Weights')
    ax.set_xticks(dims)
    ax.set_xticklabels([f'd[{i}]' for i in dims], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Cumulative weight
    ax = axes[1]
    cumulative_weights = np.cumsum(weights) / weights.sum() * 100
    ax.plot(dims, cumulative_weights, marker='o', color='green', linewidth=2)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80%')
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95%')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Cumulative Weight (%)')
    ax.set_title('Cumulative Weight Distribution')
    ax.set_xticks(dims)
    ax.set_xticklabels([f'd[{i}]' for i in dims], rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Weight vs SVD Energy comparison
    ax = axes[2]
    # Approximate SVD energy distribution (exponential)
    svd_energy = np.array([85 * (0.5 ** i) for i in range(num_dims)])
    svd_energy = np.maximum(svd_energy, 0.1)  # Minimum 0.1%

    # Normalize both for comparison
    weights_norm = weights / weights.max() * 100
    energy_norm = svd_energy / svd_energy.max() * 100

    ax.plot(dims, weights_norm, marker='o', label='Weight (normalized)', linewidth=2)
    ax.plot(dims, energy_norm, marker='s', label='SVD Energy (normalized)', linewidth=2, linestyle='--')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Normalized Value (%)')
    ax.set_title('Weight vs SVD Energy Distribution')
    ax.set_xticks(dims)
    ax.set_xticklabels([f'd[{i}]' for i in dims], rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/new_data/cyf/projects/SceneSplat/per_dimension_weights.png', dpi=150)
    print(f"\nVisualization saved to: per_dimension_weights.png")

    # Additional analysis
    print("\n" + "=" * 80)
    print("Weight Analysis")
    print("=" * 80)

    # Count dimensions by weight range
    high_weight = (weights > 0.5).sum()
    mid_weight = ((weights > 0.2) & (weights <= 0.5)).sum()
    low_weight = (weights <= 0.2).sum()

    print(f"\nWeight distribution:")
    print(f"  High (> 0.5):  {high_weight} dimensions (d[0] to d[{high_weight-1}])")
    print(f"  Mid (0.2-0.5): {mid_weight} dimensions (d[{high_weight}] to d[{high_weight+mid_weight-1}])")
    print(f"  Low (< 0.2):   {low_weight} dimensions (d[{high_weight+mid_weight}] to d[15])")

    print(f"\nEffect on training:")
    print(f"  - Top 5 dimensions (d[0]-d[4]) account for {(weights[:5].sum()/weights.sum()*100):.1f}% of total weight")
    print(f"  - Bottom 6 dimensions (d[10]-d[15]) account for {(weights[10:].sum()/weights.sum()*100):.1f}% of total weight")
    print(f"  - Minimum weight is {min_weight} (prevents zero gradients for low-energy dims)")

    print("\n" + "=" * 80)
    print("Configuration for different decay rates")
    print("=" * 80)
    print(f"{'Decay Rate':<12} {'d[0]':>8} {'d[5]':>8} {'d[10]':>8} {'d[15]':>8} {'Range':>8}")
    print("-" * 60)
    for dr in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        w = compute_svd_weights(num_dims, base_weight, dr, min_weight)
        print(f'{dr:<12.2f} {w[0]:8.3f} {w[5]:8.3f} {w[10]:8.3f} {w[15]:8.3f} {w.max()-w.min():8.3f}')

    print("\nRecommendation:")
    print("  - decay_rate=0.85: Balanced (current config)")
    print("  - decay_rate=0.80: More aggressive weighting (focus on early dims)")
    print("  - decay_rate=0.90: More uniform weighting (include later dims)")

if __name__ == '__main__':
    main()
