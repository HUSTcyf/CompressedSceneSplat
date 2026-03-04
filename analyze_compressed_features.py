#!/usr/bin/env python3
"""
Analyze compressed language features in checkpoint_with_features.pth
to understand class separability and identify improvement directions.
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set style for better visualization
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# Sampling parameters for efficiency
MAX_SAMPLES_PER_CLASS = 5000
TOTAL_SAMPLES_LIMIT = 50000


def load_checkpoint_and_labels(scene_name):
    """Load checkpoint features and corresponding segment labels."""
    t0 = time.time()
    # Load checkpoint
    ckpt_path = f'/new_data/cyf/projects/SceneSplat/output_features/{scene_name}/checkpoint_with_features.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Extract language features (item 0[7]: [N, 16])
    item0 = ckpt[0]
    lang_feat = item0[7] if isinstance(item0[7], np.ndarray) else item0[7].numpy()  # [N, 16]
    valid_mask = item0[8] if isinstance(item0[8], np.ndarray) else item0[8].numpy()  # [N]
    segment_labels = item0[13]  # Already numpy array [N]

    # Filter valid features
    valid_indices = valid_mask > 0
    lang_feat_valid = lang_feat[valid_indices]
    labels_valid = segment_labels[valid_indices]

    # Filter out background/invalid labels (assuming label >= 0 is valid)
    valid_label_mask = labels_valid >= 0
    lang_feat_valid = lang_feat_valid[valid_label_mask]
    labels_valid = labels_valid[valid_label_mask]

    print(f"Scene: {scene_name}")
    print(f"  Total gaussians: {lang_feat.shape[0]}")
    print(f"  Valid features: {lang_feat_valid.shape[0]}")
    print(f"  Feature dimension: {lang_feat_valid.shape[1]}")
    print(f"  Unique labels: {len(np.unique(labels_valid))}")
    print(f"  Loaded in {time.time() - t0:.2f}s")

    return lang_feat_valid, labels_valid


def sample_features(features, labels, max_per_class=MAX_SAMPLES_PER_CLASS, total_limit=TOTAL_SAMPLES_LIMIT):
    """Sample features to reduce computation time."""
    t0 = time.time()
    unique_labels = np.unique(labels)
    sampled_feats = []
    sampled_labels = []

    for label in unique_labels:
        mask = labels == label
        class_feats = features[mask]
        n_samples = min(len(class_feats), max_per_class)

        if n_samples < len(class_feats):
            indices = np.random.choice(len(class_feats), n_samples, replace=False)
            sampled_feats.append(class_feats[indices])
        else:
            sampled_feats.append(class_feats)
        sampled_labels.extend([label] * n_samples)

    sampled_feats = np.vstack(sampled_feats)
    sampled_labels = np.array(sampled_labels)

    # Further limit total samples if needed
    if len(sampled_feats) > total_limit:
        indices = np.random.choice(len(sampled_feats), total_limit, replace=False)
        sampled_feats = sampled_feats[indices]
        sampled_labels = sampled_labels[indices]

    print(f"Sampled {len(sampled_feats)} features from {len(features)} total ({time.time() - t0:.2f}s)")
    return sampled_feats, sampled_labels


def compute_class_statistics(features, labels):
    """Compute per-class statistics."""
    t0 = time.time()
    unique_labels = np.unique(labels)
    stats = {}

    for label in unique_labels:
        mask = labels == label
        class_feats = features[mask]

        # Compute statistics
        mean = np.mean(class_feats, axis=0)
        std = np.std(class_feats, axis=0)
        centroid = mean

        stats[label] = {
            'count': class_feats.shape[0],
            'mean': mean,
            'std': std,
            'centroid': centroid,
            'norm': np.linalg.norm(mean)
        }

    # Print summary
    print("\n" + "="*80)
    print("PER-CLASS STATISTICS")
    print("="*80)
    print(f"{'Label':<8} {'Count':<10} {'Norm':<12} {'Std-Mean':<12}")
    print("-"*50)

    for label in sorted(stats.keys()):
        s = stats[label]
        std_mean = np.mean(s['std'])
        print(f"{label:<8} {s['count']:<10} {s['norm']:<12.4f} {std_mean:<12.4f}")

    print(f"\nComputed in {time.time() - t0:.2f}s")
    return stats


def compute_inter_class_distances(stats):
    """Compute pairwise inter-class distances."""
    t0 = time.time()
    labels = sorted(stats.keys())
    n_classes = len(labels)

    print("\n" + "="*80)
    print("INTER-CLASS COSINE DISTANCES (lower = more similar, higher = more separable)")
    print("="*80)

    # Build centroid matrix
    centroids = np.stack([stats[l]['centroid'] for l in labels])

    # Normalize centroids
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    # Compute cosine distance matrix (1 - cosine similarity)
    cosine_sim = centroids_norm @ centroids_norm.T
    cosine_dist = 1 - cosine_sim

    # Print matrix
    header = f"{'':<10}" + "".join(f"{l:<10}" for l in labels)
    print(header)
    print("-"*80)

    for i, l1 in enumerate(labels):
        row = f"{l1:<10}"
        for j, l2 in enumerate(labels):
            if i == j:
                row += f"{'---':<10}"
            else:
                row += f"{cosine_dist[i, j]:<10.4f}"
        print(row)

    # Summary statistics
    upper_tri = cosine_dist[np.triu_indices(n_classes, k=1)]
    print("\nInter-class distance summary:")
    print(f"  Mean: {np.mean(upper_tri):.4f}")
    print(f"  Std:  {np.std(upper_tri):.4f}")
    print(f"  Min:  {np.min(upper_tri):.4f} (most similar pair)")
    print(f"  Max:  {np.max(upper_tri):.4f} (most separable pair)")

    # Find most similar pairs
    print("\nMost similar class pairs (potential confusion):")
    tri_indices = np.triu_indices(n_classes, k=1)
    sorted_idx = np.argsort(upper_tri)
    n_show = min(5, len(sorted_idx))
    for i in range(n_show):
        idx1, idx2 = tri_indices[0][sorted_idx[i]], tri_indices[1][sorted_idx[i]]
        print(f"  {labels[idx1]} <-> {labels[idx2]}: {upper_tri[sorted_idx[i]]:.4f}")

    print(f"\nComputed in {time.time() - t0:.2f}s")
    return cosine_dist, centroids_norm


def analyze_feature_variance(features, labels):
    """Analyze variance across dimensions."""
    t0 = time.time()
    print("\n" + "="*80)
    print("FEATURE DIMENSION ANALYSIS")
    print("="*80)

    # Global statistics
    global_mean = np.mean(features, axis=0)
    global_std = np.std(features, axis=0)
    global_var = np.var(features, axis=0)

    print("\nPer-dimension statistics (all features):")
    print(f"{'Dim':<6} {'Mean':<12} {'Std':<12} {'Var':<12} {'Var%':<10}")
    print("-"*70)

    total_var = np.sum(global_var)
    for i in range(features.shape[1]):
        var_pct = (global_var[i] / total_var) * 100
        print(f"{i:<6} {global_mean[i]:<12.4f} {global_std[i]:<12.4f} {global_var[i]:<12.6f} {var_pct:<10.2f}%")

    # Check for dead dimensions
    dead_threshold = 0.01 * total_var / features.shape[1]
    dead_dims = np.where(global_var < dead_threshold)[0]
    if len(dead_dims) > 0:
        print(f"\nPotential dead dimensions (variance < {dead_threshold:.6f}): {dead_dims}")
    else:
        print("\nNo dead dimensions detected")

    # Top contributing dimensions
    top_dims = np.argsort(global_var)[::-1][:5]
    print(f"\nTop 5 contributing dimensions: {top_dims}")
    for i, dim in enumerate(top_dims):
        print(f"  {i+1}. Dim {dim}: {global_var[dim]:.6f} ({global_var[dim]/total_var*100:.2f}%)")

    print(f"\nComputed in {time.time() - t0:.2f}s")


def analyze_separability_metrics(features, labels):
    """Compute quantitative separability metrics."""
    t0 = time.time()
    print("\n" + "="*80)
    print("CLASS SEPARABILITY METRICS")
    print("="*80)

    unique_labels = np.unique(labels)
    centroids = []
    intra_dists = []
    counts = []

    for label in unique_labels:
        class_feats = features[labels == label]
        centroids.append(np.mean(class_feats, axis=0))
        counts.append(len(class_feats))

        # Intra-class compactness (std of distances to centroid)
        centroid = np.mean(class_feats, axis=0)
        dists_to_centroid = np.linalg.norm(class_feats - centroid, axis=1)
        intra_dists.append(np.std(dists_to_centroid))

    centroids = np.array(centroids)

    # Inter-class distance (average pairwise distance between centroids)
    n_classes = len(centroids)
    inter_dists = []
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))

    # Compute metrics
    mean_intra = np.mean(intra_dists)
    mean_inter = np.mean(inter_dists)

    # Separability ratio
    separability_ratio = mean_inter / (mean_intra + 1e-8)

    print(f"Number of classes: {n_classes}")
    print(f"Mean intra-class std (compactness): {mean_intra:.4f}")
    print(f"Mean inter-class distance (separation): {mean_inter:.4f}")
    print(f"Separability ratio (inter/intra): {separability_ratio:.4f}")

    if separability_ratio < 1.0:
        print("  WARNING: Classes are poorly separated (ratio < 1.0)")
    elif separability_ratio < 2.0:
        print("  NOTE: Classes have moderate separability (ratio < 2.0)")
    else:
        print("  GOOD: Classes are well separated (ratio >= 2.0)")

    # Compute silhouette score (sampled for efficiency)
    print("\nComputing silhouette score (sampled)...")
    if len(features) > 10000:
        sample_idx = np.random.choice(len(features), 10000, replace=False)
        sample_feats = features[sample_idx]
        sample_labels = labels[sample_idx]
    else:
        sample_feats = features
        sample_labels = labels

    silhouette = silhouette_score(sample_feats, sample_labels, metric='cosine')
    print(f"Silhouette score (cosine): {silhouette:.4f}")
    print("  Range: [-1, 1], higher is better")

    print(f"\nComputed in {time.time() - t0:.2f}s")


def visualize_feature_space(features, labels, output_path):
    """Visualize features using PCA for 2D projection."""
    t0 = time.time()
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Sample for visualization
    if len(features) > 20000:
        sample_idx = np.random.choice(len(features), 20000, replace=False)
        viz_features = features[sample_idx]
        viz_labels = labels[sample_idx]
    else:
        viz_features = features
        viz_labels = labels

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(viz_features)

    # PCA to 2D
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    unique_labels = np.unique(viz_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(unique_labels))))

    # Plot 1: All points
    for i, label in enumerate(unique_labels):
        mask = viz_labels == label
        axes[0].scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors[i % len(colors)]], label=f'Class {label}',
                       alpha=0.5, s=10)

    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('PCA Projection - Sampled Points')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Plot 2: Centroids only (computed from all data)
    centroids = []
    for label in unique_labels:
        class_feats = features[labels == label]
        centroids.append(np.mean(class_feats, axis=0))
    centroids = np.array(centroids)

    centroids_scaled = scaler.transform(centroids)
    centroids_2d = pca.transform(centroids_scaled)

    for i, label in enumerate(unique_labels):
        axes[1].scatter(centroids_2d[i, 0], centroids_2d[i, 1],
                       c=[colors[i % len(colors)]], label=f'Class {label}',
                       s=200, edgecolors='black', linewidths=2)
        axes[1].annotate(f'{label}', (centroids_2d[i, 0], centroids_2d[i, 1]),
                        fontsize=9, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('PCA Projection - Class Centroids')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    print(f"Generated in {time.time() - t0:.2f}s")
    plt.close()


def main():
    """Main analysis function."""
    print("="*80)
    print("COMPRESSED LANGUAGE FEATURE ANALYSIS")
    print("="*80)

    # Scene to analyze
    scene_name = "figurines"

    # Load data
    features, labels = load_checkpoint_and_labels(scene_name)

    # Sample for efficiency
    features_samp, labels_samp = sample_features(features, labels)

    # Check minimum classes
    unique_labels = np.unique(labels_samp)
    if len(unique_labels) < 2:
        print("ERROR: Need at least 2 classes for separability analysis")
        return

    # Run analyses
    stats = compute_class_statistics(features_samp, labels_samp)
    cosine_dist, centroids_norm = compute_inter_class_distances(stats)
    analyze_feature_variance(features_samp, labels_samp)
    analyze_separability_metrics(features_samp, labels_samp)

    # Visualize
    output_dir = Path('/new_data/cyf/projects/SceneSplat/analysis_results')
    output_dir.mkdir(exist_ok=True)
    visualize_feature_space(features_samp, labels_samp, output_dir / f'{scene_name}_pca_visualization.png')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")

    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR IMPROVING CLASS SEPARABILITY")
    print("="*80)

    # Based on analysis
    upper_tri = cosine_dist[np.triu_indices_from(cosine_dist, k=1)]
    if np.min(upper_tri) < 0.1:
        print("1. CRITICAL: Some classes have very similar centroids (distance < 0.1)")
        print("   - Consider using better text prompts for these classes")
        print("   - Apply class-specific contrastive learning during training")

    separability_ratio = np.mean([np.linalg.norm(centroids_norm[i] - centroids_norm[j])
                                   for i in range(len(centroids_norm))
                                   for j in range(i+1, len(centroids_norm))]) / 0.5
    print(f"\n2. Current SVD rank (16) may be too low for preserving fine-grained distinctions")
    print("   - Try increasing SVD rank to 32 or 64")
    print("   - Or use learned projection instead of SVD")

    print("\n3. Procrustes alignment is essential but may not be enough")
    print("   - Ensure Q matrix is computed per-scene with matched labels")
    print("   - Consider learning scene-specific adapters")


if __name__ == '__main__':
    main()
