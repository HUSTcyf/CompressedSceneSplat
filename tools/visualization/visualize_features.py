#!/usr/bin/env python3
"""
Advanced Feature Visualization for 3D Semantic Segmentation

This script demonstrates advanced visualization techniques:
1. PCA projection of high-dimensional language features to 3D colors
2. Feature distribution visualization
3. SVD compression quality comparison

Usage:
    # Visualize language features with PCA
    python tools/visualize_features.py \
        --data_path /path/to/scene \
        --mode pca_colors

    # Compare original vs compressed features
    python tools/visualize_features.py \
        --data_path /path/to/scene \
        --svd_rank 16 \
        --mode compare_svd
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path


def project_features_to_rgb(features, method="pca"):
    """
    Project high-dimensional features to 3D RGB colors.

    Args:
        features: [N, D] numpy array or torch tensor
        method: 'pca', 'umap', or 'norm'

    Returns:
        colors: [N, 3] numpy array of RGB colors (0-1 range)
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    N, D = features.shape

    if method == "norm":
        # Simple normalization: use first 3 dimensions
        feat_3d = features[:, :3]
        # Min-max normalize per dimension
        colors = (feat_3d - feat_3d.min(axis=0)) / (feat_3d.max(axis=0) - feat_3d.min(axis=0) + 1e-8)
        return np.clip(colors, 0, 1)

    elif method == "pca":
        from sklearn.decomposition import PCA

        # PCA to 3 dimensions
        pca = PCA(n_components=3)
        colors = pca.fit_transform(features)

        # Normalize to [0, 1]
        colors = (colors - colors.min(axis=0)) / (colors.max(axis=0) - colors.min(axis=0) + 1e-8)
        return np.clip(colors, 0, 1)

    elif method == "umap":
        try:
            from cuml import UMAP
        except ImportError:
            from umap import UMAP

        # UMAP to 3 dimensions
        umap = UMAP(n_components=3, random_state=42)
        colors = umap.fit_transform(features)

        # Normalize to [0, 1]
        colors = (colors - colors.min(axis=0)) / (colors.max(axis=0) - colors.min(axis=0) + 1e-8)
        return np.clip(colors, 0, 1)

    else:
        raise ValueError(f"Unknown method: {method}")


def visualize_feature_colors(data, method="pca", output_path=None, interactive=False):
    """
    Visualize language features projected to RGB colors.

    Args:
        data: dict with 'coord' and 'lang_feat' keys
        method: Projection method ('pca', 'umap', 'norm')
        output_path: Path to save PLY file
        interactive: Whether to show interactive viewer
    """
    try:
        if interactive:
            from LitePT.utils.visualization import get_point_cloud
        else:
            from pointcept.utils.visualization import save_point_cloud
    except ImportError:
        print("Error: open3d not installed. Run: pip install open3d")
        return

    coord = data["coord"]
    features = data.get("lang_feat")

    if features is None:
        print("Error: No language features found in data")
        return

    print(f"Projecting {features.shape} features to RGB using {method.upper()}...")
    colors = project_features_to_rgb(features, method=method)

    print(f"Visualizing {len(coord)} points")

    if interactive:
        get_point_cloud(coord=coord, color=colors, verbose=True)
    else:
        if output_path is None:
            output_path = f"output/features_{method}.ply"
        save_point_cloud(coord=coord, color=colors, file_path=output_path)
        print(f"Saved to {output_path}")


def visualize_segmentation_with_features(data, method="pca", output_path=None, interactive=False, dataset="scannet"):
    """
    Visualize both semantic segmentation and feature projections side-by-side.

    Args:
        data: dict with 'coord', 'segment', and 'lang_feat' keys
        method: Projection method for features
        output_path: Path prefix for saving
        interactive: Whether to show interactive viewer
        dataset: Dataset type for color mapping
    """
    from tools.visualize_semantic_segmentation import get_scannet_color_map, get_matterport_color_map, labels_to_colors

    try:
        if interactive:
            from LitePT.utils.visualization import get_point_cloud
        else:
            from pointcept.utils.visualization import save_point_cloud
    except ImportError:
        print("Error: open3d not installed. Run: pip install open3d")
        return

    coord = data["coord"]
    segment = data["segment"]
    features = data.get("lang_feat")

    if segment is None:
        print("Warning: No semantic labels found")
        # Show features only
        visualize_feature_colors(data, method, output_path, interactive)
        return

    if features is None:
        print("Warning: No language features found")
        # Show segmentation only
        from tools.visualize_semantic_segmentation import visualize_gt
        visualize_gt(data, output_path, interactive, dataset)
        return

    # Get segmentation colors
    color_map = get_scannet_color_map() if dataset == "scannet" else get_matterport_color_map()
    seg_colors = labels_to_colors(segment, color_map).astype(np.float32) / 255.0

    # Get feature colors
    feat_colors = project_features_to_rgb(features, method=method)

    print(f"Visualizing {len(coord)} points")
    print(f"  - Segmentation: {len(np.unique(segment))} classes")
    print(f"  - Features: {features.shape[1]} dims projected via {method.upper()}")

    if interactive:
        print("\nShowing Segmentation (left) vs Features (right)...")
        get_point_cloud(
            coord=[coord, coord],
            color=[seg_colors, feat_colors],
            verbose=True
        )
    else:
        if output_path is None:
            output_path = "output/comparison"
        save_point_cloud(coord=coord, color=seg_colors, file_path=f"{output_path}_segment.ply")
        save_point_cloud(coord=coord, color=feat_colors, file_path=f"{output_path}_features_{method}.ply")
        print(f"Saved to {output_path}_segment.ply and {output_path}_features_{method}.ply")


def compare_svd_compression(data, svd_ranks=[8, 16, 32], output_path=None, interactive=False):
    """
    Compare different SVD compression ranks.

    Args:
        data: dict with 'coord' key
        svd_ranks: List of SVD ranks to compare
        output_path: Path prefix for saving
        interactive: Whether to show interactive viewer
    """
    try:
        if interactive:
            from LitePT.utils.visualization import get_point_cloud
        else:
            from pointcept.utils.visualization import save_point_cloud
    except ImportError:
        print("Error: open3d not installed. Run: pip install open3d")
        return

    coord = data["coord"]

    # Try to load compressed features for each rank
    features_list = []
    valid_ranks = []

    data_path = Path(data.get("data_path", "."))

    for rank in svd_ranks:
        svd_path = data_path / f"lang_feat_grid_svd_r{rank}.npz"
        if svd_path.exists():
            feat_data = np.load(svd_path)
            if "lang_feat" in feat_data:
                features_list.append(feat_data["lang_feat"])
                valid_ranks.append(rank)
                print(f"Loaded SVD-r{rank} features: {feat_data['lang_feat'].shape}")

    if not features_list:
        print("Error: No SVD-compressed features found")
        return

    # Project each feature set to colors
    color_list = []
    for features in features_list:
        colors = project_features_to_rgb(features, method="pca")
        color_list.append(colors)

    if interactive:
        print(f"\nShowing {len(valid_ranks)} SVD compression ranks side-by-side...")
        print(f"Ranks: {valid_ranks}")
        get_point_cloud(
            coord=[coord] * len(color_list),
            color=color_list,
            verbose=True
        )
    else:
        if output_path is None:
            output_path = "output/svd_comparison"
        for i, (rank, colors) in enumerate(zip(valid_ranks, color_list)):
            save_point_cloud(coord=coord, color=colors, file_path=f"{output_path}_r{rank}.ply")
            print(f"Saved to {output_path}_r{rank}.ply")


def visualize_feature_statistics(data, output_path="output/feature_stats.png"):
    """
    Create statistical plots of language features.

    Args:
        data: dict with 'lang_feat' key
        output_path: Path to save plot
    """
    import matplotlib.pyplot as plt

    features = data.get("lang_feat")

    if features is None:
        print("Error: No language features found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Feature distribution (histogram)
    ax = axes[0, 0]
    ax.hist(features.flatten(), bins=100, alpha=0.7, edgecolor='black')
    ax.set_title("Feature Value Distribution")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Frequency")

    # 2. Per-dimension mean and std
    ax = axes[0, 1]
    dim_means = features.mean(axis=0)
    dim_stds = features.std(axis=0)
    ax.errorbar(range(len(dim_means)), dim_means, yerr=dim_stds, alpha=0.7, capsize=3)
    ax.set_title("Per-Dimension Statistics")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

    # 3. Correlation matrix (first 32 dims)
    ax = axes[1, 0]
    n_dims_show = min(32, features.shape[1])
    corr = np.corrcoef(features[:, :n_dims_show].T)
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(f"Feature Correlation Matrix (First {n_dims_show} Dims)")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Dimension")
    plt.colorbar(im, ax=ax)

    # 4. Singular values (SVD energy)
    ax = axes[1, 1]
    U, S, Vt = np.linalg.svd(features - features.mean(axis=0), full_matrices=False)
    ax.plot(S, 'o-', alpha=0.7)
    ax.set_title("Singular Values (Energy Distribution)")
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular Value")
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature statistics to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Advanced feature visualization")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to scene directory")
    parser.add_argument("--mode", type=str,
                        choices=["pca_colors", "seg_and_features", "compare_svd", "stats"],
                        default="pca_colors",
                        help="Visualization mode")
    parser.add_argument("--method", type=str, choices=["pca", "umap", "norm"], default="pca",
                        help="Feature projection method")
    parser.add_argument("--svd_rank", type=int, nargs="+", default=[8, 16, 32],
                        help="SVD ranks to compare")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path")
    parser.add_argument("--interactive", action="store_true",
                        help="Show interactive viewer")
    parser.add_argument("--dataset", type=str, choices=["scannet", "matterport"], default="scannet",
                        help="Dataset type")

    args = parser.parse_args()

    # Load scene data
    print(f"Loading data from {args.data_path}...")
    from tools.visualize_semantic_segmentation import load_scene_data
    data = load_scene_data(args.data_path, load_features=True)
    data["data_path"] = args.data_path  # Store for SVD loading
    print(f"Loaded {len(data['coord'])} points")

    # Visualize based on mode
    if args.mode == "pca_colors":
        visualize_feature_colors(data, args.method, args.output_path, args.interactive)

    elif args.mode == "seg_and_features":
        visualize_segmentation_with_features(data, args.method, args.output_path, args.interactive, args.dataset)

    elif args.mode == "compare_svd":
        compare_svd_compression(data, args.svd_rank, args.output_path, args.interactive)

    elif args.mode == "stats":
        visualize_feature_statistics(data, args.output_path)


if __name__ == "__main__":
    main()
