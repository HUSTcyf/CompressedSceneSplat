#!/usr/bin/env python3
"""
3D Semantic Segmentation Visualization Tool

This script demonstrates how to visualize semantic segmentation results
for 3D Gaussian Splatting point clouds using SceneSplat utilities.

Usage:
    # Visualize ground truth
    python tools/visualize_semantic_segmentation.py \
        --data_path /path/to/scene \
        --mode gt

    # Visualize predictions
    python tools/visualize_semantic_segmentation.py \
        --data_path /path/to/scene \
        --pred_path /path/to/predictions.npy \
        --mode pred

    # Compare side-by-side
    python tools/visualize_semantic_segmentation.py \
        --data_path /path/to/scene \
        --pred_path /path/to/predictions.npy \
        --mode compare
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path


def get_scannet_color_map():
    """
    ScanNet 20-class color map for visualization.
    Returns a dict mapping class IDs to RGB colors.
    """
    colors = {
        0: [0, 0, 0],         # unlabelled/black
        1: [174, 199, 232],   # wall/light blue
        2: [152, 223, 138],   # floor/light green
        3: [31, 119, 180],    # cabinet/blue
        4: [255, 187, 120],   # bed/light orange
        5: [188, 189, 34],    # chair/yellow green
        6: [140, 86, 75],     # table/brown
        7: [255, 152, 150],   # door/light red
        8: [214, 39, 40],     # window/red
        9: [197, 176, 213],   # bookshelf/light purple
        10: [148, 103, 189],  # picture/purple
        11: [196, 156, 148],  # counter/light brown
        12: [23, 190, 207],   # desk/cyan
        13: [247, 182, 210],  # curtain/pink
        14: [219, 219, 141],  # refrigerator/light yellow
        15: [255, 127, 0],    # shower curtain/orange
        16: [158, 218, 229],  # toilet/light cyan
        17: [44, 160, 44],    # sink/green
        18: [112, 128, 144],  # bathtub/gray
        19: [227, 119, 194],  # other furniture/pink
    }
    return colors


def get_matterport_color_map():
    """
    Matterport3D color map (simplified).
    Returns a dict mapping class IDs to RGB colors.
    """
    colors = {
        0: [0, 0, 0],         # other/black
        1: [174, 199, 232],   # wall
        2: [152, 223, 138],   # floor
        3: [31, 119, 180],    # chair
        4: [255, 187, 120],   # door
        5: [188, 189, 34],    # table
        6: [140, 86, 75],     # window
        7: [255, 152, 150],   # bookshelf
        8: [214, 39, 40],     # picture
        9: [197, 176, 213],   # counter
        10: [148, 103, 189],  # desk
    }
    return colors


def labels_to_colors(labels, color_map, num_classes=160):
    """
    Convert semantic labels to RGB colors.

    Args:
        labels: [N] numpy array of semantic labels
        color_map: dict mapping class_id -> [R, G, B]
        num_classes: total number of classes (for random colors)

    Returns:
        colors: [N, 3] numpy array of RGB colors (0-255 range)
    """
    colors = np.zeros((len(labels), 3), dtype=np.uint8)

    for i, label in enumerate(labels):
        if label in color_map:
            colors[i] = color_map[label]
        else:
            # Generate random color for unmapped classes
            np.random.seed(int(label) % 1000)
            colors[i] = np.random.randint(0, 256, 3)

    return colors


def load_scene_data(data_path, load_features=False):
    """
    Load 3DGS scene data from disk.

    Args:
        data_path: Path to scene directory
        load_features: Whether to load language features

    Returns:
        dict with keys: coord, color, opacity, quat, scale, segment, lang_feat
    """
    data_path = Path(data_path)
    data = {}

    # Load required files
    data["coord"] = np.load(data_path / "coord.npy")
    data["color"] = np.load(data_path / "color.npy")
    data["opacity"] = np.load(data_path / "opacity.npy")
    data["quat"] = np.load(data_path / "quat.npy")
    data["scale"] = np.load(data_path / "scale.npy")

    # Load optional files
    segment_path = data_path / "segment.npy"
    if segment_path.exists():
        data["segment"] = np.load(segment_path)
    else:
        data["segment"] = None

    if load_features:
        lang_feat_path = data_path / "lang_feat.npy"
        if lang_feat_path.exists():
            data["lang_feat"] = np.load(lang_feat_path)
        else:
            # Try compressed features
            for svd_rank in [8, 16, 32]:
                svd_path = data_path / f"lang_feat_grid_svd_r{svd_rank}.npz"
                if svd_path.exists():
                    data["lang_feat"] = np.load(svd_path)["lang_feat"]
                    print(f"Loaded SVD-compressed features (r={svd_rank})")
                    break

    return data


def visualize_gt(data, output_path=None, interactive=False, dataset="scannet"):
    """
    Visualize ground truth semantic segmentation.

    Args:
        data: dict with 'coord' and 'segment' keys
        output_path: Path to save PLY file (optional)
        interactive: Whether to show interactive viewer
        dataset: 'scannet' or 'matterport'
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
    segment = data["segment"]

    if segment is None:
        print("Warning: No ground truth labels found")
        return

    # Get color map
    color_map = get_scannet_color_map() if dataset == "scannet" else get_matterport_color_map()

    # Convert labels to colors
    colors = labels_to_colors(segment, color_map)
    colors = colors.astype(np.float32) / 255.0  # Normalize to [0, 1]

    print(f"Visualizing {len(coord)} points with {len(np.unique(segment))} classes")

    if interactive:
        get_point_cloud(coord=coord, color=colors, verbose=True)
    else:
        if output_path is None:
            output_path = "output/gt_segmentation.ply"
        save_point_cloud(coord=coord, color=colors, file_path=output_path)
        print(f"Saved to {output_path}")


def visualize_prediction(data, predictions, output_path=None, interactive=False, dataset="scannet"):
    """
    Visualize predicted semantic segmentation.

    Args:
        data: dict with 'coord' key
        predictions: [N] numpy array of predicted labels
        output_path: Path to save PLY file (optional)
        interactive: Whether to show interactive viewer
        dataset: 'scannet' or 'matterport'
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

    # Get color map
    color_map = get_scannet_color_map() if dataset == "scannet" else get_matterport_color_map()

    # Convert labels to colors
    colors = labels_to_colors(predictions, color_map)
    colors = colors.astype(np.float32) / 255.0  # Normalize to [0, 1]

    print(f"Visualizing {len(coord)} points with {len(np.unique(predictions))} classes")

    if interactive:
        get_point_cloud(coord=coord, color=colors, verbose=True)
    else:
        if output_path is None:
            output_path = "output/pred_segmentation.ply"
        save_point_cloud(coord=coord, color=colors, file_path=output_path)
        print(f"Saved to {output_path}")


def visualize_comparison(data, predictions, output_path=None, interactive=False, dataset="scannet"):
    """
    Visualize ground truth vs predictions side-by-side.

    Args:
        data: dict with 'coord' and 'segment' keys
        predictions: [N] numpy array of predicted labels
        output_path: Path prefix for saving PLY files (optional)
        interactive: Whether to show interactive viewer
        dataset: 'scannet' or 'matterport'
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
    gt = data["segment"]

    if gt is None:
        print("Warning: No ground truth labels found for comparison")
        return

    # Get color map
    color_map = get_scannet_color_map() if dataset == "scannet" else get_matterport_color_map()

    # Convert labels to colors
    gt_colors = labels_to_colors(gt, color_map).astype(np.float32) / 255.0
    pred_colors = labels_to_colors(predictions, color_map).astype(np.float32) / 255.0

    # Calculate accuracy
    accuracy = (gt == predictions).mean() * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Show class-wise stats
    unique_classes = np.unique(gt)
    print(f"\nPer-class IoU:")
    for cls in unique_classes:
        if cls == -1:  # ignore index
            continue
        mask = (gt == cls)
        if mask.sum() > 0:
            iou = (gt[mask] == predictions[mask]).mean() * 100
            print(f"  Class {cls}: {iou:.2f}%")

    if interactive:
        print("\nShowing GT (left) vs Prediction (right)...")
        get_point_cloud(
            coord=[coord, coord],
            color=[gt_colors, pred_colors],
            verbose=True
        )
    else:
        if output_path is None:
            output_path = "output/comparison"
        save_point_cloud(coord=coord, color=gt_colors, file_path=f"{output_path}_gt.ply")
        save_point_cloud(coord=coord, color=pred_colors, file_path=f"{output_path}_pred.ply")
        print(f"Saved to {output_path}_gt.ply and {output_path}_pred.ply")


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D semantic segmentation results")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to scene directory containing .npy files")
    parser.add_argument("--pred_path", type=str, default=None,
                        help="Path to prediction .npy file (for pred/compare modes)")
    parser.add_argument("--mode", type=str, choices=["gt", "pred", "compare"], default="gt",
                        help="Visualization mode: gt (ground truth), pred (prediction), compare (both)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for saving PLY files")
    parser.add_argument("--interactive", action="store_true",
                        help="Show interactive Open3D viewer")
    parser.add_argument("--dataset", type=str, choices=["scannet", "matterport"], default="scannet",
                        help="Dataset type for color mapping")

    args = parser.parse_args()

    # Load scene data
    print(f"Loading data from {args.data_path}...")
    data = load_scene_data(args.data_path, load_features=False)
    print(f"Loaded {len(data['coord'])} points")

    # Load predictions if needed
    predictions = None
    if args.mode in ["pred", "compare"]:
        if args.pred_path is None:
            print("Error: --pred_path required for pred/compare modes")
            return
        predictions = np.load(args.pred_path)
        print(f"Loaded predictions: {predictions.shape}")

    # Visualize
    if args.mode == "gt":
        visualize_gt(data, args.output_path, args.interactive, args.dataset)
    elif args.mode == "pred":
        visualize_prediction(data, predictions, args.output_path, args.interactive, args.dataset)
    elif args.mode == "compare":
        visualize_comparison(data, predictions, args.output_path, args.interactive, args.dataset)


if __name__ == "__main__":
    main()
