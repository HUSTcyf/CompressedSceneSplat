#!/usr/bin/env python3
"""
ScanNet++ Mesh Semantic Segmentation Visualization Tool

This script visualizes semantic segmentation results for ScanNet++ mesh data using SceneSplat's neighbor voting mechanism.

Features:
- Ground Truth visualization from labeled mesh files
- SceneSplat prediction visualization with neighbor voting (k=25)
- Intermediate state: partial correction by fixing N random error classes
- Top-down view rendering for easy comparison
- Statistical analysis and comparison

Usage:
    # Visualize GT only
    python tools/visualize_semantic_segmentation.py \\
        --scannet_mesh_path /path/to/scene_vh_clean_2.labels.ply \\
        --mode gt

    # Visualize predictions with SceneSplat neighbor voting
    python tools/visualize_semantic_segmentation.py \\
        --scannet_mesh_path /path/to/scene_vh_clean_2.labels.ply \\
        --gaussian_data_path /path/to/gaussian/scene \\
        --mode pred

    # Full comparison: GT + Pred + Intermediate (partial correction)
    python tools/visualize_semantic_segmentation.py \\
        --scannet_mesh_path /path/to/scene_vh_clean_2.labels.ply \\
        --gaussian_data_path /path/to/gaussian/scene \\
        --mode compare \\
        --num_correct_classes 5

    # Using scene ID (auto-detect paths)
    python tools/visualize_semantic_segmentation.py \\
        --scene_id scene0000_00 \\
        --mode compare \\
        --num_correct_classes 5
"""

import argparse
import sys
import json
from pathlib import Path

# Import PROJECT_ROOT - handle both script and module execution
try:
    from .. import PROJECT_ROOT  # Relative import when run as module
except ImportError:
    # Fallback when run as script: add parent dir to sys.path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import trimesh

from pointcept.utils.misc import neighbor_voting


def load_mesh_with_labels(mesh_path):
    """Load mesh with vertex colors (labels encoded as colors)"""
    print(f"Loading mesh from: {mesh_path}")

    if not Path(mesh_path).exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    mesh = trimesh.load(mesh_path)
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Faces: {len(mesh.faces):,}")

    if hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors
        print(f"  Has vertex colors: {colors.shape}")
        return mesh, colors
    else:
        print(f"  No vertex colors found")
        return mesh, None


def load_gaussian_data(scene_path):
    """Load Gaussian data and labels from preprocessed scene"""
    scene_path = Path(scene_path)

    coord_path = scene_path / "coord.npy"
    segment_path = scene_path / "segment.npy"

    if not coord_path.exists():
        raise FileNotFoundError(f"Coordinates not found: {coord_path}")

    coords = np.load(coord_path).astype(np.float32)

    if segment_path.exists():
        labels = np.load(segment_path)
        print(f"  Original labels shape: {labels.shape}")

        # Handle different label shapes
        if labels.ndim > 1:
            if labels.shape[1] == 1:
                labels = labels[:, 0]
            elif labels.shape[1] == 3:
                labels = labels[:, 0]
        labels = labels.flatten()
    else:
        raise FileNotFoundError(f"Segment labels not found: {segment_path}")

    print(f"Loaded Gaussian data:")
    print(f"  Points: {len(coords):,}")
    print(f"  Labels: {len(labels):,}")
    print(f"  Unique labels: {len(np.unique(labels[labels >= 0]))}")

    # Ensure labels and coords match
    if len(labels) != len(coords):
        print(f"  Warning: Label count ({len(labels):,}) != coord count ({len(coords):,})")
        min_len = min(len(labels), len(coords))
        coords = coords[:min_len]
        labels = labels[:min_len]
        print(f"  Truncated to: {min_len:,}")

    return coords, labels


def extract_gt_labels_from_ply_labels(mesh_path):
    """Extract GT labels from ScanNet++ .labels.ply file (colors encode labels)"""
    mesh = trimesh.load(mesh_path)

    if hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors

        # Generate standard color map
        color_map = get_scannetpp_color_map()

        # Create reverse lookup: color_tuple -> label
        color_to_label = {}
        for label, color in color_map.items():
            color_tuple = tuple(color)
            color_to_label[color_tuple] = label

        # Extract labels from vertex colors
        labels = np.zeros(len(colors), dtype=np.int32)
        unknown_count = 0

        for i, color in enumerate(colors):
            color_tuple = tuple(color[:3])  # RGB only, ignore alpha
            if color_tuple in color_to_label:
                labels[i] = color_to_label[color_tuple]
            else:
                # Find closest color
                min_dist = float('inf')
                best_label = 0
                for label, ref_color in color_map.items():
                    dist = np.sum((np.array(color_tuple) - ref_color) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_label = label
                labels[i] = best_label
                unknown_count += 1

        print(f"  Extracted labels from GT mesh:")
        print(f"    Total vertices: {len(labels):,}")
        print(f"    Unique classes: {len(np.unique(labels))}")
        print(f"    Unknown colors: {unknown_count}")

        return labels

    return None


def apply_neighbor_voting_to_mesh(
    gaussian_coords,
    gaussian_labels,
    mesh_vertices,
    vote_k=25,
    num_classes=100,
    ignore_label=-1
):
    """
    Apply SceneSplat's neighbor_voting mechanism to map Gaussian labels to mesh vertices

    This is SceneSplat's official neighbor voting implementation used during inference
    to improve segmentation quality through KNN-based majority voting.
    """
    print(f"\nApplying SceneSplat neighbor voting (k={vote_k})...")

    # Filter valid labels
    valid_mask = gaussian_labels >= 0
    valid_coords = gaussian_coords[valid_mask]
    valid_labels = gaussian_labels[valid_mask]

    print(f"  Valid labeled points: {np.sum(valid_mask):,} / {len(gaussian_labels):,}")

    # Downsample for speed
    max_points = 200000
    if len(valid_coords) > max_points:
        print(f"  Downsampling from {len(valid_coords):,} to {max_points:,} points...")
        indices = np.random.choice(len(valid_coords), max_points, replace=False)
        valid_coords = valid_coords[indices]
        valid_labels = valid_labels[indices]

    print(f"  Using {len(valid_coords):,} reference points for KNN query...")
    print(f"  Querying {len(mesh_vertices):,} mesh vertices...")

    # Use SceneSplat's neighbor_voting function
    mesh_labels = neighbor_voting(
        coords=valid_coords,
        pred=valid_labels.astype(np.int32),
        vote_k=vote_k,
        ignore_label=ignore_label,
        num_classes=num_classes,
        query_coords=mesh_vertices
    )

    print(f"  Mapping complete:")
    print(f"    Total vertices: {len(mesh_labels):,}")
    print(f"    Valid labels: {np.sum(mesh_labels >= 0):,}")
    print(f"    Invalid labels: {np.sum(mesh_labels < 0):,}")
    print(f"    Unique classes: {len(np.unique(mesh_labels[mesh_labels >= 0]))}")

    return mesh_labels


def create_intermediate_labels(pred_labels, gt_labels, num_wrong_classes=5, seed=None):
    """
    Create intermediate state labels: randomly select N error classes and replace with GT labels

    Args:
        pred_labels: Predicted labels (M,)
        gt_labels: Ground truth labels (M,)
        num_wrong_classes: Number of error classes to correct
        seed: Random seed for reproducibility

    Returns:
        intermediate_labels: Intermediate state labels
        corrected_classes: List of corrected classes
        statistics: Statistics dictionary
    """
    if seed is not None:
        np.random.seed(seed)

    print(f"\nCreating intermediate state (randomly selecting {num_wrong_classes} error classes to correct)...")

    # Find prediction errors
    error_mask = (pred_labels != gt_labels) & (pred_labels >= 0) & (gt_labels >= 0)
    error_count = np.sum(error_mask)

    print(f"  Error vertices: {error_count:,} / {len(pred_labels):,} ({error_count/len(pred_labels)*100:.2f}%)")

    if error_count == 0:
        print("  No prediction errors, returning original predictions")
        return pred_labels.copy(), [], {"total_errors": 0, "corrected_vertices": 0}

    # Find which classes were predicted incorrectly
    error_pred_classes = pred_labels[error_mask]

    # Count errors per class
    unique_pred_classes, pred_counts = np.unique(error_pred_classes, return_counts=True)

    print(f"  Classes with prediction errors: {len(unique_pred_classes)}")

    # Sort by error count (descending)
    sorted_indices = np.argsort(pred_counts)[::-1]

    # Select N random error classes (from those with most errors)
    num_to_select = min(num_wrong_classes, len(unique_pred_classes))

    # Randomly select from top-k error classes
    top_k = min(10, len(unique_pred_classes))
    candidate_indices = sorted_indices[:top_k]

    if len(candidate_indices) > num_to_select:
        selected_indices = np.random.choice(candidate_indices, num_to_select, replace=False)
    else:
        selected_indices = candidate_indices

    classes_to_correct = unique_pred_classes[selected_indices]

    print(f"  Selected {len(classes_to_correct)} classes to correct:")
    for cls in classes_to_correct:
        cls_errors = np.sum((pred_labels == cls) & (pred_labels != gt_labels) & (gt_labels >= 0))
        print(f"    Class {cls}: {cls_errors:,} error predictions")

    # Create intermediate state labels
    intermediate_labels = pred_labels.copy()

    # Replace selected error classes with GT labels
    corrected_mask = np.zeros(len(intermediate_labels), dtype=bool)

    for cls in classes_to_correct:
        # Find all vertices predicted as this class with wrong prediction
        mask = (pred_labels == cls) & (pred_labels != gt_labels) & (gt_labels >= 0)
        intermediate_labels[mask] = gt_labels[mask]
        corrected_mask |= mask

    corrected_count = np.sum(corrected_mask)

    # Calculate statistics after correction
    new_error_mask = (intermediate_labels != gt_labels) & (intermediate_labels >= 0) & (gt_labels >= 0)
    new_error_count = np.sum(new_error_mask)

    statistics = {
        "total_errors": int(error_count),
        "corrected_classes": [int(c) for c in classes_to_correct],
        "corrected_vertices": int(corrected_count),
        "remaining_errors": int(new_error_count),
        "accuracy_before": float(1 - error_count / len(pred_labels)),
        "accuracy_after": float(1 - new_error_count / len(pred_labels)),
        "accuracy_improvement": float((error_count - new_error_count) / len(pred_labels))
    }

    print(f"\n  Correction statistics:")
    print(f"    Corrected vertices: {corrected_count:,}")
    print(f"    Remaining errors: {new_error_count:,}")
    print(f"    Accuracy improvement: {statistics['accuracy_improvement']*100:.2f}%")
    print(f"    Accuracy before: {statistics['accuracy_before']*100:.2f}%")
    print(f"    Accuracy after: {statistics['accuracy_after']*100:.2f}%")

    return intermediate_labels, list(classes_to_correct), statistics


def get_scannetpp_color_map():
    """Get ScanNet++ color mapping (100 classes)"""
    color_map = {}
    for i in range(100):
        np.random.seed(i * 42 + 123)
        color_map[i] = np.random.randint(50, 256, 3)
    return color_map


def labels_to_colors(labels, color_map, ignore_label=-1):
    """Convert labels to RGB colors"""
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        label_int = int(label)
        if label_int == ignore_label:
            colors[i] = [128, 128, 128]  # Gray for invalid
        elif label_int in color_map:
            colors[i] = color_map[label_int]
        else:
            np.random.seed(label_int % 1000)
            colors[i] = np.random.randint(50, 256, 3)
    return colors


def render_top_view(mesh, output_path, width=1920, height=1080, tight_bbox=True):
    """Render top-down view of mesh with no borders and tight bounding box"""
    print(f"\nRendering top view to: {output_path}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Create figure with no margins
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        vertices = mesh.vertices
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
        else:
            colors = np.ones((len(vertices), 3))

        # Downsample for faster rendering
        step = max(1, len(vertices) // 100000)
        vertices_sampled = vertices[::step]
        colors_sampled = colors[::step]

        print(f"  Downsampled rendering: {len(vertices_sampled):,} / {len(vertices):,} points")

        ax.scatter(vertices_sampled[:, 0], vertices_sampled[:, 1], vertices_sampled[:, 2],
                  c=colors_sampled, s=0.5, alpha=0.6)

        # Top-down view
        ax.view_init(elev=90, azim=-90)

        # Set axis limits to match bounding box
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        ax.set_xlim(bbox_min[0], bbox_max[0])
        ax.set_ylim(bbox_min[1], bbox_max[1])
        ax.set_zlim(bbox_min[2], bbox_max[2])

        # Remove all axes, ticks, and borders
        ax.set_axis_off()
        ax.grid(False)

        # Remove pane backgrounds
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Make panes transparent
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.zaxis.set_visible(False)

        # Set white background
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Remove margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        # Save with tight bounding box
        save_kwargs = {'facecolor': 'white', 'dpi': 100}
        if tight_bbox:
            save_kwargs['bbox_inches'] = 'tight'
            save_kwargs['pad_inches'] = 0

        plt.savefig(output_path, **save_kwargs)
        plt.close()

        print(f"  ✓ Rendered: {output_path}")
    except Exception as e:
        print(f"  ✗ Render failed: {e}")


def find_scene_paths(scene_id, base_path="/new_data/cyf/Datasets/ScanNet/scans"):
    """Auto-detect mesh paths from scene ID"""
    base_path = Path(base_path)
    scene_dir = base_path / scene_id

    gt_mesh_path = scene_dir / f"{scene_id}_vh_clean_2.labels.ply"
    mesh_path = scene_dir / f"{scene_id}_vh_clean_2.ply"

    return {
        "gt_mesh": str(gt_mesh_path) if gt_mesh_path.exists() else None,
        "mesh": str(mesh_path) if mesh_path.exists() else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ScanNet++ mesh semantic segmentation with SceneSplat neighbor voting",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input paths
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--scannet_mesh_path",
        type=str,
        help="Path to ScanNet mesh with labels (.labels.ply)"
    )
    input_group.add_argument(
        "--scene_id",
        type=str,
        help="Scene ID (e.g., scene0000_00), will auto-detect paths"
    )

    parser.add_argument(
        "--mesh_path",
        type=str,
        help="Path to original mesh without labels (for predictions)"
    )
    parser.add_argument(
        "--gaussian_data_path",
        type=str,
        help="Path to Gaussian data directory (containing coord.npy, segment.npy)"
    )
    parser.add_argument(
        "--scannet_base_path",
        type=str,
        default="/new_data/cyf/Datasets/ScanNet/scans",
        help="Base path for ScanNet scans directory"
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gt", "pred", "compare"],
        default="compare",
        help="Visualization mode: gt (ground truth only), pred (prediction only), compare (all three)"
    )

    # Intermediate state options
    parser.add_argument(
        "--num_correct_classes",
        type=int,
        default=5,
        help="Number of error classes to correct for intermediate state (default: 5)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    # Neighbor voting options
    parser.add_argument(
        "--vote_k",
        type=int,
        default=25,
        help="Number of neighbors for voting (SceneSplat default: 25)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=100,
        help="Number of semantic classes (default: 100 for ScanNet++)"
    )

    # Rendering options
    parser.add_argument(
        "--render_width",
        type=int,
        default=1920,
        help="Render image width (default: 1920)"
    )
    parser.add_argument(
        "--render_height",
        type=int,
        default=1080,
        help="Render image height (default: 1080)"
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Skip rendering images"
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="Scene name for output files (default: extracted from scene_id or mesh path)"
    )

    args = parser.parse_args()

    # Determine scene name
    if args.scene_name:
        scene_name = args.scene_name
    elif args.scene_id:
        scene_name = args.scene_id
    else:
        scene_name = Path(args.scannet_mesh_path).parent.name

    # Determine paths
    if args.scene_id:
        paths = find_scene_paths(args.scene_id, args.scannet_base_path)
        gt_mesh_path = paths["gt_mesh"]
        mesh_path = args.mesh_path or paths["mesh"]
        if not gt_mesh_path:
            raise ValueError(f"Could not find GT mesh for scene: {args.scene_id}")
    else:
        gt_mesh_path = args.scannet_mesh_path
        mesh_path = args.mesh_path

    gaussian_data_path = args.gaussian_data_path

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("ScanNet++ Mesh Semantic Segmentation Visualization")
    print("="*70)
    print(f"Scene: {scene_name}")
    print(f"GT Mesh: {gt_mesh_path}")
    if mesh_path:
        print(f"Mesh: {mesh_path}")
    if gaussian_data_path:
        print(f"Gaussian Data: {gaussian_data_path}")
    print(f"Mode: {args.mode}")
    print("="*70)

    color_map = get_scannetpp_color_map()
    results = {}

    # ===== 1. Load GT =====
    if args.mode in ["gt", "compare"]:
        print("\n[1/4] Loading Ground Truth...")
        gt_mesh = load_mesh_with_labels(gt_mesh_path)[0]

        # Extract GT labels
        gt_labels = extract_gt_labels_from_ply_labels(gt_mesh_path)
        results["gt_labels"] = gt_labels

        # Export GT mesh
        gt_output = output_dir / f"{scene_name}_gt_colored.ply"
        print(f"\nExporting GT mesh to: {gt_output}")
        gt_mesh.export(gt_output)
        results["gt_mesh"] = str(gt_output)

        # Render GT
        if not args.no_render:
            gt_render_output = output_dir / f"{scene_name}_gt_topview.png"
            render_top_view(gt_mesh, gt_render_output)
            results["gt_render"] = str(gt_render_output)

    # ===== 2. Load base mesh =====
    if args.mode in ["pred", "compare"] and gaussian_data_path:
        print("\n[2/4] Loading base mesh...")
        if mesh_path and Path(mesh_path).exists():
            pred_mesh = trimesh.load(mesh_path)
            print(f"  Vertices: {len(pred_mesh.vertices):,}")
        else:
            print("  Using GT mesh geometry")
            pred_mesh = trimesh.load(gt_mesh_path)
            pred_mesh.visual = trimesh.visual.ColorVisuals()

        mesh_vertices = pred_mesh.vertices
    else:
        mesh_vertices = None
        pred_mesh = None

    # ===== 3. Load and apply predictions =====
    if args.mode in ["pred", "compare"] and gaussian_data_path and mesh_vertices is not None:
        print("\n[3/4] Loading Gaussian predictions and applying SceneSplat neighbor voting...")
        gaussian_coords, gaussian_labels = load_gaussian_data(gaussian_data_path)

        # Apply neighbor voting
        pred_labels = apply_neighbor_voting_to_mesh(
            gaussian_coords,
            gaussian_labels,
            mesh_vertices,
            vote_k=args.vote_k,
            num_classes=args.num_classes,
            ignore_label=-1
        )
        results["pred_labels"] = pred_labels

        # Convert to colors
        pred_colors = labels_to_colors(pred_labels, color_map, ignore_label=-1)
        pred_mesh.visual.vertex_colors = pred_colors

        # Export prediction mesh
        pred_output = output_dir / f"{scene_name}_pred_scenesplat_colored.ply"
        print(f"\nExporting prediction mesh to: {pred_output}")
        pred_mesh.export(pred_output)
        results["pred_mesh"] = str(pred_output)

        # Render prediction
        if not args.no_render:
            pred_render_output = output_dir / f"{scene_name}_pred_topview.png"
            render_top_view(pred_mesh, pred_render_output)
            results["pred_render"] = str(pred_render_output)

    # ===== 4. Create intermediate state =====
    if args.mode == "compare" and gaussian_data_path and mesh_vertices is not None:
        if "pred_labels" not in results or "gt_labels" not in results:
            print("\n[4/4] Skipping intermediate state (missing predictions or GT)")
        else:
            print("\n[4/4] Creating intermediate state (partial correction)...")

            if args.random_seed is not None:
                np.random.seed(args.random_seed)
                print(f"  Using random seed: {args.random_seed}")

            intermediate_labels, corrected_classes, statistics = create_intermediate_labels(
                results["pred_labels"],
                results["gt_labels"],
                num_wrong_classes=args.num_correct_classes
            )
            results["intermediate_labels"] = intermediate_labels
            results["statistics"] = statistics

            # Create intermediate mesh
            intermediate_mesh = trimesh.load(mesh_path) if Path(mesh_path).exists() else pred_mesh
            intermediate_colors = labels_to_colors(intermediate_labels, color_map, ignore_label=-1)
            intermediate_mesh.visual.vertex_colors = intermediate_colors

            # Export intermediate mesh
            intermediate_output = output_dir / f"{scene_name}_intermediate_colored.ply"
            print(f"\nExporting intermediate mesh to: {intermediate_output}")
            intermediate_mesh.export(intermediate_output)
            results["intermediate_mesh"] = str(intermediate_output)

            # Render intermediate
            if not args.no_render:
                intermediate_render_output = output_dir / f"{scene_name}_intermediate_topview.png"
                render_top_view(intermediate_mesh, intermediate_render_output)
                results["intermediate_render"] = str(intermediate_render_output)

            # Save statistics
            stats_output = output_dir / f"{scene_name}_statistics.txt"
            with open(stats_output, 'w') as f:
                f.write("="*70 + "\n")
                f.write("SceneSplat Segmentation Statistics\n")
                f.write("="*70 + "\n\n")

                f.write(f"Scene: {scene_name}\n")
                f.write(f"Vertices: {len(pred_labels):,}\n\n")

                f.write("Corrected Classes:\n")
                for i, cls in enumerate(corrected_classes, 1):
                    f.write(f"  {i}. Class {cls}\n")

                f.write(f"\nTotal Prediction Errors: {statistics['total_errors']:,}\n")
                f.write(f"Corrected Vertices: {statistics['corrected_vertices']:,}\n")
                f.write(f"Remaining Errors: {statistics['remaining_errors']:,}\n\n")

                f.write(f"Initial Accuracy: {statistics['accuracy_before']*100:.2f}%\n")
                f.write(f"Corrected Accuracy: {statistics['accuracy_after']*100:.2f}%\n")
                f.write(f"Accuracy Improvement: {statistics['accuracy_improvement']*100:.2f}%\n")

            # Also save JSON
            stats_json_output = output_dir / f"{scene_name}_statistics.json"
            with open(stats_json_output, 'w') as f:
                json.dump({
                    "scene": scene_name,
                    "num_vertices": int(len(pred_labels)),
                    "corrected_classes": corrected_classes,
                    **statistics
                }, f, indent=2)

            results["statistics_file"] = str(stats_output)
            results["statistics_json"] = str(stats_json_output)

    # ===== Summary =====
    print("\n" + "="*70)
    print("✓ Complete!")
    print(f"\nGenerated files:")

    if "gt_mesh" in results:
        print(f"  Ground Truth:")
        print(f"    Mesh: {results['gt_mesh']}")
        if "gt_render" in results:
            print(f"    Image: {results['gt_render']}")

    if "pred_mesh" in results:
        print(f"  Prediction:")
        print(f"    Mesh: {results['pred_mesh']}")
        if "pred_render" in results:
            print(f"    Image: {results['pred_render']}")

    if "intermediate_mesh" in results:
        print(f"  Intermediate (Partially Corrected):")
        print(f"    Mesh: {results['intermediate_mesh']}")
        if "intermediate_render" in results:
            print(f"    Image: {results['intermediate_render']}")
        if "statistics_file" in results:
            print(f"    Statistics: {results['statistics_file']}")

    print("="*70)


if __name__ == "__main__":
    main()
