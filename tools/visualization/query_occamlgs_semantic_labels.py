#!/usr/bin/env python3
"""
Semantic Label Query Visualization for OccamLGS Scenes

This script loads ScanNet semantic labels from .labels.ply files,
queries by category name (e.g., "chair"), finds 3D points with that label,
finds intersecting Gaussians, colors them red, and renders the result.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Set
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree

# Add SceneSplat to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.scene.dataset_readers import readCamerasFromTransforms
from tools.gaussian_renderer.gaussian_model import GaussianModel
from tools.gaussian_renderer import render
from tools.utils.camera_utils import cameraList_from_camInfos
from tools.utils.graphics_utils import focal2fov


# ScanNet label to category mapping
SCANNET_LABELS = {
    "wall": 0,
    "floor": 1,
    "ceiling": 2,
    "chair": 3,
    "table": 4,
    "desk": 5,
    "bed": 6,
    "bookshelf": 7,
    "sofa": 8,
    "sink": 9,
    "bathtub": 10,
    "toilet": 11,
    "curtain": 12,
    "counter": 13,
    "door": 14,
    "window": 15,
    "picture": 16,
    "blind": 17,
    "shelves": 18,
    "cabinet": 19,
    "monitor": 20,
    "lamp": 21,
    "trash can": 22,
    "box": 23,
    "pillow": 24,
    "rail": 25,
    "board": 26,
    "stove": 27,
    "refrigerator": 28,
    "plant": 29,
    "tv": 30,
    "microwave": 31,
    "kitchen counter": 32,
    "coffee table": 33,
    "nightstand": 34,
    "toilet paper": 35,
    "paper towel dispenser": 36,
    "soap dispenser": 37,
    "tissue box": 38,
    "mirror": 39,
    "bathroom": 40,
    "bathroom counter": 41,
    "bathroom cabinet": 42,
    "bathroom shelf": 43,
    "shower": 44,
    "shower curtain": 45,
    "bath mat": 46,
    "towel": 47,
    "laundry basket": 48,
    "drying rack": 49,
    "scale": 50,
    "bucket": 51,
    "brush": 52,
    "trash bag": 53,
    "trash bin": 54,
    "trash pile": 55,
    "clothes": 56,
    "dish rack": 57,
    "toaster": 58,
    "toaster oven": 59,
    "dishwasher": 60,
    "oven": 61,
    "washing machine": 62,
    "dryer": 63,
    "ironing board": 64,
    "iron": 65,
    "vacuum": 66,
    "broom": 67,
    "mop": 68,
    "cleaning supplies": 69,
    "cart": 70,
    "office chair": 71,
    "storage bin": 72,
    "crate": 73,
    "bar stool": 74,
    "folding chair": 75,
    "office supplies": 76,
    "printer": 77,
    "computer": 78,
    "keyboard": 79,
    "mouse": 80,
    "laptop": 81,
    "fan": 82,
    "heater": 83,
    "air conditioner": 84,
    "dehumidifier": 85,
    "coat rack": 86,
    "hat rack": 87,
    "shoe rack": 88,
    "umbrella stand": 89,
    "clock": 90,
    "calendar": 91,
    "poster": 92,
    "whiteboard": 93,
    "blackboard": 94,
}

# Reverse mapping for fallback
SCANNET_LABELS_REVERSE = {v: k for k, v in SCANNET_LABELS.items()}


def load_labels_ply(labels_ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ScanNet labels PLY file.

    Returns:
        points: (N, 3) array of 3D coordinates
        labels: (N,) array of label IDs (objectIds)
    """
    print(f"Loading labels PLY: {labels_ply_path}")
    plydata = PlyData.read(labels_ply_path)
    vertex = plydata['vertex'].data

    points = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
    labels = vertex['label']  # objectId in ScanNet format

    print(f"  Loaded {len(points)} points with {len(np.unique(labels))} unique labels")
    return points, labels


def load_aggregation(aggregation_path: str) -> Tuple[dict, dict, dict]:
    """
    Load ScanNet aggregation.json file.

    Returns:
        seg_groups: list of segment groups
        label_to_object_ids: mapping from label name to list of objectIds
        object_id_to_label: mapping from objectId to label name
    """
    print(f"Loading aggregation: {aggregation_path}")
    with open(aggregation_path) as f:
        agg_data = json.load(f)

    seg_groups = agg_data['segGroups']

    # Build mappings
    object_id_to_label = {}  # objectId -> label name
    label_to_object_ids = {}  # label name -> list of objectIds

    for seg in seg_groups:
        object_id = seg['objectId']
        label = seg['label']
        object_id_to_label[object_id] = label

        if label not in label_to_object_ids:
            label_to_object_ids[label] = []
        label_to_object_ids[label].append(object_id)

    print(f"  Loaded {len(seg_groups)} segment groups")
    print(f"  Found {len(label_to_object_ids)} unique labels")

    return seg_groups, label_to_object_ids, object_id_to_label


def get_available_labels(aggregation_path: str) -> Set[str]:
    """Get set of available label names in a scene."""
    with open(aggregation_path) as f:
        agg_data = json.load(f)

    labels = set()
    for seg in agg_data['segGroups']:
        labels.add(seg['label'])
    return labels


def find_points_by_label(
    points: np.ndarray,
    labels: np.ndarray,
    object_id_to_label: dict,
    target_label: str
) -> np.ndarray:
    """
    Find points with the specified label.

    Args:
        points: (N, 3) array of 3D coordinates
        labels: (N,) array of objectIds
        object_id_to_label: mapping from objectId to label name
        target_label: label name to search for (e.g., "chair")

    Returns:
        (M, 3) array of points with the target label
    """
    # Find all objectIds with the target label
    target_object_ids = [
        obj_id for obj_id, label in object_id_to_label.items()
        if label.lower() == target_label.lower()
    ]

    if not target_object_ids:
        print(f"  WARNING: No objects found with label '{target_label}'")
        return np.zeros((0, 3), dtype=points.dtype)

    print(f"  Found {len(target_object_ids)} objects with label '{target_label}': {target_object_ids[:5]}...")

    # Find all points with those objectIds
    mask = np.isin(labels, target_object_ids)
    labeled_points = points[mask]

    print(f"  Found {len(labeled_points)} points with label '{target_label}'")
    return labeled_points


def find_intersecting_gaussians(
    gaussian_xyz: torch.Tensor,
    labeled_points: np.ndarray,
    distance_threshold: float = 0.05
) -> np.ndarray:
    """
    Find Gaussians that intersect with labeled points.

    Args:
        gaussian_xyz: (N, 3) tensor of Gaussian centers
        labeled_points: (M, 3) array of labeled 3D points
        distance_threshold: maximum distance for intersection

    Returns:
        (N,) boolean array indicating which Gaussians intersect
    """
    print(f"Finding intersecting Gaussians (threshold={distance_threshold}m)...")

    N = gaussian_xyz.shape[0]
    M = labeled_points.shape[0]

    if M == 0:
        return np.zeros(N, dtype=bool)

    # Convert to numpy (detach first to avoid grad error)
    gaussian_xyz_np = gaussian_xyz.detach().cpu().numpy()

    # Build KD-tree for Gaussians (faster spatial queries)
    print(f"  Building KD-tree for {N} Gaussians...")
    gaussian_tree = cKDTree(gaussian_xyz_np)

    # For each labeled point, find nearby Gaussians
    print(f"  Querying {M} labeled points...")
    gaussian_intersect = np.zeros(N, dtype=bool)

    # Use query_ball_point to find all Gaussians within radius for each point
    # This is more efficient for many-to-many queries
    for point in tqdm(labeled_points, desc="  Finding intersections"):
        indices = gaussian_tree.query_ball_point(point, distance_threshold)
        gaussian_intersect[indices] = True

    num_intersect = gaussian_intersect.sum()
    print(f"  Found {num_intersect} intersecting Gaussians out of {N}")

    return gaussian_intersect


def render_with_override(
    camera,
    gaussians: GaussianModel,
    pipeline,
    background,
    override_mask: np.ndarray
) -> torch.Tensor:
    """Render the scene with color override for masked Gaussians."""
    if override_mask is not None and override_mask.any():
        original_features_dc = gaussians._features_dc.clone()
        original_features_rest = gaussians._features_rest.clone()

        # Create override colors
        num_gaussians = gaussians.get_xyz.shape[0]
        override_colors = original_features_dc.squeeze(1).clone()

        # Set masked Gaussians to red
        red_color = torch.tensor([1.0, 0.0, 0.0], device=override_colors.device)
        mask_tensor = torch.tensor(override_mask, device=override_colors.device, dtype=torch.bool)
        override_colors[mask_tensor] = red_color

        gaussians._features_dc.data = override_colors.unsqueeze(1).contiguous()
        gaussians._features_rest.data.zero_()

    render_result = render(camera, gaussians, pipeline, background)
    rendered_image = render_result["render"]

    if override_mask is not None and override_mask.any():
        gaussians._features_dc.data = original_features_dc
        gaussians._features_rest.data = original_features_rest

    return rendered_image


def concatenate_images(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> np.ndarray:
    """Concatenate two images horizontally."""
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()

    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)

    if img1_np.shape[0] != img2_np.shape[0]:
        min_h = min(img1_np.shape[0], img2_np.shape[0])
        img1_np = img1_np[:min_h]
        img2_np = img2_np[:min_h]

    concat = np.concatenate([img1_np, img2_np], axis=1)
    return concat


def find_ply_file(scene_path: Path) -> Path:
    """Find the PLY file in the scene directory."""
    # Check for point_cloud directory (trained model)
    point_cloud_dir = scene_path / "point_cloud"
    if point_cloud_dir.exists():
        # Find highest iteration
        iterations = []
        for it_dir in point_cloud_dir.iterdir():
            if it_dir.name.startswith("iteration_"):
                try:
                    it_num = int(it_dir.name.split("_")[1])
                    iterations.append((it_num, it_dir))
                except:
                    pass
        if iterations:
            iterations.sort(key=lambda x: x[0], reverse=True)
            ply_path = iterations[0][1] / "point_cloud.ply"
            if ply_path.exists():
                return ply_path

    # Check for input.ply
    input_ply = scene_path / "input.ply"
    if input_ply.exists():
        return input_ply

    raise FileNotFoundError(f"No PLY file found in {scene_path}")


def process_scene(
    scene_path: str,
    query_labels: List[str],
    checkpoint_name: str = "chkpnt30000_langfeat_1.pth",
    distance_threshold: float = 0.05,
    output_dir: str = None,
    device: str = "cuda"
) -> dict:
    """Process a single scene: load labels, find points, render results."""
    scene_path = Path(scene_path)
    scene_name = scene_path.name

    print(f"\n{'='*60}")
    print(f"Processing Scene: {scene_name}")
    print(f"{'='*60}")

    # Determine source path (original ScanNet dataset)
    output_dir_str = str(scene_path)
    if "/output/" in output_dir_str:
        parts = output_dir_str.split("/output/")
        base_path = parts[0]
        source_path_str = f"{base_path}/datasets/scannet/{scene_name}"
    else:
        source_path_str = output_dir_str

    source_path = Path(source_path_str)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path_str}")

    # Find labels PLY and aggregation files
    labels_ply_path = source_path / f"{scene_name}_vh_clean_2.labels.ply"
    aggregation_path = source_path / f"{scene_name}_vh_clean.aggregation.json"

    if not labels_ply_path.exists():
        raise FileNotFoundError(f"Labels PLY not found: {labels_ply_path}")
    if not aggregation_path.exists():
        raise FileNotFoundError(f"Aggregation file not found: {aggregation_path}")

    # Load labels and aggregation
    points, labels = load_labels_ply(str(labels_ply_path))
    seg_groups, label_to_object_ids, object_id_to_label = load_aggregation(str(aggregation_path))

    # Show available labels
    available_labels = set(label_to_object_ids.keys())
    print(f"\nAvailable labels in scene: {sorted(available_labels)}")

    # Check if query labels exist
    for query_label in query_labels:
        if query_label.lower() not in [l.lower() for l in available_labels]:
            print(f"  WARNING: Query label '{query_label}' not found in scene")
            print(f"  Available labels: {sorted(available_labels)}")

    # Load cameras
    print(f"\nLoading cameras from: {source_path}")
    all_cam_infos = readCamerasFromTransforms(
        str(source_path),
        "transforms_train.json",
        depths_folder="",
        white_background=False,
        is_test=False,
        extension=".jpg"
    )

    # Split into train/test
    llffhold = 8
    train_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llffhold != 0]
    test_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llffhold == 0]

    print(f"  Train cameras: {len(train_cam_infos)}")
    print(f"  Test cameras: {len(test_cam_infos)}")

    # Use test cameras if available
    all_cameras = test_cam_infos if len(test_cam_infos) > 0 else train_cameras

    # Use all test cameras (no sampling)
    cameras_info = all_cameras

    print(f"\n  Using all {len(cameras_info)} test cameras")

    # Convert to camera objects
    from types import SimpleNamespace
    camera_args = SimpleNamespace(
        resolution=-1,
        data_device=device,
        train_test_exp=False
    )
    cameras = cameraList_from_camInfos(cameras_info, 1.0, camera_args, False, True)

    # Find and load PLY file
    ply_path = find_ply_file(scene_path)
    print(f"\nLoading Gaussians from PLY: {ply_path}")

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(str(ply_path))

    # Get Gaussian centers
    gaussian_xyz = gaussians.get_xyz
    print(f"  Loaded {gaussian_xyz.shape[0]} Gaussians")

    # Find labeled points for each query
    query_results = {}
    for query_label in query_labels:
        print(f"\n{'='*40}")
        print(f"Query: '{query_label}'")
        print(f"{'='*40}")

        # Find points with this label
        labeled_points = find_points_by_label(points, labels, object_id_to_label, query_label)

        if len(labeled_points) == 0:
            print(f"  No points found for label '{query_label}', skipping")
            continue

        # Find intersecting Gaussians
        gaussian_mask = find_intersecting_gaussians(gaussian_xyz, labeled_points, distance_threshold)

        query_results[query_label] = {
            'labeled_points': labeled_points,
            'gaussian_mask': gaussian_mask,
            'num_points': len(labeled_points),
            'num_gaussians': gaussian_mask.sum()
        }

    if not query_results:
        print(f"\n  No valid query results, skipping rendering")
        return {}

    # Render
    print(f"\nProcessing {len(cameras)} cameras...")
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    results = {}
    for camera_idx, camera in enumerate(cameras):
        print(f"\n{'='*40}")
        print(f"Camera {camera_idx}/{len(cameras)}: {camera.image_name}")
        print(f"{'='*40}")

        # Render original scene
        print("  Rendering original scene...")
        with torch.no_grad():
            original_result = render(camera, gaussians, None, background)
            original_image = original_result["render"]

        print(f"    Shape: {original_image.shape}, Range: [{original_image.min():.3f}, {original_image.max():.3f}]")

        # Render for each query label
        for query_label, query_data in query_results.items():
            print(f"\n  Rendering with query: '{query_label}'")
            print(f"    {query_data['num_points']} points, {query_data['num_gaussians']} Gaussians")

            with torch.no_grad():
                query_image = render_with_override(
                    camera, gaussians, None, background, query_data['gaussian_mask']
                )

            print(f"    Shape: {query_image.shape}, Range: [{query_image.min():.3f}, {query_image.max():.3f}]")

            # Concatenate images
            concat_image = concatenate_images(original_image, query_image)

            # Save results
            if output_dir is not None:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Sanitize label for filename
                label_filename = query_label.replace(" ", "_").replace("/", "_")[:50]
                label_filename = "".join(c for c in label_filename if c.isalnum() or c in "_-_")

                output_file = output_path / f"{scene_name}_{label_filename}_{camera.uid}.png"
                concat_image_uint8 = (concat_image * 255).astype(np.uint8)
                Image.fromarray(concat_image_uint8).save(output_file)
                print(f"    Saved: {output_file}")

            # Store results
            key = f"{query_label}_{camera.uid}"
            results[key] = {
                "query": query_label,
                "camera_id": camera.uid,
                "camera_name": camera.image_name,
                "num_points": query_data['num_points'],
                "num_gaussians": query_data['num_gaussians'],
                "concat_image": concat_image
            }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Label Query Visualization for OccamLGS Scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query by single label
  python tools/visualization/query_occamlgs_semantic_labels.py \\
      --scenes /new_data/cyf/projects/OccamLGS/output/scannet-origin/scene0000_00 \\
      --query "chair" \\
      --output-dir ./semantic_query_results

  # Query by multiple labels
  python tools/visualization/query_occamlgs_semantic_labels.py \\
      --scenes /new_data/cyf/projects/OccamLGS/output/scannet-origin/scene0000_00 \\
      --query "chair" "table" "bed" \\
      --num-views 10 \\
      --output-dir ./semantic_query_results

  # Use custom distance threshold
  python tools/visualization/query_occamlgs_semantic_labels.py \\
      --scenes /new_data/cyf/projects/OccamLGS/output/scannet-origin/scene0000_00 \\
      --query "wall" \\
      --distance-threshold 0.1 \\
      --output-dir ./semantic_query_results
        """
    )

    parser.add_argument(
        "--scenes", "-s",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to scene directories"
    )
    parser.add_argument(
        "--all-scenes",
        type=str,
        default=None,
        metavar="BASE_DIR",
        help="Process all scenes in the specified base directory (e.g., /path/to/OccamLGS/output/scannet-origin)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="chkpnt30000_langfeat_1.pth",
        help="Checkpoint filename (default: chkpnt30000_langfeat_1.pth)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        nargs="+",
        required=True,
        help="Label query/queries (e.g., 'chair', 'table', 'wall')"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./semantic_query_results",
        help="Output directory (default: ./semantic_query_results)"
    )
    parser.add_argument(
        "--distance-threshold", "-d",
        type=float,
        default=0.05,
        help="Distance threshold for Gaussian-point intersection in meters (default: 0.05)"
    )
    parser.add_argument(
        "--num-views", "-n",
        type=int,
        default=None,
        help="[DEPRECATED] Now renders all test views by default"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.scenes is None and args.all_scenes is None:
        parser.error("Either --scenes or --all-scenes must be specified")
    if args.scenes is not None and args.all_scenes is not None:
        parser.error("Cannot specify both --scenes and --all-scenes")

    # Determine scene list
    if args.all_scenes is not None:
        base_dir = Path(args.all_scenes)
        if not base_dir.exists():
            print(f"Error: Base directory not found: {base_dir}")
            sys.exit(1)

        # Discover all scene subdirectories
        scene_paths = []
        for item in sorted(base_dir.iterdir()):
            if item.is_dir() and (item.name.startswith("scene") or
                                  (item / "point_cloud").exists() or
                                  (item / "input.ply").exists() or
                                  (item / args.checkpoint).exists()):
                scene_paths.append(str(item))

        if not scene_paths:
            print(f"Error: No scene directories found in {base_dir}")
            print("  Looking for directories starting with 'scene' or containing checkpoint files")
            sys.exit(1)

        print(f"Found {len(scene_paths)} scenes in {base_dir}")
    else:
        scene_paths = args.scenes

    for scene_path in scene_paths:
        try:
            results = process_scene(
                scene_path=scene_path,
                query_labels=args.query,
                checkpoint_name=args.checkpoint,
                distance_threshold=args.distance_threshold,
                output_dir=args.output_dir,
                device=args.device
            )
            if results:
                print(f"\n✓ Successfully processed: {scene_path}")
            else:
                print(f"\n⚠ No results for: {scene_path}")
        except Exception as e:
            print(f"\n✗ Error processing {scene_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("All scenes processed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
