#!/usr/bin/env python3
"""
Extract and Convert OccamLGS Data to SceneSplat Format

This script extracts Gaussian parameters and language features from OccamLGS
checkpoints and converts them to SceneSplat format for training and evaluation.

OccamLGS Format:
- chkpnt30000_langfeat_{1,2,3}.pth - Checkpoints with 3 feature levels
- point_cloud/iteration_30000/point_cloud.ply - PLY file
- test/ours_30000_langfeat_{1,2,3}/renders/ - Test render images

SceneSplat Format:
- coord.npy, color.npy, opacity.npy, quat.npy, scale.npy - Gaussian parameters
- lang_feat.npy - Language features [N, D]
- valid_feat_mask.npy - Valid feature mask [N]
- renders_npy/ - Rendered feature maps (optional)
- lang_feat_grid_svd_r{8,16,32}.npz - SVD compressed features (optional)

Usage:
    # Extract single scene
    python tools/extract_occamlgs_to_scenesplat.py \\
        --input /new_data/cyf/projects/OccamLGS/output/LERF-origin/figurines \\
        --output gaussian_train_clip/lerf_ovs/train

    # Extract all scenes (recursive)
    python tools/extract_occamlgs_to_scenesplat.py \\
        --input /new_data/cyf/projects/OccamLGS/output/LERF-origin \\
        --output gaussian_train_clip/lerf_ovs/train \\
        --recursive

    # Extract with SVD compression
    python tools/extract_occamlgs_to_scenesplat.py \\
        --input /new_data/cyf/projects/OccamLGS/output/LERF-origin/figurines \\
        --output gaussian_train_clip/lerf_ovs/train \\
        --compress_svd
"""

import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm
from plyfile import PlyData


def np_sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def load_occamlgs_checkpoint(ckpt_path: str) -> Dict:
    """
    Load OccamLGS checkpoint and extract Gaussian parameters and language features.

    Checkpoint structure: ((..., lang_feat, ...), step_count)
    - [0]: int (unknown)
    - [1]: [N, 3] - xyz/coord
    - [2]: [N, 1, 3] - SH DC (degree 0)
    - [3]: [N, 15, 3] - SH higher order (degree 1-3)
    - [4]: [N, 3] - color
    - [5]: [N, 4] - quaternion (wxyz)
    - [6]: [N, 1] - opacity
    - [7]: [N, D] - **language features** (512-dim for OpenCLIP)
    - [8]: [N] - likely valid_feat_mask or something similar
    - [9]: [N, 1] - unknown
    - [10]: [N, 1] - unknown
    - [11]: dict - optimizer state
    - [12]: scalar - step count or loss

    Args:
        ckpt_path: Path to checkpoint file

    Returns:
        Dictionary with extracted data
    """
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if not isinstance(checkpoint, tuple) or len(checkpoint) < 1:
        raise ValueError(f"Invalid checkpoint format: {type(checkpoint)}")

    inner_tuple = checkpoint[0]
    if not isinstance(inner_tuple, tuple) or len(inner_tuple) < 13:
        raise ValueError(f"Invalid inner tuple length: {len(inner_tuple)}, expected >= 13")

    data = {}

    # Extract Gaussian parameters
    # Note: OccamLGS checkpoints may have tensors with requires_grad=True
    # Use .detach() before converting to numpy
    def to_numpy(t):
        """Convert tensor to numpy, handling requires_grad."""
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy().astype(np.float32)
        return np.array(t, dtype=np.float32)

    # [1]: coord/xyz [N, 3]
    data['coord'] = to_numpy(inner_tuple[1])

    # [4]: color [N, 3] - stored as float32, need to check range
    color = to_numpy(inner_tuple[4])
    # Check if color needs sigmoid/clipping (OccamLGS may store in different format)
    # Assuming color is in [0, 1] range, convert to uint8 [0, 255]
    color = np.clip(color, 0, 1)
    data['color'] = (color * 255).astype(np.uint8)

    # [5]: quaternion [N, 4] - (wxyz format, need to normalize)
    quat = to_numpy(inner_tuple[5])
    # Normalize quaternions
    quat_norms = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / (quat_norms + 1e-9)
    # Ensure w (first component) is positive for uniqueness
    signs = np.sign(quat[:, 0])
    quat = quat * signs[:, None]
    data['quat'] = quat

    # [6]: opacity [N, 1] - may need sigmoid
    opacity = to_numpy(inner_tuple[6])
    # Check if opacity is in raw or activated form
    if opacity.max() > 1.0 or opacity.min() < 0:
        # Likely raw opacity values, apply sigmoid
        opacity = np_sigmoid(opacity.reshape(-1)).reshape(-1, 1)
    else:
        opacity = opacity.reshape(-1, 1)
    data['opacity'] = opacity

    # Extract scale from SH features (approximate from SH degree 0 coefficients)
    # OccamLGS stores SH features differently than standard Gaussian Splatting
    # For now, we'll compute scale from the spatial extent or use default
    # [2]: SH DC [N, 1, 3] - can be used to estimate scale
    sh_dc = to_numpy(inner_tuple[2])  # [N, 1, 3]
    # Use a default scale for now (can be refined later)
    n_points = data['coord'].shape[0]
    data['scale'] = np.ones((n_points, 3), dtype=np.float32) * 0.01  # Default scale

    # [7]: language features [N, D]
    data['lang_feat'] = to_numpy(inner_tuple[7])

    # [8]: valid feature mask [N]
    valid_mask = to_numpy(inner_tuple[8])
    if valid_mask.dtype == np.float32 or valid_mask.dtype == np.float64:
        # Convert float mask to boolean
        data['valid_feat_mask'] = (valid_mask > 0).astype(np.uint8)
    else:
        data['valid_feat_mask'] = valid_mask.astype(np.uint8)
    valid_mask = np.any(data['lang_feat'] != 0.0, axis=1).astype(int)
    data["lang_feat"] = data["lang_feat"][valid_mask > 0]
    data["valid_feat_mask"] = valid_mask
    print((valid_mask > 0).sum(), "valid features out of", len(valid_mask))

    print(f"  Extracted: coord={data['coord'].shape}, "
          f"color={data['color'].shape}, "
          f"opacity={data['opacity'].shape}, "
          f"quat={data['quat'].shape}, "
          f"scale={data['scale'].shape}, "
          f"lang_feat={data['lang_feat'].shape}, "
          f"valid_feat_mask={data['valid_feat_mask'].shape}")

    return data


def combine_multi_level_features(
    data_level1: Dict,
    data_level2: Dict,
    data_level3: Dict
) -> Dict:
    """
    Combine 3 feature levels into a single dataset.

    OccamLGS has 3 feature levels (langfeat_1, langfeat_2, langfeat_3).
    We need to combine them appropriately.

    Args:
        data_level1: Data from level 1 checkpoint
        data_level2: Data from level 2 checkpoint
        data_level3: Data from level 3 checkpoint

    Returns:
        Combined data dictionary
    """
    # Use level 1 as base (most detailed)
    combined = data_level1.copy()

    # Check if all levels have the same number of points
    n1 = data_level1['lang_feat'].shape[0]
    n2 = data_level2['lang_feat'].shape[0]
    n3 = data_level3['lang_feat'].shape[0]

    print(f"Level sizes: L1={n1}, L2={n2}, L3={n3}")

    # For SceneSplat format, we typically store the first level's features
    # and optionally store multiple levels separately
    # For now, just use level 1 features (can be extended later)

    # Alternative: Stack features if dimensions are small
    # Or store all levels as separate files

    return combined


def save_scenesplat_format(data: Dict, output_dir: Path, scene_name: str, level_idx: int = 0):
    """
    Save data in SceneSplat NPY format.

    Args:
        data: Dictionary with Gaussian parameters and features
        output_dir: Output directory
        scene_name: Scene name for logging
        level_idx: Feature level index (1, 2, 3) for multi-level features
    """
    output_dir = Path(output_dir) / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save standard attributes for level 1 only (base Gaussian parameters)
    if level_idx == 1:
        np.save(output_dir / "coord.npy", data["coord"])
        np.save(output_dir / "color.npy", data["color"])
        np.save(output_dir / "opacity.npy", data["opacity"])
        np.save(output_dir / "scale.npy", data["scale"])
        np.save(output_dir / "quat.npy", data["quat"])
        print(f"Saved base Gaussian parameters to: {output_dir}")
        print(f"  - coord: {data['coord'].shape}, {data['coord'].dtype}")
        print(f"  - color: {data['color'].shape}, {data['color'].dtype}")
        print(f"  - opacity: {data['opacity'].shape}, {data['opacity'].dtype}")
        print(f"  - scale: {data['scale'].shape}, {data['scale'].dtype}")
        print(f"  - quat: {data['quat'].shape}, {data['quat'].dtype}")

    # Save language features for each level
    lang_feat_path = output_dir / f"lang_feat_{level_idx}.npy"
    np.save(lang_feat_path, data["lang_feat"])
    valid_mask_path = output_dir / f"valid_feat_mask_{level_idx}.npy"
    np.save(valid_mask_path, data["valid_feat_mask"])
    print(f"Saved level {level_idx} language features to: {lang_feat_path}")
    print(f"  - lang_feat_{level_idx}: {data['lang_feat'].shape}, {data['lang_feat'].dtype}")
    print(f"  - valid_feat_mask_{level_idx}: {data['valid_feat_mask'].shape}, {data['valid_feat_mask'].dtype}")


def copy_test_renders(
    test_dir: Path,
    output_dir: Path,
    scene_name: str
):
    """
    Copy test renders to output directory.

    OccamLGS test directory structure:
    test/
    ├── ours_30000_langfeat_1/
    │   ├── renders/
    │   └── gt/
    ├── ours_30000_langfeat_2/
    │   ├── renders/
    │   └── gt/
    └── ours_30000_langfeat_3/
        ├── renders/
        └── gt/

    Args:
        test_dir: Test directory in OccamLGS format
        output_dir: Output directory
        scene_name: Scene name
    """
    output_test_dir = Path(output_dir) / scene_name / "test"
    output_test_dir.mkdir(parents=True, exist_ok=True)

    # Check for OccamLGS format with subdirectories
    langfeat_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir() and 'langfeat' in d.name])

    if not langfeat_dirs:
        # Try direct renders directory
        renders_dir = test_dir / "renders"
        if renders_dir.exists():
            output_renders_dir = output_test_dir / "renders"
            shutil.copytree(renders_dir, output_renders_dir, dirs_exist_ok=True)
            print(f"  Copied test renders to: {output_renders_dir}")
        else:
            print(f"  No renders directory found in {test_dir}")
        return

    # Copy renders from each langfeat level
    for langfeat_dir in langfeat_dirs:
        level_name = langfeat_dir.name
        renders_dir = langfeat_dir / "renders"

        if renders_dir.exists():
            # Copy to output with level name
            output_renders_dir = output_test_dir / level_name / "renders"
            shutil.copytree(renders_dir, output_renders_dir, dirs_exist_ok=True)
            print(f"  Copied {level_name}/ renders to: {output_renders_dir}")

        # Also copy gt if it exists
        gt_dir = langfeat_dir / "gt"
        if gt_dir.exists():
            output_gt_dir = output_test_dir / level_name / "gt"
            shutil.copytree(gt_dir, output_gt_dir, dirs_exist_ok=True)
            print(f"  Copied {level_name}/ gt to: {output_gt_dir}")


def process_scene(
    scene_dir: Path,
    output_root: Path,
    compress_svd: bool = False
) -> bool:
    """
    Process a single scene from OccamLGS to SceneSplat format.

    Args:
        scene_dir: Scene directory path
        output_root: Output root directory
        compress_svd: Whether to apply SVD compression

    Returns:
        True if successful, False otherwise
    """
    scene_name = scene_dir.name
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*60}")

    try:
        # Check for required checkpoint files
        ckpt_files = []
        for level in [1, 2, 3]:
            ckpt_path = scene_dir / f"chkpnt30000_langfeat_{level}.pth"
            if not ckpt_path.exists():
                print(f"Warning: Checkpoint not found: {ckpt_path}")
                # Try alternative naming pattern
                ckpt_path = scene_dir / f"occam-chkpnt30000_langfeat_{level}.pth"
                if not ckpt_path.exists():
                    print(f"Error: Required checkpoint file not found")
                    return False
            ckpt_files.append(ckpt_path)

        # Load and save each level separately
        for level_idx, ckpt_path in enumerate(ckpt_files, start=1):
            print(f"\n--- Processing Level {level_idx} ---")
            data = load_occamlgs_checkpoint(str(ckpt_path))

            # Save in SceneSplat format
            save_scenesplat_format(data, output_root, scene_name, level_idx=level_idx)

        # Copy test renders if they exist
        test_dir = scene_dir / "test"
        if test_dir.exists():
            copy_test_renders(test_dir, output_root, scene_name)

        # Apply SVD compression if requested
        if compress_svd:
            print(f"\nApplying SVD compression...")
            output_scene_dir = output_root / scene_name
            apply_svd_compression(output_scene_dir)

        return True

    except Exception as e:
        print(f"Error processing scene {scene_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def apply_svd_compression(scene_dir: Path):
    """
    Apply SVD compression to lang_feat.npy.

    This is a simplified version - for full RPCA compression,
    use the compress_grid_svd.py script separately.

    Args:
        scene_dir: Scene directory path
    """
    # Import sklearn locally for this function
    from sklearn.decomposition import PCA

    lang_feat_path = scene_dir / "lang_feat.npy"
    coord_path = scene_dir / "coord.npy"

    if not lang_feat_path.exists() or not coord_path.exists():
        print(f"Warning: Missing required files for SVD compression")
        return

    # Load data
    lang_feat = np.load(lang_feat_path).astype(np.float32)
    coord = np.load(coord_path).astype(np.float32)

    print(f"  lang_feat shape: {lang_feat.shape}")

    # Simple grid-based aggregation (can be improved)
    grid_size = 0.01
    grid_indices = np.floor(coord / grid_size).astype(np.int32)

    # Create unique grid cells
    unique_grids, inverse_indices = np.unique(
        grid_indices, axis=0, return_inverse=True
    )

    # Average features per grid cell
    n_grids = len(unique_grids)
    feat_dim = lang_feat.shape[1]

    grid_features = np.zeros((n_grids, feat_dim), dtype=np.float32)
    grid_counts = np.zeros(n_grids, dtype=np.int32)

    for i in range(len(lang_feat)):
        grid_idx = inverse_indices[i]
        grid_features[grid_idx] += lang_feat[i]
        grid_counts[grid_idx] += 1

    # Normalize by count
    grid_counts_expanded = grid_counts.reshape(-1, 1).astype(np.float32)
    grid_features /= (grid_counts_expanded + 1e-9)

    print(f"  Grid features: {grid_features.shape}")

    # Apply SVD for compression
    for rank in [8, 16, 32]:
        if feat_dim < rank:
            print(f"  Skipping rank {rank} (feature dim {feat_dim} < {rank})")
            continue

        # Simple SVD (not RPCA)
        U, S, Vt = np.linalg.svd(grid_features, full_matrices=False)

        # Compress to rank
        U_r = U[:, :rank]  # [n_grids, rank]
        S_r = S[:rank]     # [rank]

        # Store compressed features and indices
        compressed = U_r * S_r[np.newaxis, :]  # [n_grids, rank]

        # Create indices mapping
        indices = inverse_indices  # [N] - maps each point to its grid index

        # Save compressed data
        output_path = scene_dir / f"lang_feat_grid_svd_r{rank}.npz"
        np.savez_compressed(
            output_path,
            compressed=compressed,
            indices=indices
        )
        print(f"  Saved SVD r{rank}: {output_path}")

        # Save metadata
        meta_path = scene_dir / "grid_meta_data.json"
        metadata = {
            "grid_size": grid_size,
            "n_grids": int(n_grids),
            "n_points": int(len(lang_feat)),
            "feat_dim": int(feat_dim),
            "rank": int(rank),
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def find_scenes(input_path: Path, recursive: bool = False) -> List[Path]:
    """
    Find all scene directories in the input path.

    Args:
        input_path: Input path
        recursive: Whether to search recursively

    Returns:
        List of scene directory paths
    """
    scenes = []

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return scenes

    if recursive:
        # Search for all directories containing checkpoint files
        for root, dirs, files in os.walk(input_path):
            root_path = Path(root)
            # Check if this directory has checkpoint files
            has_checkpoint = any(
                f.startswith("chkpnt") and "langfeat" in f
                for f in files
            )
            if has_checkpoint:
                scenes.append(root_path)
    else:
        # Single directory mode - assume input_path is a scene directory
        scenes = [input_path]

    return sorted(scenes)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and convert OccamLGS data to SceneSplat format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input directory (OccamLGS format)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        help="Output directory (SceneSplat format)",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search for scene directories",
    )
    parser.add_argument(
        "--compress_svd",
        action="store_true",
        help="Apply SVD compression to language features",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Comma-separated list of specific scene names to process",
    )

    args = parser.parse_args()

    # Find scenes to process
    if args.scenes:
        # Process specific scenes
        scene_names = [s.strip() for s in args.scenes.split(',')]
        scenes = [args.input / s for s in scene_names]
    else:
        # Find all scenes
        scenes = find_scenes(args.input, args.recursive)

    if not scenes:
        print("No scene directories found!")
        print("\nHint: Scene directories should contain checkpoint files:")
        print("  - chkpnt30000_langfeat_1.pth")
        print("  - chkpnt30000_langfeat_2.pth")
        print("  - chkpnt30000_langfeat_3.pth")
        return 1

    print(f"Found {len(scenes)} scene(s) to process:")
    for s in scenes[:5]:
        print(f"  - {s}")
    if len(scenes) > 5:
        print(f"  ... and {len(scenes) - 5} more")

    # Process each scene
    success_count = 0
    for scene_dir in tqdm(scenes, desc="Extracting scenes"):
        if process_scene(scene_dir, args.output, args.compress_svd):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Extraction complete: {success_count}/{len(scenes)} successful")
    print(f"{'='*60}")

    if success_count > 0:
        print("\nNext steps:")
        print("1. Verify the extracted data:")
        print(f"   ls -la {args.output}/*/ ")
        print("")
        print("2. For full SVD compression with RPCA:")
        print(f"   python tools/compress_grid_svd.py \\")
        print(f"     --data_root {args.output.parent} \\")
        print(f"     --dataset lerf_ovs \\")
        print(f"     --split train")
        print("")
        print("3. For rendering feature maps:")
        print(f"   python tools/feature_map_renderer.py \\")
        print(f"     -m /path/to/occamlgs_model \\")
        print(f"     --iteration 30000 \\")
        print(f"     --feature_level 0 \\")
        print(f"     --src_dim 512")

    return 0 if success_count == len(scenes) else 1


if __name__ == "__main__":
    exit(main())
