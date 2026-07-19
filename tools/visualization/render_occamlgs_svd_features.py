#!/usr/bin/env python3
"""
Render OccamLGS Language Features with SVD Compression and PCA Visualization

This script loads OccamLGS checkpoints with language features, applies SVD
compression to reduce dimensionality (default 16), renders the compressed
features, and visualizes them using PCA (3 components).

Usage:
    python tools/visualization/render_occamlgs_svd_features.py \\
        --scene /new_data/cyf/projects/OccamLGS/output/scannet-origin/scene0000_00 \\
        --checkpoint chkpnt30000_langfeat_1.pth \\
        --svd-rank 16 \\
        --output-dir ./svd_render_results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.decomposition import PCA

# Add SceneSplat to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.scene.dataset_readers import readCamerasFromTransforms
from tools.gaussian_renderer.gaussian_model import GaussianModel
from tools.gaussian_renderer import render
from tools.utils.camera_utils import cameraList_from_camInfos
from tools.utils.graphics_utils import focal2fov


def load_occamlgs_checkpoint(
    checkpoint_path: str,
    sh_degree: int = 3,
    device: str = "cuda"
) -> Tuple[GaussianModel, dict]:
    """
    Load OccamLGS checkpoint and return GaussianModel with language features.

    Args:
        checkpoint_path: Path to checkpoint file
        sh_degree: Spherical harmonics degree
        device: Device to load tensors on

    Returns:
        GaussianModel with loaded language features
        Dictionary with checkpoint metadata
    """
    print(f"Loading OccamLGS checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Parse checkpoint format
    if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        model_params, iteration = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

    print(f"  Checkpoint iteration: {iteration}")
    print(f"  Model params tuple has {len(model_params)} elements")

    # Extract Gaussian parameters
    # OccamLGS format: (active_sh_degree, _xyz, _features_dc, _features_rest,
    #                  _scaling, _rotation, _opacity, _language_feature, ...)
    if len(model_params) >= 8:
        (active_sh_degree, xyz, features_dc, features_rest,
         scaling, rotation, opacity, language_features) = model_params[:8]

        # Create Gaussian model
        gaussians = GaussianModel(sh_degree)
        gaussians.active_sh_degree = active_sh_degree
        gaussians._xyz = xyz.to(device).requires_grad_(True)
        gaussians._features_dc = features_dc.to(device)
        gaussians._features_rest = features_rest.to(device)
        gaussians._scaling = scaling.to(device)
        gaussians._rotation = rotation.to(device)
        gaussians._opacity = opacity.to(device)

        # Store language features
        gaussians._language_feature = language_features.to(device).detach()

        num_gaussians = xyz.shape[0]
        lang_feat_dim = language_features.shape[1]

        print(f"  Loaded {num_gaussians} Gaussians")
        print(f"  Language features: shape={language_features.shape}, dtype={language_features.dtype}")

        metadata = {
            'iteration': iteration,
            'num_gaussians': num_gaussians,
            'lang_feat_dim': lang_feat_dim,
        }

        return gaussians, metadata
    else:
        raise ValueError(f"Expected at least 8 elements in model_params, got {len(model_params)}")


def apply_svd_to_language_features(
    gaussians: GaussianModel,
    svd_rank: int = 16,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply SVD compression to language features.

    Args:
        gaussians: GaussianModel with _language_feature attribute
        svd_rank: Target rank for SVD compression
        device: Device for computations

    Returns:
        compressed_features: [N, svd_rank] compressed language features
        singular_values: [svd_rank] singular values
        svd_info: Dictionary with SVD statistics
    """
    print(f"\nApplying SVD compression to {svd_rank} dimensions...")

    # Get language features
    lang_features = gaussians._language_feature  # [N, D]
    N, D = lang_features.shape

    print(f"  Original features: {N} x {D}")

    # Move to CPU for sklearn SVD (more stable)
    lang_features_np = lang_features.detach().cpu().numpy()

    # Apply SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=svd_rank, random_state=42)
    compressed_features = svd.fit_transform(lang_features_np)  # [N, svd_rank]

    # Get singular values
    singular_values = svd.singular_values_  # [svd_rank]

    # Compute explained variance
    explained_variance = svd.explained_variance_ratio_
    cumulative_variance = np.sum(explained_variance)

    # Update language features with compressed version
    gaussians._language_feature = torch.from_numpy(compressed_features).to(device).float()

    print(f"  Compressed features: {N} x {svd_rank}")
    print(f"  Singular values range: [{singular_values[0]:.4f}, {singular_values[-1]:.4f}]")
    print(f"  Explained variance ratio: {cumulative_variance:.4f} ({cumulative_variance*100:.2f}%)")

    svd_info = {
        'original_dim': D,
        'compressed_dim': svd_rank,
        'singular_values': singular_values,
        'explained_variance_ratio': explained_variance,
        'cumulative_variance': cumulative_variance,
    }

    return compressed_features, singular_values, svd_info


def find_ply_file(scene_path: Path) -> Path:
    """Find the PLY file in the scene directory."""
    point_cloud_dir = scene_path / "point_cloud"
    if point_cloud_dir.exists():
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

    input_ply = scene_path / "input.ply"
    if input_ply.exists():
        return input_ply

    raise FileNotFoundError(f"No PLY file found in {scene_path}")


def process_scene(
    scene_path: str,
    checkpoint_name: str = "chkpnt30000_langfeat_1.pth",
    svd_rank: int = 16,
    feature_level: int = 1,
    output_dir: str = None,
    device: str = "cuda"
) -> dict:
    """Process a single scene: load checkpoint, apply SVD, render with PCA visualization."""
    scene_path = Path(scene_path)
    scene_name = scene_path.name

    print(f"\n{'='*60}")
    print(f"Processing Scene: {scene_name}")
    print(f"{'='*60}")

    checkpoint_path = scene_path / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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

    # Load checkpoint with language features
    print("\nLoading OccamLGS checkpoint...")
    gaussians, metadata = load_occamlgs_checkpoint(str(checkpoint_path), device=device)

    # Apply SVD compression
    compressed_features, singular_values, svd_info = apply_svd_to_language_features(
        gaussians, svd_rank=svd_rank, device=device
    )

    # Create output directories
    if output_dir is not None:
        output_path = Path(output_dir) / scene_name / f"svd_r{svd_rank}_level{feature_level}"
        output_path.mkdir(parents=True, exist_ok=True)

        render_path = output_path / "renders"
        pca_path = output_path / "pca_vis"
        render_path.mkdir(exist_ok=True)
        pca_path.mkdir(exist_ok=True)
    else:
        render_path = None
        pca_path = None

    # Render
    print(f"\nProcessing {len(cameras)} cameras...")
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    results = {}
    for camera_idx, camera in enumerate(cameras):
        print(f"\n{'='*40}")
        print(f"Camera {camera_idx}/{len(cameras)}: {camera.image_name}")
        print(f"{'='*40}")

        # Clear GPU cache
        torch.cuda.empty_cache()

        # Render feature map with include_feature=True
        print("  Rendering feature map...")
        with torch.no_grad():
            render_pkg = render(camera, gaussians, None, background,
                               include_feature=True, feature_level=feature_level)
            feature_map = render_pkg["render"]  # [D, H, W] where D=svd_rank

        # Move to CPU to free GPU memory
        feature_map = feature_map.cpu()
        torch.cuda.empty_cache()

        feat_dim, H, W = feature_map.shape
        print(f"    Feature map shape: {feature_map.shape}, Range: [{feature_map.min():.3f}, {feature_map.max():.3f}]")

        # Save original feature map (rendered with channels as RGB)
        # Take first 3 channels for RGB visualization
        if feat_dim >= 3:
            # Use first 3 channels as RGB, normalize to [0, 1]
            render_rgb = feature_map[:3].detach()
            render_rgb_min = render_rgb.amin(dim=[1, 2], keepdim=True)
            render_rgb_max = render_rgb.amax(dim=[1, 2], keepdim=True)
            render_rgb_norm = (render_rgb - render_rgb_min) / (render_rgb_max - render_rgb_min + 1e-8)

            if render_path is not None:
                render_file = render_path / f"{camera.image_name}.jpg"
                torchvision.utils.save_image(render_rgb_norm, str(render_file))
                print(f"    Saved feature render: {render_file}")

        # Apply PCA visualization (following feature_map_renderer.py approach)
        print("  Applying PCA visualization...")
        feature_map_reshaped = feature_map.reshape(feat_dim, -1).T.numpy()  # (H*W, feat_dim)

        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(feature_map_reshaped)  # (H*W, 3)

        # Normalize to [0, 1]
        pca_normalized = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0) + 1e-8)

        # Reshape back to [H, W, 3]
        pca_vis = pca_normalized.reshape(H, W, 3)

        # Convert to tensor and save
        pca_tensor = torch.from_numpy(pca_vis).permute(2, 0, 1)  # [3, H, W]

        if pca_path is not None:
            # Load original rendered image for comparison
            checkpoint_basename = Path(checkpoint_name).stem  # Remove .pth extension
            # Remove "chkpnt" prefix if present to match directory naming convention
            if checkpoint_basename.startswith("chkpnt"):
                checkpoint_basename = checkpoint_basename.replace("chkpnt", "", 1)
            original_render_dir = scene_path / "test" / f"ours_{checkpoint_basename}" / "renders"

            concat_tensor = pca_tensor  # Default to PCA only
            loaded_original = False

            print(f"    Looking for original renders in: {original_render_dir}")

            if original_render_dir.exists():
                # Try to find the original rendered image
                original_image_path = original_render_dir / f"{camera.image_name}.png"
                if not original_image_path.exists():
                    original_image_path = original_render_dir / f"{camera.image_name}.jpg"

                if original_image_path.exists():
                    try:
                        # Load original image
                        from PIL import Image as PILImage
                        original_img = PILImage.open(original_image_path).convert("RGB")
                        original_tensor = torch.from_numpy(np.array(original_img)).permute(2, 0, 1).float() / 255.0

                        # Resize to match PCA tensor if needed
                        if original_tensor.shape[1:] != pca_tensor.shape[1:]:
                            original_tensor = torch.nn.functional.interpolate(
                                original_tensor.unsqueeze(0),
                                size=pca_tensor.shape[1:],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)

                        # Concatenate horizontally: original (left) + PCA (right)
                        concat_tensor = torch.cat([original_tensor, pca_tensor], dim=2)
                        loaded_original = True
                        print(f"    Loaded original render: {original_image_path}")
                    except Exception as e:
                        print(f"    Warning: Failed to load original render: {e}")
                else:
                    print(f"    Original render not found: {original_image_path}")
            else:
                print(f"    Original render directory not found: {original_render_dir}")

            # Save concatenated or PCA-only image
            pca_file = pca_path / f"{camera.image_name}.jpg"
            torchvision.utils.save_image(concat_tensor, str(pca_file))
            if loaded_original:
                print(f"    Saved concatenated (original + PCA): {pca_file}")
            else:
                print(f"    Saved PCA visualization: {pca_file}")

        # Store results
        key = f"{camera.image_name}"
        results[key] = {
            "camera_id": camera.uid,
            "camera_name": camera.image_name,
            "feat_dim": feat_dim,
            "height": H,
            "width": W,
        }

    # Save SVD info
    if output_dir is not None:
        info_file = output_path / "svd_info.json"
        with open(info_file, 'w') as f:
            # Convert numpy types for JSON serialization
            svd_info_json = {
                'original_dim': int(svd_info['original_dim']),
                'compressed_dim': int(svd_info['compressed_dim']),
                'cumulative_variance': float(svd_info['cumulative_variance']),
                'explained_variance_ratio': [float(v) for v in svd_info['explained_variance_ratio']],
                'singular_values': [float(v) for v in svd_info['singular_values']],
            }
            json.dump(svd_info_json, f, indent=2)
        print(f"\n  Saved SVD info: {info_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Render OccamLGS Language Features with SVD Compression and PCA Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render with SVD-16 compression
  python tools/visualization/render_occamlgs_svd_features.py \\
      --scenes /new_data/cyf/projects/OccamLGS/output/scannet-origin/scene0000_00 \\
      --svd-rank 16 \\
      --output-dir ./svd_render_results

  # Use different SVD rank
  python tools/visualization/render_occamlgs_svd_features.py \\
      --scenes /new_data/cyf/projects/OccamLGS/output/scannet-origin/scene0000_00 \\
      --svd-rank 32 \\
      --num-views 5 \\
      --output-dir ./svd_render_results

  # Use different feature level
  python tools/visualization/render_occamlgs_svd_features.py \\
      --scenes /new_data/cyf/projects/OccamLGS/output/scannet-origin/scene0000_00 \\
      --checkpoint chkpnt30000_langfeat_2.pth \\
      --feature-level 2 \\
      --svd-rank 16 \\
      --output-dir ./svd_render_results
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
        "--svd-rank", "-r",
        type=int,
        default=16,
        help="SVD rank for compression (default: 16)"
    )
    parser.add_argument(
        "--feature-level", "-l",
        type=int,
        default=1,
        help="Feature level to use (default: 1)"
    )
    parser.add_argument(
        "--num-views", "-n",
        type=int,
        default=None,
        help="[DEPRECATED] Now renders all test views by default"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./svd_render_results",
        help="Output directory (default: ./svd_render_results)"
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
                checkpoint_name=args.checkpoint,
                svd_rank=args.svd_rank,
                feature_level=args.feature_level,
                output_dir=args.output_dir,
                device=args.device
            )
            print(f"\n✓ Successfully processed: {scene_path}")
        except Exception as e:
            print(f"\n✗ Error processing {scene_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("All scenes processed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
