#!/usr/bin/env python3
"""
Open-Vocabulary Query Visualization for OccamLGS Scenes

This script loads OccamLGS scenes with language features and performs
open-vocabulary text queries, highlighting matched Gaussians in red.
Uses SceneSplat modules with ScanNet-compatible camera loading.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData

# Add SceneSplat to path - use for all imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.scene.dataset_readers import readCamerasFromTransforms, CameraInfo, SceneInfo
from tools.gaussian_renderer.gaussian_model import GaussianModel, BasicPointCloud
from tools.gaussian_renderer import render
from tools.utils.camera_utils import cameraList_from_camInfos
from tools.utils.graphics_utils import focal2fov


SCANNETPP_CATEGORIES = [
    "wall", "floor", "ceiling", "table", "chair", "sofa", "bed",
    "door", "window", "bookshelf", "desk", "monitor", "lamp",
    "cabinet", "trash can", "box", "pillow", "blind", "curtain",
    "picture", "counter", "shelves", "radiator", "rail", "board",
    "bathtub", "toilet", "sink", "paper", "bin", "bottle", "bag",
    "whiteboard", "tv", "keyboard", "mouse", "laptop", "fan",
    "mirror", "refrigerator", "shoe", "plant", "stove"
]


def load_clip_model(clip_model_name: str = "ViT-B/32", device: str = "cuda"):
    """Load CLIP model for text encoding."""
    try:
        import clip
    except ImportError:
        raise ImportError(
            "CLIP is required for text encoding. "
            "Install with: pip install git+https://github.com/openai/CLIP.git"
        )

    print(f"Loading CLIP model: {clip_model_name}")
    model, _ = clip.load(clip_model_name, device=device, download_root="./.cache/clip")
    model.eval()
    return model


def encode_text_queries(
    clip_model,
    queries: List[str],
    device: str = "cuda"
) -> torch.Tensor:
    """Encode text queries using CLIP."""
    import clip

    text_tokens = clip.tokenize(queries).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, p=2, dim=1)

    return text_features


class MockOptimizationParams:
    """Mock OptimizationParams for restore_language_features."""
    def __init__(self):
        pass


def load_occamlgs_checkpoint(
    checkpoint_path: str,
    gaussian_model: GaussianModel,
    device: str = "cuda"
) -> int:
    """
    Load OccamLGS checkpoint and restore language features.
    """
    print(f"Loading OccamLGS checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        model_params, iteration = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

    print(f"  Checkpoint iteration: {iteration}")
    print(f"  Model params tuple has {len(model_params)} elements")

    # Extract language features from model_params tuple
    # OccamLGS format: (active_sh_degree, _xyz, _features_dc, _features_rest,
    #                  _scaling, _rotation, _opacity, _language_feature, ...)
    if len(model_params) >= 8:
        language_feature = model_params[7]

        # Get number of Gaussians from xyz
        num_gaussians = model_params[1].shape[0]

        # Convert to tensor and assign
        if isinstance(language_feature, np.ndarray):
            language_feature = torch.from_numpy(language_feature).float()

        gaussian_model._language_feature = language_feature.to(device).detach()

        print(f"  Loaded {num_gaussians} Gaussians")
        print(f"  Language features: shape={language_feature.shape}, dtype={language_feature.dtype}")

        # Debug: Print Gaussian statistics
        xyz = model_params[1]
        opacity = model_params[6]
        print(f"  XYZ: shape={xyz.shape}, min={xyz.min():.3f}, max={xyz.max():.3f}, mean={xyz.mean():.3f}")
        print(f"  Opacity: shape={opacity.shape}, min={opacity.min():.3f}, max={opacity.max():.3f}, mean={opacity.mean():.3f}")
    else:
        raise ValueError(f"Expected at least 8 elements in model_params, got {len(model_params)}")

    return iteration


def compute_similarity(
    lang_features: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> np.ndarray:
    """Compute cosine similarity between language features and text embeddings."""
    lang_features = lang_features.detach().float()
    text_embeddings = text_embeddings.detach().float()

    lang_feat_norm = F.normalize(lang_features, p=2, dim=1)

    if lang_feat_norm.shape[1] != text_embeddings.shape[1]:
        print(f"Warning: Dimension mismatch - lang_features: {lang_feat_norm.shape[1]}, "
              f"text_embeddings: {text_embeddings.shape[1]}")
        similarity = -torch.cdist(lang_feat_norm, text_embeddings, p=2)
    else:
        similarity = torch.mm(lang_feat_norm, text_embeddings.t())

    return similarity.cpu().numpy()


def render_with_override(
    camera,
    gaussians: GaussianModel,
    pipeline,
    background,
    override_color: torch.Tensor = None
) -> torch.Tensor:
    """Render the scene with optional color override."""
    if override_color is not None:
        original_features_dc = gaussians._features_dc.clone()
        original_features_rest = gaussians._features_rest.clone()

        gaussians._features_dc.data = override_color.unsqueeze(1).contiguous()
        gaussians._features_rest.data.zero_()

    render_result = render(camera, gaussians, pipeline, background)
    rendered_image = render_result["render"]

    if override_color is not None:
        gaussians._features_dc.data = original_features_dc
        gaussians._features_rest.data = original_features_rest

    return rendered_image


def render_and_query(
    gaussians: GaussianModel,
    camera,
    pipeline,
    background: torch.Tensor,
    query_similarities: np.ndarray,
    threshold: float = None,
    percentile: float = 90.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render the scene with and without query highlighting."""
    print("  Rendering original scene...")
    with torch.no_grad():
        original_result = render(camera, gaussians, pipeline, background)
        original_image = original_result["render"]

    print(f"    Original render shape: {original_image.shape}, range: [{original_image.min():.3f}, {original_image.max():.3f}]")

    # Check if any pixels were actually rendered
    nonzero_pixels = (original_image > 0).any().item()
    print(f"    Non-zero pixels: {nonzero_pixels}")

    num_gaussians = gaussians.get_xyz.shape[0]
    original_colors = gaussians._features_dc.squeeze(1)

    # Determine threshold
    if threshold is not None:
        # Use fixed threshold
        matched_mask = query_similarities >= threshold
        threshold_used = threshold
        print(f"    Using fixed threshold: {threshold_used}")
    else:
        # Use percentile-based threshold
        threshold_value = np.percentile(query_similarities, percentile)
        matched_mask = query_similarities >= threshold_value
        threshold_used = f"{percentile}th percentile ({threshold_value:.4f})"
        print(f"    Using percentile threshold: {threshold_used}")

    override_colors = original_colors.clone()
    matched_mask_tensor = torch.tensor(matched_mask, device=original_colors.device)
    override_colors[matched_mask_tensor] = torch.tensor([1.0, 0.0, 0.0], device=original_colors.device)

    print(f"    Matched {matched_mask_tensor.sum()} / {num_gaussians} Gaussians (threshold={threshold_used})")

    print("  Rendering with query highlighting...")
    with torch.no_grad():
        query_image = render_with_override(camera, gaussians, pipeline, background, override_colors)

    print(f"    Query render shape: {query_image.shape}, range: [{query_image.min():.3f}, {query_image.max():.3f}]")

    return original_image, query_image


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
    checkpoint_name: str = "chkpnt30000_langfeat_1.pth",
    queries: List[str] = None,
    clip_model=None,
    clip_model_name: str = "ViT-B/32",
    threshold: float = None,
    percentile: float = 90.0,
    output_dir: str = None,
    device: str = "cuda",
    random_seed: int = None
) -> dict:
    """Process a single scene: load data, perform query, render results."""
    scene_path = Path(scene_path)
    scene_name = scene_path.name

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    print(f"\n{'='*60}")
    print(f"Processing Scene: {scene_name}")
    print(f"{'='*60}")

    checkpoint_path = scene_path / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if queries is None:
        queries = random.sample(SCANNETPP_CATEGORIES, 3)
        print(f"Randomly selected queries: {queries}")
    else:
        print(f"Using provided queries: {queries}")

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

    # Load cameras using SceneSplat's readCamerasFromTransforms with ScanNet format (.jpg)
    print(f"\nLoading cameras from: {source_path}")
    print("  Reading transforms_train.json (ScanNet format)...")
    all_cam_infos = readCamerasFromTransforms(
        str(source_path),
        "transforms_train.json",
        depths_folder="",
        white_background=False,
        is_test=False,
        extension=".jpg"  # ScanNet uses .jpg
    )

    # Split into train/test
    llffhold = 8
    train_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llffhold != 0]
    test_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llffhold == 0]

    print(f"  Train cameras: {len(train_cam_infos)}")
    print(f"  Test cameras: {len(test_cam_infos)}")

    # Use test cameras if available
    all_cameras = test_cam_infos if len(test_cam_infos) > 0 else train_cam_infos
    print(f"  Total available cameras: {len(all_cameras)}")

    # Use all test cameras (no sampling)
    cameras_info = all_cameras

    print(f"\n  Using all {len(cameras_info)} test cameras")

    # Convert to camera objects for rendering
    # Create args object with required attributes
    from types import SimpleNamespace
    camera_args = SimpleNamespace(
        resolution=-1,  # Use original resolution
        data_device=device,
        train_test_exp=False
    )
    cameras = cameraList_from_camInfos(cameras_info, 1.0, camera_args, False, True)

    # Find and load PLY file
    ply_path = find_ply_file(scene_path)
    print(f"\nLoading Gaussians from PLY: {ply_path}")

    # Initialize Gaussian model from PLY
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(str(ply_path))

    # Get the number of Gaussians in PLY file
    num_ply_gaussians = gaussians.get_xyz.shape[0]
    print(f"  PLY file has {num_ply_gaussians} Gaussians")

    # Load checkpoint with language features
    print("\nLoading language features from checkpoint...")
    with torch.no_grad():
        iteration = load_occamlgs_checkpoint(
            str(checkpoint_path),
            gaussians,
            device=device
        )

    # Handle shape mismatch between PLY and checkpoint
    num_lang_gaussians = gaussians._language_feature.shape[0]
    if num_ply_gaussians != num_lang_gaussians:
        print(f"\n  WARNING: Shape mismatch detected!")
        print(f"    PLY Gaussians: {num_ply_gaussians}")
        print(f"    Language features: {num_lang_gaussians}")
        print(f"    Truncating to first {min(num_ply_gaussians, num_lang_gaussians)} Gaussians")

        # Truncate to the smaller size
        min_gaussians = min(num_ply_gaussians, num_lang_gaussians)
        gaussians._language_feature = gaussians._language_feature[:min_gaussians]

        # Also truncate other Gaussian properties to match
        gaussians._xyz = gaussians._xyz[:min_gaussians]
        gaussians._features_dc = gaussians._features_dc[:min_gaussians]
        gaussians._features_rest = gaussians._features_rest[:min_gaussians]
        gaussians._scaling = gaussians._scaling[:min_gaussians]
        gaussians._rotation = gaussians._rotation[:min_gaussians]
        gaussians._opacity = gaussians._opacity[:min_gaussians]

    # Load CLIP model
    if clip_model is None:
        clip_model = load_clip_model(clip_model_name, device)

    print(f"\nProcessing {len(cameras)} cameras for scene...")

    # Set background color (OccamLGS uses black background)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    # Encode text queries
    text_embeddings = encode_text_queries(clip_model, queries, device)
    print(f"\nText embeddings shape: {text_embeddings.shape}")

    # Get language features
    lang_features = gaussians._language_feature
    print(f"Language features shape: {lang_features.shape}")

    # Compute similarity for each query (once per scene, not per camera)
    query_similarities_dict = {}
    for i, query in enumerate(queries):
        query_similarities = compute_similarity(lang_features, text_embeddings[i:i+1])
        query_similarities = query_similarities[:, 0]

        print(f"\nQuery '{query}': mean={query_similarities.mean():.4f}, "
              f"std={query_similarities.std():.4f}, "
              f"max={query_similarities.max():.4f}")

        # Show threshold info
        if threshold is not None:
            print(f"  Using fixed threshold: {threshold}")
        else:
            threshold_value = np.percentile(query_similarities, percentile)
            print(f"  Using {percentile}th percentile threshold: {threshold_value:.4f}")
            print(f"  Matched (score>={threshold_value}): {(query_similarities >= threshold_value).sum()} / {len(query_similarities)}")

        query_similarities_dict[query] = query_similarities

    # Process each camera and query
    results = {}
    for camera_idx, camera in enumerate(cameras):
        print(f"\n{'='*40}")
        print(f"Camera {camera_idx}/{len(cameras)}: {camera.image_name} (uid={camera.uid})")
        print(f"{'='*40}")

        for query in queries:
            print(f"\n  Processing query: '{query}'")

            query_similarities = query_similarities_dict[query]

            # Render and highlight
            original_render, query_render = render_and_query(
                gaussians,
                camera,
                None,  # pipeline
                background,
                query_similarities,
                threshold=threshold,
                percentile=percentile
            )

            # Concatenate images
            concat_image = concatenate_images(original_render, query_render)

            # Save results
            if output_dir is not None:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Sanitize query for filename
                query_filename = query.replace(" ", "_").replace("/", "_")[:50]
                query_filename = "".join(c for c in query_filename if c.isalnum() or c in "_-_")

                # Format: {scene_name}_{query}_{camera_id}.png
                output_file = output_path / f"{scene_name}_{query_filename}_{camera.uid}.png"
                concat_image_uint8 = (concat_image * 255).astype(np.uint8)
                Image.fromarray(concat_image_uint8).save(output_file)
                print(f"    Saved: {output_file}")

            # Store results
            key = f"{query}_{camera.uid}"
            # Calculate matched count properly
            if threshold is not None:
                matched_count = (query_similarities >= threshold).sum()
            else:
                threshold_value = np.percentile(query_similarities, percentile)
                matched_count = (query_similarities >= threshold_value).sum()

            results[key] = {
                "query": query,
                "camera_id": camera.uid,
                "camera_name": camera.image_name,
                "similarities": query_similarities,
                "matched_count": matched_count,
                "concat_image": concat_image
            }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Open-Vocabulary Query Visualization for OccamLGS Scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        default=None,
        help="Text query/queries. If not specified, randomly select from ScanNet++ categories."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./occamlgs_query_results",
        help="Output directory (default: ./occamlgs_query_results)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Fixed similarity threshold (if specified, overrides percentile)"
    )
    parser.add_argument(
        "--percentile", "-p",
        type=float,
        default=90.0,
        help="Percentile threshold (0-100). Highlight top P%% most similar Gaussians (default: 90.0)"
    )
    parser.add_argument(
        "--num-views", "-n",
        type=int,
        default=None,
        help="[DEPRECATED] Now renders all test views by default"
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14-336"],
        help="CLIP model name (default: ViT-B/32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
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

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    clip_model = None
    for scene_path in scene_paths:
        try:
            results = process_scene(
                scene_path=scene_path,
                checkpoint_name=args.checkpoint,
                queries=args.query,
                clip_model=clip_model,
                clip_model_name=args.clip_model,
                threshold=args.threshold,
                percentile=args.percentile,
                output_dir=args.output_dir,
                device=args.device,
                random_seed=args.seed
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
