#!/usr/bin/env python3
"""
Render 2D feature maps from 3D Gaussian Splatting with language features.

This script renders 2D feature maps from a 3DGS checkpoint containing
SigLIP2 language features. The rendered features can then be used for
open-vocabulary querying and instance segmentation evaluation.

Usage:
    python tools/render_2d_features_replica.py \
        --source_path /path/to/replica/office_0 \
        --output_path /path/to/output \
        --checkpoint /path/to/checkpoint_with_features_s.pth \
        --split test
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import json

# Gaussian Splatting imports
from tools.gaussian_renderer import GaussianModel, render
from tools.scene import Scene


def load_checkpoint_with_features(checkpoint_path: str):
    """Load checkpoint containing Gaussian parameters and language features.

    Args:
        checkpoint_path: Path to checkpoint_with_features_s.pth

    Returns:
        Tuple of (model_params, first_iter) where model_params contains:
        - xyz: [N, 3] Gaussian centers
        - features_dc: [N, 3] SH DC coefficients
        - features_rest: [N, 45] SH rest coefficients
        - scaling: [N, 3] Scaling factors
        - rotation: [N, 4] Rotation quaternions
        - opacity: [N, 1] Opacity values
        - language_features: [N, 768] SigLIP2 features
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        model_params, first_iter = checkpoint
    elif isinstance(checkpoint, tuple) and len(checkpoint) >= 2:
        # SceneSplat format
        model_params = checkpoint[0]
        first_iter = checkpoint[1] if len(checkpoint) > 1 else 0
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    print(f"Checkpoint iteration: {first_iter}")
    print(f"Model params length: {len(model_params)}")

    # Parse model_params tuple (13 elements)
    # Format from render_replica_predictions.py:
    # (active_sh_degree, xyz, features_dc, features_rest,
    #  scaling, rotation, opacity, language_features,
    #  max_radii2D, xyz_gradient_accum, denom,
    #  opt_dict, spatial_lr_scale)

    if len(model_params) == 13:
        (_, xyz, features_dc, features_rest,
         scaling, rotation, opacity, language_features,
         max_radii2D, xyz_gradient_accum, denom,
         opt_dict, spatial_lr_scale) = model_params
    elif len(model_params) == 14:
        # Includes valid_feat_mask
        (_, xyz, features_dc, features_rest,
         scaling, rotation, opacity, language_features,
         max_radii2D, xyz_gradient_accum, denom,
         opt_dict, spatial_lr_scale, _) = model_params
    else:
        raise ValueError(f"Expected 13 or 14 elements, got {len(model_params)}")

    gaussian_data = {
        'xyz': xyz,
        'features_dc': features_dc,
        'features_rest': features_rest,
        'scaling': scaling,
        'rotation': rotation,
        'opacity': opacity,
        'language_features': language_features,
    }

    print(f"Loaded {xyz.shape[0]} Gaussians with {language_features.shape[1]}-dim features")

    return gaussian_data, first_iter


def render_2d_features(
    source_path: str,
    output_path: str,
    checkpoint_path: str,
    split: str = "train",
    feature_level: int = 0,
    visualize: bool = False,
    camera_filter: str = None,
    **kwargs
):
    """Render 2D feature maps from 3D Gaussians.

    Args:
        source_path: Path to Replica scene (e.g., /path/to/replica/office_0)
        output_path: Path to save rendered features
        checkpoint_path: Path to checkpoint_with_features_s.pth
        split: 'train' or 'test' (Replica uses 'train' containing all images)
        feature_level: Feature level (default: 0)
        visualize: Whether to save PCA visualization
        camera_filter: Filter cameras by image name pattern (e.g., 'test_rgb')
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    output_dir = Path(output_path)
    renders_npy_dir = output_dir / "renders_npy"
    renders_vis_dir = output_dir / "renders_vis" if visualize else None

    renders_npy_dir.mkdir(parents=True, exist_ok=True)
    if visualize:
        renders_vis_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    gaussian_data, iteration = load_checkpoint_with_features(checkpoint_path)

    # Apply inverse normalization transform to Gaussian coordinates
    # During training, coordinates are normalized: xyz_norm = (xyz_orig + translate) / radius
    # For rendering, we need to denormalize: xyz_orig = xyz_norm * radius - translate
    print("Applying inverse normalization to Gaussian coordinates...")

    # Get normalization parameters from the scene
    from tools.scene.dataset_readers import readColmapSceneInfo
    temp_scene_info = readColmapSceneInfo(
        source_path,
        "input",  # Use input directory for consistent scene loading
        "",
        eval=False,
        train_test_exp=False,
        llffhold=None
    )
    nerf_norm = temp_scene_info.nerf_normalization

    print(f"NERF normalization: translate={nerf_norm['translate']}, radius={nerf_norm['radius']}")

    # Denormalize Gaussian coordinates
    xyz = gaussian_data['xyz']
    translate = torch.tensor(nerf_norm['translate'], dtype=xyz.dtype, device=xyz.device)
    radius = nerf_norm['radius']

    xyz_denorm = xyz * radius - translate
    gaussian_data['xyz'] = xyz_denorm

    print(f"Original XYZ range: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]")
    print(f"Denorm XYZ range: [{xyz_denorm[:, 0].min():.2f}, {xyz_denorm[:, 0].max():.2f}]")

    # Load camera info directly using dataset_readers
    print("Loading cameras...")
    from tools.scene.dataset_readers import readColmapSceneInfo
    from tools.utils.camera_utils import cameraList_from_camInfos

    class SimpleModelParams:
        def __init__(self, source_path):
            self.source_path = source_path
            self.images = "input"  # Use input directory for 640x480 images
            self.depths = ""
            self.resolution = 1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = False
            self.train_test_exp = False

    model_params = SimpleModelParams(source_path)

    # Use distorted/sparse/0 which has all 360 images at 640x480
    # We'll manually handle the OPENCV camera model by treating it as PINHOLE
    print("Loading cameras from distorted/sparse/0 (640x480, OPENCV model)...")

    # Import necessary functions
    from tools.scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
    from tools.scene.dataset_readers import CameraInfo
    from tools.utils.graphics_utils import focal2fov, getWorld2View2

    # Manually load from distorted/sparse/0
    sparse_path = os.path.join(source_path, "distorted/sparse/0")
    images_folder = os.path.join(source_path, "input")

    # Read camera data from distorted/sparse/0
    cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_path, "images.bin"))
    cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_path, "cameras.bin"))

    # Read test camera names
    test_file = os.path.join(source_path, "sparse/0/test.txt")
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            test_cam_names_list = [line.strip() for line in f]
    else:
        test_cam_names_list = []

    print(f"Found {len(cam_extrinsics)} cameras in distorted/sparse/0")
    print(f"Camera model: {list(cam_intrinsics.values())[0].model}")

    # Manually create camera infos for OPENCV model
    cam_infos = []
    for idx, (key, extr) in enumerate(cam_extrinsics.items()):
        intr = cam_intrinsics[extr.camera_id]

        # For OPENCV model: params = [fx, fy, cx, cy, k1, k2, p1, p2]
        # We'll use only fx, fy, cx, cy (ignore distortion for rendering)
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
        width = intr.width
        height = intr.height

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        image_name = extr.name
        is_test = image_name in test_cam_names_list

        cam_info = CameraInfo(
            uid=intr.id,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            depth_params={},
            image_path=os.path.join(images_folder, image_name),
            image_name=image_name,
            depth_path="",
            width=width,
            height=height,
            is_test=is_test
        )
        cam_infos.append(cam_info)

    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

    # Split into train/test
    train_cam_infos = [c for c in cam_infos if not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    print(f"Loaded {len(train_cam_infos)} train cameras and {len(test_cam_infos)} test cameras")
    print(f"Camera resolution: {train_cam_infos[0].width}x{train_cam_infos[0].height}")

    # Create camera lists
    train_cameras = cameraList_from_camInfos(
        train_cam_infos, 1.0, model_params, False, False
    )
    test_cameras = cameraList_from_camInfos(
        test_cam_infos, 1.0, model_params, False, True
    )

    # Get cameras for specified split
    if split == "train":
        cameras = train_cameras
    elif split == "test":
        cameras = test_cameras
    else:
        raise ValueError(f"Unknown split: {split}")

    # Filter cameras by image name pattern if specified
    if camera_filter:
        cameras = [c for c in cameras if camera_filter in c.image_name]
        print(f"Filtered to {len(cameras)} cameras matching '{camera_filter}'")

    # Create Gaussian model
    print("Creating Gaussian model...")
    gaussians = GaussianModel(sh_degree=0)

    # Set Gaussian parameters from checkpoint
    print("Setting Gaussian parameters...")
    xyz = gaussian_data['xyz'].to(device)
    features_dc = gaussian_data['features_dc'].to(device)
    features_rest = gaussian_data['features_rest'].to(device)
    scaling = gaussian_data['scaling'].to(device)
    rotation = gaussian_data['rotation'].to(device)
    opacity = gaussian_data['opacity'].to(device)
    language_features = gaussian_data['language_features'].to(device)

    gaussians._xyz = xyz
    gaussians._features_dc = features_dc
    gaussians._features_rest = features_rest
    gaussians._scaling = scaling
    gaussians._rotation = rotation
    gaussians._opacity = opacity

    # Set language features
    if not hasattr(gaussians, "_language_features_dict"):
        gaussians._language_features_dict = {}
    gaussians._language_features_dict[feature_level] = language_features

    print(f"Rendering {len(cameras)} views from {split} split...")

    # Background color
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Render each view
    from sklearn.decomposition import PCA

    with torch.no_grad():
        for idx, view in enumerate(tqdm(cameras, desc=f"Rendering {split}")):
            # Clear GPU cache
            torch.cuda.empty_cache()

            # Render with language features
            render_pkg = render(
                view,
                gaussians,
                None,  # pipeline
                background,
                include_feature=True,
                feature_level=feature_level,
            )

            rendered_features = render_pkg["render"]  # [C, H, W]

            # Get image name
            img_name = view.image_name.split('.')[0]

            # Save features as numpy
            features_np = rendered_features.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            np.save(renders_npy_dir / f"{img_name}.npy", features_np)

            # Optional: save PCA visualization
            if visualize:
                H, W, C = features_np.shape
                features_flat = features_np.reshape(-1, C)  # [H*W, C]

                # PCA to 3 components for visualization
                pca = PCA(n_components=3)
                features_3d = pca.fit_transform(features_flat)
                features_3d = (features_3d - features_3d.min(axis=0)) / (features_3d.max(axis=0) - features_3d.min(axis=0) + 1e-8)
                features_3d = features_3d.reshape(H, W, 3)

                # Save as image
                from PIL import Image
                img = Image.fromarray((features_3d * 255).astype(np.uint8))
                img.save(renders_vis_dir / f"{img_name}.png")

    print(f"Saved {len(cameras)} feature maps to {renders_npy_dir}")
    if visualize:
        print(f"Saved visualizations to {renders_vis_dir}")

    # Save metadata
    metadata = {
        'source_path': source_path,
        'checkpoint_path': checkpoint_path,
        'split': split,
        'num_views': len(cameras),
        'feature_dim': int(language_features.shape[1]),
        'num_gaussians': int(xyz.shape[0]),
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {output_dir / 'metadata.json'}")


def main():
    parser = ArgumentParser(description="Render 2D feature maps from 3DGS with language features")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to Replica scene directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save rendered features")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint_with_features_s.pth")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="Which split to render (Replica uses 'train' split containing all images)")
    parser.add_argument("--camera_filter", type=str, default="test_rgb",
                        help="Filter cameras by image name pattern (e.g., 'test_rgb' for Replica test images)")
    parser.add_argument("--feature_level", type=int, default=0,
                        help="Feature level (default: 0)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save PCA visualization of features")

    args = parser.parse_args()

    render_2d_features(
        source_path=args.source_path,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint,
        split=args.split,
        feature_level=args.feature_level,
        visualize=args.visualize,
        camera_filter=args.camera_filter,
    )


if __name__ == "__main__":
    main()
