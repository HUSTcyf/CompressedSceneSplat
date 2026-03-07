#!/usr/bin/env python3
"""
Render Gaussian Splatting view with mask overlay using gsplat.

This script:
1. Loads Gaussian model checkpoint from gaussian_results/lerf_ovs
2. Loads scene info from datasets/lerf_ovs (COLMAP format)
3. Renders the specified view using gsplat
4. Loads mask from eval_results/LERF-SceneSplat
5. Applies mask to rendered image (darkens areas outside mask)
6. Saves result with same filename as mask

Usage:
    python tools/render_view_with_mask_gsplat.py --scene figurines --view-id 40
    python tools/render_view_with_mask_gsplat.py --scene figurines --view-id 40 --mask-name "chosen_waldo.png"
    python tools/render_view_with_mask_gsplat.py --scene figurines --view-id 40 --all-masks
"""

import os
import sys
import argparse
import math
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image

# Import PROJECT_ROOT - handle both script and module execution
try:
    from .. import PROJECT_ROOT  # Relative import when run as module
except ImportError:
    # Fallback when run as script: add project root to sys.path FIRST
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

# Import Gaussian Splatting modules using absolute imports
from tools.gaussian_renderer import GaussianModel, render
from tools.scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat

# Default paths
CHECKPOINT_ROOT = "/new_data/cyf/projects/SceneSplat/gaussian_results/lerf_ovs"
DATASET_ROOT = "/new_data/cyf/projects/SceneSplat/datasets/lerf_ovs"
EVAL_RESULTS_ROOT = "/new_data/cyf/projects/SceneSplat/eval_results/LERF-SceneSplat"
OUTPUT_ROOT = "/new_data/cyf/projects/SceneSplat/output_rendered_with_mask"
EVAL_RESULTS_ROOT = "/new_data/cyf/projects/OccamLGS/eval_results/LERF-origin"


class MiniCamera:
    """Minimal camera class compatible with gaussian_renderer.render()"""
    def __init__(
        self,
        image_width: int,
        image_height: int,
        FoVx: float,
        FoVy: float,
        world_view_transform: torch.Tensor,
        full_proj_transform: torch.Tensor,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform


def find_checkpoint(scene: str) -> str:
    """Find available checkpoint file for a scene.

    Args:
        scene: Scene name

    Returns:
        Path to checkpoint file

    Raises:
        FileNotFoundError: If no checkpoint found
    """
    scene_dir = f"{CHECKPOINT_ROOT}/{scene}"

    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    # List of possible checkpoint paths to try, in order of preference
    possible_paths = [
        # Direct checkpoint files in scene root
        f"{scene_dir}/chkpnt30000_langfeat_0.pth",
        f"{scene_dir}/occam-chkpnt30000_langfeat_0.pth",
        f"{scene_dir}/chkpnt30000.pth",
        f"{scene_dir}/checkpoint.pth",

        # Checkpoints in ckpts/ subdirectory
        f"{scene_dir}/ckpts/chkpnt30000.pth",
        f"{scene_dir}/ckpts/chkpnt30000_langfeat_0.pth",
        f"{scene_dir}/ckpts/chkpnt_last.pth",

        # Checkpoints in test/ subdirectory
        f"{scene_dir}/test/chkpnt30000.pth",
        f"{scene_dir}/test/chkpnt_last.pth",
    ]

    # Try each possible path
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found checkpoint: {path}")
            return path

    # If none found, list available files
    print(f"No checkpoint found in standard locations.")
    print(f"Available files in {scene_dir}:")
    for root, dirs, files in os.walk(scene_dir):
        for file in files:
            if file.endswith('.pth'):
                full_path = os.path.join(root, file)
                print(f"  {full_path}")

    raise FileNotFoundError(f"No checkpoint found for scene: {scene}")


def load_checkpoint(scene: str, checkpoint_name: str = None) -> GaussianModel:
    """Load Gaussian model checkpoint.

    Args:
        scene: Scene name (e.g., "figurines")
        checkpoint_name: Checkpoint file name (if None, auto-detect)

    Returns:
        GaussianModel instance
    """
    # Find checkpoint if not specified
    if checkpoint_name is None:
        checkpoint_path = find_checkpoint(scene)
    else:
        # Use specified checkpoint path (can be relative or absolute)
        if os.path.isabs(checkpoint_name):
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = f"{CHECKPOINT_ROOT}/{scene}/{checkpoint_name}"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle SceneSplat/OccamLGS tuple format
    if isinstance(checkpoint, tuple):
        inner_tuple, iteration = checkpoint

        # Extract Gaussian parameters
        # inner_tuple contains:
        # 0: flags/N
        # 1: xyz/means [N, 3]
        # 2: features_dc/sh0 [N, 1, 3]
        # 3: features_rest/shN [N, 15, 3]
        # 4: scales [N, 3]
        # 5: rotations/quats [N, 4]
        # 6: opacities [N]
        # 7: lang_feat [N, 768]
        # ...

        # Create GaussianModel
        gaussian_model = GaussianModel(sh_degree=3)

        # Load parameters
        xyz = inner_tuple[1].detach().cpu()  # [N, 3]
        features_dc = inner_tuple[2].detach().cpu()  # [N, 1, 3]
        features_rest = inner_tuple[3].detach().cpu()  # [N, 15, 3]
        scales = inner_tuple[4].detach().cpu()  # [N, 3]
        rotations = inner_tuple[5].detach().cpu()  # [N, 4]
        opacities = inner_tuple[6].detach().cpu()  # [N]

        # Set as model parameters
        gaussian_model._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
        gaussian_model._features_dc = nn.Parameter(features_dc.cuda().requires_grad_(True))
        gaussian_model._features_rest = nn.Parameter(features_rest.cuda().requires_grad_(True))
        gaussian_model._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        gaussian_model._rotation = nn.Parameter(rotations.cuda().requires_grad_(True))
        gaussian_model._opacity = nn.Parameter(opacities.cuda().requires_grad_(True))

        # Initialize auxiliary tensors
        gaussian_model.xyz_gradient_accum = torch.zeros((xyz.shape[0], 1), device="cuda")
        gaussian_model.denom = torch.zeros((xyz.shape[0], 1), device="cuda")
        gaussian_model.max_radii2D = torch.zeros((xyz.shape[0]), device="cuda")
        gaussian_model.active_sh_degree = 3

        print(f"  Loaded {xyz.shape[0]} Gaussians")
        print(f"  SH degree: {gaussian_model.active_sh_degree}")

        return gaussian_model

    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")


def load_colmap_cameras(scene: str) -> Tuple[dict, dict]:
    """Load COLMAP cameras from dataset.

    Args:
        scene: Scene name

    Returns:
        Tuple of (cameras_dict, images_dict)
    """
    colmap_path = f"{DATASET_ROOT}/{scene}/sparse/0"

    if not os.path.exists(colmap_path):
        raise FileNotFoundError(f"COLMAP directory not found: {colmap_path}")

    print(f"Loading COLMAP data from: {colmap_path}")

    # Load cameras
    cameras_path = os.path.join(colmap_path, "cameras.bin")
    cameras = read_intrinsics_binary(cameras_path)

    # Load images (extrinsics)
    images_path = os.path.join(colmap_path, "images.bin")
    images = read_extrinsics_binary(images_path)

    print(f"  Loaded {len(cameras)} cameras, {len(images)} images")

    return cameras, images


def find_camera_by_view_id(
    cameras: dict,
    images: dict,
    view_id: int
) -> Optional[Tuple[dict, dict]]:
    """Find camera and image by view ID.

    The view_id from eval_results folder names maps to frame numbers as:
    - view_id = 40 -> frame_00041.jpg (frame_num = 41)
    - view_id = 104 -> frame_00105.jpg (frame_num = 105)
    So: frame_num = view_id + 1

    Args:
        cameras: COLMAP cameras dict
        images: COLMAP images dict
        view_id: View ID from eval_results folder

    Returns:
        Tuple of (camera, image) or None
    """
    # Convert view_id to frame number (add 1 offset)
    frame_num = view_id + 1

    # Try to find image by matching frame number in name
    for img_id, image in images.items():
        img_name = image.name

        # Extract frame number from name (e.g., "frame_00041.jpg" -> 41)
        if 'frame_' in img_name:
            name_frame_num = int(img_name.split('_')[1].split('.')[0])
            if name_frame_num == frame_num:
                camera_id = image.camera_id
                if camera_id in cameras:
                    print(f"  Matched view_id={view_id} to frame_{name_frame_num:05d}.jpg (COLMAP image_id={img_id})")
                    return cameras[camera_id], image

    # Try direct ID match (fallback)
    if view_id in images:
        image = images[view_id]
        camera_id = image.camera_id
        if camera_id in cameras:
            print(f"  Matched view_id={view_id} to COLMAP image_id={view_id} (direct match)")
            return cameras[camera_id], image

    return None


def build_camera(camera: dict, image: dict) -> MiniCamera:
    """Build MiniCamera from COLMAP camera and image.

    Args:
        camera: COLMAP camera dict
        image: COLMAP image dict

    Returns:
        MiniCamera instance
    """
    # Extract camera parameters
    width = camera.width
    height = camera.height

    # Extract focal lengths
    if camera.model == "PINHOLE":
        fx, fy = camera.params[0], camera.params[1]
        cx, cy = camera.params[2], camera.params[3]
    elif camera.model == "SIMPLE_PINHOLE":
        f = camera.params[0]
        fx = fy = f
        cx, cy = width / 2.0, height / 2.0
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")

    # Calculate FoV
    FoVx = 2 * math.atan(width / (2 * fx))
    FoVy = 2 * math.atan(height / (2 * fy))

    # Build world view transformation matrix
    # COLMAP stores: qvec (quaternion for rotation), tvec (translation)
    # World-to-camera transformation
    qvec = image.qvec
    tvec = image.tvec

    # Convert quaternion to rotation matrix
    R = qvec2rotmat(qvec)  # [3, 3]

    # Build world-to-camera transformation matrix [4, 4]
    # | R   t |
    # | 0   1 |
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec

    # Convert to torch tensor and transpose (world_view_transform is [4, 4])
    world_view_transform = torch.from_numpy(w2c).float().cuda().transpose(0, 1)

    # Build full projection matrix (not really used in rendering, but needed for API)
    full_proj_transform = torch.eye(4, device="cuda")

    return MiniCamera(
        image_width=width,
        image_height=height,
        FoVx=FoVx,
        FoVy=FoVy,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )


def render_view(
    gaussian_model: GaussianModel,
    camera: MiniCamera,
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """Render Gaussian model for a camera view.

    Args:
        gaussian_model: GaussianModel instance
        camera: MiniCamera instance
        bg_color: Background color (RGB, 0-1 range)

    Returns:
        Rendered RGB image [H, W, 3]
    """
    bg_color_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Render
    print(f"  Rendering view of size {camera.image_width}x{camera.image_height}...")
    output = render(camera, gaussian_model, None, bg_color_tensor)

    # Extract rendered image
    rendered_image = output["render"]  # [3, H, W]
    rendered_image = rendered_image.permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 3]

    # Convert to uint8
    rendered_image = np.clip(rendered_image, 0, 1)
    rendered_image = (rendered_image * 255).astype(np.uint8)

    return rendered_image


def load_mask(scene: str, view_id: int, mask_name: str) -> np.ndarray:
    """Load mask from eval results.

    Args:
        scene: Scene name
        view_id: View ID
        mask_name: Mask filename (e.g., "chosen_waldo.png")

    Returns:
        Mask array [H, W] with values 0-255
    """
    mask_path = f"{EVAL_RESULTS_ROOT}/{scene}/{view_id}/{mask_name}"

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    print(f"Loading mask from: {mask_path}")
    mask = Image.open(mask_path)
    mask_array = np.array(mask)  # [H, W]

    print(f"  Mask shape: {mask_array.shape}, dtype: {mask_array.dtype}")

    return mask_array


def apply_mask_to_image(
    image: np.ndarray,
    mask: np.ndarray,
    darkening_factor: float = 0.3
) -> np.ndarray:
    """Apply mask to image (darken areas outside mask).

    Args:
        image: RGB image [H, W, 3]
        mask: Binary mask [H, W] with values 0-255
        darkening_factor: Factor to darken non-mask regions (0-1)

    Returns:
        Image with mask applied [H, W, 3]
    """
    # Ensure mask is same size as image
    if mask.shape != image.shape[:2]:
        print(f"  Warning: Mask shape {mask.shape} != image shape {image.shape[:2]}")
        # Resize mask to match image
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize((image.shape[1], image.shape[0]))
        mask = np.array(mask_pil)
        print(f"  Resized mask to: {mask.shape}")

    # Normalize mask to [0, 1]
    mask_normalized = mask / 255.0  # [H, W]

    # Create result image
    result = image.copy().astype(np.float32)

    # Darken regions where mask is 0
    mask_expanded = mask_normalized[:, :, np.newaxis]

    # Apply darkening
    result = result * mask_expanded + result * (1 - mask_expanded) * darkening_factor

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def save_result(image: np.ndarray, output_path: str):
    """Save result image.

    Args:
        image: RGB image [H, W, 3]
        output_path: Output file path
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save image
    img_pil = Image.fromarray(image)
    img_pil.save(output_path)
    print(f"Saved result to: {output_path}\n")


def list_masks_in_view(scene: str, view_id: int) -> List[str]:
    """List all mask files in a view directory.

    Args:
        scene: Scene name
        view_id: View ID

    Returns:
        List of mask filenames
    """
    view_dir = f"{EVAL_RESULTS_ROOT}/{scene}/{view_id}"

    if not os.path.exists(view_dir):
        return []

    mask_files = []
    for file in os.listdir(view_dir):
        if file.endswith('.png'):
            mask_files.append(file)

    return sorted(mask_files)


def process_single_mask(
    scene: str,
    view_id: int,
    mask_name: str,
    checkpoint_name: str = "chkpnt30000_langfeat_0.pth",
    darkening_factor: float = 0.3
):
    """Process a single mask: render, apply mask, save.

    Args:
        scene: Scene name
        view_id: View ID
        mask_name: Mask filename
        checkpoint_name: Checkpoint file name
        darkening_factor: Darkening factor for non-mask regions
    """
    print(f"\n{'='*70}")
    print(f"Processing: scene={scene}, view={view_id}, mask={mask_name}")
    print(f"{'='*70}\n")

    # Load Gaussian model
    gaussian_model = load_checkpoint(scene, checkpoint_name)

    # Load COLMAP cameras
    cameras, images = load_colmap_cameras(scene)

    # Find camera for this view
    result = find_camera_by_view_id(cameras, images, view_id)
    if result is None:
        print(f"Error: Camera not found for view_id={view_id}")
        print(f"  Available images:")
        for img_id, img in list(images.items())[:10]:
            print(f"    {img_id}: {img.name}")
        if len(images) > 10:
            print(f"    ... and {len(images) - 10} more")
        return

    camera_colmap, image_colmap = result
    print(f"  Found camera: id={image_colmap.id}, name={image_colmap.name}")

    # Build camera
    camera = build_camera(camera_colmap, image_colmap)

    # Load mask
    mask = load_mask(scene, view_id, mask_name)

    # Render view
    print(f"\nRendering...")
    rendered = render_view(gaussian_model, camera)

    # Apply mask
    print(f"\nApplying mask...")
    result_image = apply_mask_to_image(rendered, mask, darkening_factor)

    # Save result
    output_path = f"{OUTPUT_ROOT}/{scene}/{view_id}/{mask_name}"
    save_result(result_image, output_path)

    print(f"\n{'='*70}")
    print(f"Done!")
    print(f"{'='*70}\n")


def process_all_masks(
    scene: str,
    view_id: int,
    checkpoint_name: str = "chkpnt30000_langfeat_0.pth",
    darkening_factor: float = 0.3
):
    """Process all masks in a view directory.

    Args:
        scene: Scene name
        view_id: View ID
        checkpoint_name: Checkpoint file name
        darkening_factor: Darkening factor for non-mask regions
    """
    # Get all mask files
    mask_files = list_masks_in_view(scene, view_id)

    if not mask_files:
        print(f"No mask files found in {EVAL_RESULTS_ROOT}/{scene}/{view_id}")
        return

    # Filter out subdirectories
    mask_files = [f for f in mask_files if not f.startswith('.')]

    print(f"Found {len(mask_files)} mask files")

    # Load Gaussian model and COLMAP data once
    gaussian_model = load_checkpoint(scene, checkpoint_name)
    cameras, images = load_colmap_cameras(scene)

    # Find camera
    result = find_camera_by_view_id(cameras, images, view_id)
    if result is None:
        print(f"Error: Camera not found for view_id={view_id}")
        return

    camera_colmap, image_colmap = result
    print(f"Using camera: id={image_colmap.id}, name={image_colmap.name}")

    # Build camera
    camera = build_camera(camera_colmap, image_colmap)

    # Render once
    print(f"\nRendering view...")
    rendered = render_view(gaussian_model, camera)

    # Process each mask
    for i, mask_name in enumerate(mask_files):
        print(f"\n[{i+1}/{len(mask_files)}] Processing: {mask_name}")

        # Load mask
        mask = load_mask(scene, view_id, mask_name)

        # Apply mask
        result_image = apply_mask_to_image(rendered, mask, darkening_factor)

        # Save result
        output_path = f"{OUTPUT_ROOT}/{scene}/{view_id}/{mask_name}"
        save_result(result_image, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Render Gaussian view with mask overlay using gsplat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single mask (auto-detect checkpoint)
  python tools/render_view_with_mask_gsplat.py --scene figurines --view-id 40 --mask-name "chosen_waldo.png"

  # Process single mask with specific checkpoint
  python tools/render_view_with_mask_gsplat.py --scene ramen --view-id 104 --mask-name "chosen_jake.png" --checkpoint "ckpts/chkpnt30000.pth"

  # Process all masks in view
  python tools/render_view_with_mask_gsplat.py --scene figurines --view-id 40 --all-masks

  # List available masks
  python tools/render_view_with_mask_gsplat.py --scene figurines --view-id 40 --list-masks

  # Custom darkening factor (default 0.3, lower = darker)
  python tools/render_view_with_mask_gsplat.py --scene figurines --view-id 40 --mask-name "chosen_waldo.png" --darken 0.2
        """
    )

    parser.add_argument("--scene", type=str, required=True,
                        help="Scene name (e.g., figurines)")
    parser.add_argument("--view-id", type=int, required=True,
                        help="View ID (e.g., 40)")
    parser.add_argument("--mask-name", type=str, default=None,
                        help="Mask filename (e.g., chosen_waldo.png)")
    parser.add_argument("--all-masks", action="store_true",
                        help="Process all masks in the view directory")
    parser.add_argument("--list-masks", action="store_true",
                        help="List all available masks and exit")
    parser.add_argument("--darken", type=float, default=0.3,
                        help="Darkening factor for non-mask regions (0-1, default 0.3)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint file name or path (default: auto-detect)")

    args = parser.parse_args()

    # List masks mode
    if args.list_masks:
        mask_files = list_masks_in_view(args.scene, args.view_id)
        print(f"Found {len(mask_files)} mask files in {args.scene}/{args.view_id}:")
        for mask_file in mask_files:
            print(f"  - {mask_file}")
        return

    # Validate arguments
    if not args.all_masks and args.mask_name is None:
        parser.error("Either --mask-name or --all-masks must be specified")

    if args.all_masks and args.mask_name is not None:
        parser.error("Cannot specify both --mask-name and --all-masks")

    # Process
    if args.all_masks:
        process_all_masks(args.scene, args.view_id, args.checkpoint, args.darken)
    else:
        process_single_mask(args.scene, args.view_id, args.mask_name, args.checkpoint, args.darken)


if __name__ == "__main__":
    # Import nn module for Parameter creation
    from torch import nn

    main()
