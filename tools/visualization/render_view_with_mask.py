#!/usr/bin/env python3
"""
Render Gaussian Splatting view with mask overlay.

This script:
1. Loads Gaussian model checkpoint from gaussian_results/lerf_ovs
2. Renders the specified view
3. Loads mask from eval_results/LERF-SceneSplat
4. Applies mask to rendered image (darkens areas outside mask)
5. Saves result with same filename as mask

Usage:
    python tools/render_view_with_mask.py --scene figurines --view-id 40
    python tools/render_view_with_mask.py --scene figurines --view-id 40 --mask-name "chosen_waldo.png"
    python tools/render_view_with_mask.py --scene figurines --view-id 40 --all-masks
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from PIL import Image

# Try to import gsplat for rendering
try:
    from gsplat.rendering import rasterize_gaussians
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Warning: gsplat not available, will try alternative rendering")

# Default paths
CHECKPOINT_ROOT = "/new_data/cyf/projects/SceneSplat/gaussian_results/lerf_ovs"
EVAL_RESULTS_ROOT = "/new_data/cyf/projects/SceneSplat/eval_results/LERF-SceneSplat"
OUTPUT_ROOT = "/new_data/cyf/projects/SceneSplat/output_rendered_with_mask"


def load_checkpoint(scene: str, checkpoint_name: str = "chkpnt30000_langfeat_0.pth") -> dict:
    """Load Gaussian model checkpoint.

    Args:
        scene: Scene name (e.g., "figurines")
        checkpoint_name: Checkpoint file name

    Returns:
        Dictionary containing Gaussian model parameters
    """
    checkpoint_path = f"{CHECKPOINT_ROOT}/{scene}/{checkpoint_name}"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, tuple):
        # OccamLGS/SceneSplat format: (inner_tuple, iteration)
        inner_tuple, iteration = checkpoint

        # inner_tuple contains:
        # 0: N (number of Gaussians or flags)
        # 1: means [N, 3]
        # 2: sh0 [N, 1, 3] or [N, 3] - DC (spherical harmonics degree 0, i.e., color)
        # 3: shN [N, 15, 3] - higher degree spherical harmonics
        # 4: scales [N, 3]
        # 5: quats [N, 4] - rotation quaternions
        # 6: opacities [N]
        # 7: lang_feat [N, 768] - optional
        # 8+: other fields

        means = inner_tuple[1].cpu()  # [N, 3]
        sh0 = inner_tuple[2].cpu()  # [N, 1, 3] or [N, 3]
        scales = inner_tuple[4].cpu()  # [N, 3]
        quats = inner_tuple[5].cpu()  # [N, 4]
        opacities = inner_tuple[6].cpu()  # [N]

        # Extract colors from sh0
        if sh0.dim() == 3:
            colors = sh0.squeeze(1)  # [N, 3]
        else:
            colors = sh0  # [N, 3]

        return {
            'means': means.numpy(),
            'scales': scales.numpy(),
            'quats': quats.numpy(),
            'opacities': opacities.numpy(),
            'colors': colors.numpy(),
        }

    elif isinstance(checkpoint, dict):
        if 'splats' in checkpoint:
            # gsplat format
            splats = checkpoint['splats']
            means = splats['means']  # [N, 3]
            scales = splats['scales']  # [N, 3]
            quats = splats['quats']  # [N, 4]
            opacities = splats['opacities']  # [N]

            # Extract colors from spherical harmonics
            sh0 = splats['sh0']  # [N, 1, 3] or [N, 3]
            if sh0.dim() == 3:
                colors = sh0.squeeze(1)  # [N, 3]
            else:
                colors = sh0  # [N, 3]

            return {
                'means': means.numpy(),
                'scales': scales.numpy(),
                'quats': quats.numpy(),
                'opacities': opacities.numpy(),
                'colors': colors.numpy(),
            }
        else:
            # Plain dict format
            means = checkpoint['means'] if 'means' in checkpoint else checkpoint.get('xyz', None)
            scales = checkpoint.get('scales', None)
            quats = checkpoint.get('quats', checkpoint.get('rotations', None))
            opacities = checkpoint.get('opacities', checkpoint.get('opacity', None))

            if sh0 := checkpoint.get('sh0', checkpoint.get('features_dc', None)):
                colors = sh0.squeeze(-2) if sh0.dim() == 3 else sh0
            else:
                colors = None

            return {
                'means': means.cpu().numpy() if torch.is_tensor(means) else means,
                'scales': scales.cpu().numpy() if torch.is_tensor(scales) else scales,
                'quats': quats.cpu().numpy() if torch.is_tensor(quats) else quats,
                'opacities': opacities.cpu().numpy() if torch.is_tensor(opacities) else opacities,
                'colors': colors.cpu().numpy() if torch.is_tensor(colors) else colors,
            }
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")


def load_cameras(scene: str) -> List[dict]:
    """Load camera parameters from cameras.json.

    Args:
        scene: Scene name

    Returns:
        List of camera dictionaries
    """
    cameras_path = f"{CHECKPOINT_ROOT}/{scene}/cameras.json"

    if not os.path.exists(cameras_path):
        raise FileNotFoundError(f"Cameras file not found: {cameras_path}")

    print(f"Loading cameras from: {cameras_path}")
    with open(cameras_path, 'r') as f:
        cameras = json.load(f)

    print(f"  Loaded {len(cameras)} cameras")
    return cameras


def find_camera_by_view_id(cameras: List[dict], view_id: int) -> Optional[dict]:
    """Find camera by view ID (matches id field or frame number).

    Args:
        cameras: List of camera dictionaries
        view_id: View ID to search for

    Returns:
        Camera dictionary or None
    """
    # First try direct ID match
    for cam in cameras:
        if cam['id'] == view_id:
            return cam

    # Then try to extract from img_name (e.g., "frame_00041.jpg" -> 41)
    for cam in cameras:
        img_name = cam.get('img_name', '')
        if 'frame_' in img_name:
            # Extract frame number from "frame_00041.jpg"
            frame_num = int(img_name.split('_')[1].split('.')[0])
            if frame_num == view_id:
                return cam

    return None


def render_view_gaussians(
    gaussians: dict,
    camera: dict,
    image_size: Tuple[int, int] = (986, 728)
) -> np.ndarray:
    """Render Gaussian model for a single camera view.

    Args:
        gaussians: Gaussian model parameters
        camera: Camera parameters
        image_size: Output image size (width, height)

    Returns:
        Rendered RGB image [H, W, 3]
    """
    if GSPLAT_AVAILABLE:
        return render_view_gsplat(gaussians, camera, image_size)
    else:
        return render_view_simple(gaussians, camera, image_size)


def render_view_gsplat(
    gaussians: dict,
    camera: dict,
    image_size: Tuple[int, int]
) -> np.ndarray:
    """Render using gsplat library."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    width, height = image_size

    # Convert to tensors
    means = torch.from_numpy(gaussians['means']).float().to(device)  # [N, 3]
    scales = torch.from_numpy(gaussians['scales']).float().to(device)  # [N, 3]
    quats = torch.from_numpy(gaussians['quats']).float().to(device)  # [N, 4]
    opacities = torch.from_numpy(gaussians['opacities']).float().to(device)  # [N]
    colors = torch.from_numpy(gaussians['colors']).float().to(device)  # [N, 3]

    # Camera parameters
    position = torch.from_numpy(np.array(camera['position'])).float().to(device)  # [3]
    rotation = torch.from_numpy(np.array(camera['rotation'])).float().to(device)  # [3, 3]
    fx = camera['fx']
    fy = camera['fy']

    # Build view matrix (world to camera)
    # position is camera center in world coords
    # rotation is camera rotation matrix (world to camera)
    view_matrix = torch.eye(4, device=device)
    view_matrix[:3, :3] = rotation
    view_matrix[:3, 3] = position

    # Projection matrix
    cx, cy = width / 2.0, height / 2.0
    near, far = 0.01, 100.0

    # For gsplat, we need to transform Gaussians to camera space
    # This is a simplified rendering - for production, use proper gsplat pipeline
    print(f"  Rendering with gsplat (device: {device})")
    print(f"  Gaussians: {means.shape[0]}")

    # Simplified: just project and render (not full gsplat pipeline)
    # For proper rendering, would need full camera setup
    return render_view_simple(gaussians, camera, image_size)


def render_view_simple(
    gaussians: dict,
    camera: dict,
    image_size: Tuple[int, int]
) -> np.ndarray:
    """Simple projection-based rendering (fallback)."""
    width, height = image_size

    # Extract Gaussian parameters
    means = gaussians['means']  # [N, 3]
    colors = gaussians['colors']  # [N, 3]
    scales = gaussians['scales']  # [N, 3]
    opacities = gaussians['opacities']  # [N]

    # Camera parameters
    position = np.array(camera['position'])  # [3]
    rotation = np.array(camera['rotation'])  # [3, 3]
    fx = camera['fx']
    fy = camera['fy']

    cx, cy = width / 2.0, height / 2.0

    # Transform points to camera space
    # World to camera: p_cam = R * (p_world - t)
    points_cam = (rotation @ (means - position).T).T  # [N, 3]

    # Filter points in front of camera
    valid_depth = points_cam[:, 2] > 0
    points_cam = points_cam[valid_depth]
    colors = colors[valid_depth]
    opacities = opacities[valid_depth]
    scales = scales[valid_depth]

    if len(points_cam) == 0:
        print("  Warning: No points in front of camera")
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Project to image plane
    depths = points_cam[:, 2]
    u = fx * points_cam[:, 0] / depths + cx
    v = fy * points_cam[:, 1] / depths + cy

    # Filter points within image bounds
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_bounds].astype(int)
    v = v[in_bounds].astype(int)
    colors = colors[in_bounds]
    opacities = opacities[in_bounds]
    depths = depths[in_bounds]

    print(f"  Rendering {len(u)} visible points")

    # Simple depth buffer rendering
    rendered = np.zeros((height, width, 3), dtype=np.float32)
    depth_buffer = np.zeros((height, width), dtype=np.float32) + np.inf

    # Sort by depth (far to near) for proper blending
    sort_order = np.argsort(depths)[::-1]

    for i in sort_order:
        x, y = u[i], v[i]
        if depths[i] < depth_buffer[y, x]:
            alpha = opacities[i]
            # Alpha blend
            rendered[y, x] = rendered[y, x] * (1 - alpha) + colors[i] * alpha
            depth_buffer[y, x] = depths[i]

    # Clip and convert to uint8
    rendered = np.clip(rendered, 0, 1)
    rendered = (rendered * 255).astype(np.uint8)

    return rendered


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
    print(f"  Mask unique values: {np.unique(mask_array)}")

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
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(mask)
        mask_pil = mask_pil.resize((image.shape[1], image.shape[0]))
        mask = np.array(mask_pil)
        print(f"  Resized mask to: {mask.shape}")

    # Normalize mask to [0, 1]
    mask_normalized = mask / 255.0  # [H, W]

    # Create result image
    result = image.copy().astype(np.float32)

    # Darken regions where mask is 0
    # Expand mask to [H, W, 1] for broadcasting
    mask_expanded = mask_normalized[:, :, np.newaxis]

    # Apply darkening: result = image * mask + image * (1 - mask) * darkening_factor
    # This means: masked regions stay bright, unmasked regions are darkened
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
    print(f"Saved result to: {output_path}")


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
    darkening_factor: float = 0.3
):
    """Process a single mask: render, apply mask, save.

    Args:
        scene: Scene name
        view_id: View ID
        mask_name: Mask filename
        darkening_factor: Darkening factor for non-mask regions
    """
    print(f"\n{'='*70}")
    print(f"Processing: scene={scene}, view={view_id}, mask={mask_name}")
    print(f"{'='*70}\n")

    # Load checkpoint
    gaussians = load_checkpoint(scene)
    print(f"  Loaded {gaussians['means'].shape[0]} Gaussians")

    # Load cameras
    cameras = load_cameras(scene)

    # Find camera for this view
    camera = find_camera_by_view_id(cameras, view_id)
    if camera is None:
        print(f"Error: Camera not found for view_id={view_id}")
        print(f"  Available camera IDs: {[c['id'] for c in cameras[:10]]}...")
        return

    print(f"  Found camera: id={camera['id']}, img_name={camera.get('img_name', 'N/A')}")

    # Load mask
    mask = load_mask(scene, view_id, mask_name)

    # Render view
    print(f"\nRendering view...")
    rendered = render_view_gaussians(gaussians, camera, image_size=(mask.shape[1], mask.shape[0]))

    # Apply mask
    print(f"\nApplying mask...")
    result = apply_mask_to_image(rendered, mask, darkening_factor)

    # Save result
    output_path = f"{OUTPUT_ROOT}/{scene}/{view_id}/{mask_name}"
    save_result(result, output_path)

    print(f"\n{'='*70}")
    print(f"Done!")
    print(f"{'='*70}\n")


def process_all_masks(
    scene: str,
    view_id: int,
    darkening_factor: float = 0.3
):
    """Process all masks in a view directory.

    Args:
        scene: Scene name
        view_id: View ID
        darkening_factor: Darkening factor for non-mask regions
    """
    # Get all mask files
    mask_files = list_masks_in_view(scene, view_id)

    if not mask_files:
        print(f"No mask files found in {EVAL_RESULTS_ROOT}/{scene}/{view_id}")
        return

    # Filter out comparison_maps subdirectory
    mask_files = [f for f in mask_files if not f.startswith('.')]

    print(f"Found {len(mask_files)} mask files")

    # Load checkpoint and cameras once (shared across all masks)
    gaussians = load_checkpoint(scene)
    cameras = load_cameras(scene)

    # Find camera
    camera = find_camera_by_view_id(cameras, view_id)
    if camera is None:
        print(f"Error: Camera not found for view_id={view_id}")
        return

    print(f"Using camera: id={camera['id']}, img_name={camera.get('img_name', 'N/A')}")

    # Render once (shared across all masks)
    print(f"\nRendering view...")
    # Get image size from first mask
    first_mask = load_mask(scene, view_id, mask_files[0])
    rendered = render_view_gaussians(gaussians, camera, image_size=(first_mask.shape[1], first_mask.shape[0]))

    # Process each mask
    for i, mask_name in enumerate(mask_files):
        print(f"\n[{i+1}/{len(mask_files)}] Processing: {mask_name}")

        # Load mask
        mask = load_mask(scene, view_id, mask_name)

        # Apply mask
        result = apply_mask_to_image(rendered, mask, darkening_factor)

        # Save result
        output_path = f"{OUTPUT_ROOT}/{scene}/{view_id}/{mask_name}"
        save_result(result, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Render Gaussian view with mask overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single mask
  python tools/render_view_with_mask.py --scene figurines --view-id 40 --mask-name "chosen_waldo.png"

  # Process all masks in view
  python tools/render_view_with_mask.py --scene figurines --view-id 40 --all-masks

  # List available masks
  python tools/render_view_with_mask.py --scene figurines --view-id 40 --list-masks

  # Custom darkening factor (default 0.3, lower = darker)
  python tools/render_view_with_mask.py --scene figurines --view-id 40 --mask-name "chosen_waldo.png" --darken 0.2
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
    parser.add_argument("--checkpoint", type=str, default="chkpnt30000_langfeat_0.pth",
                        help="Checkpoint file name (default: chkpnt30000_langfeat_0.pth)")

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
        process_all_masks(args.scene, args.view_id, args.darken)
    else:
        process_single_mask(args.scene, args.view_id, args.mask_name, args.darken)


if __name__ == "__main__":
    main()
