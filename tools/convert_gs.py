#!/usr/bin/env python3
"""
Convert gsplat trained models (PLY files) to SceneSplat dataset format (NPY files).

This script reads Gaussian Splatting PLY files from gaussian_results/ and converts
them to SceneSplat's NPY format, saving to gaussian_train/{dataset_name}/.

Output format:
    scene_folder/
    ├── coord.npy           # 3D coordinates [N, 3]
    ├── color.npy          # RGB colors [N, 3] (uint8, 0-255)
    ├── opacity.npy        # Opacity values [N, 1] (float, 0-1)
    ├── quat.npy           # Quaternion rotation [N, 4] (w, x, y, z)
    ├── scale.npy          # Scale parameters [N, 3]
    ├── lang_feat.npy      # Language features [N, D] (optional, not from PLY)
    ├── valid_feat_mask.npy # Valid feature mask [N] (optional, not from PLY)
    └── segment.npy        # Semantic labels [N] (optional, needs to be provided)

Usage:
    # Convert specific scene
    python tools/convert_gs.py --input gaussian_results/lerf_ovs/figurines/point_cloud.ply

    # Convert all scenes in a directory
    python tools/convert_gs.py --input gaussian_results/lerf_ovs --output gaussian_train/lerf_ovs

    # Convert with recursive search
    python tools/convert_gs.py --input gaussian_results --output gaussian_train --recursive
"""

import argparse
import numpy as np
import os
from pathlib import Path
from plyfile import PlyData
from tqdm import tqdm


def np_sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def read_gsplat_ply(ply_path):
    """
    Read a gsplat PLY file and extract Gaussian attributes.

    Args:
        ply_path: Path to the PLY file

    Returns:
        Dictionary with keys: coord, color, opacity, scale, quat
    """
    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]

    # Access the actual data array from PlyElement
    vertex_data = vertex.data if hasattr(vertex, 'data') else vertex

    n_gaussians = len(vertex_data)
    data = {}

    # Coordinates (xyz)
    x = vertex_data["x"].astype(np.float32)
    y = vertex_data["y"].astype(np.float32)
    z = vertex_data["z"].astype(np.float32)
    data["coord"] = np.stack((x, y, z), axis=-1)  # [N, 3]

    # Normals (not used in SceneSplat, but present in PLY)
    # nx = vertex_data["nx"].astype(np.float32)
    # ny = vertex_data["ny"].astype(np.float32)
    # nz = vertex_data["nz"].astype(np.float32)

    # SH DC coefficients (f_dc_0, f_dc_1, f_dc_2) -> Color
    f_dc_0 = vertex_data["f_dc_0"].astype(np.float32)
    f_dc_1 = vertex_data["f_dc_1"].astype(np.float32)
    f_dc_2 = vertex_data["f_dc_2"].astype(np.float32)

    # Convert SH DC to RGB color
    # SH DC to RGB conversion factor (0.28209479177387814)
    C0 = 0.28209479177387814
    feature_dc = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=-1)  # [N, 3]
    feature_pc = (feature_dc * C0).astype(np.float32) + 0.5
    feature_pc = np.clip(feature_pc, 0, 1)
    data["color"] = (feature_pc * 255).astype(np.uint8)  # [N, 3], uint8, 0-255

    # Opacity
    # Note: gsplat stores raw opacity values, need sigmoid
    if "opacity" in vertex_data.dtype.names:
        opacity_raw = vertex_data["opacity"].astype(np.float32)
        data["opacity"] = np_sigmoid(opacity_raw)  # [N], 0-1
    else:
        # Fallback: if no opacity field, use default value
        data["opacity"] = np.ones(n_gaussians, dtype=np.float32) * 0.5

    # Scales (scale_0, scale_1, scale_2)
    # gsplat stores log(scale), need to exponentiate
    scale_names = sorted([p for p in vertex_data.dtype.names if p.startswith("scale_")],
                         key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((n_gaussians, 3), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names[:3]):  # Take first 3
        scales[:, idx] = vertex_data[attr_name].astype(np.float32)
    data["scale"] = np.exp(scales)  # [N, 3]

    # Rotation/Quaternion (rot_0, rot_1, rot_2, rot_3)
    # gsplat stores quaternions in (w, x, y, z) format
    rot_names = sorted([p for p in vertex_data.dtype.names if p.startswith("rot")],
                       key=lambda x: int(x.split("_")[-1]))
    quats = np.zeros((n_gaussians, 4), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names[:4]):  # Take first 4
        quats[:, idx] = vertex_data[attr_name].astype(np.float32)

    # Normalize quaternions
    quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / (quat_norms + 1e-9)

    # Ensure w (first component) is positive for uniqueness
    signs = np.sign(quats[:, 0])
    quats = quats * signs[:, None]

    data["quat"] = quats  # [N, 4], (w, x, y, z)

    return data


def save_scenesplat_format(data, output_dir, scene_name=""):
    """
    Save Gaussian data in SceneSplat NPY format.

    Args:
        data: Dictionary with Gaussian attributes
        output_dir: Output directory path
        scene_name: Scene name (for logging)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save standard attributes
    np.save(output_dir / "coord.npy", data["coord"])
    np.save(output_dir / "color.npy", data["color"])
    np.save(output_dir / "opacity.npy", data["opacity"].reshape(-1, 1))  # [N, 1]
    np.save(output_dir / "scale.npy", data["scale"])
    np.save(output_dir / "quat.npy", data["quat"])

    # Create placeholder files for optional attributes
    # These would need to be generated separately

    print(f"Saved to: {output_dir}")
    print(f"  - coord: {data['coord'].shape}, {data['coord'].dtype}")
    print(f"  - color: {data['color'].shape}, {data['color'].dtype}")
    print(f"  - opacity: {data['opacity'].shape}, {data['opacity'].dtype}")
    print(f"  - scale: {data['scale'].shape}, {data['scale'].dtype}")
    print(f"  - quat: {data['quat'].shape}, {data['quat'].dtype}")
    print(f"  - Total Gaussians: {len(data['coord'])}")

    # Note: lang_feat.npy needs to be generated separately using SAM2/SigLIP
    print(f"  Note: lang_feat.npy needs to be generated separately (e.g., using SAM2/SigLIP)")


def process_ply_file(ply_path, output_dir, relative_path):
    """Process a single PLY file."""
    try:
        # Read gsplat PLY
        data = read_gsplat_ply(ply_path)

        if relative_path:
            # For directory input: preserve structure
            # e.g., gaussian_results/lerf_ovs/figurines/point_cloud.ply
            # -> gaussian_train/lerf_ovs/figurines/
            output_subdir = output_dir / relative_path.parent
        else:
            # For single file input
            output_subdir = output_dir / ply_path.stem

        save_scenesplat_format(data, output_subdir, scene_name=ply_path.stem)
        return True
    except Exception as e:
        print(f"Error processing {ply_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_ply_files(input_path, recursive=False, iteration=None):
    """
    Find all PLY files in the input path.
    Looks for files at {scene_path}/ckpts/point_cloud_{iteration}.ply

    Args:
        input_path: Input file or directory path
        recursive: Whether to search recursively across all scene directories
        iteration: Specific iteration number to find (e.g., 30000)
                   If specified, looks for point_cloud_{iteration}.ply

    Returns:
        List of PLY file paths
    """
    ply_files = []

    if input_path.is_file():
        if input_path.suffix == ".ply":
            ply_files = [input_path]
        else:
            print(f"Error: {input_path} is not a .ply file")
            return []
    elif input_path.is_dir():
        def find_in_directory(scene_dir):
            """Find PLY files in a single scene's ckpts directory."""
            ckpts_dir = scene_dir / "ckpts"
            if not ckpts_dir.exists():
                return None

            if iteration is not None:
                # Directly access the specific iteration file
                target_file = f"point_cloud_{iteration}.ply"
                target_path = ckpts_dir / target_file
                if target_path.exists():
                    return target_path
                return None
            else:
                # Find the latest iteration by listing ckpts directory
                latest_iter = -1
                latest_file = None
                for entry in ckpts_dir.iterdir():
                    if entry.is_file() and entry.suffix == ".ply":
                        # Extract iteration number from filename (point_cloud_12345.ply -> 12345)
                        match = re.match(r"point_cloud_(\d+)\.ply", entry.name)
                        if match:
                            iter_num = int(match.group(1))
                            if iter_num > latest_iter:
                                latest_iter = iter_num
                                latest_file = entry
                        elif entry.name == "point_cloud.ply":
                            # Base point_cloud.ply (no iteration number)
                            if latest_iter < 0:
                                latest_file = entry

                return latest_file

        import re

        if recursive:
            # Traverse all subdirectories to find scenes
            for scene_dir in sorted(os.listdir(input_path)):
                scene_dir = input_path / scene_dir
                if scene_dir.is_dir():
                    ply_file = find_in_directory(scene_dir)
                    if ply_file is not None:
                        ply_files.append(ply_file)
                        print(f"  Found: {ply_file}")
        else:
            # Single directory mode
            ckpts_dir = input_path / "ckpts"
            if not ckpts_dir.exists():
                print(f"Error: Checkpoints directory not found: {ckpts_dir}")
                print("Expected structure: {{scene_path}}/ckpts/point_cloud_{{iteration}}.ply")
                return []

            ply_file = find_in_directory(input_path)
            if ply_file is not None:
                ply_files = [ply_file]
                if iteration is None:
                    # Show which iteration was found
                    match = re.match(r"point_cloud_(\d+)\.ply", ply_file.name)
                    if match:
                        print(f"Found latest iteration: {match.group(1)}")
                    elif ply_file.name == "point_cloud.ply":
                        print(f"Found: point_cloud.ply (base)")
            else:
                if iteration is not None:
                    print(f"Error: PLY file not found: {ckpts_dir / f'point_cloud_{iteration}.ply'}")
                else:
                    print(f"Error: No point_cloud*.ply files found in {ckpts_dir}")
                return []

    return sorted(ply_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert gsplat PLY files to SceneSplat NPY format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert specific scene with iteration 30000
  python tools/convert_gs.py --input gaussian_results/lerf_ovs/figurines --iteration 30000

  # Convert specific PLY file
  python tools/convert_gs.py --input gaussian_results/lerf_ovs/figurines/ckpts/point_cloud_30000.ply

  # Convert all scenes in a dataset (latest iteration)
  python tools/convert_gs.py --input gaussian_results/lerf_ovs --output gaussian_train/lerf_ovs

  # Convert all datasets (recursive)
  python tools/convert_gs.py --input gaussian_results --output gaussian_train --recursive --iteration 30000
        """
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input PLY file or directory containing gsplat trained models",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory for SceneSplat NPY files (default: gaussian_train/)",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search for PLY files in subdirectories",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Specific training iteration to convert (e.g., 30000 for point_cloud_30000.ply)",
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output is None:
        if args.input.is_dir():
            # Use directory name as output name
            # e.g., gaussian_results/lerf_ovs -> gaussian_train/lerf_ovs
            args.output = Path("gaussian_train") / args.input.name
        else:
            args.output = Path("gaussian_train")

    args.input = args.input.resolve()
    args.output = args.output.resolve()

    print("=" * 60)
    print("gsplat PLY to SceneSplat NPY Converter")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Recursive: {args.recursive}")
    if args.iteration is not None:
        print(f"Iteration: {args.iteration}")
    print("=" * 60)

    # Find PLY files
    ply_files = find_ply_files(args.input, args.recursive, args.iteration)

    if not ply_files:
        print("No PLY files found!")
        print("\nHint: gsplat saves models as 'point_cloud_30000.ply' in ckpts/ directory")
        print("Example path: gaussian_results/lerf_ovs/figurines/ckpts/point_cloud_30000.ply")
        if args.iteration is None:
            print("\nYou can use --iteration to specify a specific training iteration.")
            print("Example: --iteration 30000")
        return 1

    print(f"Found {len(ply_files)} PLY file(s) to process:")
    for f in ply_files[:5]:  # Show first 5
        print(f"  - {f}")
    if len(ply_files) > 5:
        print(f"  ... and {len(ply_files) - 5} more")
    print("=" * 60)

    # Process each PLY file
    success_count = 0
    for ply_file in tqdm(ply_files, desc="Converting"):
        if process_ply_file(ply_file, args.output, ply_file.relative_to(args.input) if args.input.is_dir() else None):
            success_count += 1

    print("=" * 60)
    print(f"Conversion complete: {success_count}/{len(ply_files)} successful")
    print("=" * 60)

    # Next steps reminder
    print("\nNext steps:")
    print("1. Generate language features using SAM2/SigLIP:")
    print("   python scripts/preprocess_siglip2_sam2.py \\")
    print("     --input_root gaussian_train/lerf_ovs \\")
    print("     --output_root gaussian_train/lerf_ovs \\")
    print("     --sam2_model_path /path/to/sam2_model.pth")
    print("")
    print("2. (Optional) Chunk scenes for training:")
    print("   python -u pointcept/datasets/preprocessing/sampling_chunking_data_gs.py \\")
    print("     --dataset_root gaussian_train/lerf_ovs \\")
    print("     --output_dir chunks/lerf_ovs \\")
    print("     --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3")

    return 0 if success_count == len(ply_files) else 1


# python tools/convert_gs.py --input gaussian_results/lerf_ovs --output gaussian_train/lerf_ovs --iteration 30000 --recursive
if __name__ == "__main__":
    exit(main())
