#!/usr/bin/env python3
"""
Batch Compression Script for Replica Dataset (Dual-GPU Parallel)

This script processes all scenes in the replica dataset using compress_grid_svd.py
with dual-GPU parallel processing support.
Unlike LERF dataset, replica does not have multiple feature sequences (lang_feat_1, lang_feat_2, etc.),
so each scene is processed only once.

Usage:
    # Process all replica scenes with dual-GPU parallel (default: cuda:0 and cuda:1)
    python batch_compress_replica.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train \\
        --dataset replica

    # Single GPU mode
    python batch_compress_replica.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train \\
        --dataset replica \\
        --single_gpu

    # Specify custom GPUs
    python batch_compress_replica.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train \\
        --dataset replica \\
        --gpus cuda:0,cuda:1

    # Specify custom ranks
    python batch_compress_replica.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train \\
        --dataset replica \\
        --ranks 8,16,32

    # All other parameters are passed through to compress_grid_svd.py
    python batch_compress_replica.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train \\
        --dataset replica \\
        --grid_size 0.01 \\
        --ranks 16,32
"""

import os
import sys
import argparse
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Tuple

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_scenes_in_dataset(data_root: str, dataset: str, split: str = "train") -> List[str]:
    """
    Get all scene directories in the specified dataset.

    Args:
        data_root: Root directory containing datasets
        dataset: Dataset name (e.g., 'replica')
        split: Dataset split (train, val, test)

    Returns:
        List of scene directory names
    """
    dataset_path = Path(data_root) / dataset / split
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # Get all subdirectories that could be scenes
    scenes = []
    for item in dataset_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it contains coord.npy (basic validation for a scene)
            coord_path = item / "coord.npy"
            if coord_path.exists():
                scenes.append(item.name)

    return sorted(scenes)


def split_scenes(scenes: List[str], num_parts: int = 2) -> List[List[str]]:
    """
    Split scenes into roughly equal parts for parallel processing.

    Args:
        scenes: List of scene names
        num_parts: Number of parts to split into

    Returns:
        List of scene lists (one for each GPU)
    """
    scenes_per_part = (len(scenes) + num_parts - 1) // num_parts  # Ceiling division
    result = []
    for i in range(num_parts):
        start = i * scenes_per_part
        end = min((i + 1) * scenes_per_part, len(scenes))
        if start < len(scenes):
            result.append(scenes[start:end])
        else:
            result.append([])
    return result


def run_compression_worker(
    scenes: List[str],
    gpu: str,
    compress_args: List[str],
    worker_id: int,
    result_dict: dict,
) -> None:
    """
    Worker function to run compression on a specific GPU.

    Args:
        scenes: List of scene names to process
        gpu: GPU device string (e.g., 'cuda:0')
        compress_args: Arguments to pass to compress_grid_svd.py
        worker_id: Worker identifier (0 or 1)
        result_dict: Dictionary to store result (thread-safe)
    """
    compress_script = PROJECT_ROOT / "tools" / "compression" / "compress_grid_svd.py"

    print(f"\n[Worker {worker_id}] Starting on {gpu}")
    print(f"[Worker {worker_id}] Scenes: {scenes}")

    # Build command with device and scenes
    cmd = [
        sys.executable,
        str(compress_script),
    ] + compress_args + [
        "--device", gpu,
        "--scenes", ",".join(scenes),
    ]

    print(f"[Worker {worker_id}] Running: {' '.join(cmd)}")

    # Run compress_grid_svd.py
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        result_dict[worker_id] = {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'scenes': scenes,
        }
        if result.returncode == 0:
            print(f"[Worker {worker_id}] ✓ Completed successfully")
        else:
            print(f"[Worker {worker_id}] ✗ Failed with return code {result.returncode}")
    except Exception as e:
        result_dict[worker_id] = {
            'success': False,
            'error': str(e),
            'scenes': scenes,
        }
        print(f"[Worker {worker_id}] ✗ Exception: {e}")


def run_compression_parallel(
    scenes: List[str],
    gpus: List[str],
    compress_args: List[str],
) -> None:
    """
    Run compression in parallel on multiple GPUs.

    Args:
        scenes: List of scene names to process
        gpus: List of GPU device strings (e.g., ['cuda:0', 'cuda:1'])
        compress_args: Arguments to pass to compress_grid_svd.py
    """
    print("=" * 70)
    print(f"Batch Compression for Replica Dataset (Parallel - {len(gpus)} GPUs)")
    print("=" * 70)
    print(f"Total scenes: {len(scenes)}")
    print(f"GPUs: {gpus}")
    print("=" * 70)

    # Split scenes among GPUs
    scene_groups = split_scenes(scenes, len(gpus))

    for i, (gpu, group) in enumerate(zip(gpus, scene_groups)):
        print(f"GPU {i} ({gpu}): {len(group)} scenes - {group}")

    print("=" * 70)

    # Launch workers in parallel threads
    threads = []
    result_dict = {}

    for worker_id, (gpu, scene_group) in enumerate(zip(gpus, scene_groups)):
        if scene_group:  # Only launch if there are scenes to process
            thread = threading.Thread(
                target=run_compression_worker,
                args=(scene_group, gpu, compress_args, worker_id, result_dict),
                daemon=True,
            )
            threads.append(thread)
            thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Print summary
    print("\n" + "=" * 70)
    print("Parallel Compression Summary")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for worker_id in sorted(result_dict.keys()):
        result = result_dict[worker_id]
        if result['success']:
            success_count += len(result['scenes'])
            print(f"Worker {worker_id}: ✓ Success - {result['scenes']}")
        else:
            fail_count += len(result['scenes'])
            error_msg = result.get('error', f"returncode={result.get('returncode')}")
            print(f"Worker {worker_id}: ✗ Failed - {result['scenes']} ({error_msg})")

    print(f"\nTotal scenes processed: {len(scenes)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print("=" * 70)


def run_compression_single(
    scenes: List[str],
    gpu: str,
    compress_args: List[str],
) -> None:
    """
    Run compression on a single GPU.

    Args:
        scenes: List of scene names to process
        gpu: GPU device string (e.g., 'cuda:0')
        compress_args: Arguments to pass to compress_grid_svd.py
    """
    compress_script = PROJECT_ROOT / "tools" / "compression" / "compress_grid_svd.py"

    print("=" * 70)
    print(f"Batch Compression for Replica Dataset (Single GPU: {gpu})")
    print("=" * 70)
    print(f"Scenes to process ({len(scenes)}): {scenes}")
    print(f"Arguments passed to compress_grid_svd.py: {' '.join(compress_args)}")
    print("=" * 70)

    # Build command with device and scenes
    cmd = [
        sys.executable,
        str(compress_script),
    ] + compress_args + [
        "--device", gpu,
        "--scenes", ",".join(scenes),
    ]

    print(f"\nRunning command:")
    print(f"  {' '.join(cmd)}")
    print()

    # Run compress_grid_svd.py
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("[Success] Batch compression completed successfully")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print(f"[Failed] Batch compression failed with return code {result.returncode}")
            print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"[Error] Exception during batch compression: {e}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Batch compression for replica dataset with dual-GPU parallel support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all replica scenes with dual-GPU (default: cuda:0, cuda:1)
    python batch_compress_replica.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train \\
        --dataset replica

    # Single GPU mode
    python batch_compress_replica.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train \\
        --dataset replica \\
        --single_gpu

    # Specify custom GPUs
    python batch_compress_replica.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train \\
        --dataset replica \\
        --gpus cuda:0,cuda:2

    # Specify custom ranks
    python batch_compress_replica.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train \\
        --dataset replica \\
        --ranks 8,16,32

    # All other parameters are passed through to compress_grid_svd.py
    python batch_compress_replica.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train \\
        --dataset replica \\
        --grid_size 0.01 \\
        --ranks 16,32
        """
    )

    # GPU selection arguments
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPUs to use for parallel processing (default: auto-detect from CUDA_VISIBLE_DEVICES or cuda:0,cuda:1)",
    )
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        help="Use single GPU mode (only use first GPU from --gpus)",
    )

    # Scene selection arguments
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Comma-separated list of specific scenes to process (default: all scenes in dataset)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to process (default: train)",
    )

    # Parse known args to extract batch-specific parameters
    # Remaining args will be passed to compress_grid_svd.py
    args, compress_args = parser.parse_known_args()

    # Parse GPUs - auto-detect from CUDA_VISIBLE_DEVICES if not specified
    if args.gpus is None:
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            # Map CUDA_VISIBLE_DEVICES to cuda:0, cuda:1, etc.
            num_gpus = len(cuda_visible.split(','))
            gpus = [f'cuda:{i}' for i in range(num_gpus)]
            print(f"Auto-detected GPUs from CUDA_VISIBLE_DEVICES={cuda_visible}: {gpus}")
        else:
            gpus = ['cuda:0', 'cuda:1']
            print(f"Using default GPUs: {gpus}")
    else:
        gpus = [gpu.strip() for gpu in args.gpus.split(',')]

    if args.single_gpu:
        gpus = [gpus[0]]  # Use only first GPU

    # Parse scenes if provided, otherwise auto-discover from dataset
    scenes = []
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(',')]
    else:
        # Auto-discover scenes from --data_root, --dataset, and --split
        # Extract these from compress_args or use args
        data_root = None
        dataset = None
        split = args.split

        for i, arg in enumerate(compress_args):
            if arg == '--data_root' and i + 1 < len(compress_args):
                data_root = compress_args[i + 1]
            elif arg == '--dataset' and i + 1 < len(compress_args):
                dataset = compress_args[i + 1]
            elif arg == '--split' and i + 1 < len(compress_args):
                split = compress_args[i + 1]

        if data_root and dataset:
            print(f"Auto-discovering scenes from {data_root}/{dataset}/{split}...")
            scenes = get_scenes_in_dataset(data_root, dataset, split)
            print(f"Found {len(scenes)} scenes: {scenes}")
        else:
            print("Error: --scenes not provided and cannot auto-discover scenes.")
            print("Please provide either --scenes or both --data_root and --dataset.")
            return

    # Run compression
    if len(gpus) == 1:
        run_compression_single(scenes, gpus[0], compress_args)
    else:
        run_compression_parallel(scenes, gpus, compress_args)


if __name__ == "__main__":
    main()
