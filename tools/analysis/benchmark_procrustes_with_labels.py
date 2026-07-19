#!/usr/bin/env python3
"""
Benchmark script for compute_procrustes_Q_cuda_with_labels algorithm.

This script measures:
1. Average execution time per scene
2. GPU memory consumption (peak and average)
3. Per-scene statistics (N, M, d)

Usage:
    python tools/analysis/benchmark_procrustes_with_labels.py \\
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/val \\
        --text_embed /new_data/cyf/projects/SceneSplat/pointcept/datasets/preprocessing/scannet/meta_data/scannet20_text_embeddings_siglip2.pt \\
        --svd_rank 16 \\
        --label_file lang_label.npy \\
        --num_runs 10 \\
        --warmup_runs 2
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools" / "projection"))
from compute_procrustes_alignment_simple import (
    load_grid_svd_features,
    compute_procrustes_Q_cuda_with_labels,
)


def load_text_embeddings_dict(
    embed_file: str,
    svd_rank: int,
    normalize: bool = True,
    use_torch: bool = False,
    benchmark: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Load and SVD-reduce text embeddings from a dictionary format file.

    The dictionary format contains:
        - embeddings: [N, D] text embeddings
        - categories: list of class names
        - model_name, model_type, embed_dim: metadata

    Args:
        embed_file: Path to text embeddings file (.pt or .npy)
        svd_rank: Target SVD rank
        normalize: Whether to normalize features
        use_torch: Whether to use PyTorch for SVD (default: False, use NumPy)
        benchmark: If True, return timing info

    Returns:
        embeddings: [N, svd_rank] SVD-reduced embeddings
        timing: Dictionary with timing information
    """
    from compute_procrustes_alignment_simple import perform_svd_reduction

    if embed_file.endswith('.pt'):
        data = torch.load(embed_file, weights_only=False)

        # Handle dictionary format
        if isinstance(data, dict):
            embeddings = data['embeddings'].numpy() if isinstance(data['embeddings'], torch.Tensor) else data['embeddings']
        else:
            embeddings = data.numpy() if isinstance(data, torch.Tensor) else data
    elif embed_file.endswith('.npy'):
        embeddings = np.load(embed_file)
    else:
        raise ValueError(f"Unsupported file format: {embed_file}")

    embeddings = embeddings.astype(np.float32)

    timing = {}
    if embeddings.shape[1] > svd_rank:
        embeddings, components, timing = perform_svd_reduction(
            embeddings, svd_rank, normalize, use_torch, 'cuda', benchmark
        )
    else:
        components = None

    return embeddings, timing


def get_gpu_memory_info(device: torch.device) -> Dict[str, float]:
    """Get GPU memory information in MB."""
    if not device.type == 'cuda':
        return {
            'allocated_mb': 0,
            'reserved_mb': 0,
            'free_mb': 0,
            'total_mb': 0,
        }

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)

    if hasattr(torch.cuda, 'mem_get_info'):
        free, total = torch.cuda.mem_get_info(device)
        free_mb = free / (1024 ** 2)
        total_mb = total / (1024 ** 2)
    else:
        # Fallback for older PyTorch versions
        try:
            props = torch.cuda.get_device_properties(device)
            total_mb = props.total_memory / (1024 ** 2)
            free_mb = total_mb - reserved
        except:
            free_mb = 0
            total_mb = 0

    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'free_mb': free_mb,
        'total_mb': total_mb,
    }


def clear_cuda_cache(device: torch.device):
    """Clear CUDA cache and force garbage collection."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def benchmark_single_scene(
    X_c: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    num_runs: int = 10,
    warmup_runs: int = 2,
    expand_factor: int = 1,
) -> Dict:
    """
    Benchmark compute_procrustes_Q_cuda_with_labels on a single scene.

    Args:
        X_c: [N, d] source features
        Y: [M, d] target features (text embeddings)
        labels: [N] label indices
        device: CUDA device
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs (not counted in timing)
        expand_factor: Factor to expand data by repeating (for testing larger scales)

    Returns:
        Dictionary with benchmark results
    """
    # Expand data by repeating if requested
    N_original = X_c.shape[0]
    if expand_factor > 1:
        X_c = np.repeat(X_c, expand_factor, axis=0)
        labels = np.repeat(labels, expand_factor, axis=0)

    results = {
        'N_original': N_original,
        'N': X_c.shape[0],
        'M': Y.shape[0],
        'd': X_c.shape[1],
        'expand_factor': expand_factor,
        'num_runs': num_runs,
        'warmup_runs': warmup_runs,
        'times_ms': [],
        'peak_memory_mb': [],
        'memory_allocated_mb': [],
        'metrics_list': [],
    }

    # Calculate memory size of input tensors (for reference)
    # float32 = 4 bytes per element
    # int64 = 8 bytes per element
    X_c_memory_mb = X_c.nbytes / (1024 ** 2)
    Y_memory_mb = Y.nbytes / (1024 ** 2)
    labels_memory_mb = labels.nbytes / (1024 ** 2)
    total_input_memory_mb = X_c_memory_mb + Y_memory_mb + labels_memory_mb

    results['input_memory_mb'] = {
        'X_c_mb': X_c_memory_mb,
        'Y_mb': Y_memory_mb,
        'labels_mb': labels_memory_mb,
        'total_mb': total_input_memory_mb,
    }

    # Warmup runs
    for _ in range(warmup_runs):
        clear_cuda_cache(device)

        # Create fresh tensors for warmup
        X_c_tensor = torch.from_numpy(X_c).to(device)
        Y_tensor = torch.from_numpy(Y).to(device)
        labels_tensor = torch.from_numpy(labels).to(device)

        with torch.no_grad():
            Q, metrics = compute_procrustes_Q_cuda_with_labels(
                X_c_tensor, Y_tensor, labels_tensor
            )
        torch.cuda.synchronize(device)

        # Delete tensors to free memory
        del X_c_tensor, Y_tensor, labels_tensor, Q

    # Benchmark runs - measure peak memory during each run
    for _ in range(num_runs):
        clear_cuda_cache(device)

        # Reset peak memory stats before measurement
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        # Get baseline memory before tensor creation
        baseline_memory = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0

        # Create fresh tensors for this run
        X_c_tensor = torch.from_numpy(X_c).to(device)
        Y_tensor = torch.from_numpy(Y).to(device)
        labels_tensor = torch.from_numpy(labels).to(device)

        # Memory after tensor creation
        after_tensors_memory = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0
        tensors_memory_mb = after_tensors_memory - baseline_memory

        # Reset peak memory stats before computation
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        # Run computation and measure time
        t_start = time.perf_counter()

        with torch.no_grad():
            Q, metrics = compute_procrustes_Q_cuda_with_labels(
                X_c_tensor, Y_tensor, labels_tensor
            )

        torch.cuda.synchronize(device)
        t_end = time.perf_counter()

        # Get peak memory during computation
        if device.type == 'cuda':
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_memory_mb = 0

        # Get current allocated memory after computation
        final_memory = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0

        elapsed_ms = (t_end - t_start) * 1000
        results['times_ms'].append(elapsed_ms)
        results['peak_memory_mb'].append(peak_memory_mb)
        results['memory_allocated_mb'].append(final_memory)
        results['metrics_list'].append(metrics)

        # Delete tensors to free memory for next run
        del X_c_tensor, Y_tensor, labels_tensor, Q

    # Compute statistics
    results['time_mean_ms'] = float(np.mean(results['times_ms']))
    results['time_std_ms'] = float(np.std(results['times_ms']))
    results['time_min_ms'] = float(np.min(results['times_ms']))
    results['time_max_ms'] = float(np.max(results['times_ms']))

    results['peak_memory_mean_mb'] = float(np.mean(results['peak_memory_mb']))
    results['peak_memory_std_mb'] = float(np.std(results['peak_memory_mb']))
    results['peak_memory_min_mb'] = float(np.min(results['peak_memory_mb']))
    results['peak_memory_max_mb'] = float(np.max(results['peak_memory_mb']))

    results['memory_allocated_mean_mb'] = float(np.mean(results['memory_allocated_mb']))

    # Memory above input tensors (computation overhead)
    results['computation_overhead_mean_mb'] = results['peak_memory_mean_mb'] - total_input_memory_mb
    results['computation_overhead_max_mb'] = results['peak_memory_max_mb'] - total_input_memory_mb

    # Average metrics across runs
    for key in ['residual_norm', 'relative_error', 'cosine_before', 'cosine_after',
                'cosine_improvement', 'orthogonality_error']:
        values = [m.get(key, 0) for m in results['metrics_list'] if key in m]
        if values:
            results[f'{key}_mean'] = float(np.mean(values))
            results[f'{key}_std'] = float(np.std(values))

    # Memory per data point
    results['memory_per_point_kb'] = (results['peak_memory_mean_mb'] * 1024) / results['N']
    results['memory_per_class_kb'] = (results['peak_memory_mean_mb'] * 1024) / results['M']

    return results


def find_scenes_with_data(
    data_root: str,
    svd_rank: int,
    label_filename: str,
    svd_suffix: Optional[str] = None,
    find_all: bool = False,
) -> List[Tuple[str, str, str]]:
    """
    Find all scenes with required data files.

    Args:
        data_root: Root directory containing scene subdirectories
        svd_rank: SVD rank for feature file name
        label_filename: Name of label file
        svd_suffix: Optional suffix for SVD files (e.g., '_1', '_2', '_3')
                    If None and find_all=False, searches for any SVD file matching the rank
        find_all: If True, find ALL SVD files matching the rank pattern for each scene

    Returns:
        List of tuples (scene_name, svd_file, label_file)
    """
    scenes = []
    data_root_path = Path(data_root)

    for scene_dir in sorted(data_root_path.iterdir()):
        if not scene_dir.is_dir():
            continue

        label_file = scene_dir / label_filename
        if not label_file.exists():
            continue  # Skip scenes without label file

        # Find SVD files
        if find_all:
            # Find ALL SVD files matching the rank
            svd_candidates = sorted(scene_dir.glob(f'lang_feat_grid_svd_r{svd_rank}*.npz'))
            for svd_file in svd_candidates:
                scenes.append((scene_dir.name, str(svd_file), str(label_file)))
        else:
            # Find single SVD file
            svd_file = None
            if svd_suffix is not None:
                # Try with specific suffix first
                svd_file = scene_dir / f'lang_feat_grid_svd_r{svd_rank}{svd_suffix}.npz'

            if svd_file is None or not svd_file.exists():
                # Try without suffix
                svd_file = scene_dir / f'lang_feat_grid_svd_r{svd_rank}.npz'

            if not svd_file.exists():
                # Try to find any SVD file matching the rank
                svd_candidates = list(scene_dir.glob(f'lang_feat_grid_svd_r{svd_rank}*.npz'))
                if svd_candidates:
                    svd_file = svd_candidates[0]  # Use the first match
                else:
                    continue  # No SVD file found for this scene

            if svd_file.exists():
                scenes.append((scene_dir.name, str(svd_file), str(label_file)))

    return scenes


def load_and_prepare_labels(
    label_file: str,
    svd_file: str,
    num_classes: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load and filter labels for Procrustes alignment.

    Args:
        label_file: Path to label file
        svd_file: Path to SVD feature file (to get expected point count)
        num_classes: Expected number of classes (Y.shape[0])

    Returns:
        Tuple of (filtered_labels, valid_mask) or (None, None) if loading failed
        valid_mask: Boolean mask to apply to X_c to keep only points with valid labels
    """
    try:
        labels = np.load(label_file).astype(np.int64)

        # Get expected number of points from SVD file
        svd_data = np.load(svd_file, mmap_mode='r')
        num_points = svd_data['indices'].shape[0]

        # Check if we need to filter labels
        if labels.shape[0] != num_points:
            # Try to load valid_feat_mask to filter labels
            svd_dir = Path(label_file).parent
            valid_mask_path = svd_dir / 'valid_feat_mask.npy'

            if valid_mask_path.exists():
                valid_mask = np.load(valid_mask_path).astype(bool)
                labels_filtered = labels[valid_mask]

                if labels_filtered.shape[0] == num_points:
                    labels = labels_filtered
                else:
                    return None, None
            else:
                return None, None

        # Create a mask for valid labels (ignore_index -1 and out-of-range labels)
        valid_label_mask = (labels >= 0) & (labels < num_classes)

        # Return both the filtered labels and the mask to apply to X_c
        return labels[valid_label_mask], valid_label_mask

    except Exception as e:
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark compute_procrustes_Q_cuda_with_labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing scene subdirectories')
    parser.add_argument('--text_embed', type=str, required=True,
                       help='Path to text embeddings file (.pt or .npy)')
    parser.add_argument('--svd_rank', type=int, default=16,
                       help='SVD rank (default: 16)')
    parser.add_argument('--svd_suffix', type=str, default='_1',
                       help='SVD file suffix (default: "_1"). Use "" for no suffix, or specific suffix like "_1", "_2", "_3". Ignored if --find_all is set.')
    parser.add_argument('--find_all', action='store_true', default=False,
                       help='Find ALL SVD files matching the rank for each scene and compute average statistics across them')
    parser.add_argument('--label_file', type=str, default='lang_label.npy',
                       help='Name of label file (default: lang_label.npy)')

    # Benchmark arguments
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of benchmark runs per scene (default: 10)')
    parser.add_argument('--warmup_runs', type=int, default=2,
                       help='Number of warmup runs (default: 2)')
    parser.add_argument('--expand_factor', type=int, default=1,
                       help='Factor to expand data by repeating (for testing larger scales, default: 1). Ignored if --target_points is set.')
    parser.add_argument('--target_points', type=int, default=None,
                       help='Target number of points to expand each scene to (e.g., 1500000). Each scene will use a different expand_factor to reach this target.')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    # Output arguments
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save benchmark results (JSON format)')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    print("=" * 80)
    print("Procrustes Alignment Benchmark - compute_procrustes_Q_cuda_with_labels")
    print("=" * 80)
    print(f"Data root: {args.data_root}")
    print(f"Text embeddings: {args.text_embed}")
    print(f"SVD rank: {args.svd_rank}")
    print(f"Label file: {args.label_file}")
    print(f"Device: {device}")
    print(f"Benchmark runs per scene: {args.num_runs}")
    print(f"Warmup runs: {args.warmup_runs}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Total GPU memory: {props.total_memory / (1024**3):.2f} GB")

    print("=" * 80)
    print()

    # Load text embeddings
    print("Loading text embeddings...")
    Y, _ = load_text_embeddings_dict(args.text_embed, args.svd_rank, normalize=True, use_torch=False)
    print(f"  Y shape: {Y.shape}")
    num_classes = Y.shape[0]
    feature_dim = Y.shape[1]
    print(f"  Number of classes: {num_classes}")
    print(f"  Feature dimension: {feature_dim}")
    print()

    # Find scenes with required data
    print(f"Scanning {args.data_root} for scenes...")
    scenes = find_scenes_with_data(args.data_root, args.svd_rank, args.label_file, args.svd_suffix, args.find_all)
    print(f"  Found {len(scenes)} scene-SVD file combinations")
    if not scenes:
        print("  ERROR: No scenes found with both SVD features and label files!")
        return 1
    print()

    # If find_all is True, group results by scene name
    if args.find_all:
        # Group by scene name
        from collections import defaultdict
        scene_files = defaultdict(list)
        for scene_name, svd_file, label_file in scenes:
            scene_files[scene_name].append((svd_file, label_file))

        print(f"  Found {len(scene_files)} unique scenes")
        for scene_name, files in sorted(scene_files.items()):
            print(f"    {scene_name}: {len(files)} SVD files")
        print()

        # Benchmark each scene
        all_results = []
        scene_names = []

        for scene_name, svd_files in sorted(scene_files.items()):
            print(f"Processing scene: {scene_name}")
            print(f"  Number of SVD files: {len(svd_files)}")

            scene_results = []
            for svd_file, label_file in svd_files:
                suffix = Path(svd_file).stem.split('_r')[-1]  # Extract suffix like '16_1' -> '_1'
                print(f"  SVD file: {svd_file}")

                # Load features
                X_c = load_grid_svd_features(svd_file)
                print(f"    X_c shape: {X_c.shape}")

                # Load and prepare labels (returns filtered labels and mask)
                labels, valid_label_mask = load_and_prepare_labels(label_file, svd_file, num_classes)
                if labels is None or valid_label_mask is None:
                    print(f"    WARNING: Failed to load/prepare labels, skipping")
                    continue

                # Apply mask to X_c to keep only points with valid labels
                X_c = X_c[valid_label_mask]
                print(f"    X_c filtered shape: {X_c.shape}")
                print(f"    Labels shape: {labels.shape}")
                print(f"    Unique labels: {len(np.unique(labels))}")

                # Run benchmark
                print(f"    Running benchmark ({args.num_runs} runs + {args.warmup_runs} warmup)...")

                # Calculate expand_factor if target_points is specified
                if args.target_points is not None:
                    expand_factor = max(1, int(np.round(args.target_points / X_c.shape[0])))
                else:
                    expand_factor = args.expand_factor

                result = benchmark_single_scene(
                    X_c, Y, labels, device, args.num_runs, args.warmup_runs, expand_factor
                )
                result['scene_name'] = scene_name
                result['svd_file'] = svd_file
                result['label_file'] = label_file
                result['svd_suffix'] = suffix

                scene_results.append(result)

                # Print per-file results
                print(f"    Results:")
                if result['expand_factor'] > 1:
                    print(f"      Data expanded: {result['N_original']:,} -> {result['N']:,} points (x{result['expand_factor']})")
                print(f"      Time: {result['time_mean_ms']:.2f} ms")
                print(f"      Peak Memory: {result['peak_memory_mean_mb']:.2f} MB (input: {result['input_memory_mb']['total_mb']:.2f} MB)")
                print(f"      Computation Overhead: {result['computation_overhead_mean_mb']:.2f} MB")
                print()

            if not scene_results:
                print(f"  WARNING: No valid results for scene {scene_name}, skipping")
                continue

            # Compute average across all SVD files for this scene
            avg_result = {
                'scene_name': scene_name,
                'num_svd_files': len(scene_results),
                'svd_files': [r['svd_file'] for r in scene_results],
                'time_mean_ms': float(np.mean([r['time_mean_ms'] for r in scene_results])),
                'time_std_ms': float(np.std([r['time_mean_ms'] for r in scene_results])),
                'time_min_ms': float(np.min([r['time_min_ms'] for r in scene_results])),
                'time_max_ms': float(np.max([r['time_max_ms'] for r in scene_results])),
                'peak_memory_mean_mb': float(np.mean([r['peak_memory_mean_mb'] for r in scene_results])),
                'peak_memory_max_mb': float(np.max([r['peak_memory_max_mb'] for r in scene_results])),
                'computation_overhead_mean_mb': float(np.mean([r['computation_overhead_mean_mb'] for r in scene_results])),
                'cosine_improvement_mean': float(np.mean([r['cosine_improvement_mean'] for r in scene_results])),
                'N': scene_results[0]['N'],  # Same for all SVD files
                'M': scene_results[0]['M'],
                'input_memory_mb': scene_results[0]['input_memory_mb'],
                'individual_results': scene_results,
            }
            all_results.append(avg_result)
            scene_names.append(scene_name)

            # Print averaged results
            print(f"  Averaged Results (across {len(scene_results)} SVD files):")
            print(f"    Time: {avg_result['time_mean_ms']:.2f} ± {avg_result['time_std_ms']:.2f} ms "
                  f"(min: {avg_result['time_min_ms']:.2f}, max: {avg_result['time_max_ms']:.2f})")
            print(f"    Peak Memory: {avg_result['peak_memory_mean_mb']:.2f} MB "
                  f"(peak: {avg_result['peak_memory_max_mb']:.2f} MB)")
            print(f"      Input tensors: {avg_result['input_memory_mb']['total_mb']:.2f} MB "
                  f"(X_c: {avg_result['input_memory_mb']['X_c_mb']:.2f} MB, "
                  f"Y: {avg_result['input_memory_mb']['Y_mb']:.2f} MB, "
                  f"labels: {avg_result['input_memory_mb']['labels_mb']:.2f} MB)")
            print(f"      Computation overhead: {avg_result['computation_overhead_mean_mb']:.2f} MB")
            print(f"    Cosine improvement: {avg_result['cosine_improvement_mean']:.4f}")
            print()
    else:
        # Original behavior: process each scene-SVD combination separately
        all_results = []
        scene_names = []

        for scene_name, svd_file, label_file in scenes:
            print(f"Processing scene: {scene_name}")
            print(f"  SVD file: {svd_file}")
            print(f"  Label file: {label_file}")

            # Load features
            X_c = load_grid_svd_features(svd_file)
            print(f"  X_c shape: {X_c.shape}")

            # Load and prepare labels (returns filtered labels and mask)
            labels, valid_label_mask = load_and_prepare_labels(label_file, svd_file, num_classes)
            if labels is None or valid_label_mask is None:
                print(f"  WARNING: Failed to load/prepare labels, skipping scene")
                continue

            # Apply mask to X_c to keep only points with valid labels
            X_c = X_c[valid_label_mask]
            print(f"  X_c filtered shape: {X_c.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Unique labels: {len(np.unique(labels))}")

            # Run benchmark
            print(f"  Running benchmark ({args.num_runs} runs + {args.warmup_runs} warmup)...")

            # Calculate expand_factor if target_points is specified
            if args.target_points is not None:
                expand_factor = max(1, int(np.round(args.target_points / X_c.shape[0])))
            else:
                expand_factor = args.expand_factor

            result = benchmark_single_scene(
                X_c, Y, labels, device, args.num_runs, args.warmup_runs, expand_factor
            )
            result['scene_name'] = scene_name
            result['svd_file'] = svd_file
            result['label_file'] = label_file

            all_results.append(result)
            scene_names.append(scene_name)

            # Print per-scene results
            print(f"  Results:")
            if result['expand_factor'] > 1:
                print(f"    Data expanded: {result['N_original']:,} -> {result['N']:,} points (x{result['expand_factor']})")
            print(f"    Time: {result['time_mean_ms']:.2f} ± {result['time_std_ms']:.2f} ms "
                  f"(min: {result['time_min_ms']:.2f}, max: {result['time_max_ms']:.2f})")
            print(f"    Peak Memory: {result['peak_memory_mean_mb']:.2f} MB (input: {result['input_memory_mb']['total_mb']:.2f} MB)")
            print(f"    Computation Overhead: {result['computation_overhead_mean_mb']:.2f} MB")
            print(f"    Memory per point: {result['memory_per_point_kb']:.2f} KB")
            print(f"    Cosine improvement: {result['cosine_improvement_mean']:.4f}")
            print()

    # Compute aggregate statistics
    print("=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    if all_results:
        # Aggregate time statistics
        all_times = [r['time_mean_ms'] for r in all_results]
        all_time_stds = [r['time_std_ms'] for r in all_results]

        # Aggregate memory statistics
        all_peak_memory = [r.get('peak_memory_mean_mb', 0) for r in all_results]
        all_computation_overhead = [r.get('computation_overhead_mean_mb', 0) for r in all_results]

        # Aggregate data sizes
        all_N = [r['N'] for r in all_results]
        all_M = [r['M'] for r in all_results]

        # Compute average time per scene
        avg_time_mean = np.mean(all_times)
        avg_time_std = np.mean(all_time_stds)
        avg_peak_memory = np.mean(all_peak_memory)
        avg_computation_overhead = np.mean(all_computation_overhead)

        print(f"Number of scenes benchmarked: {len(all_results)}")
        print()
        print("Time Statistics:")
        print(f"  Average time per scene: {avg_time_mean:.2f} ± {avg_time_std:.2f} ms")
        print(f"  Min time: {np.min(all_times):.2f} ms")
        print(f"  Max time: {np.max(all_times):.2f} ms")
        print()
        print("Memory Statistics:")
        print(f"  Average peak memory: {avg_peak_memory:.2f} MB")
        print(f"  Average computation overhead: {avg_computation_overhead:.2f} MB")
        print()
        print("Data Size Statistics:")
        print(f"  N (points): min={np.min(all_N):,}, max={np.max(all_N):,}, mean={np.mean(all_N):.0f}")
        print(f"  M (classes): {all_M[0]} (same for all scenes)")
        print()

        # Per-scene breakdown table
        print("Per-Scene Breakdown:")
        print("-" * 80)
        print(f"{'Scene':<20} {'N (points)':<12} {'Time (ms)':<18} {'Peak Mem (MB)':<15} {'Cos Δ':<10}")
        print("-" * 80)

        for i, result in enumerate(all_results):
            peak_mem = result.get('peak_memory_mean_mb', 0)
            print(f"{result['scene_name']:<20} "
                  f"{result['N']:<12,} "
                  f"{result['time_mean_ms']:.2f} ± {result['time_std_ms']:.2f}  "
                  f"{peak_mem:.2f}  "
                  f"{result['cosine_improvement_mean']:+.4f}")

        print("-" * 80)
        print()

        # Efficiency metrics
        print("Efficiency Metrics:")
        throughput_points_per_ms = np.mean(all_N) / avg_time_mean
        throughput_points_per_sec = throughput_points_per_ms * 1000
        print(f"  Throughput: {throughput_points_per_sec:.0f} points/second")
        print(f"  Memory efficiency: {avg_peak_memory / np.mean(all_N) * 1024:.4f} KB/point")
        print()

        # Save results if requested
        if args.output_file:
            output_data = {
                'config': {
                    'data_root': args.data_root,
                    'text_embed': args.text_embed,
                    'svd_rank': args.svd_rank,
                    'label_file': args.label_file,
                    'device': str(device),
                    'num_runs': args.num_runs,
                    'warmup_runs': args.warmup_runs,
                    'num_classes': int(num_classes),
                    'feature_dim': int(feature_dim),
                },
                'scenes': scene_names,
                'aggregate': {
                    'avg_time_mean_ms': float(avg_time_mean),
                    'avg_time_std_ms': float(avg_time_std),
                    'min_time_ms': float(np.min(all_times)),
                    'max_time_ms': float(np.max(all_times)),
                    'avg_peak_memory_mb': float(avg_peak_memory),
                    'avg_computation_overhead_mb': float(avg_computation_overhead),
                    'throughput_points_per_sec': float(throughput_points_per_sec),
                },
                'per_scene': all_results,
            }

            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"Results saved to: {args.output_file}")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
