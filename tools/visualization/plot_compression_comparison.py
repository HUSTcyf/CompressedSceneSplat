#!/usr/bin/env python3
"""
Visualization script for comparing language feature storage before and after compression.

This script creates bar charts comparing original vs compressed language feature sizes,
matching the style of reference scientific publications.

Usage:
    # Basic comparison with custom data root
    python tools/plot_compression_comparison.py --data-root /path/to/data

    # Compare SceneSplat datasets (ScanNet, ScanNet++ with HF lang_feat)
    python tools/plot_compression_comparison.py --dataset scannet --svd-rank 128

    # Compare Matterport3D with HF lang_feat
    python tools/plot_compression_comparison.py --dataset matterport3d --svd-rank 64

    # Compare multiple ranks (all 3 ranks in one chart)
    python tools/plot_compression_comparison.py --dataset scannetpp --show-all-ranks

    # Load compressed stats from JSON file (when compressed files are not local)
    python tools/plot_compression_comparison.py --stats-json scannetpp.json --dataset scannetpp --svd-rank 32

    # Compare all datasets from JSON files
    python tools/plot_compression_comparison.py --compare-datasets --show-all-ranks

    # Save to file
    python tools/plot_compression_comparison.py --dataset scannet --save-plot scannet_comparison.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec

# Import PROJECT_ROOT - handle both script and module execution
try:
    from .. import PROJECT_ROOT  # Relative import when run as module
except ImportError:
    # Fallback when run as script: add parent dir to sys.path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

# SceneSplat dataset configurations
DATASET_CONFIGS = {
    "scannet": {
        "root": Path("/new_data/cyf/Datasets/SceneSplat7k/scannet"),
        "subdirs": {
            "train": "train_grid1.0cm_chunk6x6_stride3x3",
            "test": "test_grid1.0cm_chunk6x6_stride3x3",
        },
        "hf_repo_id": "clapfor/scannet_mcmc_3dgs_lang_base",
        "hf_subfolders": [
            "train_grid1.0cm_chunk6x6_stride3x3",
            "test_grid1.0cm_chunk6x6_stride3x3",
        ],
        "excluded_scenes": [],
    },
    "scannetpp": {
        "root": Path("/new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2"),
        "subdirs": {
            "train": "train_grid1.0cm_chunk6x6_stride3x3",
            "test": "test_grid1.0cm_chunk6x6_stride3x3",
        },
        "hf_repo_id": "clapfor/scannetpp_v2_mcmc_3dgs_lang_base",
        "hf_subfolders": [
            "train_grid1.0cm_chunk6x6_stride3x3",
            "test_grid1.0cm_chunk6x6_stride3x3",
        ],
        "excluded_scenes": [
            "1b379f1114_0", "5c215ef3b0_1", "5d591ff74d_1",
            "b4b39438f0_7", "ea42cd27e6_1",
        ],
    },
    "matterport3d": {
        "root": Path("/new_data/cyf/Datasets/SceneSplat7k/matterport3d"),
        "subdirs": {
            "train": "train_grid1.0cm_chunk6x6x4_stride4x4x4",
            "val": "val_grid1.0cm_chunk6x6x4_stride4x4x4",
        },
        "hf_repo_id": "clapfor/matterport3d_scene_mcmc_3dgs_lang_base",
        "hf_subfolders": [
            "train_grid1.0cm_chunk6x6x4_stride4x4x4",
            "val_grid1.0cm_chunk6x6x4_stride4x4x4",
        ],
        "excluded_scenes": [],
    },
}

# Constants for base-10 (decimal) units
BYTES_PER_GB = 1_000_000_000
BYTES_PER_MB = 1_000_000


def bytes_to_gb(size_bytes: int) -> float:
    """Convert bytes to decimal GB (1 GB = 1,000,000,000 bytes)."""
    return size_bytes / BYTES_PER_GB


def bytes_to_mb(size_bytes: int) -> float:
    """Convert bytes to decimal MB (1 MB = 1,000,000 bytes)."""
    return size_bytes / BYTES_PER_MB


def get_file_size_bytes(file_path: str) -> int:
    """Get file size in bytes."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    return 0


def get_file_size_gb(file_path: str) -> float:
    """Get file size in decimal GB."""
    return bytes_to_gb(get_file_size_bytes(file_path))


# Legacy function for backward compatibility
def get_file_size_mb(file_path: str) -> float:
    """Get file size in decimal MB."""
    return bytes_to_mb(get_file_size_bytes(file_path))


def get_huggingface_file_info(
    repo_id: str,
    subfolders: str | List[str],
    excluded_scenes: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Get file sizes from HuggingFace repository.

    Args:
        repo_id: HuggingFace repository ID
        subfolders: Subfolder path(s) in the repository (can be a single string or a list)
        excluded_scenes: Optional list of scene names to exclude from the results

    Returns:
        Dictionary mapping file names to sizes in bytes
    """
    # Normalize subfolders to a list
    if isinstance(subfolders, str):
        subfolders = [subfolders]

    # Create excluded scenes set for faster lookup
    excluded_scene_set = set(excluded_scenes) if excluded_scenes else set()

    try:
        from huggingface_hub import HfApi
        api = HfApi()

        # Get repo info with file sizes
        repo_info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)

        file_sizes = defaultdict(int)

        # Filter files by each subfolder
        for subfolder in subfolders:
            prefix = subfolder.rstrip("/") + "/"
            for file_info in repo_info.siblings:
                if file_info.rfilename.startswith(prefix):
                    # Extract scene name and file name
                    rel_path = file_info.rfilename[len(prefix):]
                    parts = rel_path.split("/")
                    if len(parts) >= 2:
                        scene_name = parts[0]
                        # Skip excluded scenes
                        if scene_name in excluded_scene_set:
                            continue
                        file_name = parts[1]
                        size = file_info.size if hasattr(file_info, 'size') else 0
                        if size > 0:
                            # Store as scene_name/file_name -> size in bytes
                            key = f"{scene_name}/{file_name}"
                            file_sizes[key] = size

        return dict(file_sizes)

    except ImportError:
        print("[WARNING] huggingface_hub not installed. Cannot fetch HF file sizes.")
        print("  Install with: pip install huggingface_hub")
        return {}
    except Exception as e:
        print(f"[WARNING] Failed to fetch HuggingFace file info: {e}")
        return {}


def load_stats_from_json(json_path: str) -> Dict:
    """
    Load grid SVD statistics from a JSON file.

    Args:
        json_path: Path to the JSON stats file

    Returns:
        Dictionary containing the stats data
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"[ERROR] JSON stats file not found: {json_path}")
        return {}

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded stats from {json_path}")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load JSON stats: {e}")
        return {}


def get_compressed_size_from_json(
    stats_data: Dict,
    svd_rank: int,
    file_count: int = None
) -> int:
    """
    Get compressed size from JSON stats data.

    Args:
        stats_data: JSON stats data from analyze_disk_usage.py
        svd_rank: SVD rank to get size for
        file_count: Number of files (to calculate average if needed)

    Returns:
        Compressed size in bytes (average per file)
    """
    if not stats_data or "files" not in stats_data:
        return 0

    file_name = f"lang_feat_grid_svd_r{svd_rank}.npz"
    if file_name in stats_data["files"]:
        file_data = stats_data["files"][file_name]
        count = file_data.get("count", 1)
        total_bytes = file_data.get("size_bytes", 0)
        # Return average size per file in bytes
        return total_bytes // count if count > 0 else total_bytes

    return 0


def load_datasets_from_jsons(
    json_files: Dict[str, str],
    svd_rank: int,
) -> Dict[str, Dict]:
    """
    Load dataset statistics from multiple JSON files.

    Args:
        json_files: Dictionary mapping dataset names to JSON file paths
                   e.g., {'scannet': 'scannet.json', 'scannetpp': 'scannetpp.json'}
        svd_rank: SVD rank to get compressed size for

    Returns:
        Dictionary mapping dataset names to their stats
        {'scannet': {'original': total_gb, 'compressed': total_gb, 'count': n}, ...}
    """
    results = {}

    for dataset_name, json_path in json_files.items():
        stats_data = load_stats_from_json(json_path)
        if not stats_data:
            print(f"[WARNING] Failed to load stats from {json_path}")
            continue

        # Get compressed size for this rank (in bytes)
        compressed_avg_bytes = get_compressed_size_from_json(stats_data, svd_rank)
        if compressed_avg_bytes == 0:
            print(f"[WARNING] No data for rank {svd_rank} in {json_path}")
            continue

        # Get file count
        file_name = f"lang_feat_grid_svd_r{svd_rank}.npz"
        count = stats_data["files"].get(file_name, {}).get("count", 0)

        # Calculate totals in bytes
        compressed_total_bytes = compressed_avg_bytes * count

        # For original, fetch from HuggingFace
        if dataset_name in DATASET_CONFIGS:
            config = DATASET_CONFIGS[dataset_name]
            hf_repo_id = config["hf_repo_id"]
            hf_subfolders = config["hf_subfolders"]
            excluded_scenes = config.get("excluded_scenes", [])

            print(f"Fetching original lang_feat sizes from HuggingFace: {hf_repo_id}")
            print(f"  Subfolders: {hf_subfolders}")
            if excluded_scenes:
                print(f"  Excluding {len(excluded_scenes)} scenes: {', '.join(excluded_scenes)}")
            hf_data = get_huggingface_file_info(hf_repo_id, hf_subfolders, excluded_scenes)

            # Sum up all lang_feat.npy sizes (in bytes)
            original_total_bytes = 0
            for key, size in hf_data.items():
                if "/lang_feat.npy" in key:
                    original_total_bytes += size

            if original_total_bytes == 0:
                print(f"[WARNING] No original lang_feat size found in HuggingFace for {dataset_name}")
                continue

        else:
            print(f"[WARNING] No HuggingFace config for {dataset_name}")
            continue

        # Convert to GB
        original_total_gb = bytes_to_gb(original_total_bytes)
        compressed_total_gb = bytes_to_gb(compressed_total_bytes)

        results[dataset_name] = {
            'original': original_total_gb,
            'compressed': compressed_total_gb,
            'count': count
        }
        print(f"Loaded {dataset_name}: {count} files, original={original_total_gb:.2f}GB, compressed={compressed_total_gb:.2f}GB")

    return results


def scan_compression_data(
    data_root: str = None,
    dataset_name: str = None,
    split: str = "train",
    svd_rank: int = None,
    pattern: str = "lang_feat_grid_svd_r{}.npz",
    include_hf: bool = False,
    excluded_scenes: Optional[List[str]] = None,
    stats_json_path: str = None,
) -> Dict[str, Dict[str, float]]:
    """
    Scan directory for original and compressed language features.

    Args:
        data_root: Direct path to data directory (if dataset_name is None)
        dataset_name: Name of SceneSplat dataset (scannet, scannetpp, matterport3d)
        split: Dataset split to analyze (train, test, val)
        svd_rank: Specific SVD rank to compare
        pattern: Pattern for compressed files
        include_hf: Include HuggingFace lang_feat files
        excluded_scenes: List of scenes to exclude from analysis
        stats_json_path: Path to JSON file with compressed stats (from analyze_disk_usage.py)

    Returns:
        Dictionary mapping scene names to {'original': size_gb, 'compressed': size_gb}
    """
    results = {}
    hf_lang_feat_sizes = {}
    stats_json_data = None
    compressed_size_from_json = 0

    # Load compressed size from JSON if provided (returns bytes)
    if stats_json_path:
        stats_json_data = load_stats_from_json(stats_json_path)
        if stats_json_data and svd_rank:
            compressed_size_from_json = get_compressed_size_from_json(stats_json_data, svd_rank)
            if compressed_size_from_json > 0:
                print(f"Using compressed size from JSON: {bytes_to_mb(compressed_size_from_json):.2f} MB per file (rank={svd_rank})")

    # Determine data root and get HF data if needed
    if dataset_name and dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        if split not in config["subdirs"] or config["subdirs"][split] is None:
            print(f"[WARNING] Split '{split}' not available for dataset '{dataset_name}'")
            return {}

        data_root = str(config["root"] / config["subdirs"][split])
        excluded_scenes = config.get("excluded_scenes", [])

        # Fetch HF lang_feat sizes if requested (returns bytes)
        if include_hf:
            hf_repo_id = config["hf_repo_id"]
            hf_subfolders = config["hf_subfolders"]
            hf_data = get_huggingface_file_info(hf_repo_id, hf_subfolders, excluded_scenes)
            # Extract only lang_feat.npy files and convert to GB
            for key, size in hf_data.items():
                if "/lang_feat.npy" in key:
                    scene_name = key.split("/")[0]
                    hf_lang_feat_sizes[scene_name] = bytes_to_gb(size)

    if not data_root or not os.path.exists(data_root):
        print(f"[WARNING] Data root {data_root} does not exist!")
        return {}

    # Get file count from JSON or scan local directory
    file_count = 0
    if stats_json_data and "files" in stats_json_data:
        file_name = f"lang_feat_grid_svd_r{svd_rank}.npz" if svd_rank else None
        if file_name and file_name in stats_json_data["files"]:
            file_count = stats_json_data["files"][file_name].get("count", 0)
            print(f"Using file count from JSON: {file_count} files")
        else:
            # Count local directories to estimate file count
            file_count = sum(1 for d in Path(data_root).iterdir() if d.is_dir())
            if excluded_scenes:
                file_count -= sum(1 for s in excluded_scenes if s in [d.name for d in Path(data_root).iterdir() if d.is_dir()])
            print(f"Using local file count: {file_count} files")

    # Scan local files
    for scene_dir in Path(data_root).iterdir():
        if not scene_dir.is_dir():
            continue

        scene_name = scene_dir.name

        # Skip excluded scenes
        if excluded_scenes and scene_name in excluded_scenes:
            continue

        # Get compressed size (from JSON or local file) in GB
        compressed_size = 0.0
        if compressed_size_from_json > 0:
            # Use average size from JSON (convert bytes to GB)
            compressed_size = bytes_to_gb(compressed_size_from_json)
        else:
            # Scan local file
            compressed_path = None
            if svd_rank is not None:
                compressed_path = scene_dir / pattern.format(svd_rank)
                if not compressed_path.exists():
                    compressed_path = None
            else:
                # Find any compressed version
                compressed_files = list(scene_dir.glob("lang_feat_grid_svd_r*.npz"))
                if compressed_files:
                    compressed_path = compressed_files[0]

            if compressed_path is None:
                continue

            compressed_size = get_file_size_gb(str(compressed_path))

        # Get original size (prefer HF if available and include_hf is True) - already in GB
        original_size = 0.0
        if include_hf and scene_name in hf_lang_feat_sizes:
            original_size = hf_lang_feat_sizes[scene_name]
        else:
            original_path = scene_dir / "lang_feat.npy"
            if original_path.exists():
                original_size = get_file_size_gb(str(original_path))
            else:
                # Skip scene if no original data available
                continue

        results[scene_name] = {
            'original': original_size,
            'compressed': compressed_size,
            'from_hf': include_hf and scene_name in hf_lang_feat_sizes
        }

    # If using JSON compressed sizes but no local files found, create aggregate entry
    if not results and compressed_size_from_json > 0 and include_hf:
        # Create a single aggregate entry using total sizes
        if stats_json_data and "files" in stats_json_data:
            # Get original lang_feat size from stats if available (in bytes)
            original_total_bytes = 0
            for fname, fdata in stats_json_data["files"].items():
                if "lang_feat" in fname and "grid" not in fname and "index" not in fname:
                    original_total_bytes += fdata.get("size_bytes", 0)

            if original_total_bytes > 0:
                # Use total sizes from JSON (convert bytes to GB)
                count = file_count or 1
                original_avg = bytes_to_gb(original_total_bytes) / count
                compressed_avg = bytes_to_gb(compressed_size_from_json)
                results["aggregate"] = {
                    'original': original_avg,
                    'compressed': compressed_avg,
                    'from_hf': False
                }
                print(f"Created aggregate entry using JSON stats (original: {original_avg:.2f} GB, compressed: {compressed_avg:.2f} GB)")

    return results


def create_comparison_bar_chart(
    data: Dict[str, Dict[str, float]],
    svd_rank: int = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 7),
    title: str = None,
    use_log_scale: bool = True,
) -> None:
    """
    Create a bar chart comparing original vs compressed sizes.

    Style matching pie chart colors:
    - Red (#e41a1c) for original (lang_feat)
    - Gray (#969696) for compressed
    - Log scale for y-axis to handle large size differences
    - Grid lines for readability
    - Large, clear labels

    Args:
        data: Dictionary mapping scene names to size data
        svd_rank: SVD rank for label
        save_path: Path to save the plot
        figsize: Figure size
        title: Custom title for the chart
        use_log_scale: Whether to use logarithmic y-axis
    """
    if not data:
        print("No data found to visualize!")
        return

    scenes = list(data.keys())
    original_sizes = [data[s]['original'] for s in scenes]
    compressed_sizes = [data[s]['compressed'] for s in scenes]

    # Calculate compression statistics
    original_total = sum(original_sizes)
    compressed_total = sum(compressed_sizes)
    compression_ratio = compressed_total / original_total if original_total > 0 else 0
    space_saved = original_total - compressed_total
    space_saved_pct = (space_saved / original_total * 100) if original_total > 0 else 0

    # Create figure with reference style
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Bar positions
    x = np.arange(len(scenes))
    width = 0.35

    # Colors matching pie chart style (red for original, gray for compressed)
    color_original = '#e41a1c'  # Red (like lang_feat in pie chart)
    color_compressed = '#969696'  # Gray (like others in pie chart)

    # Create bars
    bars1 = ax.bar(x - width/2, original_sizes, width,
                   label='Original', color=color_original, alpha=0.9)
    bars2 = ax.bar(x + width/2, compressed_sizes, width,
                   label=f'Compressed (SVD rank={svd_rank})' if svd_rank else 'Compressed',
                   color=color_compressed, alpha=0.9)

    # Reference style: grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    # Set log scale if requested
    if use_log_scale:
        ax.set_yscale('log')
        # Set specific ticks for log scale: 1, 10, 100, 1000
        ax.set_yticks([1, 10, 100, 1000])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y)}'))
    else:
        # Y-axis formatting for linear scale
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0f}'))

    # Labels and title (large, bold fonts like reference)
    ax.set_xlabel('Scenes', fontsize=18, fontweight='bold')
    ax.set_ylabel('Storage Size (GB)' + (' (log scale)' if use_log_scale else ''), fontsize=18, fontweight='bold')

    rank_text = f" (SVD rank={svd_rank})" if svd_rank else ""
    if title:
        ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
    else:
        ax.set_title(f'Language Feature Storage Comparison{rank_text}',
                     fontsize=22, fontweight='bold', pad=20)

    # X-axis labels with scene names
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha='right', fontsize=12)

    # Legend (top right, like reference)
    ax.legend(fontsize=14, frameon=True, shadow=False,
              loc='upper right', fancybox=False)

    # Add value labels on bars (for clarity)
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                if use_log_scale:
                    # For log scale, show value in appropriate units
                    if height >= 1000:
                        label = f'{height/1000:.1f}K'
                    else:
                        label = f'{height:.1f}'
                else:
                    label = f'{height:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label,
                       ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Add summary statistics as text box
    stats_text = f'Total Original: {original_total:.1f} GB\n'
    stats_text += f'Total Compressed: {compressed_total:.1f} GB\n'
    stats_text += f'Compression Ratio: {compression_ratio:.2f}x\n'
    stats_text += f'Space Saved: {space_saved_pct:.1f}%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()

    plt.close()

    # Print statistics
    print("\n" + "="*60)
    print("COMPRESSION STATISTICS")
    print("="*60)
    print(f"Items analyzed: {len(scenes)}")
    print(f"Total original size:     {original_total:10.1f} GB")
    print(f"Total compressed size:   {compressed_total:10.1f} GB")
    print(f"Compression ratio:       {compression_ratio:10.2f}x")
    print(f"Space saved:             {space_saved:10.1f} GB ({space_saved_pct:.1f}%)")
    print("="*60)


def create_dataset_comparison_chart(
    datasets_data: Dict[str, Dict],
    svd_rank: int = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 7),
    use_log_scale: bool = True,
) -> None:
    """
    Create a comparison chart for multiple datasets.

    Args:
        datasets_data: Dictionary mapping dataset names to their stats
                       {'scannet': {'original': total_mb, 'compressed': total_mb, 'count': n}, ...}
        svd_rank: SVD rank for label
        save_path: Path to save the plot
        figsize: Figure size
        use_log_scale: Whether to use logarithmic y-axis
    """
    if not datasets_data:
        print("No data found to visualize!")
        return

    dataset_names = list(datasets_data.keys())
    original_totals = [datasets_data[d]['original'] for d in dataset_names]
    compressed_totals = [datasets_data[d]['compressed'] for d in dataset_names]
    counts = [datasets_data[d]['count'] for d in dataset_names]

    # Calculate per-file averages
    original_avg = [datasets_data[d]['original'] / datasets_data[d]['count'] for d in dataset_names]
    compressed_avg = [datasets_data[d]['compressed'] / datasets_data[d]['count'] for d in dataset_names]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=150)

    x = np.arange(len(dataset_names))
    width = 0.35

    # Colors matching pie chart style (red for original, gray for compressed)
    color_original = '#e41a1c'  # Red (like lang_feat in pie chart)
    color_compressed = '#969696'  # Gray (like others in pie chart)

    # Left plot: Total sizes
    bars1 = ax1.bar(x - width/2, original_totals, width,
                   label='Original', color=color_original, alpha=0.9)
    bars2 = ax1.bar(x + width/2, compressed_totals, width,
                   label='Compressed', color=color_compressed, alpha=0.9)

    ax1.set_xlabel('Dataset', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Total Size (GB)' + (' (log scale)' if use_log_scale else ''), fontsize=16, fontweight='bold')
    ax1.set_title('Total Storage by Dataset', fontsize=18, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.upper() for d in dataset_names], fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)

    # Set log scale for left plot if requested
    if use_log_scale:
        ax1.set_yscale('log')
        ax1.set_yticks([1, 10, 100, 1000])
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y)}'))

    # Add value labels on total bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                if use_log_scale:
                    # Values are now in GB, so use appropriate thresholds
                    if height >= 1000:
                        label = f'{height/1000:.1f}K'
                    else:
                        label = f'{height:.1f}'
                else:
                    label = f'{height:.1f} GB'
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=10)

    # Right plot: Per-file averages
    bars3 = ax2.bar(x - width/2, original_avg, width,
                   label='Original', color=color_original, alpha=0.9)
    bars4 = ax2.bar(x + width/2, compressed_avg, width,
                   label='Compressed', color=color_compressed, alpha=0.9)

    ax2.set_xlabel('Dataset', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Avg Size Per File (GB)', fontsize=16, fontweight='bold')
    rank_text = f" (SVD rank={svd_rank})" if svd_rank else ""
    ax2.set_title(f'Average Size Per File{rank_text}', fontsize=18, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.upper() for d in dataset_names], fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)

    # Set log scale for right plot if requested
    if use_log_scale:
        ax2.set_yscale('log')
        ax2.set_yticks([1, 10, 100, 1000])
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y)}'))

    # Add value labels on average bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                if use_log_scale:
                    if height >= 1000:
                        label = f'{height/1000:.1f}K'
                    else:
                        label = f'{height:.1f}'
                else:
                    label = f'{height:.1f}'
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dataset comparison to {save_path}")
    else:
        plt.show()

    plt.close()

    # Print statistics
    print("\n" + "="*70)
    print("DATASET COMPARISON STATISTICS")
    print("="*70)
    print(f"{'Dataset':<15} {'Count':>8} {'Original (GB)':>18} {'Compressed (GB)':>18} {'Ratio':>10} {'Saved':>10}")
    print("-"*70)

    total_original_all = 0
    total_compressed_all = 0

    for name in dataset_names:
        data = datasets_data[name]
        orig = data['original']
        comp = data['compressed']
        count = data['count']
        ratio = comp / orig if orig > 0 else 0
        saved = (orig - comp) / orig * 100 if orig > 0 else 0

        total_original_all += orig
        total_compressed_all += comp

        print(f"{name.upper():<15} {count:>8} {orig:>18.1f} {comp:>18.1f} {ratio:>10.3f} {saved:>9.1f}%")

    print("-"*70)
    overall_ratio = total_compressed_all / total_original_all if total_original_all > 0 else 0
    overall_saved = (total_original_all - total_compressed_all) / total_original_all * 100 if total_original_all > 0 else 0
    print(f"{'OVERALL':<15} {'':>8} {total_original_all:>18.1f} {total_compressed_all:>18.1f} {overall_ratio:>10.3f} {overall_saved:>9.1f}%")
    print("="*70)


def create_multi_rank_comparison(
    dataset_name: str = None,
    data_root: str = None,
    split: str = "train",
    ranks: List[int] = None,
    include_hf: bool = False,
    stats_json_path: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 7)
):
    """
    Create a comparison chart for multiple SVD ranks.

    Shows how compression ratio changes with different ranks.
    """
    all_data = {}
    for rank in ranks:
        rank_data = scan_compression_data(
            data_root=data_root,
            dataset_name=dataset_name,
            split=split,
            svd_rank=rank,
            include_hf=include_hf,
            stats_json_path=stats_json_path
        )
        if rank_data:
            all_data[rank] = rank_data

    if not all_data:
        print("No compressed data found for any rank!")
        return

    # Get common scenes across all ranks
    common_scenes = set(all_data[ranks[0]].keys())
    for rank in ranks[1:]:
        common_scenes &= set(all_data[rank].keys())
    common_scenes = sorted(list(common_scenes))

    if not common_scenes:
        print("No common scenes found across all ranks!")
        return

    # Aggregate data
    original_sizes = [all_data[ranks[0]][scene]['original'] for scene in common_scenes]
    rank_sizes = {rank: [all_data[rank][scene]['compressed'] for scene in common_scenes]
                  for rank in ranks}

    # Calculate compression ratios
    original_total = sum(original_sizes)
    ratios = {rank: sum(rank_sizes[rank]) / original_total for rank in ranks}

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=150)

    # Left plot: Storage comparison
    x = np.arange(len(common_scenes))
    width = 0.8 / (len(ranks) + 1)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ranks)))

    # Original bar
    ax1.bar(x - width * len(ranks) / 2, original_sizes, width,
           label='Original', color='#3274A1', alpha=0.9)

    # Compressed bars for each rank
    for i, rank in enumerate(ranks):
        ax1.bar(x - width * len(ranks) / 2 + width * (i + 1),
               rank_sizes[rank], width,
               label=f'Rank {rank}', color=colors[i], alpha=0.9)

    ax1.set_xlabel('Scenes', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Storage Size (GB)', fontsize=16, fontweight='bold')
    ax1.set_title('Storage by SVD Rank', fontsize=18, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(common_scenes, rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)

    # Right plot: Compression ratio trend
    rank_labels = [f'R{r}' for r in ranks]
    ratio_values = [ratios[r] * 100 for r in ranks]

    bars = ax2.bar(rank_labels, ratio_values, color=colors, alpha=0.9)
    ax2.set_xlabel('SVD Rank', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Compressed Size (% of Original)', fontsize=14, fontweight='bold')
    ax2.set_title('Compression Ratio vs Rank', fontsize=18, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)

    # Add value labels
    for bar, val in zip(bars, ratio_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-rank comparison to {save_path}")
    else:
        plt.show()

    plt.close()


def create_all_ranks_comparison_chart(
    json_files: Dict[str, str],
    ranks: List[int] = [8, 16, 32],
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 7),
    use_log_scale: bool = True,
) -> None:
    """
    Create a comparison chart showing all 3 compression ranks together.

    For each dataset, shows:
    - Original lang_feat size
    - Compressed sizes for ranks 32, 16, 8 (in that order)

    Displays two plots: total storage and average per file.

    Args:
        json_files: Dictionary mapping dataset names to JSON file paths
        ranks: List of SVD ranks to show (default: [8, 16, 32], displayed in descending order)
        save_path: Path to save the plot
        figsize: Figure size
        use_log_scale: Whether to use logarithmic y-axis
    """
    # Load data for each rank
    all_data = {}
    for rank in ranks:
        rank_data = load_datasets_from_jsons(json_files, rank)
        if rank_data:
            all_data[rank] = rank_data

    if not all_data:
        print("[ERROR] Failed to load data for any rank!")
        return

    # Get common datasets across all ranks
    common_datasets = set(all_data[ranks[0]].keys())
    for rank in ranks[1:]:
        common_datasets &= set(all_data[rank].keys())
    common_datasets = sorted(list(common_datasets))

    if not common_datasets:
        print("[ERROR] No common datasets found across all ranks!")
        return

    # Prepare data: for each dataset, get original and all compressed sizes
    dataset_names = common_datasets
    original_totals = [all_data[ranks[0]][d]['original'] for d in dataset_names]
    counts = [all_data[ranks[0]][d]['count'] for d in dataset_names]

    # Compressed totals by rank (display order: 32, 16, 8 - descending)
    display_ranks = sorted(ranks, reverse=True)  # [32, 16, 8]
    compressed_totals_by_rank = {
        rank: [all_data[rank][d]['compressed'] for d in dataset_names]
        for rank in display_ranks
    }

    # Calculate per-file averages (in MB for better readability)
    original_avg = [all_data[ranks[0]][d]['original'] / all_data[ranks[0]][d]['count'] * 1000 for d in dataset_names]
    compressed_avg_by_rank = {
        rank: [all_data[rank][d]['compressed'] / all_data[rank][d]['count'] * 1000 for d in dataset_names]
        for rank in display_ranks
    }

    # Create figure with two subplots (like create_dataset_comparison_chart)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=150)

    x = np.arange(len(dataset_names))
    num_bars = len(display_ranks) + 1  # original + compressed for each rank
    width = 0.8 / num_bars

    # Colors matching pie chart style
    color_original = '#e41a1c'  # Red (like lang_feat in pie chart)

    # Colorful colors for each rank
    # Rank 32: blue (largest compression), Rank 16: green (medium), Rank 8: orange (smallest)
    rank_color_map = {32: '#377eb8', 16: '#4daf4a', 8: '#ff7f00'}

    # === LEFT PLOT: Total sizes ===
    # Original bar (leftmost)
    bars1_orig = ax1.bar(x - width * (num_bars - 1) / 2, original_totals, width,
                        label='Original', color=color_original, alpha=0.9)

    # Compressed bars for each rank (in descending order: 32, 16, 8)
    bars1_compressed = {}
    for i, rank in enumerate(display_ranks):
        offset = x - width * (num_bars - 1) / 2 + width * (i + 1)
        bars1_compressed[rank] = ax1.bar(offset, compressed_totals_by_rank[rank], width,
                                        label=f'Rank {rank}', color=rank_color_map.get(rank, '#969696'),
                                        alpha=0.9)

    ax1.set_xlabel('Dataset', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Total Size (GB)' + (' (log scale)' if use_log_scale else ''), fontsize=16, fontweight='bold')
    ax1.set_title('Total Storage by Dataset', fontsize=18, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.upper() for d in dataset_names], fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)

    # Set log scale for left plot if requested
    if use_log_scale:
        ax1.set_yscale('log')
        ax1.set_yticks([1, 10, 100, 1000])
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y)}'))

    # Add value labels on total bars
    def add_value_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                if use_log_scale:
                    if height >= 1000:
                        label = f'{height/1000:.1f}K'
                    else:
                        label = f'{height:.1f}'
                else:
                    label = f'{height:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=10)

    add_value_labels(bars1_orig, ax1)
    for rank in display_ranks:
        add_value_labels(bars1_compressed[rank], ax1)

    # === RIGHT PLOT: Per-file averages ===
    # Original bar (leftmost)
    bars2_orig = ax2.bar(x - width * (num_bars - 1) / 2, original_avg, width,
                        label='Original', color=color_original, alpha=0.9)

    # Compressed bars for each rank (in descending order: 32, 16, 8)
    bars2_compressed = {}
    for i, rank in enumerate(display_ranks):
        offset = x - width * (num_bars - 1) / 2 + width * (i + 1)
        bars2_compressed[rank] = ax2.bar(offset, compressed_avg_by_rank[rank], width,
                                        label=f'Rank {rank}', color=rank_color_map.get(rank, '#969696'),
                                        alpha=0.9)

    ax2.set_xlabel('Dataset', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Avg Size Per File (MB)', fontsize=16, fontweight='bold')
    ax2.set_title('Average Size Per File', fontsize=18, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.upper() for d in dataset_names], fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)

    # Set log scale for right plot
    ax2.set_yscale('log')
    ax2.set_yticks([1, 10, 100, 1000, 10000])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.1f}'))

    # Add value labels on average bars (log scale, MB units)
    def add_value_labels_log(bars, ax):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                if height >= 10000:
                    label = f'{height/1000:.1f}K'
                else:
                    label = f'{height:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=10)

    add_value_labels_log(bars2_orig, ax2)
    for rank in display_ranks:
        add_value_labels_log(bars2_compressed[rank], ax2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved all-ranks comparison to {save_path}")
    else:
        plt.show()

    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("ALL RANKS COMPARISON STATISTICS")
    print("="*80)

    for dataset in dataset_names:
        orig = all_data[ranks[0]][dataset]['original']
        count = all_data[ranks[0]][dataset]['count']
        print(f"\n{dataset.upper()} ({count} files):")
        print(f"  Total Original:   {orig:10.2f} GB")
        for rank in display_ranks:
            comp = all_data[rank][dataset]['compressed']
            ratio = comp / orig if orig > 0 else 0
            saved = (orig - comp) / orig * 100 if orig > 0 else 0
            print(f"  Total Rank {rank:2d}:   {comp:10.2f} GB (ratio: {ratio:.3f}x, saved: {saved:5.1f}%)")

    print("\n" + "="*80)
    print("TOTALS ACROSS ALL DATASETS:")
    print("="*80)

    total_original = sum(all_data[ranks[0]][d]['original'] for d in dataset_names)
    print(f"  Total Original:   {total_original:.2f} GB")

    for rank in display_ranks:
        total_compressed = sum(all_data[rank][d]['compressed'] for d in dataset_names)
        ratio = total_compressed / total_original if total_original > 0 else 0
        saved = (total_original - total_compressed) / total_original * 100 if total_original > 0 else 0
        print(f"  Total Rank {rank:2d}:   {total_compressed:.2f} GB (ratio: {ratio:.3f}x, saved: {saved:5.1f}%)")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize language feature compression comparison for SceneSplat datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare ScanNet with HF lang_feat
  python tools/plot_compression_comparison.py --dataset scannet --include-hf-lang-feat --svd-rank 128

  # Compare Matterport3D val split
  python tools/plot_compression_comparison.py --dataset matterport3d --split val --svd-rank 64

  # Compare multiple ranks
  python tools/plot_compression_comparison.py --dataset scannetpp --compare-ranks 64 128 256

  # Load compressed stats from JSON (when compressed files are not local)
  python tools/plot_compression_comparison.py --stats-json scannetpp.json --dataset scannetpp --svd-rank 32 --include-hf-lang-feat

  # Compare all 3 datasets from JSON files (HuggingFace + JSON compressed stats)
  python tools/plot_compression_comparison.py --compare-datasets --svd-rank 32 --save-plot datasets_comparison.png

  # Show all 3 compression ranks (8, 16, 32) together
  python tools/plot_compression_comparison.py --show-all-ranks --save-plot all_ranks_comparison.png

  # Use custom data root (no HF support)
  python tools/plot_compression_comparison.py --data-root /path/to/data --svd-rank 32
        """
    )

    # Dataset selection
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['scannet', 'scannetpp', 'matterport3d'],
        default=None,
        help='SceneSplat dataset name (uses predefined paths and HF configs)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'test', 'val'],
        help='Dataset split to analyze (default: train)'
    )

    parser.add_argument(
        '--include-hf-lang-feat',
        action='store_true',
        help='Include HuggingFace lang_feat.npy files for comparison (requires --dataset)'
    )

    # Custom data root (for non-SceneSplat data)
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='Custom root directory containing scene folders (disables --dataset config)'
    )

    # Compression options
    parser.add_argument(
        '--svd-rank',
        type=int,
        default=None,
        help='SVD rank to compare (if None, searches for any compressed file)'
    )

    parser.add_argument(
        '--compare-ranks',
        type=int,
        nargs='+',
        default=None,
        help='Compare multiple SVD ranks (e.g., --compare-ranks 64 128 256)'
    )

    # Stats JSON options
    parser.add_argument(
        '--stats-json',
        type=str,
        default=None,
        help='Path to JSON file with compressed stats (from analyze_disk_usage.py --save-stats)'
    )

    parser.add_argument(
        '--compare-datasets',
        action='store_true',
        help='Compare all 3 datasets (scannet, scannetpp, matterport3d) from JSON files. Requires --svd-rank.'
    )

    parser.add_argument(
        '--show-all-ranks',
        action='store_true',
        help='Show all 3 compression ranks (8, 16, 32) together in a grouped bar chart.'
    )

    # Output options
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Path to save the plot (if None, displays interactively)'
    )

    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[12, 7],
        help='Figure size (width height) in inches'
    )

    args = parser.parse_args()

    # Handle --compare-datasets mode
    if args.compare_datasets:
        if not args.svd_rank:
            print("[ERROR] --compare-datasets requires --svd-rank to be specified")
            sys.exit(1)

        # Look for JSON files in project root
        project_root = Path(__file__).resolve().parent.parent
        json_files = {}
        for dataset_name in ['scannet', 'scannetpp', 'matterport3d']:
            json_path = project_root / f"{dataset_name}.json"
            if json_path.exists():
                json_files[dataset_name] = str(json_path)
            else:
                print(f"[WARNING] JSON file not found: {json_path}")

        if not json_files:
            print("[ERROR] No JSON files found for dataset comparison!")
            print("  Expected files: scannet.json, scannetpp.json, matterport3d.json")
            print("  Generate them with: python tools/analyze_disk_usage.py --dataset <name> --grid-svd-stats --save-stats <name>.json")
            sys.exit(1)

        # Load data from all JSON files
        datasets_data = load_datasets_from_jsons(json_files, args.svd_rank)

        if not datasets_data:
            print("[ERROR] Failed to load data from any JSON file!")
            sys.exit(1)

        # Create comparison chart
        create_dataset_comparison_chart(
            datasets_data,
            svd_rank=args.svd_rank,
            save_path=args.save_plot,
            figsize=tuple(args.figsize) if args.figsize else (14, 7),
        )
        return

    # Handle --show-all-ranks mode
    if args.show_all_ranks:
        # Look for JSON files in project root
        project_root = Path(__file__).resolve().parent.parent
        json_files = {}
        for dataset_name in ['scannet', 'scannetpp', 'matterport3d']:
            json_path = project_root / f"{dataset_name}.json"
            if json_path.exists():
                json_files[dataset_name] = str(json_path)
            else:
                print(f"[WARNING] JSON file not found: {json_path}")

        if not json_files:
            print("[ERROR] No JSON files found for all-ranks comparison!")
            print("  Expected files: scannet.json, scannetpp.json, matterport3d.json")
            print("  Generate them with: python tools/analyze_disk_usage.py --dataset <name> --grid-svd-stats --save-stats <name>.json")
            sys.exit(1)

        # Create all-ranks comparison chart
        create_all_ranks_comparison_chart(
            json_files,
            ranks=[8, 16, 32],
            save_path=args.save_plot,
            figsize=tuple(args.figsize) if args.figsize else (14, 7),
        )
        return

    # Validate arguments
    if args.include_hf_lang_feat and not args.dataset:
        print("[ERROR] --include-hf-lang-feat requires --dataset to be specified")
        sys.exit(1)

    # Determine data root
    data_root = args.data_root
    if args.dataset:
        if args.data_root:
            print("[WARNING] Both --dataset and --data-root specified. Using --data-root.")
        else:
            # data_root will be set by scan_compression_data based on dataset config
            pass
    elif not args.data_root and not args.stats_json:
        # Default to gaussian_train if neither dataset nor data-root specified
        data_root = '/new_data/cyf/projects/SceneSplat/gaussian_train'

    # When using stats-json without data-root, we still need dataset config for HF data
    if args.stats_json and not data_root and not args.dataset:
        print("[ERROR] --stats-json requires --dataset (for HF lang_feat) or --data-root")
        print("  Usage: --stats-json scannetpp.json --dataset scannetpp --svd-rank 32 --include-hf-lang-feat")
        sys.exit(1)

    if data_root and not os.path.exists(data_root):
        print(f"[ERROR] Data root {data_root} does not exist!")
        sys.exit(1)

    # Run comparison
    if args.compare_ranks:
        # Multi-rank comparison
        create_multi_rank_comparison(
            dataset_name=args.dataset,
            data_root=data_root,
            split=args.split,
            ranks=args.compare_ranks,
            include_hf=args.include_hf_lang_feat,
            stats_json_path=args.stats_json,
            save_path=args.save_plot,
            figsize=tuple(args.figsize)
        )
    else:
        # Single rank comparison
        data = scan_compression_data(
            data_root=data_root,
            dataset_name=args.dataset,
            split=args.split,
            svd_rank=args.svd_rank,
            include_hf=args.include_hf_lang_feat,
            stats_json_path=args.stats_json
        )

        if not data:
            print("[ERROR] No data found!")
            if args.dataset:
                print(f"  Dataset: {args.dataset}, Split: {args.split}")
                if args.include_hf_lang_feat:
                    print(f"  Using HuggingFace for original lang_feat sizes")
                if args.stats_json:
                    print(f"  Using compressed stats from {args.stats_json}")
                else:
                    print("  Make sure compressed files exist locally:")
                    print(f"    lang_feat_grid_svd_r{args.svd_rank or '<rank>'}.npz")
            else:
                print("  Make sure your data root contains scene folders with:")
                print("    - lang_feat.npy (original)")
                print("    - lang_feat_grid_svd_r<rank>.npz (compressed)")
            sys.exit(1)

        create_comparison_bar_chart(
            data,
            svd_rank=args.svd_rank,
            save_path=args.save_plot,
            figsize=tuple(args.figsize)
        )


if __name__ == '__main__':
    main()
