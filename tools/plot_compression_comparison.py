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

    # Compare multiple ranks
    python tools/plot_compression_comparison.py --dataset scannetpp --compare-ranks 64 128 256

    # Load compressed stats from JSON file (when compressed files are not local)
    python tools/plot_compression_comparison.py --stats-json scannetpp.json --svd-rank 32

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

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
        "hf_subfolder": "train_grid1.0cm_chunk6x6_stride3x3",
        "excluded_scenes": [],
    },
    "scannetpp": {
        "root": Path("/new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2"),
        "subdirs": {
            "train": "train_grid1.0cm_chunk6x6_stride3x3",
            "test": "test_grid1.0cm_chunk6x6_stride3x3",
        },
        "hf_repo_id": "clapfor/scannetpp_v2_mcmc_3dgs_lang_base",
        "hf_subfolder": "train_grid1.0cm_chunk6x6_stride3x3",
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
        "hf_subfolder": "train_grid1.0cm_chunk6x6x4_stride4x4x4",
        "excluded_scenes": [],
    },
}


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0


def get_huggingface_file_info(
    repo_id: str,
    subfolder: str,
) -> Dict[str, float]:
    """
    Get file sizes from HuggingFace repository.

    Args:
        repo_id: HuggingFace repository ID
        subfolder: Subfolder path in the repository

    Returns:
        Dictionary mapping file names to sizes in MB
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()

        # Get repo info with file sizes
        repo_info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)

        file_sizes = defaultdict(float)

        # Filter files by subfolder
        prefix = subfolder.rstrip("/") + "/"
        for file_info in repo_info.siblings:
            if file_info.rfilename.startswith(prefix):
                # Extract scene name and file name
                rel_path = file_info.rfilename[len(prefix):]
                parts = rel_path.split("/")
                if len(parts) >= 2:
                    scene_name = parts[0]
                    file_name = parts[1]
                    size = file_info.size if hasattr(file_info, 'size') else 0
                    if size > 0:
                        # Store as scene_name/file_name -> size in MB
                        key = f"{scene_name}/{file_name}"
                        file_sizes[key] = size / (1024 * 1024)

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
) -> float:
    """
    Get compressed size from JSON stats data.

    Args:
        stats_data: JSON stats data from analyze_disk_usage.py
        svd_rank: SVD rank to get size for
        file_count: Number of files (to calculate average if needed)

    Returns:
        Compressed size in MB (average per file)
    """
    if not stats_data or "files" not in stats_data:
        return 0.0

    file_name = f"lang_feat_grid_svd_r{svd_rank}.npz"
    if file_name in stats_data["files"]:
        file_data = stats_data["files"][file_name]
        count = file_data.get("count", 1)
        total_mb = file_data.get("size_mb", 0)
        # Return average size per file
        return total_mb / count if count > 0 else total_mb

    return 0.0


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
        Dictionary mapping scene names to {'original': size_mb, 'compressed': size_mb}
    """
    results = {}
    hf_lang_feat_sizes = {}
    stats_json_data = None
    compressed_size_from_json = 0.0

    # Load compressed size from JSON if provided
    if stats_json_path:
        stats_json_data = load_stats_from_json(stats_json_path)
        if stats_json_data and svd_rank:
            compressed_size_from_json = get_compressed_size_from_json(stats_json_data, svd_rank)
            if compressed_size_from_json > 0:
                print(f"Using compressed size from JSON: {compressed_size_from_json:.2f} MB per file (rank={svd_rank})")

    # Determine data root and get HF data if needed
    if dataset_name and dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        if split not in config["subdirs"] or config["subdirs"][split] is None:
            print(f"[WARNING] Split '{split}' not available for dataset '{dataset_name}'")
            return {}

        data_root = str(config["root"] / config["subdirs"][split])
        excluded_scenes = config.get("excluded_scenes", [])

        # Fetch HF lang_feat sizes if requested
        if include_hf:
            hf_repo_id = config["hf_repo_id"]
            hf_subfolder = config["hf_subfolder"]
            hf_data = get_huggingface_file_info(hf_repo_id, hf_subfolder)
            # Extract only lang_feat.npy files
            for key, size in hf_data.items():
                if "/lang_feat.npy" in key:
                    scene_name = key.split("/")[0]
                    hf_lang_feat_sizes[scene_name] = size

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

        # Get compressed size (from JSON or local file)
        compressed_size = 0.0
        if compressed_size_from_json > 0:
            # Use average size from JSON
            compressed_size = compressed_size_from_json
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

            compressed_size = get_file_size_mb(str(compressed_path))

        # Get original size (prefer HF if available and include_hf is True)
        original_size = 0.0
        if include_hf and scene_name in hf_lang_feat_sizes:
            original_size = hf_lang_feat_sizes[scene_name]
        else:
            original_path = scene_dir / "lang_feat.npy"
            if original_path.exists():
                original_size = get_file_size_mb(str(original_path))
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
            # Get original lang_feat size from stats if available
            original_total_mb = 0
            for fname, fdata in stats_json_data["files"].items():
                if "lang_feat" in fname and "grid" not in fname and "index" not in fname:
                    original_total_mb += fdata.get("size_mb", 0)

            if original_total_mb > 0:
                # Use total sizes from JSON
                count = file_count or 1
                original_avg = original_total_mb / count
                results["aggregate"] = {
                    'original': original_avg,
                    'compressed': compressed_size_from_json,
                    'from_hf': False
                }
                print(f"Created aggregate entry using JSON stats (original: {original_avg:.2f} MB, compressed: {compressed_size_from_json:.2f} MB)")

    return results


def create_comparison_bar_chart(
    data: Dict[str, Dict[str, float]],
    svd_rank: int = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 7)
):
    """
    Create a bar chart comparing original vs compressed sizes.

    Style based on reference publication:
    - Side-by-side bars for each scene
    - Blue for original, orange for compressed
    - Grid lines for readability
    - Large, clear labels
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

    # Colors matching reference style (blue and orange)
    color_original = '#3274A1'  # Blue
    color_compressed = '#E1812C'  # Orange

    # Create bars
    bars1 = ax.bar(x - width/2, original_sizes, width,
                   label='Original', color=color_original, alpha=0.9)
    bars2 = ax.bar(x + width/2, compressed_sizes, width,
                   label=f'Compressed (SVD rank={svd_rank})' if svd_rank else 'Compressed',
                   color=color_compressed, alpha=0.9)

    # Reference style: grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    # Labels and title (large, bold fonts like reference)
    ax.set_xlabel('Scenes', fontsize=18, fontweight='bold')
    ax.set_ylabel('Storage Size (MB)', fontsize=18, fontweight='bold')

    rank_text = f" (SVD rank={svd_rank})" if svd_rank else ""
    ax.set_title(f'Language Feature Storage Comparison{rank_text}',
                 fontsize=22, fontweight='bold', pad=20)

    # X-axis labels with scene names
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha='right', fontsize=12)

    # Y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0f}'))

    # Legend (top right, like reference)
    ax.legend(fontsize=14, frameon=True, shadow=False,
              loc='upper right', fancybox=False)

    # Add value labels on bars (for clarity)
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Add summary statistics as text box
    stats_text = f'Total Original: {original_total:.1f} MB\n'
    stats_text += f'Total Compressed: {compressed_total:.1f} MB\n'
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
    print(f"Scenes analyzed: {len(scenes)}")
    print(f"Total original size:     {original_total:10.1f} MB")
    print(f"Total compressed size:   {compressed_total:10.1f} MB")
    print(f"Compression ratio:       {compression_ratio:10.2f}x")
    print(f"Space saved:             {space_saved:10.1f} MB ({space_saved_pct:.1f}%)")
    print("="*60)


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
    ax1.set_ylabel('Storage Size (MB)', fontsize=16, fontweight='bold')
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
  python tools/plot_compression_comparison.py --stats-json scannetpp.json --svd-rank 32 --include-hf-lang-feat

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

    # Stats JSON option
    parser.add_argument(
        '--stats-json',
        type=str,
        default=None,
        help='Path to JSON file with compressed stats (from analyze_disk_usage.py --save-stats)'
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
