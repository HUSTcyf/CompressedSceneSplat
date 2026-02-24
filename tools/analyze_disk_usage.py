#!/usr/bin/env python3
"""
Analyze disk usage of ScanNet/ScanNetPP/Matterport3D dataset.

This script scans the train, test, and val directories and provides
statistics on file types, their sizes, and percentage of total space.

Usage:
    # Basic analysis (ScanNet)
    python tools/analyze_disk_usage.py

    # Analyze ScanNetPP dataset
    python tools/analyze_disk_usage.py --dataset scannetpp

    # Analyze Matterport3D dataset
    python tools/analyze_disk_usage.py --dataset matterport3d

    # Show per-file-type breakdown
    python tools/analyze_disk_usage.py --detailed

    # Only analyze specific splits
    python tools/analyze_disk_usage.py --splits train test

    # Include HuggingFace lang_feat files
    python tools/analyze_disk_usage.py --include-hf-lang-feat

    # Merge splits and show combined statistics
    python tools/analyze_disk_usage.py --include-hf-lang-feat --merge-splits

    # Exclude specific file patterns
    python tools/analyze_disk_usage.py --exclude-files "grid_meta_data.json" "lang_feat_grid_svd_r*.npz"

    # Show detailed grid SVD file statistics
    python tools/analyze_disk_usage.py --grid-svd-stats

    # Show grid SVD stats for specific dataset
    python tools/analyze_disk_usage.py --dataset scannetpp --grid-svd-stats

    # Save visualization as image
    python tools/analyze_disk_usage.py --dataset scannetpp --merge-splits --save-plot scannetpp_disk_usage.png

    # Merge all datasets and save combined visualization
    python tools/analyze_disk_usage.py --merge-all-datasets --save-plot scenesplat.png
"""

import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse
import fnmatch
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Default paths
DATASET_CONFIGS = {
    "scannet": {
        "root": Path("/new_data/cyf/Datasets/SceneSplat7k/scannet"),
        "subdirs": {
            "train": "train_grid1.0cm_chunk6x6_stride3x3",
            "test": "test_grid1.0cm_chunk6x6_stride3x3",
            "val": None,
        },
        "default_splits": ["train", "test"],
        "hf_repo_id": "clapfor/scannet_mcmc_3dgs_lang_base",
        "hf_subfolder": "train_grid1.0cm_chunk6x6_stride3x3",
        "excluded_scenes": [],  # Scenes with missing/incomplete files to exclude
    },
    "scannetpp": {
        "root": Path("/new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2"),
        "subdirs": {
            "train": "train_grid1.0cm_chunk6x6_stride3x3",
            "test": "test_grid1.0cm_chunk6x6_stride3x3",
            "val": None,
        },
        "default_splits": ["train", "test"],
        "hf_repo_id": "clapfor/scannetpp_v2_mcmc_3dgs_lang_base",
        "hf_subfolder": "train_grid1.0cm_chunk6x6_stride3x3",
        # Exclude scenes with missing/incomplete core files
        "excluded_scenes": [
            "1b379f1114_0",  # missing: coord, color, opacity, scale
            "5c215ef3b0_1",  # missing: coord, color, opacity, scale
            "5d591ff74d_1",  # missing: quat, valid_feat_mask
            "b4b39438f0_7",  # missing: coord, color, opacity
            "ea42cd27e6_1",  # missing: quat, valid_feat_mask
        ],
    },
    "matterport3d": {
        "root": Path("/new_data/cyf/Datasets/SceneSplat7k/matterport3d"),
        "subdirs": {
            "train": "train_grid1.0cm_chunk6x6x4_stride4x4x4",
            "val": "val_grid1.0cm_chunk6x6x4_stride4x4x4",
            "test": None,  # Matterport3D doesn't have a test split
        },
        "default_splits": ["train", "val"],
        "hf_repo_id": "clapfor/matterport3d_scene_mcmc_3dgs_lang_base",
        "hf_subfolder": "train_grid1.0cm_chunk6x6x4_stride4x4x4",
        "excluded_scenes": [],
    },
}


def should_exclude_file(filename: str, exclude_patterns: Optional[List[str]]) -> bool:
    """
    Check if a file should be excluded based on patterns.

    Args:
        filename: Name of the file to check
        exclude_patterns: List of patterns (supports wildcards like *.npz)

    Returns:
        True if file should be excluded, False otherwise
    """
    if not exclude_patterns:
        return False

    for pattern in exclude_patterns:
        # Handle patterns with * wildcard
        if fnmatch.fnmatch(filename, pattern):
            return True

    return False


def format_size(size_bytes: int, base10: bool = False) -> str:
    """
    Format bytes to human readable size.

    Args:
        size_bytes: Size in bytes
        base10: If True, use base-10 units (TB=10^12), else base-2 (TiB=2^40)
    """
    if base10:
        # Base-10 (decimal) - like HuggingFace web: 1 KB = 1000 bytes
        for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
            if size_bytes < 1000.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1000.0
        return f"{size_bytes:.2f} EB"
    else:
        # Base-2 (binary) - traditional: 1 KiB = 1024 bytes
        for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} EiB"


def create_pie_chart(
    file_sizes: Dict[str, int],
    file_counts: Dict[str, int],
    total_size: int,
    dataset_name: str,
    split_name: str,
    output_path: Optional[Path] = None,
    base10: bool = False,
    bar_chart: bool = True,
    detailed_pie: bool = False,
    common_files: Optional[set] = None,
) -> None:
    """
    Create a pie chart visualization of file type distribution.
    Pie chart shows only lang_feat.npy vs others, bar chart shows detailed breakdown of others.

    Args:
        file_sizes: Dictionary mapping file names to sizes in bytes
        file_counts: Dictionary mapping file names to counts
        total_size: Total size in bytes
        dataset_name: Name of the dataset
        split_name: Name of the split (e.g., "MERGED", "TRAIN")
        output_path: Path to save the figure (if None, uses current directory)
        base10: If True, use base-10 units for display
        bar_chart: If True, include bar chart; otherwise pie chart only
        detailed_pie: If True, show detailed breakdown in pie chart with leader lines
        common_files: Set of files common to all splits (for merged splits, None means no special grouping)
    """
    if total_size == 0:
        print("[WARNING] No data to visualize")
        return

    # Separate lang_feat files from others (for pie chart)
    lang_feat_size = 0
    lang_feat_count = 0

    for file_name, size in file_sizes.items():
        if file_name.startswith("lang_feat"):
            lang_feat_size += size
            lang_feat_count += file_counts.get(file_name, 0)

    # Pie chart: lang_feat vs others (all other files)
    pie_sizes = [lang_feat_size, total_size - lang_feat_size]
    pie_labels = ["lang_feat.npy", "others"]
    pie_colors = ['#FF6B6B', '#4ECDC4']

    # Prepare data for bar chart
    # If common_files is provided, group non-common files into "other files"
    # Otherwise show all non-lang_feat files individually
    others_files = {}
    if common_files is not None:
        # Show common files individually, group non-common files as "other files"
        other_files_size = 0
        other_files_count = 0

        for file_name, size in file_sizes.items():
            if file_name.startswith("lang_feat"):
                continue
            if file_name in common_files:
                # Common file: show individually
                others_files[file_name] = size
            else:
                # Non-common file: group into "other files"
                count = file_counts.get(file_name, 0)
                other_files_size += size
                other_files_count += count

        # Add "other files" category if there are any non-common files
        if other_files_size > 0:
            others_files["other files (not in all splits)"] = other_files_size
    else:
        # Show all non-lang_feat files
        for file_name, size in file_sizes.items():
            if not file_name.startswith("lang_feat"):
                others_files[file_name] = size

    if bar_chart:
        # Prepare data for bar chart (detailed breakdown of others)
        bar_file_names = list(others_files.keys())
        bar_file_sizes = [others_files[name] for name in bar_file_names]

        # Sort bar data by size (descending)
        bar_data = sorted(zip(bar_file_names, bar_file_sizes), key=lambda x: x[1], reverse=True)
        bar_file_names, bar_file_sizes = zip(*bar_data) if bar_data else ([], [])

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Pie chart (only lang_feat vs others)
        wedges, texts, autotexts = ax1.pie(
            pie_sizes,
            labels=pie_labels,
            colors=pie_colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'},
            explode=(0.05, 0),  # Slightly explode lang_feat slice
        )

        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_weight('bold')

        # Set titles with consistent font size
        ax1.set_title(f'{dataset_name.upper()} - {split_name} SET\nFile Type Distribution',
                      fontsize=12, fontweight='bold', pad=20)
        ax2.set_title(f'Detailed Breakdown of "others" (excluding lang_feat.npy)',
                      fontsize=12, fontweight='bold', pad=20)

        # Bar chart for absolute sizes of others (excluding lang_feat)
        # Convert to appropriate units
        if total_size - lang_feat_size >= 1024**4:  # TiB/TB range
            unit_factor = 1024**4 if not base10 else 1000**4
            unit_label = "TiB" if not base10 else "TB"
        elif total_size - lang_feat_size >= 1024**3:  # GiB/GB range
            unit_factor = 1024**3 if not base10 else 1000**3
            unit_label = "GiB" if not base10 else "GB"
        else:
            unit_factor = 1024**2 if not base10 else 1000**2
            unit_label = "MiB" if not base10 else "MB"

        sizes_in_unit = [s / unit_factor for s in bar_file_sizes]

        # Define colors for bar chart
        color_palette = plt.cm.Set3.colors
        bar_colors = [color_palette[i % len(color_palette)] for i in range(len(bar_file_names))]

        bars = ax2.barh(range(len(bar_file_names)), sizes_in_unit, color=bar_colors)
        ax2.set_xlabel(f'Size ({unit_label})', fontsize=11, fontweight='bold')
        ax2.set_yticks(range(len(bar_file_names)))
        ax2.set_yticklabels(bar_file_names)
        ax2.invert_yaxis()

        # Add value labels on bars
        for i, (bar, size) in enumerate(zip(bars, sizes_in_unit)):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f' {size:.2f} {unit_label}',
                    ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()

        # Add total info at the bottom
        total_size_str = f"{format_size(total_size, base10=False)} ({format_size(total_size, base10=True)})"
        total_files = sum(file_counts.values())
        fig.text(0.5, 0.02, f'Total: {total_size_str} | Files: {total_files:,}',
                 ha='center', fontsize=12, fontweight='bold')
    else:
        # Pie chart only
        if detailed_pie:
            # Detailed pie chart with legend on the side
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))

            # Group file types for detailed breakdown
            # Priority order: lang_feat, coord, color, opacity, quat, scale, valid_feat_mask, others
            priority_files = {
                "lang_feat.npy": None,
                "coord.npy": None,
                "color.npy": None,
                "opacity.npy": None,
                "quat.npy": None,
                "scale.npy": None,
                "valid_feat_mask.npy": None,
            }

            others_size = 0
            for file_name, size in file_sizes.items():
                if file_name in priority_files:
                    priority_files[file_name] = size
                # elif file_name.startswith("lang_feat"):
                #     # Other lang_feat files (index, grid_svd, etc.) go into lang_feat
                #     if priority_files["lang_feat.npy"] is None:
                #         priority_files["lang_feat.npy"] = 0
                #     priority_files["lang_feat.npy"] += size
                elif file_name.startswith("lang_feat"):
                    # Other lang_feat files (index, grid_svd, etc.) go into others
                    # to keep lang_feat.npy showing its standalone percentage
                    others_size += size
                else:
                    others_size += size

            # Prepare data for pie chart
            pie_data = []
            pie_labels = []
            pie_colors_list = []

            # Color scheme
            color_map = {
                "lang_feat.npy": '#FF6B6B',
                "coord.npy": '#4ECDC4',
                "color.npy": '#45B7D1',
                "opacity.npy": '#FFA07A',
                "quat.npy": '#98D8C8',
                "scale.npy": '#F7DC6F',
                "valid_feat_mask.npy": '#BB8FCE',
                "others": '#D5D8DC',
            }

            # Add files in priority order (only include non-zero sizes)
            for file_name, size in priority_files.items():
                if size is not None and size > 0:
                    pie_data.append(size)
                    pie_labels.append(file_name)
                    pie_colors_list.append(color_map.get(file_name, '#95A5A6'))

            # Add others if non-zero
            if others_size > 0:
                pie_data.append(others_size)
                pie_labels.append("others")
                pie_colors_list.append(color_map["others"])

            # Sort by size (descending) but keep lang_feat first
            if len(pie_data) > 1:
                # Separate lang_feat and others
                if pie_labels[0] == "lang_feat.npy":
                    lang_feat_data = pie_data[0]
                    lang_feat_label = pie_labels[0]
                    lang_feat_color = pie_colors_list[0]
                    remaining_data = pie_data[1:]
                    remaining_labels = pie_labels[1:]
                    remaining_colors = pie_colors_list[1:]

                    # Sort remaining by size
                    sorted_remaining = sorted(zip(remaining_data, remaining_labels, remaining_colors),
                                             key=lambda x: x[0], reverse=True)
                    remaining_data, remaining_labels, remaining_colors = zip(*sorted_remaining) if sorted_remaining else ([], [], [])

                    pie_data = [lang_feat_data] + list(remaining_data)
                    pie_labels = [lang_feat_label] + list(remaining_labels)
                    pie_colors_list = [lang_feat_color] + list(remaining_colors)
                else:
                    # No lang_feat, sort all by size
                    sorted_all = sorted(zip(pie_data, pie_labels, pie_colors_list),
                                       key=lambda x: x[0], reverse=True)
                    pie_data, pie_labels, pie_colors_list = zip(*sorted_all) if sorted_all else ([], [], [])

            # Custom autopct function: only show label + percentage for lang_feat.npy and others
            def autopct_only_major(pct):
                # Find which slice this percentage belongs to
                for i, (label, size) in enumerate(zip(pie_labels, pie_data)):
                    label_pct = size / total_size * 100
                    if abs(pct - label_pct) < 0.1:  # Match the percentage to the label
                        if label in ["lang_feat.npy", "others"]:
                            # Show short label name with percentage
                            short_label = label.replace(".npy", "") if label.endswith(".npy") else label
                            return f'{short_label}\n{pct:.1f}%'
                        else:
                            return ''
                return ''

            # Create pie chart
            # Calculate explode: others slice gets emphasis (0.08)
            explode_list = []
            pcount = 0
            for label in pie_labels:
                if label == "others":
                    explode_list.append(0.08)
                else:
                    pcount += 1
                    explode_list.append(0.08 + 0.02 * pcount)  # Slightly increase explode for larger slices

            wedges, texts, autotexts = ax.pie(
                pie_data,
                labels=None,  # No labels directly on pie
                colors=pie_colors_list,
                autopct=autopct_only_major,
                startangle=90,
                pctdistance=0.85,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1},
                textprops={'fontsize': 14, 'weight': 'bold'},
                explode=explode_list,
            )

            # Make percentage text more readable (same as simple mode)
            for autotext in autotexts:
                if autotext.get_text():  # Only style non-empty text
                    autotext.set_color('black')
                    autotext.set_fontsize(16)
                    autotext.set_weight('bold')

            # Create legend with file type, size, and percentage (without lang_feat.npy and others)
            legend_labels = []
            legend_wedges = []
            for i, (label, size) in enumerate(zip(pie_labels, pie_data)):
                percentage = size / total_size * 100

                # Skip lang_feat.npy and others in legend (they're shown on pie)
                if label not in ["lang_feat.npy", "others"]:
                    # Format size for label
                    if size >= 1024**4:  # TiB/TB range
                        size_str = format_size(size, base10=False)
                    elif size >= 1024**3:  # GiB/GB range
                        size_str = format_size(size, base10=False)
                    else:
                        size_str = format_size(size, base10=False)

                    legend_labels.append(f"{label}: {percentage:.1f}% ({size_str})")
                    legend_wedges.append(wedges[i])

            # Add legend in top-right corner (further right to avoid overlap)
            if legend_labels:
                legend = ax.legend(
                    legend_wedges,
                    legend_labels,
                    title="File Types",
                    loc="upper right",
                    bbox_to_anchor=(1.2, 1),
                    fontsize=11,
                    title_fontsize=13,
                    frameon=True,
                )

            ax.set_title(f'{dataset_name.upper()} - {split_name} SET\nFile Type Distribution',
                        fontsize=16, fontweight='bold', pad=20)

            # Add total info to the right of legend area (moved down to avoid title overlap)
            total_size_str = f"{format_size(total_size, base10=False)} ({format_size(total_size, base10=True)})"
            total_files = sum(file_counts.values())
            fig.text(0.87, 0.92, f'Total: {total_size_str} | Files: {total_files:,}',
                    ha='right', va='top', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.subplots_adjust(top=0.88)  # Make room for legend on the right
        else:
            # Simple pie chart (lang_feat vs others)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            wedges, texts, autotexts = ax.pie(
                pie_sizes,
                labels=pie_labels,
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 14, 'weight': 'bold'},
                explode=(0.05, 0),
                pctdistance=0.85,
            )

            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(16)
                autotext.set_weight('bold')

            ax.set_title(f'{dataset_name.upper()} - {split_name} SET\nFile Type Distribution',
                        fontsize=16, fontweight='bold', pad=20)

            # Add total info at the bottom
            total_size_str = f"{format_size(total_size, base10=False)} ({format_size(total_size, base10=True)})"
            total_files = sum(file_counts.values())
            fig.text(0.5, 0.02, f'Total: {total_size_str} | Files: {total_files:,}',
                    ha='center', fontsize=14, fontweight='bold')

            plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = Path(f"{dataset_name}_{split_name.lower()}_disk_usage.png")

    # Use bbox_extra_artists to ensure footer is included
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    print(f"\n[INFO] Saved visualization to: {output_path}")

    plt.close()


def analyze_directory(
    root_dir: Path,
    exclude_patterns: Optional[List[str]] = None,
    excluded_scenes: Optional[List[str]] = None,
) -> Tuple[Dict[str, int], Dict[str, int], int]:
    """
    Analyze a directory for file sizes by type.

    Args:
        root_dir: Root directory to analyze
        exclude_patterns: Optional list of file patterns to exclude
        excluded_scenes: Optional list of scene names to exclude

    Returns:
        (file_sizes, file_counts, total_size)
        - file_sizes: {file_name: total_size}
        - file_counts: {file_name: count}
        - total_size: total size in bytes
    """
    file_sizes = defaultdict(int)
    file_counts = defaultdict(int)
    total_size = 0
    excluded_scene_set = set(excluded_scenes) if excluded_scenes else set()

    if not root_dir.exists():
        return file_sizes, file_counts, 0

    # Walk through all subdirectories (scene folders)
    for scene_dir in root_dir.iterdir():
        if not scene_dir.is_dir() or scene_dir.name.startswith('.'):
            continue

        # Skip excluded scenes
        if scene_dir.name in excluded_scene_set:
            continue

        # Process all files in the scene directory
        for file_path in scene_dir.iterdir():
            if file_path.is_file():
                file_name = file_path.name
                # Skip excluded files
                if should_exclude_file(file_name, exclude_patterns):
                    continue
                try:
                    size = file_path.stat().st_size
                    file_sizes[file_name] += size
                    file_counts[file_name] += 1
                    total_size += size
                except (OSError, FileNotFoundError) as e:
                    print(f"[WARNING] Could not read file: {file_path}: {e}")

    return file_sizes, file_counts, total_size


def get_huggingface_file_info(
    repo_id: str,
    subfolder: str,
) -> Tuple[Dict[str, int], Dict[str, int], int]:
    """
    Get file sizes from HuggingFace repository.

    Args:
        repo_id: HuggingFace repository ID
        subfolder: Subfolder path in the repository

    Returns:
        (file_sizes, file_counts, total_size)
        - file_sizes: {file_name: total_size}
        - file_counts: {file_name: count}
        - total_size: total size in bytes
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()

        # Get repo info with file sizes
        repo_info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)

        file_sizes = defaultdict(int)
        file_counts = defaultdict(int)
        total_size = 0

        # Filter files by subfolder
        prefix = subfolder.rstrip("/") + "/"
        for file_info in repo_info.siblings:
            if file_info.rfilename.startswith(prefix):
                file_name = Path(file_info.rfilename).name
                size = file_info.size if hasattr(file_info, 'size') else 0
                if size > 0:
                    file_sizes[file_name] += size
                    file_counts[file_name] += 1
                    total_size += size

        return dict(file_sizes), dict(file_counts), total_size

    except ImportError:
        print("[WARNING] huggingface_hub not installed. Cannot fetch HF file sizes.")
        print("  Install with: pip install huggingface_hub")
        return {}, {}, 0
    except Exception as e:
        print(f"[WARNING] Failed to fetch HuggingFace file info: {e}")
        return {}, {}, 0


def merge_hf_lang_feat(
    train_data: Dict,
    hf_data: Tuple[Dict[str, int], Dict[str, int], int],
) -> Tuple[Dict[str, int], Dict[str, int], int, Dict[str, str]]:
    """
    Merge HuggingFace lang_feat files into train data.

    Args:
        train_data: Train split data (with file_sizes, file_counts, total_size)
        hf_data: HuggingFace data (file_sizes, file_counts, total_size)

    Returns:
        (merged_file_sizes, merged_file_counts, merged_total_size, hf_only_files)
        - merged_file_sizes: Train file sizes with HF lang_feat added
        - merged_file_counts: Train file counts with HF lang_feat added
        - merged_total_size: Train total size with HF lang_feat added
        - hf_only_files: Files that only exist on HF (added to train)
    """
    hf_file_sizes, hf_file_counts, hf_total = hf_data
    if hf_total == 0:
        return train_data["file_sizes"], train_data["file_counts"], train_data["total_size"], {}

    # Copy train data
    merged_file_sizes = dict(train_data["file_sizes"])
    merged_file_counts = dict(train_data["file_counts"])
    hf_only_files = {}

    # Files to merge from HF (lang_feat files that don't exist in local train)
    lang_feat_files = ["lang_feat.npy", "lang_feat_index.npy"]

    for file_name in lang_feat_files:
        if file_name in hf_file_sizes and file_name not in merged_file_sizes:
            merged_file_sizes[file_name] = hf_file_sizes[file_name]
            merged_file_counts[file_name] = hf_file_counts[file_name]
            hf_only_files[file_name] = format_size(hf_file_sizes[file_name])

    # Calculate merged total size
    merged_total_size = sum(merged_file_sizes.values())

    return merged_file_sizes, merged_file_counts, merged_total_size, hf_only_files


def print_split_stats(
    split_name: str,
    file_sizes: Dict[str, int],
    file_counts: Dict[str, int],
    total_size: int,
    detailed: bool = False,
    show_base10: bool = True,
):
    """Print statistics for a single split."""
    if total_size == 0:
        print(f"\n{split_name.upper()}: No data found")
        return

    print(f"\n{'=' * 80}")
    print(f"{split_name.upper()} SET")
    print(f"{'=' * 80}")
    print(f"Total size: {format_size(total_size, base10=False)} ({format_size(total_size, base10=True)})")
    print(f"            ({total_size:,} bytes)")
    print(f"Total files: {sum(file_counts.values()):,}")

    if detailed:
        print(f"\n{'-' * 80}")
        print(f"{'File Name':<30} {'Count':>10} {'Size (GiB/TiB)':>18} {'Size (GB/TB)':>15} {'Percentage':>12}")
        print(f"{'-' * 80}")

        # Sort by size (descending)
        sorted_files = sorted(file_sizes.items(), key=lambda x: x[1], reverse=True)

        for file_name, size in sorted_files:
            count = file_counts[file_name]
            percentage = (size / total_size) * 100 if total_size > 0 else 0
            print(f"{file_name:<30} {count:>10} {format_size(size, base10=False):>18} {format_size(size, base10=True):>15} {percentage:>11.2f}%")
    else:
        # Group by file extension/base type
        print(f"\n{'-' * 80}")
        print(f"{'File Type':<30} {'Count':>10} {'Size (GiB/TiB)':>18} {'Size (GB/TB)':>15} {'Percentage':>12}")
        print(f"{'-' * 80}")

        # Group files by pattern
        groups = defaultdict(lambda: {"size": 0, "count": 0})

        for file_name, size in file_sizes.items():
            count = file_counts[file_name]

            # Group by base name pattern
            if file_name.startswith("lang_feat_grid_svd"):
                group_name = "lang_feat_grid_svd*.npz (SVD)"
            elif file_name == "grid_meta_data.json":
                group_name = "grid_meta_data.json"
            elif file_name.endswith(".npy"):
                base = file_name.replace(".npy", "")
                group_name = f"{base}*.npy"
            else:
                group_name = file_name

            groups[group_name]["size"] += size
            groups[group_name]["count"] += count

        # Sort by size (descending)
        sorted_groups = sorted(groups.items(), key=lambda x: x[1]["size"], reverse=True)

        for group_name, data in sorted_groups:
            size = data["size"]
            count = data["count"]
            percentage = (size / total_size) * 100 if total_size > 0 else 0
            print(f"{group_name:<30} {count:>10} {format_size(size, base10=False):>18} {format_size(size, base10=True):>15} {percentage:>11.2f}%")


def print_grid_svd_stats(
    split_name: str,
    file_sizes: Dict[str, int],
    file_counts: Dict[str, int],
    total_size: int,
) -> None:
    """
    Print detailed statistics for grid SVD related files only.

    Grid SVD related files include:
    - lang_feat_grid_svd*.npz (compressed features)
    - grid_meta_data.json (metadata)
    - lang_feat_index.npy (index file)

    Args:
        split_name: Name of the split (e.g., "train", "test")
        file_sizes: Dictionary mapping file names to sizes in bytes
        file_counts: Dictionary mapping file names to counts
        total_size: Total size of all files in the split
    """
    # Filter grid SVD related files
    grid_svd_files = {}
    grid_svd_total_size = 0
    grid_svd_total_count = 0

    # Patterns for grid SVD files
    for file_name, size in file_sizes.items():
        is_grid_svd = (
            file_name.startswith("lang_feat_grid_svd") or
            file_name == "grid_meta_data.json" or
            file_name == "lang_feat_index.npy"
        )
        if is_grid_svd:
            count = file_counts.get(file_name, 0)
            grid_svd_files[file_name] = {
                "size": size,
                "count": count,
            }
            grid_svd_total_size += size
            grid_svd_total_count += count

    if not grid_svd_files:
        print(f"\n[Grid SVD Stats] No grid SVD files found in {split_name.upper()} set")
        return

    print(f"\n{'=' * 80}")
    print(f"GRID SVD FILES - {split_name.upper()} SET")
    print(f"{'=' * 80}")

    # Summary
    percentage_of_total = (grid_svd_total_size / total_size * 100) if total_size > 0 else 0
    print(f"Grid SVD total size: {format_size(grid_svd_total_size, base10=False)} ({format_size(grid_svd_total_size, base10=True)})")
    print(f"                     ({grid_svd_total_size:,} bytes)")
    print(f"Grid SVD files: {grid_svd_total_count:,}")
    print(f"Percentage of total: {percentage_of_total:.2f}%")

    # Detailed breakdown
    print(f"\n{'-' * 80}")
    print(f"{'File Name':<40} {'Count':>10} {'Size (GiB/TiB)':>18} {'Size (GB/TB)':>15} {'% of SVD':>12}")
    print(f"{'-' * 80}")

    # Sort by size (descending)
    sorted_files = sorted(grid_svd_files.items(), key=lambda x: x[1]["size"], reverse=True)

    for file_name, data in sorted_files:
        size = data["size"]
        count = data["count"]
        percentage = (size / grid_svd_total_size * 100) if grid_svd_total_size > 0 else 0
        print(f"{file_name:<40} {count:>10} {format_size(size, base10=False):>18} {format_size(size, base10=True):>15} {percentage:>11.2f}%")

    print(f"{'-' * 80}")
    print(f"{'TOTAL':<40} {grid_svd_total_count:>10} {format_size(grid_svd_total_size, base10=False):>18} {format_size(grid_svd_total_size, base10=True):>15} {100.0:>11.2f}%")
    print(f"{'-' * 80}")

    # Group lang_feat_grid_svd by rank
    svd_npz_files = {k: v for k, v in grid_svd_files.items() if k.startswith("lang_feat_grid_svd") and k.endswith(".npz")}
    if svd_npz_files:
        print(f"\n{'-' * 80}")
        print(f"{'Rank':<10} {'File Count':>12} {'Total Size':>35} {'Avg Size':>25}")
        print(f"{'':>10} {'':>12} {'(GiB/TiB)':>18} {'(GB/TB)':>15} {'(MiB/GiB)':>13} {'(MB/GB)':>10}")
        print(f"{'-' * 80}")

        # Group by rank
        rank_groups = {}
        for file_name, data in svd_npz_files.items():
            # Extract rank from filename like lang_feat_grid_svd_r16.npz
            match = re.search(r'_r(\d+)\.npz$', file_name)
            if match:
                rank = int(match.group(1))
                if rank not in rank_groups:
                    rank_groups[rank] = {"size": 0, "count": 0}
                rank_groups[rank]["size"] += data["size"]
                rank_groups[rank]["count"] += data["count"]

        # Sort by rank (ascending)
        for rank in sorted(rank_groups.keys()):
            group = rank_groups[rank]
            total_size = group["size"]
            count = group["count"]
            avg_size = total_size / count if count > 0 else 0
            print(f"R{rank:<9} {count:>12} {format_size(total_size, base10=False):>18} {format_size(total_size, base10=True):>15} {format_size(avg_size, base10=False):>13} {format_size(avg_size, base10=True):>10}")

        print(f"{'-' * 80}")


def merge_all_datasets(args) -> int:
    """
    Merge all datasets (scannet, scannetpp, matterport3d) and show combined statistics.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    exclude_files = args.exclude_files if args.exclude_files else []
    use_hf = args.include_hf_lang_feat or args.include_hf_all

    # Collect data from all datasets
    all_datasets_data = {}
    total_size_all = 0
    total_files_all = 0

    for dataset_name in ["scannet", "scannetpp", "matterport3d"]:
        dataset_config = DATASET_CONFIGS[dataset_name]
        root_dir = dataset_config["root"]
        excluded_scenes = dataset_config.get("excluded_scenes", [])

        if not root_dir.exists():
            print(f"[WARNING] {dataset_name.upper()} root directory not found: {root_dir}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing {dataset_name.upper()}...")
        print(f"{'=' * 80}")

        if excluded_scenes:
            print(f"  Excluding {len(excluded_scenes)} scenes: {', '.join(excluded_scenes)}")

        # Collect all splits for this dataset
        dataset_file_sizes = defaultdict(int)
        dataset_file_counts = defaultdict(int)

        for split_name, subdir in dataset_config["subdirs"].items():
            if subdir is None:
                continue
            split_dir = root_dir / subdir

            if not split_dir.exists():
                print(f"[WARNING] {dataset_name.upper()} {split_name} directory not found: {split_dir}")
                continue

            file_sizes, file_counts, total_size = analyze_directory(split_dir, exclude_files, excluded_scenes)

            for fname, size in file_sizes.items():
                dataset_file_sizes[fname] += size
            for fname, count in file_counts.items():
                dataset_file_counts[fname] += count

            print(f"  {split_name}: {len(file_counts)} file types, {sum(file_counts.values()):,} files")

        # Include HuggingFace lang_feat files if requested
        # NOTE: For train set, use HF data; for test set, use local data
        if use_hf and "train" in dataset_config["subdirs"]:
            hf_repo_id = dataset_config["hf_repo_id"]
            hf_subfolder = dataset_config["hf_subfolder"]

            print(f"  Fetching HuggingFace data from {hf_repo_id}...")
            hf_data = get_huggingface_file_info(hf_repo_id, hf_subfolder)

            if hf_data[2] > 0:
                hf_file_sizes, hf_file_counts, hf_total = hf_data
                lang_feat_files = ["lang_feat.npy", "lang_feat_index.npy"]

                # If there are excluded scenes, filter them out from HF counts
                if excluded_scenes:
                    try:
                        # Recalculate HF counts, excluding specified scenes
                        filtered_hf_file_sizes = defaultdict(int)
                        filtered_hf_file_counts = defaultdict(int)

                        # Get repo info once with all file metadata
                        prefix = hf_subfolder.rstrip("/") + "/"
                        from huggingface_hub import HfApi
                        api = HfApi()
                        repo_info = api.repo_info(hf_repo_id, repo_type="dataset", files_metadata=True)

                        # Filter out excluded scenes directly from siblings
                        excluded_scenes_set = set(excluded_scenes)
                        for sibling in repo_info.siblings:
                            rfilename = sibling.rfilename
                            if not rfilename.startswith(prefix):
                                continue

                            # Extract scene name from path: prefix/scene_name/filename.npy
                            parts = rfilename[len(prefix):].split('/')
                            if len(parts) >= 2:
                                scene_name = parts[0]
                                # Skip excluded scenes
                                if scene_name in excluded_scenes_set:
                                    continue

                                file_name = parts[-1]
                                if file_name in lang_feat_files:
                                    size = sibling.size if hasattr(sibling, 'size') else 0
                                    filtered_hf_file_sizes[file_name] += size
                                    filtered_hf_file_counts[file_name] += 1

                        # Use filtered counts instead of original HF counts
                        for file_name in lang_feat_files:
                            if file_name in filtered_hf_file_counts:
                                original_count = hf_file_counts.get(file_name, 0)
                                filtered_count = filtered_hf_file_counts[file_name]
                                excluded_count = original_count - filtered_count

                                hf_file_counts[file_name] = filtered_count
                                hf_file_sizes[file_name] = filtered_hf_file_sizes[file_name]

                                if excluded_count > 0:
                                    print(f"    Excluded {excluded_count} {file_name} files from {len(excluded_scenes)} incomplete scenes")

                    except ImportError:
                        print(f"    [WARNING] huggingface_hub not available, cannot filter excluded scenes from HF data")
                    except Exception as e:
                        print(f"    [WARNING] Failed to filter excluded scenes from HF: {e}")

                # First, remove train split's lang_feat files from dataset totals (will be replaced with HF data)
                # We need to get the train split's lang_feat counts separately
                train_subdir = dataset_config["subdirs"].get("train")
                if train_subdir:
                    train_dir = root_dir / train_subdir
                    if train_dir.exists():
                        train_file_sizes, train_file_counts, _ = analyze_directory(train_dir, exclude_files, excluded_scenes)
                        for file_name in lang_feat_files:
                            if file_name in train_file_counts:
                                # Remove train's lang_feat files from dataset totals
                                dataset_file_sizes[file_name] -= train_file_sizes[file_name]
                                dataset_file_counts[file_name] -= train_file_counts[file_name]
                                print(f"    Removing {file_name} from train split: -{train_file_counts[file_name]} files")

                # Now add HF lang_feat files for train split
                for file_name in lang_feat_files:
                    if file_name in hf_file_sizes:
                        new_size = int(hf_file_sizes[file_name])
                        new_count = hf_file_counts[file_name]

                        dataset_file_sizes[file_name] += new_size
                        dataset_file_counts[file_name] += new_count

                        print(f"    {file_name} (train from HF): {format_size(new_size)} (+{new_count} files)")

        # Store dataset data
        dataset_total = sum(dataset_file_sizes.values())
        all_datasets_data[dataset_name] = {
            "file_sizes": dict(dataset_file_sizes),
            "file_counts": dict(dataset_file_counts),
            "total_size": dataset_total,
        }

        total_size_all += dataset_total
        total_files_all += sum(dataset_file_counts.values())

    # Merge all datasets together
    merged_file_sizes = defaultdict(int)
    merged_file_counts = defaultdict(int)

    for dataset_name, data in all_datasets_data.items():
        for file_name, size in data["file_sizes"].items():
            merged_file_sizes[file_name] += size
        for file_name, count in data["file_counts"].items():
            merged_file_counts[file_name] += count

    # Print summary
    print(f"\n{'=' * 80}")
    print("SCENESPLAT - ALL DATASETS MERGED")
    print(f"{'=' * 80}")

    print(f"\nDatasets: {', '.join(all_datasets_data.keys()).upper()}")
    print(f"Excluded patterns: {', '.join(exclude_files) if exclude_files else 'None'}")
    print(f"Include HuggingFace: {use_hf}")
    print()

    print(f"\n{'-' * 80}")
    print(f"{'Dataset':<15} {'Size (GiB/TiB)':>20} {'Size (GB/TB)':>15} {'Percentage':>12} {'Files':>12}")
    print(f"{'-' * 80}")

    for dataset_name, data in all_datasets_data.items():
        total_size = data["total_size"]
        file_count = sum(data["file_counts"].values())
        percentage = (total_size / total_size_all * 100) if total_size_all > 0 else 0

        print(f"{dataset_name.upper():<15} {format_size(total_size, base10=False):>20} {format_size(total_size, base10=True):>15} {percentage:>11.2f}% {file_count:>12,}")

    print(f"{'-' * 80}")
    print(f"{'TOTAL':<15} {format_size(total_size_all, base10=False):>20} {format_size(total_size_all, base10=True):>15} {100.0:>11.2f}% {total_files_all:>12,}")
    print(f"{'-' * 80}")

    # Print detailed file type breakdown
    print(f"\n{'=' * 80}")
    print("MERGED FILE TYPE BREAKDOWN")
    print(f"{'=' * 80}")
    print(f"Total size: {format_size(total_size_all, base10=False)} ({format_size(total_size_all, base10=True)})")
    print(f"Total files: {total_files_all:,}")

    print(f"\n{'-' * 80}")
    print(f"{'File Type':<30} {'Count':>12} {'Size (GiB/TiB)':>20} {'Size (GB/TB)':>15} {'Percentage':>12}")
    print(f"{'-' * 80}")

    # Sort by size (descending)
    sorted_files = sorted(merged_file_sizes.items(), key=lambda x: x[1], reverse=True)

    for file_name, size in sorted_files:
        count = merged_file_counts[file_name]
        percentage = (size / total_size_all * 100) if total_size_all > 0 else 0
        print(f"{file_name:<30} {count:>12,} {format_size(size, base10=False):>20} {format_size(size, base10=True):>15} {percentage:>11.2f}%")

    print(f"{'-' * 80}")

    # Print grid SVD statistics if requested
    if args.grid_svd_stats:
        # Print per-dataset grid SVD stats
        for dataset_name, data in all_datasets_data.items():
            print_grid_svd_stats(
                dataset_name,
                data["file_sizes"],
                data["file_counts"],
                data["total_size"],
            )

        # Print merged grid SVD stats
        print_grid_svd_stats(
            "all_datasets_merged",
            merged_file_sizes,
            merged_file_counts,
            total_size_all,
        )

    # Create visualization if requested
    if args.save_plot:
        output_path = Path(args.save_plot)
        create_pie_chart(
            merged_file_sizes,
            merged_file_counts,
            total_size_all,
            "scenesplat",
            "all_datasets",
            output_path,
            bar_chart=False,  # Pie chart only
            detailed_pie=True,  # Show detailed breakdown with leader lines
        )

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Analyze disk usage of ScanNet/ScanNetPP/Matterport3D dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["scannet", "scannetpp", "matterport3d"],
        default="scannet",
        help="Dataset to analyze (default: scannet)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to dataset root (auto-detected based on --dataset if not specified)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        choices=["train", "test", "val"],
        default=None,
        help="Splits to analyze (default: dataset-specific)",
    )
    parser.add_argument(
        "--subdir-train",
        type=str,
        default=None,
        help="Train subdirectory name (auto-detected based on dataset if not specified)",
    )
    parser.add_argument(
        "--subdir-test",
        type=str,
        default=None,
        help="Test subdirectory name (auto-detected based on dataset if not specified)",
    )
    parser.add_argument(
        "--subdir-val",
        type=str,
        default=None,
        help="Val subdirectory name (auto-detected based on dataset if not specified)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show per-file breakdown instead of grouped by type",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary table, no per-split details",
    )
    parser.add_argument(
        "--include-hf-lang-feat",
        action="store_true",
        help="Include lang_feat.npy and lang_feat_index.npy from HuggingFace for train set",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="HuggingFace repository ID (auto-detected based on dataset if not specified)",
    )
    parser.add_argument(
        "--hf-subfolder",
        type=str,
        default=None,
        help="HuggingFace subfolder for train lang_feat files",
    )
    parser.add_argument(
        "--merge-splits",
        action="store_true",
        help="Merge train and test sets together for combined statistics",
    )
    parser.add_argument(
        "--exclude-files",
        type=str,
        nargs="+",
        default=None,
        help="File patterns to exclude (e.g., grid_meta_data.json lang_feat_grid_svd_r*.npz)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Save visualization as image (specify output path, e.g., output.png)",
    )
    parser.add_argument(
        "--merge-all-datasets",
        action="store_true",
        help="Merge all datasets (scannet, scannetpp, matterport3d) and show combined statistics",
    )
    parser.add_argument(
        "--include-hf-all",
        action="store_true",
        help="Include HuggingFace lang_feat files for all datasets when using --merge-all-datasets",
    )
    parser.add_argument(
        "--grid-svd-stats",
        action="store_true",
        help="Show detailed statistics for grid SVD related files only (lang_feat_grid_svd*.npz, grid_meta_data.json, lang_feat_index.npy)",
    )

    args = parser.parse_args()

    # Handle --merge-all-datasets option
    if args.merge_all_datasets:
        return merge_all_datasets(args)

    # Get dataset configuration
    dataset_config = DATASET_CONFIGS[args.dataset]
    excluded_scenes = dataset_config.get("excluded_scenes", [])

    # Set root directory
    root_dir = Path(args.root) if args.root else dataset_config["root"]

    # Set default splits if not specified
    if args.splits is None:
        args.splits = dataset_config["default_splits"]

    # Set default subdirectories if not specified
    subdir_mapping = {}
    for split in ["train", "test", "val"]:
        if args.subdir_train and split == "train":
            subdir_mapping[split] = args.subdir_train
        elif args.subdir_test and split == "test":
            subdir_mapping[split] = args.subdir_test
        elif args.subdir_val and split == "val":
            subdir_mapping[split] = args.subdir_val
        elif dataset_config["subdirs"].get(split):
            subdir_mapping[split] = dataset_config["subdirs"][split]

    # Set default HF parameters if not specified
    hf_repo_id = args.hf_repo_id if args.hf_repo_id else dataset_config["hf_repo_id"]
    hf_subfolder = args.hf_subfolder if args.hf_subfolder else dataset_config["hf_subfolder"]

    if not root_dir.exists():
        print(f"[ERROR] Root directory not found: {root_dir}")
        return 1

    # Print excluded scenes info
    if excluded_scenes:
        print(f"Excluding {len(excluded_scenes)} scenes: {', '.join(excluded_scenes)}")

    # Collect data for all splits
    all_data = {}
    total_all_splits = 0

    for split in args.splits:
        if split not in subdir_mapping:
            print(f"[WARNING] Split '{split}' not found in dataset configuration, skipping...")
            continue
        subdir = subdir_mapping[split]
        split_dir = root_dir / subdir

        file_sizes, file_counts, total_size = analyze_directory(split_dir, args.exclude_files, excluded_scenes)

        all_data[split] = {
            "dir": split_dir,
            "file_sizes": file_sizes,
            "file_counts": file_counts,
            "total_size": total_size,
        }
        total_all_splits += total_size

    # Include HuggingFace lang_feat files for train set
    hf_only_files = {}
    if args.include_hf_lang_feat and "train" in all_data:
        print(f"\nFetching HuggingFace file info from {hf_repo_id}...")
        print(f"Subfolder: {hf_subfolder}")
        hf_data = get_huggingface_file_info(hf_repo_id, hf_subfolder)

        if hf_data[2] > 0:  # If HF data has content
            hf_file_sizes, hf_file_counts, hf_total = hf_data
            print(f"Found {hf_total} bytes in {sum(hf_file_counts.values())} files on HuggingFace")

            # Filter out excluded scenes from HF counts
            if excluded_scenes:
                try:
                    from huggingface_hub import HfApi

                    # Recalculate HF counts, excluding specified scenes
                    filtered_hf_file_sizes = defaultdict(int)
                    filtered_hf_file_counts = defaultdict(int)

                    # Get repo info once with all file metadata
                    prefix = hf_subfolder.rstrip("/") + "/"
                    api = HfApi()
                    repo_info = api.repo_info(hf_repo_id, repo_type="dataset", files_metadata=True)

                    # Filter out excluded scenes directly from siblings
                    lang_feat_files = ["lang_feat.npy", "lang_feat_index.npy"]
                    excluded_scenes_set = set(excluded_scenes)

                    for sibling in repo_info.siblings:
                        rfilename = sibling.rfilename
                        if not rfilename.startswith(prefix):
                            continue

                        # Extract scene name from path: prefix/scene_name/filename.npy
                        parts = rfilename[len(prefix):].split('/')
                        if len(parts) >= 2:
                            scene_name = parts[0]
                            # Skip excluded scenes
                            if scene_name in excluded_scenes_set:
                                continue

                            file_name = parts[-1]
                            if file_name in lang_feat_files:
                                size = sibling.size if hasattr(sibling, 'size') else 0
                                filtered_hf_file_sizes[file_name] += size
                                filtered_hf_file_counts[file_name] += 1

                    # Use filtered counts instead of original HF counts
                    for file_name in lang_feat_files:
                        if file_name in filtered_hf_file_counts:
                            original_count = hf_file_counts.get(file_name, 0)
                            filtered_count = filtered_hf_file_counts[file_name]
                            excluded_count = original_count - filtered_count

                            hf_file_counts[file_name] = filtered_count
                            hf_file_sizes[file_name] = filtered_hf_file_sizes[file_name]

                            if excluded_count > 0:
                                print(f"Excluded {excluded_count} {file_name} files from {len(excluded_scenes)} incomplete scenes")

                except ImportError:
                    print(f"[WARNING] huggingface_hub not available, cannot filter excluded scenes from HF data")
                except Exception as e:
                    print(f"[WARNING] Failed to filter excluded scenes from HF: {e}")

            # Replace train data with HF data for lang_feat files
            lang_feat_files = ["lang_feat.npy", "lang_feat_index.npy"]

            for file_name in lang_feat_files:
                if file_name in hf_file_sizes:
                    # Calculate size difference
                    old_size = all_data["train"]["file_sizes"].get(file_name, 0)
                    old_count = all_data["train"]["file_counts"].get(file_name, 0)
                    new_size = int(hf_file_sizes[file_name])
                    new_count = hf_file_counts[file_name]

                    # Replace with HF data
                    all_data["train"]["file_sizes"][file_name] = new_size
                    all_data["train"]["file_counts"][file_name] = new_count

                    # Update total
                    total_all_splits += (new_size - old_size)

                    if new_count != old_count:
                        hf_only_files[file_name] = f"{format_size(new_size)} ({old_count} -> {new_count} files)"

            # Update train total
            all_data["train"]["total_size"] = sum(all_data["train"]["file_sizes"].values())

            print(f"Updated {len(hf_only_files)} file types from HuggingFace for train set:")
            for fname, finfo in hf_only_files.items():
                print(f"  - {fname}: {finfo}")
        else:
            print("[WARNING] No lang_feat files found on HuggingFace or failed to fetch")

    # Merge splits if requested
    if args.merge_splits and len(all_data) > 1:
        # Store original splits before modifying
        original_splits = list(all_data.keys())

        # Find files that exist in ALL splits (intersection)
        # These will be the main categories shown in the pie chart
        common_files = set(all_data[original_splits[0]]["file_sizes"].keys())
        for split in original_splits[1:]:
            common_files &= set(all_data[split]["file_sizes"].keys())

        # Also filter out excluded files from common files
        if args.exclude_files:
            excluded_from_common = {f for f in common_files if should_exclude_file(f, args.exclude_files)}
            common_files -= excluded_from_common

        print(f"\n[INFO] Files common to all splits: {len(common_files)}")

        # Show different types of excluded files
        all_train_files = set(all_data[original_splits[0]]["file_sizes"].keys())
        not_in_all = all_train_files - common_files
        if not_in_all:
            print(f"[INFO] Files not in all splits (will be grouped as 'others'): {sorted(not_in_all)}")
        if args.exclude_files:
            excluded_matches = {f for f in all_train_files if should_exclude_file(f, args.exclude_files)}
            if excluded_matches:
                print(f"[INFO] Excluded by pattern: {sorted(excluded_matches)}")

        # Combine all splits into one, including ALL files (not just common)
        # We'll mark which files are common vs others for the pie chart
        merged_file_sizes = defaultdict(int)
        merged_file_counts = defaultdict(int)

        for split_data in all_data.values():
            for file_name, size in split_data["file_sizes"].items():
                merged_file_sizes[file_name] += size
            for file_name, count in split_data["file_counts"].items():
                merged_file_counts[file_name] += count

        # Create a "merged" entry with metadata about common files
        all_data["merged"] = {
            "dir": None,
            "file_sizes": dict(merged_file_sizes),
            "file_counts": dict(merged_file_counts),
            "total_size": sum(merged_file_sizes.values()),
            "common_files": common_files,  # Store for pie chart rendering
        }

        # Update total to reflect all files (not just common)
        total_all_splits = all_data["merged"]["total_size"]

        # Update splits to only show merged
        args.splits = ["merged"]

    # Print summary table
    print("=" * 80)
    print(f"{args.dataset.upper()} DISK USAGE ANALYSIS")
    print("=" * 80)
    print(f"Root directory: {root_dir}")
    if args.merge_splits:
        print(f"Splits analyzed: {', '.join(original_splits)} (MERGED)")
    else:
        print(f"Splits analyzed: {', '.join(args.splits)}")

    if args.exclude_files:
        print(f"Excluded patterns: {', '.join(args.exclude_files)}")
    if args.include_hf_lang_feat:
        print(f"Note: Train set includes lang_feat files from HuggingFace ({hf_repo_id})")
    print()

    print(f"\n{'-' * 80}")
    print(f"{'Split':<15} {'Size (GiB/TiB)':>20} {'Size (GB/TB)':>15} {'Percentage':>12} {'Files':>12}")
    print(f"{'-' * 80}")

    for split in args.splits:
        data = all_data[split]
        total_size = data["total_size"]
        file_count = sum(data["file_counts"].values())
        percentage = (total_size / total_all_splits * 100) if total_all_splits > 0 else 0

        print(f"{split.upper():<15} {format_size(total_size, base10=False):>20} {format_size(total_size, base10=True):>15} {percentage:>11.2f}% {file_count:>12,}")

    print(f"{'-' * 80}")
    # Calculate total files only from displayed splits
    total_files = sum(sum(all_data[s]['file_counts'].values()) for s in args.splits)
    print(f"{'TOTAL':<15} {format_size(total_all_splits, base10=False):>20} {format_size(total_all_splits, base10=True):>15} {100.0:>11.2f}% {total_files:>12,}")
    print(f"{'-' * 80}")

    # Print detailed per-split information
    if not args.summary_only:
        for split in args.splits:
            data = all_data[split]
            print_split_stats(
                split,
                data["file_sizes"],
                data["file_counts"],
                data["total_size"],
                detailed=args.detailed,
            )

    # Print grid SVD statistics if requested
    if args.grid_svd_stats:
        for split in args.splits:
            data = all_data[split]
            print_grid_svd_stats(
                split,
                data["file_sizes"],
                data["file_counts"],
                data["total_size"],
            )

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

    # Create visualization if requested
    if args.save_plot:
        # Only visualize merged data or the single split
        splits_to_visualize = ["merged"] if args.merge_splits else args.splits

        for split in splits_to_visualize:
            if split not in all_data:
                continue
            data = all_data[split]
            split_name = split.upper()
            output_path = Path(args.save_plot)

            # Get common_files if available (for merged splits)
            common_files = data.get("common_files", None)

            create_pie_chart(
                data["file_sizes"],
                data["file_counts"],
                data["total_size"],
                args.dataset,
                split_name,
                output_path,
                common_files=common_files,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
