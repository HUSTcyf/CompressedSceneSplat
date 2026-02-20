#!/usr/bin/env python3
"""
Copy or move SVD files and metadata from grid_svd_output to corresponding dataset scene directories.

This script copies or moves:
- lang_feat_grid_svd_r*.npz files (all ranks)
- grid_meta_data.json file

From: /new_data/cyf/projects/SceneSplat/grid_svd_output/{scene_name}/
To:   /new_data/cyf/projects/SceneSplat/gaussian_train/{dataset}/train/{scene_name}/

Scene-to-dataset mapping:
- 3DOVS: bed, bench, lawn, room, sofa
- lerf_ovs: figurines, ramen, teatime, waldo_kitchen

Usage:
    # Dry run (show what would be copied without actually copying)
    python tools/copy_grid_svd_to_datasets.py --dry-run

    # Copy all SVD files
    python tools/copy_grid_svd_to_datasets.py

    # Move all SVD files (removes source files after copying)
    python tools/copy_grid_svd_to_datasets.py --move

    # Copy only specific rank
    python tools/copy_grid_svd_to_datasets.py --rank 16

    # Copy specific scenes
    python tools/copy_grid_svd_to_datasets.py --scenes bed figurines

    # Verbose output
    python tools/copy_grid_svd_to_datasets.py --verbose
"""

import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Scene to dataset mapping
SCENE_DATASET_MAPPING = {
    # 3DOVS scenes
    "bed": "3DOVS",
    "bench": "3DOVS",
    "lawn": "3DOVS",
    "room": "3DOVS",
    "sofa": "3DOVS",
    # lerf_ovs scenes
    "figurines": "lerf_ovs",
    "ramen": "lerf_ovs",
    "teatime": "lerf_ovs",
    "waldo_kitchen": "lerf_ovs",
}

# Reverse mapping for reference
DATASET_SCENES = {
    "3DOVS": ["bed", "bench", "lawn", "room", "sofa"],
    "lerf_ovs": ["figurines", "ramen", "teatime", "waldo_kitchen"],
}


def get_svd_files(scene_dir: Path) -> List[Path]:
    """Get all SVD files in a scene directory."""
    svd_files = list(scene_dir.glob("lang_feat_grid_svd_r*.npz"))
    return sorted(svd_files, key=lambda x: int(x.stem.split('r')[-1]))


def get_meta_file(scene_dir: Path) -> Optional[Path]:
    """Get the metadata file in a scene directory."""
    meta_file = scene_dir / "grid_meta_data.json"
    return meta_file if meta_file.exists() else None


def transfer_file(
    src: Path,
    dst: Path,
    move: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> bool:
    """Copy or move a single file from src to dst."""
    action = "move" if move else "copy"

    if dry_run:
        print(f"[DRY RUN] Would {action}: {src.name} -> {dst}")
        return True

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)
        if verbose:
            print(f"{'Moved' if move else 'Copied'}: {src.name} -> {dst}")
        return True
    except Exception as e:
        print(f"Error {action}ing {src} to {dst}: {e}")
        return False


def transfer_scene_svd_files(
    scene_name: str,
    dataset: str,
    grid_svd_root: Path,
    data_root: Path,
    rank_filter: Optional[int] = None,
    move: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Copy or move SVD files for a single scene.

    Returns:
        (num_transferred, num_skipped)
    """
    src_dir = grid_svd_root / scene_name
    dst_dir = data_root / dataset / "train" / scene_name

    # Check if source directory exists
    if not src_dir.exists():
        print(f"[WARNING] Source directory not found: {src_dir}")
        return 0, 0

    # Check if destination directory exists
    if not dst_dir.exists():
        print(f"[WARNING] Destination directory not found: {dst_dir}")
        return 0, 0

    num_transferred = 0
    num_skipped = 0

    # Transfer SVD files
    svd_files = get_svd_files(src_dir)
    if rank_filter is not None:
        svd_files = [f for f in svd_files if f"lang_feat_grid_svd_r{rank_filter}." in f.name]

    for svd_file in svd_files:
        dst_file = dst_dir / svd_file.name
        if dst_file.exists():
            if verbose:
                print(f"[SKIP] Already exists: {dst_file}")
            num_skipped += 1
        elif transfer_file(svd_file, dst_file, move=move, dry_run=dry_run, verbose=verbose):
            num_transferred += 1
        else:
            num_skipped += 1

    # Transfer metadata file
    meta_file = get_meta_file(src_dir)
    if meta_file is not None:
        dst_meta = dst_dir / meta_file.name
        if dst_meta.exists():
            if verbose:
                print(f"[SKIP] Metadata already exists: {dst_meta}")
        elif transfer_file(meta_file, dst_meta, move=move, dry_run=dry_run, verbose=verbose):
            num_transferred += 1

    return num_transferred, num_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Copy or move SVD files from grid_svd_output to dataset directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--grid-svd-root",
        type=str,
        default="/new_data/cyf/projects/SceneSplat/grid_svd_output",
        help="Path to grid_svd_output directory",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/new_data/cyf/projects/SceneSplat/gaussian_train",
        help="Path to gaussian_train directory",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Only copy SVD files with specific rank (e.g., 8, 16, 32)",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        default=None,
        help="Specific scenes to copy (default: all scenes)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        choices=["3DOVS", "lerf_ovs"],
        help="Specific datasets to process (default: all)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (removes source files after transfer)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied/moved without actually doing it",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    grid_svd_root = Path(args.grid_svd_root)
    data_root = Path(args.data_root)

    # Validate directories
    if not grid_svd_root.exists():
        print(f"[ERROR] grid_svd_root not found: {grid_svd_root}")
        return 1

    if not data_root.exists():
        print(f"[ERROR] data_root not found: {data_root}")
        return 1

    # Determine which scenes to process
    if args.scenes:
        # Filter to specific scenes
        scenes_to_process = {s for s in args.scenes if s in SCENE_DATASET_MAPPING}
        if not scenes_to_process:
            print(f"[ERROR] No valid scenes found in: {args.scenes}")
            return 1
    elif args.datasets:
        # Filter to specific datasets
        scenes_to_process = set()
        for dataset in args.datasets:
            if dataset in DATASET_SCENES:
                scenes_to_process.update(DATASET_SCENES[dataset])
    else:
        # Process all scenes
        scenes_to_process = set(SCENE_DATASET_MAPPING.keys())

    # Sort scenes for consistent output
    scenes_sorted = sorted(scenes_to_process)

    # Summary
    print("=" * 80)
    action = "Move" if args.move else "Copy"
    print(f"Grid SVD File {action} Tool")
    print("=" * 80)
    print(f"Grid SVD root: {grid_svd_root}")
    print(f"Data root: {data_root}")
    print(f"Scenes to process: {len(scenes_sorted)}")
    if args.rank:
        print(f"Rank filter: r{args.rank}")
    print(f"Mode: {'MOVE (files will be removed from source)' if args.move else 'COPY'}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be transferred")
    print("=" * 80)
    print()

    # Process each scene
    total_transferred = 0
    total_skipped = 0

    for scene_name in scenes_sorted:
        dataset = SCENE_DATASET_MAPPING[scene_name]
        print(f"Processing: {scene_name} -> {dataset}")

        transferred, skipped = transfer_scene_svd_files(
            scene_name=scene_name,
            dataset=dataset,
            grid_svd_root=grid_svd_root,
            data_root=data_root,
            rank_filter=args.rank,
            move=args.move,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        total_transferred += transferred
        total_skipped += skipped
        print(f"  Transferred: {transferred}, Skipped: {skipped}")
        print()

    # Summary
    print("=" * 80)
    print("Summary:")
    print(f"  Total transferred: {total_transferred}")
    print(f"  Total skipped: {total_skipped}")
    if args.dry_run:
        print("  [DRY RUN] No files were actually transferred")
    else:
        if args.move:
            print(f"  Files successfully moved to dataset directories")
            print(f"  Source files were removed from grid_svd_output")
        else:
            print(f"  Files successfully copied to dataset directories")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
