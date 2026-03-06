#!/usr/bin/env python3
"""
Move or copy specific files (grid_meta_data.json and lang_feat_grid_svd_r*.npz)
from scene folders to a target directory, preserving the directory structure.
Only the specified files are copied/moved; other files remain in the source.
"""

import argparse
import shutil
from pathlib import Path
from typing import Set


def find_scenes_with_svd_features(source_dir: Path) -> Set[str]:
    """
    Find all scene folders that contain both grid_meta_data.json
    and at least one lang_feat_grid_svd_r*.npz file.

    Args:
        source_dir: Source directory containing scene folders

    Returns:
        Set of scene folder names that meet the criteria
    """
    valid_scenes = set()

    for scene_dir in source_dir.iterdir():
        if not scene_dir.is_dir():
            continue

        # Check for grid_meta_data.json
        grid_meta_path = scene_dir / "grid_meta_data.json"
        if not grid_meta_path.exists():
            continue

        # Check for at least one lang_feat_grid_svd_r*.npz file
        has_lang_feat = False
        for file in scene_dir.glob("lang_feat_grid_svd_r*.npz"):
            if file.is_file():
                has_lang_feat = True
                break

        if has_lang_feat:
            valid_scenes.add(scene_dir.name)

    return valid_scenes


def move_scenes(
    source_dir: Path,
    target_dir: Path,
    valid_scenes: Set[str],
    dry_run: bool = False,
    copy_mode: bool = False,
) -> None:
    """
    Move or copy specific files (grid_meta_data.json and lang_feat_grid_svd_r*.npz)
    from valid scene folders to target directory.

    Args:
        source_dir: Source directory
        target_dir: Target directory
        valid_scenes: Set of valid scene folder names
        dry_run: If True, only print what would be done without actually moving
        copy_mode: If True, copy files instead of moving
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    processed_scenes = 0
    skipped_count = 0
    file_count = 0
    action = "COPY" if copy_mode else "MOVE"

    for scene_name in sorted(valid_scenes):
        source_path = source_dir / scene_name
        target_path = target_dir / scene_name

        # Create target scene directory
        if not dry_run:
            target_path.mkdir(parents=True, exist_ok=True)

        # Files to copy/move
        files_to_process = [source_path / "grid_meta_data.json"]
        files_to_process.extend(source_path.glob("lang_feat_grid_svd_r*.npz"))

        files_copied = 0
        for src_file in files_to_process:
            if not src_file.exists():
                continue

            dst_file = target_path / src_file.name

            if dst_file.exists():
                continue

            if dry_run:
                print(f"[DRY RUN] Would {action.lower()}: {scene_name}/{src_file.name}")
            else:
                if copy_mode:
                    shutil.copy2(str(src_file), str(dst_file))
                else:
                    shutil.move(str(src_file), str(dst_file))

            files_copied += 1
            file_count += 1

        if files_copied > 0:
            print(f"[{action}] {scene_name} ({files_copied} files)")
            processed_scenes += 1
        else:
            skipped_count += 1

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Valid scenes found: {len(valid_scenes)}")
    print(f"  Scenes processed: {processed_scenes}")
    print(f"  Total files {'to be ' if dry_run else ''}{action.lower()}ed: {file_count}")
    if skipped_count > 0:
        print(f"  Scenes skipped (no new files): {skipped_count}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Move scene folders with SVD-compressed language features"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Source directory containing scene folders",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Target directory to move scenes to",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without actually moving/copying files",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving (source files remain unchanged)",
    )

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return 1

    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    mode_str = ("COPY" if args.copy else "MOVE") + (" (DRY RUN)" if args.dry_run else "")
    print(f"Mode: {mode_str}")
    print(f"{'=' * 60}")

    # Find valid scenes
    print("Scanning for valid scenes...")
    valid_scenes = find_scenes_with_svd_features(source_dir)
    print(f"Found {len(valid_scenes)} valid scenes\n")

    if len(valid_scenes) == 0:
        print("No valid scenes found. Exiting.")
        return 0

    # Move scenes
    move_scenes(source_dir, target_dir, valid_scenes, dry_run=args.dry_run, copy_mode=args.copy)

    return 0


if __name__ == "__main__":
    exit(main())
