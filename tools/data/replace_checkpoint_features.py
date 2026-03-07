#!/usr/bin/env python3
"""
Replace language features in checkpoint_with_features_s.pth with lang_feat.npy from gaussian_train.

This script takes the GaussianModel checkpoint from output_features and replaces
its language features (index 7) with the original lang_feat.npy from gaussian_train
or a directly specified feature file.

Usage:
    # Single scene (search in gaussian_train)
    python tools/replace_checkpoint_features.py --scene figurines

    # All scenes in output_features
    python tools/replace_checkpoint_features.py --all

    # Custom paths
    python tools/replace_checkpoint_features.py --all \\
        --output_dir /new_data/cyf/projects/SceneSplat/output_features \\
        --gaussian_train /new_data/cyf/projects/SceneSplat/gaussian_train

    # Direct feature file path (override gaussian_train search)
    python tools/replace_checkpoint_features.py --scene figurines \\
        --feat_path /new_data/cyf/projects/SceneSplat/output_features/figurines/language_features_768d.npy
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def load_checkpoint_with_features(checkpoint_path: str) -> Tuple[tuple, int]:
    """
    Load GaussianModel checkpoint in format: ((13-element tuple), iteration).

    Returns:
        (state_tuple, iteration)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        state, iteration = checkpoint
        if isinstance(state, tuple) and len(state) >= 13:
            return state, iteration
        else:
            raise ValueError(f"Expected state to be a 13+ element tuple, got {len(state)} elements")
    else:
        raise ValueError(f"Expected checkpoint format: ((13-element tuple), iteration)")


def get_checkpoint_info(state: tuple) -> dict:
    """Extract information from checkpoint state tuple."""
    # GaussianModel format (13 elements):
    # 0: active_sh_degree
    # 1: xyz [N, 3]
    # 2: features_dc [N, 3]
    # 3: features_rest [N, 45]
    # 4: scaling [N, 3]
    # 5: rotation [N, 4]
    # 6: opacity [N, 1]
    # 7: language_features [N, feat_dim]  <-- This is what we want to replace
    # 8: max_radii2D [N]
    # 9: xyz_gradient_accum [N, 1]
    # 10: denom [N, 1]
    # 11: opt_dict
    # 12: spatial_lr_scale

    xyz = state[1]
    lang_features = state[7]

    return {
        "num_gaussians": xyz.shape[0],
        "lang_feat_dim": lang_features.shape[1],
        "lang_feat_shape": lang_features.shape,
    }


def replace_language_features(
    state: tuple,
    new_lang_feat: np.ndarray,
) -> tuple:
    """
    Replace language features in checkpoint state tuple.

    Args:
        state: Original 13-element state tuple
        new_lang_feat: New language features [N, feat_dim]

    Returns:
        New state tuple with replaced language features
    """
    # Convert to tensor if needed
    if isinstance(new_lang_feat, np.ndarray):
        new_lang_feat_tensor = torch.from_numpy(new_lang_feat.astype(np.float32))
    else:
        new_lang_feat_tensor = new_lang_feat

    # Create new tuple with replaced language features (index 7)
    new_state = tuple(
        new_lang_feat_tensor if i == 7 else item
        for i, item in enumerate(state)
    )

    return new_state


def validate_replacement(
    old_lang_feat: torch.Tensor,
    new_lang_feat: np.ndarray,
    num_gaussians: int,
) -> bool:
    """Validate that new features can replace old features."""
    # Check shape compatibility
    if new_lang_feat.shape[0] != num_gaussians:
        print(f"  Warning: Feature count mismatch!")
        print(f"    Checkpoint has {num_gaussians} Gaussians")
        print(f"    lang_feat.npy has {new_lang_feat.shape[0]} points")

        # If checkpoint has fewer Gaussians, we can truncate
        if new_lang_feat.shape[0] > num_gaussians:
            print(f"  Will truncate lang_feat.npy to {num_gaussians} points")
            return True
        else:
            print(f"  Error: Not enough points in lang_feat.npy")
            return False

    # Check dimension
    old_dim = old_lang_feat.shape[1]
    new_dim = new_lang_feat.shape[1]

    if old_dim != new_dim:
        print(f"  Warning: Feature dimension mismatch!")
        print(f"    Checkpoint: {old_dim} dim")
        print(f"    lang_feat.npy: {new_dim} dim")
        print(f"  Will proceed with dimension change")

    return True


def find_lang_feat_path(
    scene_name: str,
    gaussian_train: Path,
) -> Optional[Path]:
    """
    Find lang_feat.npy for a given scene.

    Searches in multiple possible locations:
    - gaussian_train/train/{scene}/lang_feat.npy
    - gaussian_train/{scene}/lang_feat.npy
    - gaussian_train/{dataset}/train/{scene}/lang_feat.npy
    - gaussian_train/{dataset}/{scene}/lang_feat.npy

    Args:
        scene_name: Name of the scene
        gaussian_train: Base gaussian_train directory

    Returns:
        Path to lang_feat.npy if found, None otherwise
    """
    # Possible path patterns
    possible_paths = [
        gaussian_train / "train" / scene_name / "lang_feat.npy",
        gaussian_train / scene_name / "lang_feat.npy",
    ]

    # Also search in dataset subdirectories (e.g., lerf_ovs, 3DOVS)
    if gaussian_train.is_dir():
        for dataset_dir in gaussian_train.iterdir():
            if dataset_dir.is_dir():
                possible_paths.extend([
                    dataset_dir / "train" / scene_name / "lang_feat.npy",
                    dataset_dir / scene_name / "lang_feat.npy",
                ])

    for path in possible_paths:
        if path.exists():
            return path

    return None


def find_valid_feat_mask_path(
    scene_name: str,
    gaussian_train: Path,
    lang_feat_path: Path,
) -> Optional[Path]:
    """
    Find valid_feat_mask.npy for a given scene.

    Looks in the same directory as lang_feat.npy.

    Args:
        scene_name: Name of the scene
        gaussian_train: Base gaussian_train directory
        lang_feat_path: Path to lang_feat.npy (used to find the directory)

    Returns:
        Path to valid_feat_mask.npy if found, None otherwise
    """
    # Check in same directory as lang_feat.npy
    mask_path = lang_feat_path.parent / "valid_feat_mask.npy"
    if mask_path.exists():
        return mask_path
    return None


def process_scene(
    scene_name: str,
    output_dir: Path,
    gaussian_train: Path,
    backup: bool = True,
    dry_run: bool = False,
    feat_path: Optional[Path] = None,
) -> bool:
    """
    Process a single scene: replace checkpoint features with lang_feat.npy.

    Args:
        scene_name: Name of the scene
        output_dir: Directory containing checkpoint_with_features_s.pth
        gaussian_train: Directory containing scene data with lang_feat.npy
        backup: Whether to backup original checkpoint
        dry_run: If True, only print what would be done
        feat_path: Optional direct path to feature file (overrides gaussian_train search)

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'=' * 70}")
    print(f"Processing scene: {scene_name}")
    print(f"{'=' * 70}")

    # Paths
    checkpoint_path = output_dir / scene_name / "checkpoint_with_features_s.pth"

    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        return False

    # Determine lang_feat path
    if feat_path is not None:
        # Use directly provided feature path
        if not os.path.exists(feat_path):
            print(f"  ✗ Feature file not found: {feat_path}")
            return False
        lang_feat_path = feat_path
        print(f"  Using direct feature path: {lang_feat_path}")
    else:
        # Find lang_feat.npy in gaussian_train
        lang_feat_path = find_lang_feat_path(scene_name, gaussian_train)
        if lang_feat_path is None:
            print(f"  ✗ lang_feat.npy not found for scene: {scene_name}")
            print(f"      Searched in: {gaussian_train}")
            return False

    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Features: {lang_feat_path}")

    # Load checkpoint
    try:
        state, iteration = load_checkpoint_with_features(str(checkpoint_path))
        info = get_checkpoint_info(state)
        print(f"  Checkpoint info:")
        print(f"    Gaussians: {info['num_gaussians']:,}")
        print(f"    Feature dim: {info['lang_feat_dim']}")
        print(f"    Feature shape: {info['lang_feat_shape']}")
    except Exception as e:
        print(f"  ✗ Failed to load checkpoint: {e}")
        return False

    # Load lang_feat.npy
    try:
        print(f"\n  Loading lang_feat.npy...")
        loaded_lang_feat = np.load(lang_feat_path)
        print(f"    Shape: {loaded_lang_feat.shape}")
        print(f"    Data type: {loaded_lang_feat.dtype}")
        print(f"    L2 norm mean: {np.linalg.norm(loaded_lang_feat, axis=1).mean():.6f}")
    except Exception as e:
        print(f"  ✗ Failed to load lang_feat.npy: {e}")
        return False

    # Check if we need to expand using valid_feat_mask
    new_lang_feat = loaded_lang_feat
    num_gaussians = info["num_gaussians"]

    if loaded_lang_feat.shape[0] < num_gaussians:
        # Try to find valid_feat_mask
        valid_feat_mask_path = find_valid_feat_mask_path(scene_name, gaussian_train, lang_feat_path)

        if valid_feat_mask_path is not None:
            print(f"\n  Found valid_feat_mask.npy: {valid_feat_mask_path}")
            valid_feat_mask = np.load(valid_feat_mask_path)
            print(f"    Valid points: {np.sum(valid_feat_mask):,}")
            print(f"    Total points: {len(valid_feat_mask):,}")

            # Verify that loaded_lang_feat matches valid points count
            if np.sum(valid_feat_mask) == loaded_lang_feat.shape[0] and len(valid_feat_mask) == num_gaussians:
                # Expand features to all points using valid_feat_mask
                print(f"\n  Expanding features using valid_feat_mask...")
                feat_dim = loaded_lang_feat.shape[1]
                expanded_lang_feat = np.zeros((num_gaussians, feat_dim), dtype=np.float32)
                # Use boolean indexing correctly - need to assign where mask is True
                expanded_lang_feat[valid_feat_mask.astype(bool)] = loaded_lang_feat
                new_lang_feat = expanded_lang_feat
                print(f"    Expanded shape: {new_lang_feat.shape}")
                invalid_count = num_gaussians - int(np.sum(valid_feat_mask))
                print(f"    Zero features for invalid points: {invalid_count:,}")
            else:
                print(f"  ✗ valid_feat_mask doesn't match feature dimensions")
                print(f"    Expected: {num_gaussians} total, {loaded_lang_feat.shape[0]} valid")
                print(f"    Got: {len(valid_feat_mask)} total, {np.sum(valid_feat_mask)} valid")
                return False
        else:
            print(f"\n  ✗ Cannot find valid_feat_mask.npy")
            print(f"    lang_feat.npy has {loaded_lang_feat.shape[0]} points")
            print(f"    Checkpoint has {num_gaussians} Gaussians")
            print(f"    Cannot expand without valid_feat_mask")
            return False

    # Validate replacement
    print(f"\n  Validating replacement...")
    old_lang_feat = state[7]
    if not validate_replacement(old_lang_feat, new_lang_feat, num_gaussians):
        return False

    # Truncate if needed (shouldn't happen after expansion)
    if new_lang_feat.shape[0] > num_gaussians:
        print(f"  Truncating lang_feat from {new_lang_feat.shape[0]} to {num_gaussians}")
        new_lang_feat = new_lang_feat[:num_gaussians]

    # Backup original checkpoint
    if backup and not dry_run:
        backup_path = checkpoint_path.with_suffix(".pth.backup")
        print(f"\n  Backing up to: {backup_path}")
        shutil.copy(checkpoint_path, backup_path)
        print(f"  ✓ Backup created")

    # Replace features
    print(f"\n  Replacing language features...")
    new_state = replace_language_features(state, new_lang_feat)

    # Verify replacement
    new_lang_feat_tensor = new_state[7]
    print(f"    New feature shape: {new_lang_feat_tensor.shape}")
    print(f"    New feature dim: {new_lang_feat_tensor.shape[1]}")
    print(f"    L2 norm mean: {torch.norm(new_lang_feat_tensor, dim=1).mean().item():.6f}")

    # Save new checkpoint
    if dry_run:
        print(f"\n  [DRY RUN] Would save to: {checkpoint_path}")
        print(f"  [DRY RUN] Backup: {'Yes' if backup else 'No'}")
    else:
        print(f"\n  Saving modified checkpoint...")
        output_checkpoint = (new_state, iteration)
        torch.save(output_checkpoint, checkpoint_path)
        print(f"  ✓ Saved to: {checkpoint_path}")

    print(f"\n  ✓ Successfully processed {scene_name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Replace checkpoint language features with lang_feat.npy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Scene selection
    scene_group = parser.add_mutually_exclusive_group(required=True)
    scene_group.add_argument(
        "--scene",
        type=str,
        help="Single scene name to process",
    )
    scene_group.add_argument(
        "--all",
        action="store_true",
        help="Process all scenes in output_features directory",
    )

    # Path arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/new_data/cyf/projects/SceneSplat/output_features",
        help="Directory containing scene checkpoints (default: output_features)",
    )
    parser.add_argument(
        "--gaussian_train",
        type=str,
        default="/new_data/cyf/projects/SceneSplat/gaussian_train",
        help="Directory containing scene data with lang_feat.npy (default: gaussian_train)",
    )

    # Processing options
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not backup original checkpoint before modifying",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    gaussian_train = Path(args.gaussian_train)
    backup = not args.no_backup

    print("=" * 70)
    print("Checkpoint Language Feature Replacement")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"gaussian_train: {gaussian_train}")
    print(f"Backup: {'Yes' if backup else 'No'}")
    print(f"Dry run: {'Yes' if args.dry_run else 'No'}")

    # Collect scenes to process
    if args.all:
        # Find all scenes with checkpoint_with_features_s.pth
        scenes = []
        for scene_dir in output_dir.iterdir():
            if scene_dir.is_dir():
                checkpoint_path = scene_dir / "checkpoint_with_features_s.pth"
                if checkpoint_path.exists():
                    scenes.append(scene_dir.name)

        if not scenes:
            print(f"\nNo scenes found with checkpoint_with_features_s.pth in {output_dir}")
            return

        scenes = sorted(scenes)
        print(f"\nFound {len(scenes)} scenes to process:")
        for scene in scenes:
            print(f"  - {scene}")
    else:
        scenes = [args.scene]

    # Process each scene
    print("\n" + "=" * 70)
    print("Starting processing")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for scene in scenes:
        feat_path = os.path.join(args.output_dir, scene, f"language_features_768d.npy")
        print(feat_path)
        if process_scene(
            scene,
            output_dir,
            gaussian_train,
            backup=backup,
            dry_run=args.dry_run,
            feat_path=feat_path
        ):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total scenes: {len(scenes)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified. Run without --dry-run to apply changes.")

    if fail_count > 0:
        exit(1)


if __name__ == "__main__":
    main()
