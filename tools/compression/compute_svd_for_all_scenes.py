#!/usr/bin/env python3
"""
Compute SVD decomposition for all scenes in lerf_ovs/train directory.

This script:
1. Loads lang_feat.npy and valid_feat_mask.npy for each scene
2. Performs SVD decomposition on valid features only
3. Saves results as lang_feat_svd.npz containing U, S, Vt matrices

Usage:
    python tools/compute_svd_for_all_scenes.py
    python tools/compute_svd_for_all_scenes.py --scene figurines
    python tools/compute_svd_for_all_scenes.py --force-overwrite
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, List
from tqdm import tqdm


# Default paths
DEFAULT_TRAIN_ROOT = "/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train"
DATASET_ROOTS = {
    "lerf_ovs": "/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train",
    "3DOVS": "/new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS/train",
}


def get_train_root(dataset: str = "lerf_ovs") -> str:
    """Get train root path for given dataset."""
    return DATASET_ROOTS.get(dataset, DEFAULT_TRAIN_ROOT)


def get_scenes_with_lang_feat(dataset: str = "lerf_ovs") -> List[str]:
    """Get list of scenes that have lang_feat.npy files."""
    train_root = get_train_root(dataset)
    train_dir = Path(train_root)
    scenes = []

    if not train_dir.exists():
        return scenes

    for scene_dir in train_dir.iterdir():
        if scene_dir.is_dir():
            lang_feat_path = scene_dir / "lang_feat.npy"
            if lang_feat_path.exists():
                scenes.append(scene_dir.name)

    return sorted(scenes)


def compute_svd_on_scene(
    scene: str,
    dataset: str = "lerf_ovs",
    force_overwrite: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD decomposition for a single scene.

    Args:
        scene: Scene name
        dataset: Dataset name (lerf_ovs, 3DOVS, etc.)
        force_overwrite: If True, overwrite existing lang_feat_svd.npz

    Returns:
        U: [N_valid, 768] - left singular vectors (truncated to valid features only)
        S: [768] - singular values
        Vt: [768, 768] - right singular vectors (transposed)
    """
    train_root = get_train_root(dataset)
    scene_dir = Path(train_root) / scene

    # Load lang_feat
    lang_feat_path = scene_dir / "lang_feat.npy"
    if not lang_feat_path.exists():
        raise FileNotFoundError(f"lang_feat.npy not found: {lang_feat_path}")

    print(f"Loading lang_feat from: {lang_feat_path}")
    lang_feat = np.load(lang_feat_path).astype(np.float32)  # [N, 768]
    print(f"  lang_feat shape: {lang_feat.shape}")

    # Load valid_feat_mask
    valid_feat_mask_path = scene_dir / "valid_feat_mask.npy"
    if not valid_feat_mask_path.exists():
        print(f"  Warning: valid_feat_mask.npy not found, assuming all features are valid")
        valid_feat_mask = np.ones(lang_feat.shape[0], dtype=np.bool_)
    else:
        print(f"Loading valid_feat_mask from: {valid_feat_mask_path}")
        valid_feat_mask = np.load(valid_feat_mask_path).astype(np.bool_)
        print(f"  valid_feat_mask shape: {valid_feat_mask.shape}")
        print(f"  Valid count: {valid_feat_mask.sum()}, Invalid count: {(~valid_feat_mask).sum()}")

    # Check if lang_feat has been filtered
    N_original = valid_feat_mask.shape[0]
    N_features = lang_feat.shape[0]

    if N_features != N_original:
        print(f"  Warning: lang_feat has been filtered ({N_features} != {N_original})")
        print(f"  All features in lang_feat are valid")
        valid_features = lang_feat  # [N_features, 768]
    else:
        # Extract valid features for SVD
        valid_features = lang_feat[valid_feat_mask]  # [N_valid, 768]
        print(f"  Valid features shape: {valid_features.shape}")

    # Perform SVD decomposition
    print(f"Computing SVD on {valid_features.shape} features...")
    U, S, Vt = np.linalg.svd(valid_features, full_matrices=False)

    print(f"  U shape: {U.shape}")
    print(f"  S shape: {S.shape}")
    print(f"  Vt shape: {Vt.shape}")
    print(f"  Singular values (first 10): {S[:10]}")
    print(f"  Cumulative variance (first 16): {np.sum(S[:16]**2) / np.sum(S**2):.4f}")
    print(f"  Cumulative variance (first 32): {np.sum(S[:32]**2) / np.sum(S**2):.4f}")
    print(f"  Cumulative variance (first 64): {np.sum(S[:64]**2) / np.sum(S**2):.4f}")

    return U, S, Vt


def save_svd_results(
    scene: str,
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
    dataset: str = "lerf_ovs",
    force_overwrite: bool = False
):
    """Save SVD results to lang_feat_svd.npz.

    Args:
        scene: Scene name
        U: [N_valid, 768] - left singular vectors
        S: [768] - singular values
        Vt: [768, 768] - right singular vectors (transposed)
        dataset: Dataset name (lerf_ovs, 3DOVS, etc.)
        force_overwrite: If True, overwrite existing file
    """
    train_root = get_train_root(dataset)
    scene_dir = Path(train_root) / scene
    output_path = scene_dir / "lang_feat_svd.npz"

    # Check if file exists
    if output_path.exists() and not force_overwrite:
        print(f"  SVD file already exists: {output_path}")
        print(f"  Use --force-overwrite to overwrite")
        return False

    # Save SVD results
    print(f"Saving SVD results to: {output_path}")
    np.savez(output_path, U=U, S=S, Vt=Vt)

    # Print file size
    file_size = output_path.stat().st_size
    print(f"  File size: {file_size / 1024**2:.2f} MB")

    return True


def process_scene(
    scene: str,
    dataset: str = "lerf_ovs",
    force_overwrite: bool = False
) -> bool:
    """Process a single scene: compute SVD and save results.

    Args:
        scene: Scene name
        dataset: Dataset name (lerf_ovs, 3DOVS, etc.)
        force_overwrite: If True, overwrite existing lang_feat_svd.npz

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Processing scene: {scene} (dataset: {dataset})")
    print(f"{'='*70}\n")

    try:
        # Compute SVD
        U, S, Vt = compute_svd_on_scene(scene, dataset, force_overwrite)

        # Save results
        success = save_svd_results(scene, U, S, Vt, dataset, force_overwrite)

        if success:
            print(f"\n{'='*70}")
            print(f"Scene {scene} processed successfully!")
            print(f"{'='*70}\n")
        else:
            print(f"\nScene {scene} skipped (SVD file already exists)")

        return True

    except Exception as e:
        print(f"\nError processing scene {scene}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compute SVD decomposition for all scenes in lerf_ovs/train",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all scenes
  python tools/compute_svd_for_all_scenes.py

  # Process specific scene
  python tools/compute_svd_for_all_scenes.py --scene figurines

  # Overwrite existing SVD files
  python tools/compute_svd_for_all_scenes.py --force-overwrite

  # List scenes with lang_feat.npy
  python tools/compute_svd_for_all_scenes.py --list-scenes
        """
    )

    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene name to process (default: process all scenes)"
    )
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Process all scenes (default behavior)"
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing lang_feat_svd.npz files"
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List scenes with lang_feat.npy and exit"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lerf_ovs",
        choices=["lerf_ovs", "3DOVS"],
        help="Dataset to process (default: lerf_ovs)"
    )

    args = parser.parse_args()

    # List scenes mode
    if args.list_scenes:
        scenes = get_scenes_with_lang_feat(args.dataset)
        print(f"Found {len(scenes)} scenes with lang_feat.npy (dataset: {args.dataset}):")
        for scene in scenes:
            train_root = get_train_root(args.dataset)
            scene_dir = Path(train_root) / scene
            lang_feat_path = scene_dir / "lang_feat.npy"
            svd_path = scene_dir / "lang_feat_svd.npz"

            lang_feat_size = lang_feat_path.stat().st_size / 1024**2
            svd_status = "✓" if svd_path.exists() else "✗"

            print(f"  {svd_status} {scene} ({lang_feat_size:.2f} MB)")
        return

    # Determine which scenes to process
    if args.scene:
        scenes = [args.scene]
        # Verify scene exists
        train_root = get_train_root(args.dataset)
        scene_dir = Path(train_root) / args.scene
        if not scene_dir.exists():
            print(f"Error: Scene directory not found: {scene_dir}")
            return
        lang_feat_path = scene_dir / "lang_feat.npy"
        if not lang_feat_path.exists():
            print(f"Error: lang_feat.npy not found: {lang_feat_path}")
            return
    else:
        # Process all scenes
        scenes = get_scenes_with_lang_feat(args.dataset)
        if not scenes:
            print(f"No scenes found with lang_feat.npy in dataset: {args.dataset}")
            return

        print(f"Found {len(scenes)} scenes to process (dataset: {args.dataset}):")
        for scene in scenes:
            print(f"  - {scene}")
        print()

    # Process scenes
    results = {}
    for scene in scenes:
        success = process_scene(scene, args.dataset, args.force_overwrite)
        results[scene] = success

    # Print summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    successful = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Processed: {successful}/{total} scenes")

    if successful < total:
        print("\nFailed scenes:")
        for scene, success in results.items():
            if not success:
                print(f"  ✗ {scene}")
    else:
        print("\nAll scenes processed successfully!")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
