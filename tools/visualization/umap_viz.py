#!/usr/bin/env python3
"""
UMAP Visualization Tool for Language Feature Space Comparison

This script reads language features from training scenes and visualizes:
1. Original feature space (768-dim lang_feat.npy)
2. Compressed feature space (16-dim from SVD decomposition)

Results are saved to langfeat_visualizations/

GPU acceleration is available via RAPIDS cuML (use --use-gpu flag).
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import GPU-accelerated UMAP (RAPIDS cuML)
CUML_AVAILABLE = False
cuml_UMAP = None

try:
    from cuml.manifold import UMAP as cuml_UMAP
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    print("RAPIDS cuML not available, using CPU UMAP")

# Always import CPU UMAP as fallback
umap = None
try:
    import umap
except ImportError:
    print("umap-learn not found, installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn", "-q"])
    import umap


def get_umap_reducer(use_gpu: bool = False, n_components: int = 2, n_neighbors: int = 15,
                     min_dist: float = 0.1, metric: str = "cosine", random_state: int = 42):
    """
    Get UMAP reducer with optional GPU acceleration.

    Args:
        use_gpu: Whether to use GPU acceleration (requires RAPIDS cuML)
        n_components: Number of output dimensions
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        random_state: Random seed

    Returns:
        UMAP reducer (cuml.UMAP or umap.UMAP)
    """
    if use_gpu:
        if CUML_AVAILABLE:
            return cuml_UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state
            )
        else:
            print("Warning: GPU requested but RAPIDS cuML is not available. Falling back to CPU.")
            print("To install cuML: conda install -c rapidsai -c nvidia cuml")

    # Use CPU UMAP
    if umap is None:
        raise RuntimeError("UMAP is not available. Please install umap-learn: pip install umap-learn")

    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=-1,
        verbose=True
    )


def find_scene_data(data_root: str) -> Dict[str, Dict[str, Path]]:
    """
    Find all scenes with both lang_feat.npy and lang_feat_svd.npz files.

    Returns:
        Dictionary mapping scene names to paths of original and compressed files
    """
    data_root = Path(data_root)
    scenes = {}

    # Find all lang_feat.npy files
    for lang_feat_path in data_root.glob("*/train/*/lang_feat.npy"):
        scene_dir = lang_feat_path.parent
        scene_name = scene_dir.name

        # Check if corresponding SVD file exists
        svd_path = scene_dir / "lang_feat_svd.npz"
        if svd_path.exists():
            dataset_name = scene_dir.parent.parent.name  # e.g., 'lerf_ovs' or '3DOVS'
            key = f"{dataset_name}/{scene_name}"
            scenes[key] = {
                "original": lang_feat_path,
                "svd": svd_path,
                "scene_name": scene_name,
                "dataset": dataset_name,
                "scene_dir": scene_dir
            }

    return scenes


def load_semantic_labels(scene_dir: Path) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[np.ndarray]]:
    """
    Load semantic labels, label mapping, and label mask for a scene.
    Only returns labels for points where lang_label_mask is True.

    Args:
        scene_dir: Path to the scene directory

    Returns:
        Tuple of (labels_array, label_mapping_dict, label_mask)
        - labels_array: numpy array of label indices (only for masked=True points), or None if not available
        - label_mapping_dict: dict mapping indices to class names, or None if not available
        - label_mask: boolean array where True indicates labeled points, or None if not available
    """
    label_path = scene_dir / "lang_label.npy"
    mask_path = scene_dir / "lang_label_mask.npy"
    mapping_path = scene_dir / "lang_label_mapping.json"

    labels = None
    mapping = None
    label_mask = None

    if mask_path.exists():
        label_mask = np.load(mask_path)
        print(f"  Found lang_label_mask.npy: {label_mask.sum()}/{len(label_mask)} points labeled")

    if label_path.exists():
        labels = np.load(label_path)

    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        # Convert string keys to int keys for easier lookup
        if isinstance(mapping, dict):
            mapping = {int(k): v for k, v in mapping.items()}

    return labels, mapping, label_mask


def merge_label_mappings(mappings: List[dict]) -> dict:
    """
    Merge multiple label mappings into a unified mapping.
    Handles cases where different scenes have different class indices.

    Args:
        mappings: List of label mapping dictionaries

    Returns:
        Unified label mapping dictionary
    """
    unified = {}
    all_names = set()

    # Collect all unique class names
    for mapping in mappings:
        if mapping:
            all_names.update(mapping.values())

    # Assign new indices to unique class names
    for i, name in enumerate(sorted(all_names)):
        unified[i] = name

    return unified


def remap_labels(labels: np.ndarray, old_mapping: dict, new_mapping: dict) -> np.ndarray:
    """
    Remap label indices from old mapping to new mapping.

    Args:
        labels: Original label indices
        old_mapping: Original mapping (index -> name)
        new_mapping: New unified mapping (index -> name)

    Returns:
        Remapped label indices with -1 for unmapped/invalid labels
    """
    if old_mapping is None or new_mapping is None:
        return np.full_like(labels, -1)

    # Create reverse mapping from name to new index
    name_to_new_idx = {v: k for k, v in new_mapping.items()}

    # Create mapping from old index to new index
    old_to_new = {}
    for old_idx, name in old_mapping.items():
        if name in name_to_new_idx:
            old_to_new[old_idx] = name_to_new_idx[name]

    # Remap labels
    remapped = np.full_like(labels, -1)
    for old_idx, new_idx in old_to_new.items():
        mask = labels == old_idx
        remapped[mask] = new_idx

    return remapped


def load_features(scene_info: Dict, rank: int = 16, max_samples: int = None,
                  load_semantic: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[dict]]:
    """
    Load original and compressed features for a scene.

    Args:
        scene_info: Dictionary containing file paths
        rank: Rank for SVD reconstruction
        max_samples: Maximum number of samples to load (for memory efficiency)
        load_semantic: Whether to load semantic labels (default: True)

    Returns:
        Tuple of (original_features, compressed_features, semantic_labels, label_mapping)
        - semantic_labels and label_mapping are None if load_semantic=False
    """
    scene_dir = scene_info["original"].parent
    valid_feat_mask_path = scene_dir / "valid_feat_mask.npy"

    # Load original features
    original = np.load(scene_info["original"])
    n_original = len(original)

    # Load SVD components
    svd_data = np.load(scene_info["svd"])
    U = svd_data["U"]
    S = svd_data["S"]
    n_u = U.shape[0]

    # Load semantic labels and label mask first (needed for computing combined mask)
    semantic_labels = None
    label_mapping = None
    label_mask = None

    if load_semantic:
        semantic_labels, label_mapping, label_mask = load_semantic_labels(scene_dir)

    # === Handle different data space scenarios ===
    # Case 1: U and original have same dimensions (no prior filtering)
    # Case 2: U is already filtered by valid_feat_mask (U.shape[0] < original.shape[0])
    # Case 3: lang_feat.npy is pre-filtered (smaller than valid_feat_mask)

    if n_u == n_original:
        # Case 1: Same space - but need to verify if original is already filtered
        print(f"  Same space: U and original both have {n_original} dimensions")

        # Check if original is already pre-filtered (smaller than valid_feat_mask)
        combined_mask = None
        original_already_filtered = False

        if valid_feat_mask_path.exists():
            valid_feat_mask = np.load(valid_feat_mask_path).astype(bool)
            print(f"  valid_feat_mask size: {len(valid_feat_mask)}, original size: {len(original)}")

            # Check if original is already filtered
            if len(valid_feat_mask) != len(original):
                print(f"  Original is pre-filtered (lang_feat.npy already filtered)")
                print(f"  Skipping valid_feat_mask application, only using label_mask")
                original_already_filtered = True
                # Don't apply valid_feat_mask since original is already filtered
            else:
                combined_mask = valid_feat_mask
                print(f"  valid_feat_mask: {combined_mask.sum()}/{len(combined_mask)} points")

        if label_mask is not None:
            if original_already_filtered:
                # Original is already filtered, label_mask needs to be aligned
                if len(label_mask) != len(original):
                    # label_mask is in original space, need to filter it
                    if valid_feat_mask_path.exists():
                        valid_feat_mask = np.load(valid_feat_mask_path).astype(bool)
                        # First, align semantic_labels to filtered space
                        if semantic_labels is not None:
                            semantic_labels = semantic_labels[valid_feat_mask]
                            print(f"  semantic_labels aligned to filtered space: {len(semantic_labels)} points")
                        # Then align label_mask to filtered space
                        label_mask_for_filtered = label_mask[valid_feat_mask]
                        combined_mask = label_mask_for_filtered
                        print(f"  label_mask (aligned to filtered space): {combined_mask.sum()}/{len(combined_mask)} points")
                    else:
                        print(f"  Warning: Cannot align label_mask (no valid_feat_mask)")
                        combined_mask = None
                else:
                    combined_mask = label_mask
                    print(f"  label_mask: {combined_mask.sum()}/{len(combined_mask)} points")
            else:
                # Normal case: original is not pre-filtered
                if combined_mask is None:
                    combined_mask = label_mask
                else:
                    combined_mask = combined_mask & label_mask
                print(f"  Combined mask (valid & labeled): {combined_mask.sum()}/{len(combined_mask)} points")

        # Apply combined filter (only if mask size matches)
        if combined_mask is not None:
            if len(combined_mask) == len(original):
                original = original[combined_mask]
                U = U[combined_mask]
                if semantic_labels is not None:
                    semantic_labels = semantic_labels[combined_mask]
            else:
                print(f"  Warning: Mask size ({len(combined_mask)}) != data size ({len(original)}), skipping filter")

        # Reconstruct compressed features
        U_r = U[:, :rank]
        S_r = S[:rank]
        compressed = U_r * S_r

    else:
        # Case 2: U is pre-filtered, need to align masks
        print(f"  Different spaces: original={n_original}, U={n_u}")
        print(f"  U is pre-filtered by valid_feat_mask")

        # Load valid_feat_mask to understand the mapping
        if valid_feat_mask_path.exists():
            valid_feat_mask = np.load(valid_feat_mask_path).astype(bool)
            n_valid = valid_feat_mask.sum()
            print(f"  valid_feat_mask: {n_valid}/{len(valid_feat_mask)} points")

            # Verify U dimension matches valid count
            if n_u != n_valid:
                print(f"  Warning: U.shape[0]={n_u} != valid_feat_mask.sum()={n_valid}")
        else:
            print(f"  Warning: valid_feat_mask not found, cannot verify alignment")

        # Apply label_mask only (since U is already valid-feat-filtered)
        if label_mask is not None:
            # First, apply valid_feat_mask to label_mask to get the mask for U space
            if valid_feat_mask_path.exists():
                valid_feat_mask = np.load(valid_feat_mask_path).astype(bool)
                # Filter label_mask to U space (only keep valid positions)
                label_mask_for_u = label_mask[valid_feat_mask]
                print(f"  label_mask in U space: {label_mask_for_u.sum()}/{len(label_mask_for_u)} points")

                # Filter U and compressed features
                U = U[label_mask_for_u]
                if semantic_labels is not None:
                    # semantic_labels is also in original space, need to filter it
                    # First apply valid_feat_mask, then label_mask
                    sem_valid = semantic_labels[valid_feat_mask]
                    semantic_labels = sem_valid[label_mask_for_u]

                # Also filter original features for consistency
                # Apply both masks to original space
                combined_mask_original = valid_feat_mask & label_mask
                original = original[combined_mask_original]
            else:
                # No valid_feat_mask, assume U and original should align differently
                # This case shouldn't happen but handle it gracefully
                print(f"  Warning: No valid_feat_mask but U.dim != original.dim")
                # Just use label_mask on original
                original = original[label_mask]
                if semantic_labels is not None:
                    semantic_labels = semantic_labels[label_mask]
                # U is already filtered, reconstruct compressed
                U_r = U[:, :rank]
                S_r = S[:rank]
                compressed = U_r * S_r
                return original, compressed, semantic_labels, label_mapping
        else:
            # No label_mask, just apply valid_feat_mask to original
            if valid_feat_mask_path.exists():
                valid_feat_mask = np.load(valid_feat_mask_path).astype(bool)
                original = original[valid_feat_mask]
                if semantic_labels is not None:
                    semantic_labels = semantic_labels[valid_feat_mask]
            # U is already filtered
            U_r = U[:, :rank]
            S_r = S[:rank]
            compressed = U_r * S_r
            return original, compressed, semantic_labels, label_mapping

        # Reconstruct compressed features from filtered U
        U_r = U[:, :rank]
        S_r = S[:rank]
        compressed = U_r * S_r

    # Subsample if needed
    if max_samples is not None and len(original) > max_samples:
        indices = np.random.choice(len(original), max_samples, replace=False)
        original = original[indices]
        compressed = compressed[indices]
        if semantic_labels is not None:
            semantic_labels = semantic_labels[indices]

    return original, compressed, semantic_labels, label_mapping


def load_all_features(scenes: Dict, rank: int = 16, max_samples_per_scene: int = None,
                     max_total_samples: int = 50000, load_semantic: bool = True) -> Tuple:
    """
    Load and combine features from all scenes.

    NOTE: This function loads all scenes into memory at once.
    For better memory efficiency, use process_scenes_iteratively() instead.

    Args:
        scenes: Dictionary of scene information
        rank: Rank for SVD reconstruction
        max_samples_per_scene: Maximum samples per scene
        max_total_samples: Maximum total samples across all scenes
        load_semantic: Whether to load semantic labels

    Returns:
        Tuple of (original_features, compressed_features, scene_labels, semantic_labels, unified_mapping)
        - scene_labels: Scene name for each sample
        - semantic_labels: Semantic class indices for each sample (or None)
        - unified_mapping: Unified label mapping (or None)
    """
    original_list = []
    compressed_list = []
    scene_labels = []
    semantic_labels_list = []
    mappings = []

    for scene_key, scene_info in scenes.items():
        print(f"Loading {scene_key}...")

        try:
            orig, comp, sem_labels, mapping = load_features(
                scene_info, rank=rank, max_samples=max_samples_per_scene, load_semantic=load_semantic
            )

            original_list.append(orig)
            compressed_list.append(comp)
            scene_labels.extend([scene_key] * len(orig))

            if load_semantic and sem_labels is not None:
                semantic_labels_list.append(sem_labels)
            if load_semantic and mapping is not None:
                mappings.append(mapping)

        except Exception as e:
            print(f"  Error loading {scene_key}: {e}")
            continue

    # Concatenate all features
    original_all = np.vstack(original_list)
    compressed_all = np.vstack(compressed_list)

    # Handle semantic labels
    semantic_labels_all = None
    unified_mapping = None

    if load_semantic and semantic_labels_list:
        # Merge all label mappings into a unified one
        if mappings:
            unified_mapping = merge_label_mappings(mappings)

            # Remap all semantic labels to unified mapping
            remapped_list = []
            for i, (scene_labels_i, mapping) in enumerate(zip(semantic_labels_list, mappings)):
                remapped = remap_labels(scene_labels_i, mapping, unified_mapping)
                remapped_list.append(remapped)

            semantic_labels_all = np.concatenate(remapped_list)

    # Subsample total if needed
    # If max_total_samples is None, use 1/10 of total samples
    if max_total_samples is None:
        max_total_samples = len(original_all) // 10
        print(f"  Using 1/10 of total samples: {max_total_samples}")

    if len(original_all) > max_total_samples:
        indices = np.random.choice(len(original_all), max_total_samples, replace=False)
        original_all = original_all[indices]
        compressed_all = compressed_all[indices]
        scene_labels = [scene_labels[i] for i in indices]
        if semantic_labels_all is not None:
            semantic_labels_all = semantic_labels_all[indices]

    return original_all, compressed_all, scene_labels, semantic_labels_all, unified_mapping


def process_scenes_iteratively(
    scenes: Dict,
    rank: int = 16,
    max_samples_per_scene: int = None,
    use_gpu: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    load_semantic: bool = True,
    # Per-scene visualization and saving parameters
    output_dir: str = None,
    create_per_scene: bool = False,
    create_semantic_viz: bool = False,
    remove_outliers: bool = False,
    outlier_method: str = "isolation_forest",
    contamination: float = 0.05,
    percentile: float = 95.0,
    exclude_invalid: bool = True,
    save_per_scene_npz: bool = True
) -> Tuple:
    """
    Process scenes one at a time to save memory.
    Loads each scene, computes UMAP, optionally creates visualizations and saves npz files.

    Args:
        scenes: Dictionary of scene information
        rank: Rank for SVD reconstruction
        max_samples_per_scene: Maximum samples per scene
        use_gpu: Whether to use GPU acceleration
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        random_state: Random seed
        load_semantic: Whether to load semantic labels
        output_dir: Output directory for visualizations and npz files
        create_per_scene: Whether to create per-scene visualizations
        create_semantic_viz: Whether to create per-scene semantic visualizations
        remove_outliers: Whether to remove outliers for per-scene visualizations
        outlier_method: Outlier detection method
        contamination: Expected proportion of outliers
        percentile: Percentile threshold for percentile method
        exclude_invalid: Whether to exclude invalid labels (-1) from semantic visualization
        save_per_scene_npz: Whether to save individual scene npz files

    Returns:
        Tuple of (scene_info_dict, unified_mapping)
        where scene_info_dict contains per-scene statistics
    """
    import gc

    scene_info_dict = {}

    # Setup output directory for per-scene saves
    per_scene_npz_dir = None
    if output_dir and (create_per_scene or create_semantic_viz or save_per_scene_npz):
        per_scene_npz_dir = Path(output_dir) / "per_scene_npz"
        per_scene_npz_dir.mkdir(parents=True, exist_ok=True)

    # Collect all label mappings first for unified semantic labels
    all_mappings = []
    for scene_key in sorted(scenes.keys()):
        scene_info = scenes[scene_key]
        scene_dir = scene_info["scene_dir"]
        _, mapping, _ = load_semantic_labels(scene_dir)
        if mapping is not None:
            all_mappings.append(mapping)

    # Create unified mapping if needed
    unified_mapping = None
    if load_semantic and all_mappings:
        unified_mapping = merge_label_mappings(all_mappings)

    scene_keys = sorted(scenes.keys())
    total_scenes = len(scene_keys)

    for idx, scene_key in enumerate(scene_keys, 1):
        scene_info = scenes[scene_key]
        print(f"[{idx}/{total_scenes}] Processing {scene_key}...")

        try:
            # Load features for this scene
            original, compressed, sem_labels, mapping = load_features(
                scene_info, rank=rank, max_samples=max_samples_per_scene, load_semantic=load_semantic
            )

            n_samples = len(original)
            print(f"  Loaded {n_samples} samples")

            if n_samples == 0:
                print(f"  Skipping {scene_key}: no valid samples")
                continue

            # Apply UMAP to original features
            original_2d = apply_umap(
                original,
                use_gpu=use_gpu,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state
            )

            # Apply UMAP to compressed features
            compressed_2d = apply_umap(
                compressed,
                use_gpu=use_gpu,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state
            )

            # Remap semantic labels to unified mapping if needed
            if load_semantic and sem_labels is not None and mapping is not None and unified_mapping is not None:
                sem_labels = remap_labels(sem_labels, mapping, unified_mapping)

            # Save per-scene npz file if requested
            if save_per_scene_npz and per_scene_npz_dir is not None:
                safe_scene_name = scene_key.replace('/', '_').replace('\\', '_')
                npz_path = per_scene_npz_dir / f"{safe_scene_name}.npz"
                save_dict = {
                    'original': original_2d,
                    'compressed': compressed_2d,
                    'scene_labels': np.array([scene_key] * n_samples, dtype='U')
                }
                if sem_labels is not None:
                    save_dict['semantic_labels'] = sem_labels
                np.savez(npz_path, **save_dict)
                print(f"  Saved per-scene npz to {npz_path}")

            # Create per-scene visualizations if requested
            if create_per_scene or create_semantic_viz:
                # Prepare data for visualization
                orig_2d_viz = original_2d.copy()
                comp_2d_viz = compressed_2d.copy()
                scene_name_list = [scene_key] * n_samples
                sem_labels_viz = sem_labels.copy() if sem_labels is not None else None

                # Remove outliers if requested
                if remove_outliers and n_samples > 100:
                    print(f"  Removing outliers for visualization (original: {n_samples} samples)...")
                    if sem_labels_viz is not None:
                        orig_2d_viz, comp_2d_viz, scene_name_list, sem_labels_viz = remove_outliers_embeddings_with_semantic(
                            orig_2d_viz, comp_2d_viz, scene_name_list, sem_labels_viz,
                            method=outlier_method,
                            contamination=contamination,
                            percentile=percentile
                        )
                    else:
                        orig_2d_viz, comp_2d_viz, scene_name_list = remove_outliers_embeddings(
                            orig_2d_viz, comp_2d_viz, scene_name_list,
                            method=outlier_method,
                            contamination=contamination,
                            percentile=percentile
                        )

                # Create per-scene visualization directory
                if create_per_scene and output_dir:
                    viz_output_dir = Path(output_dir) / "per_scene"
                    viz_output_dir.mkdir(parents=True, exist_ok=True)

                    safe_scene_name = scene_key.replace('/', '_').replace('\\', '_')
                    if remove_outliers:
                        output_path = viz_output_dir / f"{safe_scene_name}_no_outliers_{outlier_method}.png"
                    else:
                        output_path = viz_output_dir / f"{safe_scene_name}_comparison.png"

                    # Create visualization
                    _, axes = plt.subplots(1, 2, figsize=(14, 6))

                    # Original features
                    ax1 = axes[0]
                    ax1.scatter(orig_2d_viz[:, 0], orig_2d_viz[:, 1], c='blue', s=1, alpha=0.5)
                    title = f'Original Features (768-dim)'
                    ax1.set_title(title, fontsize=24, fontweight='bold')
                    ax1.set_xlabel('UMAP 1')
                    ax1.set_ylabel('UMAP 2')
                    ax1.grid(True, alpha=0.3)
                    ax1.text(0.05, 0.95, f'Points: {len(orig_2d_viz)}',
                            transform=ax1.transAxes, fontsize=10,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    # Compressed features
                    ax2 = axes[1]
                    ax2.scatter(comp_2d_viz[:, 0], comp_2d_viz[:, 1], c='red', s=1, alpha=0.5)
                    title = f'Compressed Features ({rank}-dim SVD)'
                    ax2.set_title(title, fontsize=24, fontweight='bold')
                    ax2.set_xlabel('UMAP 1')
                    ax2.set_ylabel('UMAP 2')
                    ax2.grid(True, alpha=0.3)
                    ax2.text(0.05, 0.95, f'Points: {len(comp_2d_viz)}',
                            transform=ax2.transAxes, fontsize=10,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    plt.tight_layout()
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    print(f"  Saved per-scene visualization to {output_path}")
                    plt.close()

                    # Create semantic visualization if requested
                    if create_semantic_viz and sem_labels_viz is not None and unified_mapping is not None:
                        semantic_output_dir = Path(output_dir) / "per_scene_semantic"
                        semantic_output_dir.mkdir(parents=True, exist_ok=True)

                        # Filter out invalid labels if requested
                        if exclude_invalid:
                            valid_mask = sem_labels_viz >= 0
                            orig_2d_sem = orig_2d_viz[valid_mask]
                            comp_2d_sem = comp_2d_viz[valid_mask]
                            sem_labels_sem = sem_labels_viz[valid_mask]
                        else:
                            orig_2d_sem = orig_2d_viz
                            comp_2d_sem = comp_2d_viz
                            sem_labels_sem = sem_labels_viz

                        # Skip if no valid labels
                        if len(orig_2d_sem) > 0:
                            unique_labels = sorted(np.unique(sem_labels_sem))
                            n_classes = len(unique_labels)

                            # Create color map
                            if n_classes <= 10:
                                cmap = plt.get_cmap('tab10')
                            elif n_classes <= 20:
                                cmap = plt.get_cmap('tab20')
                            else:
                                cmap = plt.get_cmap('hsv')

                            colors = {}
                            for label_idx in unique_labels:
                                if label_idx >= 0 and label_idx in unified_mapping:
                                    color_idx = label_idx % n_classes
                                    colors[label_idx] = cmap(color_idx / max(n_classes, 1))

                            fig, axes = plt.subplots(1, 2, figsize=(20, 9))

                            # Original features with semantic colors
                            ax1 = axes[0]
                            for label_idx in unique_labels:
                                if label_idx >= 0 and label_idx in unified_mapping:
                                    mask = sem_labels_sem == label_idx
                                    n_points = np.sum(mask)
                                    if n_points > 0:
                                        class_name = unified_mapping[label_idx]
                                        ax1.scatter(orig_2d_sem[mask, 0], orig_2d_sem[mask, 1],
                                                    c=[colors[label_idx]], label=f"{class_name} ({n_points})",
                                                    s=2, alpha=0.6)

                            title = f'Original - Semantic ({n_classes} classes)'
                            ax1.set_title(title, fontsize=24, fontweight='bold')
                            ax1.set_xlabel('UMAP 1')
                            ax1.set_ylabel('UMAP 2')
                            ax1.grid(True, alpha=0.3)

                            stats_text = f'Points: {len(orig_2d_sem)}\nClasses: {n_classes}'
                            ax1.text(0.02, 0.98, stats_text,
                                    transform=ax1.transAxes, fontsize=9,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                            if n_classes > 20:
                                ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
                            else:
                                ax1.legend(loc='best', fontsize=7, ncol=2 if n_classes > 8 else 1)

                            # Compressed features with semantic colors
                            ax2 = axes[1]
                            for label_idx in unique_labels:
                                if label_idx >= 0 and label_idx in unified_mapping:
                                    mask = sem_labels_sem == label_idx
                                    n_points = np.sum(mask)
                                    if n_points > 0:
                                        class_name = unified_mapping[label_idx]
                                        ax2.scatter(comp_2d_sem[mask, 0], comp_2d_sem[mask, 1],
                                                    c=[colors[label_idx]], label=f"{class_name} ({n_points})",
                                                    s=2, alpha=0.6)

                            title = f'Compressed ({rank}-dim SVD) - Semantic ({n_classes} classes)'
                            ax2.set_title(title, fontsize=24, fontweight='bold')
                            ax2.set_xlabel('UMAP 1')
                            ax2.set_ylabel('UMAP 2')
                            ax2.grid(True, alpha=0.3)

                            ax2.text(0.02, 0.98, stats_text,
                                    transform=ax2.transAxes, fontsize=9,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                            if n_classes > 20:
                                ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
                            else:
                                ax2.legend(loc='best', fontsize=7, ncol=2 if n_classes > 8 else 1)

                            fig.suptitle(f'Semantic: {scene_key}', fontsize=28, fontweight='bold', y=0.98)
                            plt.tight_layout(rect=[0, 0, 0.85 if n_classes > 20 else 1, 0.96])

                            if remove_outliers:
                                sem_output_path = semantic_output_dir / f"{safe_scene_name}_semantic_no_outliers_{outlier_method}.png"
                            else:
                                sem_output_path = semantic_output_dir / f"{safe_scene_name}_semantic.png"

                            plt.savefig(sem_output_path, dpi=150, bbox_inches='tight')
                            print(f"  Saved semantic visualization to {sem_output_path}")
                            plt.close()

                        # Free visualization memory
                        del orig_2d_viz, comp_2d_viz, sem_labels_viz
                else:
                    # Free visualization memory
                    del orig_2d_viz, comp_2d_viz
                    if sem_labels_viz is not None:
                        del sem_labels_viz

            # Store scene info
            scene_info_dict[scene_key] = {
                'n_samples': n_samples,
                'has_semantic': sem_labels is not None
            }

            # Explicitly free memory - don't accumulate embeddings
            del original, compressed, original_2d, compressed_2d
            if sem_labels is not None:
                del sem_labels
            gc.collect()

        except Exception as e:
            print(f"  Error processing {scene_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nAll scenes processed. Saved {len(scene_info_dict)} scenes to {per_scene_npz_dir}")

    return scene_info_dict, unified_mapping


def load_embeddings_from_npz(
    npz_dir: str,
    max_total_samples: int = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Optional[dict]]:
    """
    Load and concatenate UMAP embeddings from per-scene npz files.

    Args:
        npz_dir: Directory containing per-scene npz files
        max_total_samples: Maximum total samples to load (for memory efficiency)
        random_state: Random seed for subsampling

    Returns:
        Tuple of (original_2d, compressed_2d, scene_labels, semantic_labels, unified_mapping)
    """
    npz_dir = Path(npz_dir)
    if not npz_dir.exists():
        raise ValueError(f"NPZ directory not found: {npz_dir}")

    # Find all npz files
    npz_files = sorted(list(npz_dir.glob("*.npz")))
    if not npz_files:
        raise ValueError(f"No npz files found in {npz_dir}")

    print(f"Loading embeddings from {len(npz_files)} npz files...")

    original_list = []
    compressed_list = []
    scene_labels = []
    semantic_labels_list = []

    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)

            original = data['original']
            compressed = data['compressed']
            labels = data['scene_labels'].tolist()

            original_list.append(original)
            compressed_list.append(compressed)
            scene_labels.extend(labels)

            if 'semantic_labels' in data:
                semantic_labels_list.append(data['semantic_labels'])

        except Exception as e:
            print(f"  Error loading {npz_path}: {e}")
            continue

    if not original_list:
        raise ValueError("No valid embeddings loaded from npz files")

    # Concatenate all embeddings
    original_all = np.vstack(original_list)
    compressed_all = np.vstack(compressed_list)

    # Handle semantic labels
    semantic_labels_all = None
    if semantic_labels_list:
        semantic_labels_all = np.concatenate(semantic_labels_list)

    # Subsample if needed
    if max_total_samples is not None and len(original_all) > max_total_samples:
        np.random.seed(random_state)
        indices = np.random.choice(len(original_all), max_total_samples, replace=False)
        original_all = original_all[indices]
        compressed_all = compressed_all[indices]
        scene_labels = [scene_labels[i] for i in indices]
        if semantic_labels_all is not None:
            semantic_labels_all = semantic_labels_all[indices]

    print(f"  Loaded {len(original_all)} samples from {len(original_list)} scenes")

    return original_all, compressed_all, scene_labels, semantic_labels_all, None


def apply_umap(features: np.ndarray, use_gpu: bool = False, n_components: int = 2, n_neighbors: int = 15,
               min_dist: float = 0.1, metric: str = "cosine", random_state: int = 42) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction with optional GPU acceleration.

    Args:
        features: Input features of shape (N, D)
        use_gpu: Whether to use GPU acceleration (requires RAPIDS cuML)
        n_components: Number of output dimensions
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        random_state: Random seed

    Returns:
        UMAP-reduced features of shape (N, n_components)
    """
    device_str = "GPU" if (use_gpu and CUML_AVAILABLE) else "CPU"
    print(f"Applying UMAP ({device_str}): {features.shape[0]} samples, {features.shape[1]} dimensions...")

    reducer = get_umap_reducer(
        use_gpu=use_gpu,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )

    embedding = reducer.fit_transform(features)

    # cuML returns cupy array, convert to numpy
    if hasattr(embedding, 'get'):
        embedding = embedding.get()

    return embedding


def remove_outliers_embeddings(original_2d: np.ndarray, compressed_2d: np.ndarray,
                               labels: List[str],
                               method: str = "isolation_forest",
                               contamination: float = 0.05,
                               percentile: float = 95.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Remove outliers from 2D UMAP embeddings by detecting outliers in both representations
    and taking the union (remove if outlier in either original or compressed).

    Args:
        original_2d: Original 2D embeddings of shape (N, 2)
        compressed_2d: Compressed 2D embeddings of shape (N, 2)
        labels: Labels for each point
        method: Outlier detection method
            - "isolation_forest": Uses sklearn IsolationForest
            - "percentile": Removes points based on distance percentile
            - "iqr": Uses Interquartile Range method
            - "density": Uses Local Outlier Factor (LOF)
        contamination: Expected proportion of outliers (for isolation_forest, lof)
        percentile: Percentile threshold for distance-based method

    Returns:
        Tuple of (filtered_original_2d, filtered_compressed_2d, filtered_labels)
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    n_samples = len(original_2d)
    print(f"\nRemoving outliers using method: {method} (original: {n_samples} samples)")
    print("  Detecting outliers separately in original and compressed, then taking union...")

    def detect_outliers(data: np.ndarray) -> np.ndarray:
        """Detect outliers and return boolean mask (True for inliers)."""
        if method == "isolation_forest":
            clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            outlier_pred = clf.fit_predict(data)
            return outlier_pred == 1  # 1 for inliers, -1 for outliers

        elif method == "percentile":
            centroid = np.mean(data, axis=0)
            distances = np.linalg.norm(data - centroid, axis=1)
            threshold = np.percentile(distances, percentile)
            return distances <= threshold

        elif method == "iqr":
            mask = np.ones(len(data), dtype=bool)
            for dim in range(data.shape[1]):
                q1 = np.percentile(data[:, dim], 25)
                q3 = np.percentile(data[:, dim], 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                dim_mask = (data[:, dim] >= lower_bound) & (data[:, dim] <= upper_bound)
                mask &= dim_mask
            return mask

        elif method == "density" or method == "lof":
            clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
            outlier_pred = clf.fit_predict(data)
            return outlier_pred == 1  # 1 for inliers, -1 for outliers

        else:
            raise ValueError(f"Unknown outlier removal method: {method}")

    # Detect outliers in original and compressed separately
    original_mask = detect_outliers(original_2d)
    compressed_mask = detect_outliers(compressed_2d)

    # Take intersection of inliers (union of outliers)
    # Keep only samples that are inliers in BOTH representations
    combined_mask = original_mask & compressed_mask

    # Report statistics
    n_original_outliers = np.sum(~original_mask)
    n_compressed_outliers = np.sum(~compressed_mask)
    n_total_removed = np.sum(~combined_mask)

    print(f"  Original: {n_original_outliers} outliers detected")
    print(f"  Compressed: {n_compressed_outliers} outliers detected")
    print(f"  Total removed (union): {n_total_removed} outliers ({100 * n_total_removed / n_samples:.1f}%)")
    print(f"  Remaining {np.sum(combined_mask)} samples")

    return original_2d[combined_mask], compressed_2d[combined_mask], [labels[i] for i in range(len(labels)) if combined_mask[i]]


def remove_outliers_embeddings_with_semantic(
    original_2d: np.ndarray,
    compressed_2d: np.ndarray,
    labels: List[str],
    semantic_labels: np.ndarray,
    method: str = "isolation_forest",
    contamination: float = 0.05,
    percentile: float = 95.0
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Remove outliers from 2D UMAP embeddings while preserving semantic label alignment.

    This function detects outliers in both original and compressed representations,
    removes the union of outliers (removes if outlier in either), and also filters
    the semantic labels accordingly to maintain alignment.

    Args:
        original_2d: Original 2D embeddings of shape (N, 2)
        compressed_2d: Compressed 2D embeddings of shape (N, 2)
        labels: Scene labels for each point
        semantic_labels: Semantic class indices for each point
        method: Outlier detection method
            - "isolation_forest": Uses sklearn IsolationForest
            - "percentile": Removes points based on distance percentile
            - "iqr": Uses Interquartile Range method
            - "density": Uses Local Outlier Factor (LOF)
        contamination: Expected proportion of outliers (for isolation_forest, lof)
        percentile: Percentile threshold for distance-based method

    Returns:
        Tuple of (filtered_original_2d, filtered_compressed_2d, filtered_labels, filtered_semantic_labels)
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    n_samples = len(original_2d)
    print(f"\nRemoving outliers using method: {method} (original: {n_samples} samples)")
    print("  Detecting outliers separately in original and compressed, then taking union...")

    def detect_outliers(data: np.ndarray) -> np.ndarray:
        """Detect outliers and return boolean mask (True for inliers)."""
        if method == "isolation_forest":
            clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            outlier_pred = clf.fit_predict(data)
            return outlier_pred == 1  # 1 for inliers, -1 for outliers

        elif method == "percentile":
            centroid = np.mean(data, axis=0)
            distances = np.linalg.norm(data - centroid, axis=1)
            threshold = np.percentile(distances, percentile)
            return distances <= threshold

        elif method == "iqr":
            mask = np.ones(len(data), dtype=bool)
            for dim in range(data.shape[1]):
                q1 = np.percentile(data[:, dim], 25)
                q3 = np.percentile(data[:, dim], 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                dim_mask = (data[:, dim] >= lower_bound) & (data[:, dim] <= upper_bound)
                mask &= dim_mask
            return mask

        elif method == "density" or method == "lof":
            clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
            outlier_pred = clf.fit_predict(data)
            return outlier_pred == 1  # 1 for inliers, -1 for outliers

        else:
            raise ValueError(f"Unknown outlier removal method: {method}")

    # Detect outliers in original and compressed separately
    original_mask = detect_outliers(original_2d)
    compressed_mask = detect_outliers(compressed_2d)

    # Take intersection of inliers (union of outliers)
    # Keep only samples that are inliers in BOTH representations
    combined_mask = original_mask & compressed_mask

    # Report statistics
    n_original_outliers = np.sum(~original_mask)
    n_compressed_outliers = np.sum(~compressed_mask)
    n_total_removed = np.sum(~combined_mask)

    print(f"  Original: {n_original_outliers} outliers detected")
    print(f"  Compressed: {n_compressed_outliers} outliers detected")
    print(f"  Total removed (union): {n_total_removed} outliers ({100 * n_total_removed / n_samples:.1f}%)")
    print(f"  Remaining {np.sum(combined_mask)} samples")

    # Filter all arrays using the combined mask
    filtered_original = original_2d[combined_mask]
    filtered_compressed = compressed_2d[combined_mask]
    filtered_labels = [labels[i] for i in range(len(labels)) if combined_mask[i]]
    filtered_semantic = semantic_labels[combined_mask]

    return filtered_original, filtered_compressed, filtered_labels, filtered_semantic


def load_umap_embeddings(npz_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load pre-computed UMAP embeddings from npz file.

    Args:
        npz_path: Path to the npz file containing embeddings

    Returns:
        Tuple of (original_2d, compressed_2d, labels)
    """
    print(f"Loading UMAP embeddings from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    original_2d = data['original']
    compressed_2d = data['compressed']
    labels = data['labels'].tolist()

    print(f"  Original shape: {original_2d.shape}")
    print(f"  Compressed shape: {compressed_2d.shape}")
    print(f"  Labels: {len(labels)}")

    return original_2d, compressed_2d, labels


def create_visualization(original_2d: np.ndarray, compressed_2d: np.ndarray,
                        labels: List[str], output_path: str, dataset_names: List[str] = None,
                        rank: int = 16):
    """
    Create comparison visualization of original vs compressed feature spaces.

    Args:
        original_2d: UMAP embedding of original features
        compressed_2d: UMAP embedding of compressed features
        labels: Scene labels for each point
        output_path: Path to save the figure
        dataset_names: Names of datasets for color coding
    """
    # Determine colors based on dataset
    if dataset_names is None:
        dataset_names = sorted(set(label.split('/')[0] for label in labels))

    colors = {}
    cmap = plt.get_cmap('tab10')
    for i, name in enumerate(dataset_names):
        colors[name] = cmap(i / len(dataset_names))

    # Create figure with GridSpec for better layout
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Extract dataset from labels for coloring
    label_datasets = [label.split('/')[0] for label in labels]

    # --- Plot 1: Original features - All scenes together ---
    ax1 = fig.add_subplot(gs[0, 0])
    for dataset in dataset_names:
        mask = np.array(label_datasets) == dataset
        ax1.scatter(original_2d[mask, 0], original_2d[mask, 1],
                   c=[colors[dataset]], label=dataset, s=0.5, alpha=0.6)
    ax1.set_title('Original Features (768-dim)', fontsize=28, fontweight='bold')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Compressed features - All scenes together ---
    ax2 = fig.add_subplot(gs[0, 1])
    for dataset in dataset_names:
        mask = np.array(label_datasets) == dataset
        ax2.scatter(compressed_2d[mask, 0], compressed_2d[mask, 1],
                   c=[colors[dataset]], label=dataset, s=0.5, alpha=0.6)
    ax2.set_title(f'Compressed Features ({rank}-dim SVD)', fontsize=28, fontweight='bold')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Side-by-side comparison (per-scene panels) ---
    ax3 = fig.add_subplot(gs[0, 2])

    # Get unique scenes
    unique_scenes = sorted(set(labels))
    n_scenes = len(unique_scenes)

    # Create a small panel showing per-scene structure
    scene_colors = plt.get_cmap('Set3')(np.linspace(0, 1, n_scenes))
    scene_to_color = {scene: scene_colors[i] for i, scene in enumerate(unique_scenes)}

    # Plot original with scene colors
    for i, scene in enumerate(unique_scenes[:10]):  # Show up to 10 scenes
        mask = np.array(labels) == scene
        ax3.scatter(original_2d[mask, 0], original_2d[mask, 1],
                   c=[scene_to_color[scene]], label=scene.split('/')[-1][:15],
                   s=0.5, alpha=0.6)
    ax3.set_title('Original Features (by Scene)', fontsize=24, fontweight='bold')
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    ax3.legend(fontsize=6, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Compressed with scene colors ---
    ax4 = fig.add_subplot(gs[1, 0])
    for i, scene in enumerate(unique_scenes[:10]):
        mask = np.array(labels) == scene
        ax4.scatter(compressed_2d[mask, 0], compressed_2d[mask, 1],
                   c=[scene_to_color[scene]], label=scene.split('/')[-1][:15],
                   s=0.5, alpha=0.6)
    ax4.set_title('Compressed Features (by Scene)', fontsize=24, fontweight='bold')
    ax4.set_xlabel('UMAP 1')
    ax4.set_ylabel('UMAP 2')
    ax4.legend(fontsize=6, loc='upper right')
    ax4.grid(True, alpha=0.3)

    # --- Plot 5: Distribution analysis ---
    ax5 = fig.add_subplot(gs[1, 1])

    # Plot distribution of UMAP coordinates
    ax5.hist(original_2d[:, 0], bins=50, alpha=0.5, label='Original (UMAP 1)', color='blue')
    ax5.hist(compressed_2d[:, 0], bins=50, alpha=0.5, label='Compressed (UMAP 1)', color='red')
    ax5.set_title('Distribution of UMAP-1 Coordinates', fontsize=24, fontweight='bold')
    ax5.set_xlabel('UMAP 1 Value')
    ax5.set_ylabel('Frequency')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # --- Plot 6: Statistics summary ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Calculate statistics
    orig_std = np.std(original_2d, axis=0)
    comp_std = np.std(compressed_2d, axis=0)
    orig_mean = np.mean(original_2d, axis=0)
    comp_mean = np.mean(compressed_2d, axis=0)

    stats_text = f"""
    Statistics Summary
    {'='*40}

    Total Samples: {len(original_2d):,}
    Number of Scenes: {len(unique_scenes)}

    Original Features (768-dim):
      Mean UMAP-1: {orig_mean[0]:.2f}
      Mean UMAP-2: {orig_mean[1]:.2f}
      Std UMAP-1: {orig_std[0]:.2f}
      Std UMAP-2: {orig_std[1]:.2f}

    Compressed Features ({rank}-dim SVD):
      Mean UMAP-1: {comp_mean[0]:.2f}
      Mean UMAP-2: {comp_mean[1]:.2f}
      Std UMAP-1: {comp_std[0]:.2f}
      Std UMAP-2: {comp_std[1]:.2f}

    Datasets: {', '.join(dataset_names)}
    """

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add overall title
    fig.suptitle('Language Feature Space: Original vs SVD-16 Compressed',
                fontsize=32, fontweight='bold', y=0.98)

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def create_semantic_visualization(
    original_2d: np.ndarray,
    compressed_2d: np.ndarray,
    semantic_labels: np.ndarray,
    label_mapping: dict,
    scene_labels: List[str],
    output_dir: str,
    remove_outliers: bool = False,
    outlier_method: str = "isolation_forest",
    contamination: float = 0.05,
    percentile: float = 95.0,
    exclude_invalid: bool = True,
    rank: int = 16
):
    """
    Create per-scene semantic label-based visualizations with colored classes and legend.

    Args:
        original_2d: UMAP embedding of original features
        compressed_2d: UMAP embedding of compressed features
        semantic_labels: Semantic class indices for each point
        label_mapping: Dictionary mapping indices to class names
        scene_labels: Scene name for each point
        output_dir: Directory to save individual scene figures
        remove_outliers: Whether to remove outliers for each scene individually
        outlier_method: Outlier detection method
        contamination: Expected proportion of outliers
        percentile: Percentile threshold for percentile method
        exclude_invalid: Whether to exclude invalid labels (-1) from visualization
    """
    if semantic_labels is None or label_mapping is None:
        print("Warning: No semantic labels provided, skipping semantic visualization")
        return

    output_dir = Path(output_dir) / "per_scene_semantic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique scenes
    unique_scenes = sorted(set(scene_labels))

    print(f"Creating per-scene semantic visualizations for {len(unique_scenes)} scenes...")

    for scene in unique_scenes:
        mask = np.array(scene_labels) == scene

        orig_2d = original_2d[mask]
        comp_2d = compressed_2d[mask]
        sem_labels = semantic_labels[mask]
        scene_name_list = [scene] * len(orig_2d)

        # Remove outliers for this scene individually if requested
        if remove_outliers and len(orig_2d) > 100:
            print(f"  Removing outliers for {scene} (original: {len(orig_2d)} samples)...")
            orig_2d, comp_2d, scene_name_list, sem_labels = remove_outliers_embeddings_with_semantic(
                orig_2d, comp_2d, scene_name_list, sem_labels,
                method=outlier_method,
                contamination=contamination,
                percentile=percentile
            )

        # Filter out invalid labels if requested
        if exclude_invalid:
            valid_mask = sem_labels >= 0
            orig_2d = orig_2d[valid_mask]
            comp_2d = comp_2d[valid_mask]
            sem_labels = sem_labels[valid_mask]

        # Skip if no valid labels
        if len(orig_2d) == 0:
            print(f"  Skipping {scene}: no valid semantic labels")
            continue

        # Get unique labels for this scene
        unique_labels = sorted(np.unique(sem_labels))
        n_classes = len(unique_labels)

        # Create color map for classes
        if n_classes <= 10:
            cmap = plt.get_cmap('tab10')
        elif n_classes <= 20:
            cmap = plt.get_cmap('tab20')
        else:
            cmap = plt.get_cmap('hsv')

        # Get colors for each unique label
        colors = {}
        for label_idx in unique_labels:
            if label_idx >= 0 and label_idx in label_mapping:
                color_idx = label_idx % n_classes
                colors[label_idx] = cmap(color_idx / max(n_classes, 1))

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))

        # --- Plot 1: Original features colored by semantic class ---
        ax1 = axes[0]
        for label_idx in unique_labels:
            if label_idx >= 0 and label_idx in label_mapping:
                mask = sem_labels == label_idx
                n_points = np.sum(mask)
                if n_points > 0:
                    class_name = label_mapping[label_idx]
                    ax1.scatter(orig_2d[mask, 0], orig_2d[mask, 1],
                              c=[colors[label_idx]], label=f"{class_name} ({n_points})",
                              s=2, alpha=0.6)

        title = f'Original Features (768-dim) - Semantic Classes ({n_classes})'
        ax1.set_title(title, fontsize=24, fontweight='bold')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f'Total Points: {len(orig_2d)}\nClasses: {n_classes}'
        ax1.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Create legend
        if n_classes > 20:
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
        else:
            ax1.legend(loc='best', fontsize=7, ncol=2 if n_classes > 8 else 1)

        # --- Plot 2: Compressed features colored by semantic class ---
        ax2 = axes[1]
        for label_idx in unique_labels:
            if label_idx >= 0 and label_idx in label_mapping:
                mask = sem_labels == label_idx
                n_points = np.sum(mask)
                if n_points > 0:
                    class_name = label_mapping[label_idx]
                    ax2.scatter(comp_2d[mask, 0], comp_2d[mask, 1],
                              c=[colors[label_idx]], label=f"{class_name} ({n_points})",
                              s=2, alpha=0.6)

        title = f'Compressed Features ({rank}-dim SVD) - Semantic Classes ({n_classes})'
        ax2.set_title(title, fontsize=24, fontweight='bold')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.grid(True, alpha=0.3)

        # Add statistics
        ax2.text(0.02, 0.98, stats_text,
                transform=ax2.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Create legend
        if n_classes > 20:
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
        else:
            ax2.legend(loc='best', fontsize=7, ncol=2 if n_classes > 8 else 1)

        # Add overall title
        fig.suptitle(f'Semantic Visualization: {scene}',
                    fontsize=28, fontweight='bold', y=0.98)

        # Adjust layout to make room for legend
        plt.tight_layout(rect=[0, 0, 0.85 if n_classes > 20 else 1, 0.96])

        # Create safe filename
        safe_scene_name = scene.replace('/', '_').replace('\\', '_')
        if remove_outliers:
            output_path = output_dir / f"{safe_scene_name}_semantic_no_outliers_{outlier_method}.png"
        else:
            output_path = output_dir / f"{safe_scene_name}_semantic.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {scene} semantic visualization to {output_path}")
        plt.close()

    print(f"Saved {len(unique_scenes)} per-scene semantic visualizations to {output_dir}")


def create_per_dataset_visualization(original_2d: np.ndarray, compressed_2d: np.ndarray,
                                   labels: List[str], output_dir: str,
                                   rank: int = 16):
    """
    Create individual visualizations for each dataset.

    Args:
        original_2d: UMAP embedding of original features
        compressed_2d: UMAP embedding of compressed features
        labels: Scene labels for each point
        output_dir: Directory to save individual figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by dataset
    datasets = sorted(set(label.split('/')[0] for label in labels))

    for dataset in datasets:
        mask = np.array([label.split('/')[0] == dataset for label in labels])
        orig_2d = original_2d[mask]
        comp_2d = compressed_2d[mask]
        scene_labels = [label.split('/')[-1] for label in np.array(labels)[mask]]

        # Get unique scenes in this dataset
        unique_scenes = sorted(set(scene_labels))
        n_scenes = len(unique_scenes)

        # Create visualization
        _, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Original features
        ax1 = axes[0]
        cmap = plt.get_cmap('tab20')
        for i, scene in enumerate(unique_scenes):
            scene_mask = np.array(scene_labels) == scene
            ax1.scatter(orig_2d[scene_mask, 0], orig_2d[scene_mask, 1],
                       c=[cmap(i / n_scenes)], label=scene[:20], s=1, alpha=0.6)
        ax1.set_title(f'{dataset}: Original Features (768-dim)', fontsize=28, fontweight='bold')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Compressed features
        ax2 = axes[1]
        for i, scene in enumerate(unique_scenes):
            scene_mask = np.array(scene_labels) == scene
            ax2.scatter(comp_2d[scene_mask, 0], comp_2d[scene_mask, 1],
                       c=[cmap(i / n_scenes)], label=scene[:20], s=1, alpha=0.6)
        ax2.set_title(f'{dataset}: Compressed Features ({rank}-dim SVD)', fontsize=28, fontweight='bold')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f"{dataset}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved {dataset} visualization to {output_path}")
        plt.close()


def create_per_scene_visualization(original_2d: np.ndarray, compressed_2d: np.ndarray,
                                   labels: List[str], output_dir: str,
                                   remove_outliers: bool = False,
                                   outlier_method: str = "isolation_forest",
                                   contamination: float = 0.05,
                                   percentile: float = 95.0,
                                   rank: int = 16):
    """
    Create individual comparison visualization for each scene.

    Args:
        original_2d: UMAP embedding of original features
        compressed_2d: UMAP embedding of compressed features
        labels: Scene labels for each point
        output_dir: Directory to save individual scene figures
        remove_outliers: Whether to remove outliers for each scene individually
        outlier_method: Outlier detection method
        contamination: Expected proportion of outliers
        percentile: Percentile threshold for percentile method
    """
    output_dir = Path(output_dir) / "per_scene"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique scenes
    unique_scenes = sorted(set(labels))

    for scene in unique_scenes:
        mask = np.array(labels) == scene

        orig_2d = original_2d[mask]
        comp_2d = compressed_2d[mask]
        scene_labels = [scene] * len(orig_2d)

        # Remove outliers for this scene individually if requested
        if remove_outliers and len(orig_2d) > 100:
            print(f"  Removing outliers for {scene} (original: {len(orig_2d)} samples)...")
            orig_2d, comp_2d, scene_labels = remove_outliers_embeddings(
                orig_2d, comp_2d, scene_labels,
                method=outlier_method,
                contamination=contamination,
                percentile=percentile
            )

        # Create visualization
        _, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Original features
        ax1 = axes[0]
        ax1.scatter(orig_2d[:, 0], orig_2d[:, 1], c='blue', s=1, alpha=0.5)
        title = f'Original Features (768-dim)'
        ax1.set_title(title, fontsize=28, fontweight='bold')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.grid(True, alpha=0.3)

        # Add statistics
        ax1.text(0.05, 0.95, f'Points: {len(orig_2d)}',
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Compressed features
        ax2 = axes[1]
        ax2.scatter(comp_2d[:, 0], comp_2d[:, 1], c='red', s=1, alpha=0.5)
        title = f'Compressed Features ({rank}-dim SVD)'
        ax2.set_title(title, fontsize=28, fontweight='bold')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.grid(True, alpha=0.3)

        # Add statistics
        ax2.text(0.05, 0.95, f'Points: {len(comp_2d)}',
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Create safe filename
        safe_scene_name = scene.replace('/', '_').replace('\\', '_')
        if remove_outliers:
            output_path = output_dir / f"{safe_scene_name}_no_outliers_{outlier_method}.png"
        else:
            output_path = output_dir / f"{safe_scene_name}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {scene} visualization to {output_path}")
        plt.close()

    print(f"Saved {len(unique_scenes)} individual scene visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='UMAP visualization of language features')
    parser.add_argument('--data-root', type=str,
                       default='/new_data/cyf/projects/SceneSplat/gaussian_train',
                       help='Path to training data directory')
    parser.add_argument('--output-dir', type=str,
                       default='langfeat_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--rank', type=int, default=16,
                       help='SVD rank for compression (default: 16)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum total samples to visualize (default: 1/10 of total samples)')
    parser.add_argument('--max-samples-per-scene', type=int, default=None,
                       help='Maximum samples per scene (default: no limit)')
    parser.add_argument('--n-neighbors', type=int, default=15,
                       help='UMAP n_neighbors parameter (default: 15)')
    parser.add_argument('--min-dist', type=float, default=0.1,
                       help='UMAP min_dist parameter (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--create-per-dataset', action='store_true',
                       help='Create separate visualizations per dataset')
    parser.add_argument('--create-per-scene', action='store_true',
                       help='Create separate visualizations per scene')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration (requires RAPIDS cuML)')

    # New arguments for loading pre-computed embeddings and outlier removal
    parser.add_argument('--load-npz', type=str, default=None,
                       help='Path to pre-computed UMAP embeddings npz file (skip UMAP processing)')
    parser.add_argument('--remove-outliers', action='store_true',
                       help='Remove outliers from UMAP embeddings before visualization')
    parser.add_argument('--outlier-method', type=str, default='isolation_forest',
                       choices=['isolation_forest', 'percentile', 'iqr', 'lof'],
                       help='Outlier detection method (default: isolation_forest)')
    parser.add_argument('--contamination', type=float, default=0.05,
                       help='Expected proportion of outliers for isolation_forest/lof (default: 0.05)')
    parser.add_argument('--percentile', type=float, default=95.0,
                       help='Percentile threshold for percentile method (default: 95.0)')

    # Semantic visualization arguments
    parser.add_argument('--semantic-viz', action='store_true',
                       help='Create per-scene semantic label-based visualization with class colors and legend (requires --create-per-scene)')
    parser.add_argument('--load-semantic', action='store_true', default=True,
                       help='Load semantic labels (default: True)')
    parser.add_argument('--exclude-invalid', action='store_true', default=True,
                       help='Exclude invalid labels (-1) from semantic visualization (default: True)')

    args = parser.parse_args()

    print("="*60)
    print("UMAP Visualization Tool for Language Features")
    print("="*60)

    # Initialize semantic labels and mapping
    semantic_labels = None
    label_mapping = None

    # Option 1: Load pre-computed embeddings from npz file
    if args.load_npz:
        print(f"\nLoading pre-computed UMAP embeddings from: {args.load_npz}")
        original_2d, compressed_2d, labels = load_umap_embeddings(args.load_npz)

        # Try to load semantic labels from npz if available
        data = np.load(args.load_npz, allow_pickle=True)
        if 'semantic_labels' in data and 'label_mapping' in data:
            semantic_labels = data['semantic_labels']
            label_mapping = data['label_mapping'].item() if data['label_mapping'].dtype == object else data['label_mapping']
            print(f"  Loaded semantic labels: {len(np.unique(semantic_labels))} classes")

    # Option 2: Process scenes from scratch
    else:
        # Find all scene data
        print(f"\nSearching for scenes in {args.data_root}...")
        scenes = find_scene_data(args.data_root)

        if not scenes:
            print("No scenes found with both lang_feat.npy and lang_feat_svd.npz files!")
            return

        print(f"Found {len(scenes)} scenes:")
        for scene_key in sorted(scenes.keys()):
            print(f"  - {scene_key}")

        # Process scenes iteratively (memory-efficient)
        print(f"\nProcessing scenes iteratively (max_samples={args.max_samples})...")
        print("This will process each scene separately to save memory.")

        # Create output directory for per-scene saves
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_info_dict, unified_mapping = process_scenes_iteratively(
            scenes,
            rank=args.rank,
            max_samples_per_scene=args.max_samples_per_scene,
            use_gpu=args.use_gpu,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric="cosine",
            random_state=args.seed,
            load_semantic=args.load_semantic,
            # Pass per-scene visualization parameters
            output_dir=str(output_dir),
            create_per_scene=args.create_per_scene,
            create_semantic_viz=args.semantic_viz,
            remove_outliers=args.remove_outliers,
            outlier_method=args.outlier_method,
            contamination=args.contamination,
            percentile=args.percentile,
            exclude_invalid=args.exclude_invalid,
            save_per_scene_npz=True
        )

        # Reload data from npz files for main visualization
        per_scene_npz_dir = output_dir / "per_scene_npz"
        print(f"\nReloading embeddings from {per_scene_npz_dir} for main visualization...")
        original_2d, compressed_2d, labels, semantic_labels, _ = load_embeddings_from_npz(
            str(per_scene_npz_dir),
            max_total_samples=args.max_samples,
            random_state=args.seed
        )
        label_mapping = unified_mapping  # Use the unified mapping from processing

        if len(original_2d) == 0:
            print("No data to visualize!")
            return

        print(f"\nFinal results:")
        print(f"  Total samples: {len(original_2d)}")
        print(f"  Original UMAP shape: {original_2d.shape}")
        print(f"  Compressed UMAP shape: {compressed_2d.shape}")
        if semantic_labels is not None:
            unique_labels = np.unique(semantic_labels[semantic_labels >= 0]) if args.exclude_invalid else np.unique(semantic_labels)
            print(f"  Semantic classes: {len(unique_labels)}")
        print(f"  Scenes processed: {len(scene_info_dict)}")

    # Generate suffix for output files based on processing options
    suffix = f"_rank{args.rank}"
    if args.remove_outliers and not args.create_per_scene:
        suffix += f"_no_outliers_{args.outlier_method}"

    # Create main visualization (skip if only doing per-scene with outlier removal)
    if not (args.remove_outliers and args.create_per_scene):
        print("\nCreating comparison visualization...")
        main_output = output_dir / f"umap_comparison{suffix}.png"
        dataset_names = sorted(set(label.split('/')[0] for label in labels))
        create_visualization(original_2d, compressed_2d, labels, main_output, dataset_names, args.rank)

    # Create per-dataset visualizations if requested
    if args.create_per_dataset:
        print("\nCreating per-dataset visualizations...")
        create_per_dataset_visualization(original_2d, compressed_2d, labels, output_dir, args.rank)

    # Save UMAP embeddings (only if not loaded from npz)
    if not args.load_npz:
        save_dict = {
            'original': original_2d,
            'compressed': compressed_2d,
            'labels': np.array(labels, dtype='U')
        }
        # Save semantic labels if available
        if semantic_labels is not None and label_mapping is not None:
            save_dict['semantic_labels'] = semantic_labels
            # Convert label_mapping to a format that can be saved in npz
            save_dict['label_mapping'] = np.array([label_mapping], dtype=object)

        np.savez(output_dir / f"umap_embeddings{suffix}.npz", **save_dict)
        print(f"\nSaved UMAP embeddings to {output_dir / f'umap_embeddings{suffix}.npz'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
