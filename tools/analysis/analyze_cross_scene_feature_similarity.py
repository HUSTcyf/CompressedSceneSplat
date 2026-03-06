#!/usr/bin/env python
"""
Analyze cross-scene feature similarity for Matterport3D dataset.

This script loads SVD-compressed language features (lang_feat_grid_svd_r16.npz)
from multiple scenes and computes the average similarity between features of
the same semantic category across different scenes.

Usage:
    python tools/analyze_cross_scene_feature_similarity.py

Output:
    - CSV file with per-category cross-scene similarity statistics
    - Console summary of results
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


# NYU40 class names for reference (0-39, with -1 for unlabeled)
NYU40_CLASS_NAMES = [
    "void",  # 0
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "desk",
    "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
]
# Add remaining classes (20-39)
NYU40_CLASS_NAMES += [
    "unknown_20", "unknown_21", "unknown_22", "unknown_23", "unknown_24",
    "unknown_25", "unknown_26", "unknown_27", "unknown_28", "unknown_29",
    "unknown_30", "unknown_31", "unknown_32", "unknown_33", "unknown_34",
    "unknown_35", "unknown_36", "unknown_37", "unknown_38", "unknown_39",
]


def find_all_scenes(data_root: str) -> List[str]:
    """Find all scene directories containing SVD files."""
    scenes = []
    for scene_dir in Path(data_root).iterdir():
        if scene_dir.is_dir():
            svd_file = scene_dir / "lang_feat_grid_svd_r16.npz"
            segment_file = None
            for name in ["segment_nyu_160.npy", "segment.npy"]:
                candidate = scene_dir / name
                if candidate.exists():
                    segment_file = candidate
                    break
            if svd_file.exists() and segment_file:
                scenes.append(str(scene_dir))
    return scenes


def load_scene_data(scene_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load compressed features and segment labels from a scene."""
    scene_name = Path(scene_path).name

    # Load SVD compressed features
    svd_file = Path(scene_path) / "lang_feat_grid_svd_r16.npz"
    svd_data = np.load(svd_file)
    compressed = svd_data["compressed"]  # [num_grids, 16]
    indices = svd_data["indices"]  # [num_points] - maps points to grids

    # Load segment labels
    segment_file = None
    for name in ["segment_nyu_160.npy", "segment.npy"]:
        candidate = Path(scene_path) / name
        if candidate.exists():
            segment_file = candidate
            break

    segment = np.load(segment_file)  # [num_points]

    # Map segment labels from point-level to grid-level
    # For each grid, use the most common segment among its points
    num_grids = compressed.shape[0]
    grid_segment = np.full(num_grids, -1, dtype=np.int32)

    for grid_id in range(num_grids):
        point_indices = np.where(indices == grid_id)[0]
        if len(point_indices) > 0:
            # Get most common non-negative segment
            segments = segment[point_indices]
            valid_segments = segments[segments >= 0]
            if len(valid_segments) > 0:
                # Use mode
                values, counts = np.unique(valid_segments, return_counts=True)
                grid_segment[grid_id] = values[np.argmax(counts)]

    return compressed, grid_segment, scene_name


def compute_cosine_similarity(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Compute average pairwise cosine similarity between two sets of features."""
    # Normalize features
    norm_a = np.linalg.norm(features_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(features_b, axis=1, keepdims=True)

    # Avoid division by zero
    norm_a = np.where(norm_a > 1e-8, norm_a, 1.0)
    norm_b = np.where(norm_b > 1e-8, norm_b, 1.0)

    features_a_norm = features_a / norm_a
    features_b_norm = features_b / norm_b

    # Compute all pairwise similarities
    similarities = features_a_norm @ features_b_norm.T

    # Return mean similarity
    return float(np.mean(similarities))


def analyze_cross_scene_similarity(
    scenes: List[str],
    min_samples_per_class: int = 10,
    max_scene_pairs: int = None,
    svd_rank: int = 16,
) -> Dict[int, Dict]:
    """
    Analyze cross-scene feature similarity for each semantic category.

    Args:
        scenes: List of scene directory paths
        min_samples_per_class: Minimum samples required per class per scene
        max_scene_pairs: Maximum number of scene pairs to sample (None = all pairs)
        svd_rank: SVD compression rank

    Returns:
        Dictionary mapping category_id to statistics dictionary
    """
    # Load all scene data
    print("Loading scene data...")
    scene_data_list = []
    for scene_path in tqdm(scenes):
        try:
            features, segments, scene_name = load_scene_data(scene_path)
            scene_data_list.append((features, segments, scene_name))
        except Exception as e:
            print(f"Warning: Failed to load {scene_path}: {e}")
            continue

    print(f"Successfully loaded {len(scene_data_list)} scenes")

    # Group features by category for each scene
    scene_category_features = []
    for features, segments, scene_name in scene_data_list:
        category_features = {}
        for cat_id in np.unique(segments):
            if cat_id < 0:  # Skip invalid/void
                continue
            mask = segments == cat_id
            cat_features = features[mask]
            if len(cat_features) >= min_samples_per_class:
                category_features[cat_id] = cat_features
        scene_category_features.append((category_features, scene_name))

    # Compute cross-scene similarities for each category
    results = {}

    # First, find all unique categories across scenes
    all_categories = set()
    for cat_features, _ in scene_category_features:
        all_categories.update(cat_features.keys())

    print(f"\nFound {len(all_categories)} unique categories across all scenes")

    # For each category, compute cross-scene similarities
    for cat_id in sorted(all_categories):
        print(f"\nAnalyzing category {cat_id}...")

        # Find scenes that have this category
        scene_indices = [
            i for i, (cat_features, _) in enumerate(scene_category_features)
            if cat_id in cat_features
        ]

        if len(scene_indices) < 2:
            print(f"  Category {cat_id}: Only found in {len(scene_indices)} scene(s), skipping")
            results[cat_id] = {
                "category_id": cat_id,
                "category_name": NYU40_CLASS_NAMES[cat_id] if cat_id < len(NYU40_CLASS_NAMES) else f"class_{cat_id}",
                "num_scenes": len(scene_indices),
                "num_scene_pairs": 0,
                "mean_similarity": np.nan,
                "std_similarity": np.nan,
                "min_similarity": np.nan,
                "max_similarity": np.nan,
                "num_comparisons": 0,
            }
            continue

        # Sample scene pairs if needed
        from itertools import combinations
        pairs = list(combinations(scene_indices, 2))
        if max_scene_pairs and len(pairs) > max_scene_pairs:
            np.random.seed(42)
            pair_indices = np.random.choice(len(pairs), max_scene_pairs, replace=False)
            pairs = [pairs[idx] for idx in pair_indices]

        similarities = []
        for i, j in tqdm(pairs, desc=f"  Cat {cat_id}", leave=False):
            features_a = scene_category_features[i][0][cat_id]
            features_b = scene_category_features[j][0][cat_id]

            # Sample to balance sizes if needed
            min_size = min(len(features_a), len(features_b))
            if min_size > 1000:
                # Sample 1000 from each
                idx_a = np.random.choice(len(features_a), 1000, replace=False)
                idx_b = np.random.choice(len(features_b), 1000, replace=False)
                features_a = features_a[idx_a]
                features_b = features_b[idx_b]

            sim = compute_cosine_similarity(features_a, features_b)
            similarities.append(sim)

        results[cat_id] = {
            "category_id": cat_id,
            "category_name": NYU40_CLASS_NAMES[cat_id] if cat_id < len(NYU40_CLASS_NAMES) else f"class_{cat_id}",
            "num_scenes": len(scene_indices),
            "num_scene_pairs": len(similarities),
            "mean_similarity": float(np.mean(similarities)) if similarities else np.nan,
            "std_similarity": float(np.std(similarities)) if similarities else np.nan,
            "min_similarity": float(np.min(similarities)) if similarities else np.nan,
            "max_similarity": float(np.max(similarities)) if similarities else np.nan,
            "num_comparisons": len(similarities),
        }

        print(f"  Category {cat_id} ({results[cat_id]['category_name']}): "
              f"{len(scene_indices)} scenes, "
              f"mean similarity = {results[cat_id]['mean_similarity']:.4f}")

    return results


def print_summary(results: Dict[int, Dict]):
    """Print summary of results to console."""
    print("\n" + "=" * 80)
    print("CROSS-SCENE FEATURE SIMILARITY SUMMARY")
    print("=" * 80)

    # Sort by mean similarity
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if not np.isnan(v["mean_similarity"])],
        key=lambda x: x[1]["mean_similarity"],
        reverse=True
    )

    print(f"\n{'Category ID':<12} {'Category Name':<25} {'Scenes':<8} {'Pairs':<8} {'Mean Sim':<10} {'Std':<10}")
    print("-" * 90)

    for cat_id, stats in sorted_results:
        print(f"{stats['category_id']:<12} {stats['category_name']:<25} "
              f"{stats['num_scenes']:<8} {stats['num_scene_pairs']:<8} "
              f"{stats['mean_similarity']:<10.4f} {stats['std_similarity']:<10.4f}")

    # Overall statistics
    valid_results = [v for v in results.values() if not np.isnan(v["mean_similarity"])]
    if valid_results:
        overall_mean = np.mean([v["mean_similarity"] for v in valid_results])
        print("\n" + "=" * 80)
        print(f"Overall mean similarity across all categories: {overall_mean:.4f}")
        print("=" * 80)


def save_results(results: Dict[int, Dict], output_path: str):
    """Save results to CSV file."""
    import csv

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "category_id", "category_name", "num_scenes", "num_scene_pairs",
            "mean_similarity", "std_similarity", "min_similarity", "max_similarity",
            "num_comparisons"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cat_id in sorted(results.keys()):
            writer.writerow(results[cat_id])

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cross-scene feature similarity for Matterport3D dataset"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/new_data/cyf/Datasets/SceneSplat7k/matterport3d/train_grid1.0cm_chunk6x6x4_stride4x4x4",
        help="Path to Matterport3D data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exp/cross_scene_similarity_matterport3d.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples required per class per scene"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of scene pairs to sample per category (None = all pairs)"
    )
    parser.add_argument(
        "--svd-rank",
        type=int,
        default=16,
        help="SVD compression rank"
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process (None = all)"
    )

    args = parser.parse_args()

    # Find all scenes
    print(f"Searching for scenes in: {args.data_root}")
    scenes = find_all_scenes(args.data_root)
    print(f"Found {len(scenes)} scenes")

    if args.num_scenes:
        scenes = scenes[:args.num_scenes]
        print(f"Limiting to first {args.num_scenes} scenes")

    # Analyze cross-scene similarity
    results = analyze_cross_scene_similarity(
        scenes=scenes,
        min_samples_per_class=args.min_samples,
        max_scene_pairs=args.max_pairs,
        svd_rank=args.svd_rank,
    )

    # Print summary
    print_summary(results)

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
