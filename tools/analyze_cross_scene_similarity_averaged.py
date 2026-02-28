#!/usr/bin/env python
"""
Cross-scene feature similarity analysis using averaged category features.

This version computes the mean feature vector for each category in each scene,
then computes cosine similarity between these mean vectors across scenes.

Advantages:
- No sampling needed - deterministic results
- Much faster - O(N) instead of O(N^2)
- Consistent and reproducible

Usage:
    python tools/analyze_cross_scene_similarity_averaged.py
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv

import numpy as np
from tqdm import tqdm


# NYU40 class names for reference
NYU40_CLASS_NAMES = [
    "void", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "desk",
    "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
]
NYU40_CLASS_NAMES += [f"unknown_{i}" for i in range(20, 40)]
NYU40_CLASS_NAMES += [f"class_{i}" for i in range(40, 200)]


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


def load_scene_category_features(
    scene_path: str,
    min_samples_per_class: int = 1
) -> Tuple[Dict[int, np.ndarray], str]:
    """
    Load scene data and compute mean feature vector for each category.

    Args:
        scene_path: Path to scene directory
        min_samples_per_class: Minimum samples required to compute mean

    Returns:
        (category_to_mean_features, scene_name)
        - category_to_mean_features: Dict mapping category_id to mean feature vector
        - scene_name: Name of the scene
    """
    scene_name = Path(scene_path).name

    try:
        # Load SVD compressed features
        svd_file = Path(scene_path) / "lang_feat_grid_svd_r16.npz"
        svd_data = np.load(svd_file)
        compressed = svd_data["compressed"]  # [num_grids, 16]
        indices = svd_data["indices"]  # [num_points]

        # Load segment labels
        segment_file = None
        for name in ["segment_nyu_160.npy", "segment.npy"]:
            candidate = Path(scene_path) / name
            if candidate.exists():
                segment_file = candidate
                break

        segment = np.load(segment_file)  # [num_points]

        # Map segment labels from point-level to grid-level
        num_grids = compressed.shape[0]
        grid_segment = np.full(num_grids, -1, dtype=np.int32)

        for grid_id in range(num_grids):
            point_indices = np.where(indices == grid_id)[0]
            if len(point_indices) > 0:
                segments = segment[point_indices]
                valid_segments = segments[segments >= 0]
                if len(valid_segments) > 0:
                    values, counts = np.unique(valid_segments, return_counts=True)
                    grid_segment[grid_id] = values[np.argmax(counts)]

        # Compute mean feature vector for each category
        category_means = {}
        for cat_id in np.unique(grid_segment):
            if cat_id < 0:
                continue
            mask = grid_segment == cat_id
            cat_features = compressed[mask]
            if len(cat_features) >= min_samples_per_class:
                category_means[cat_id] = np.mean(cat_features, axis=0)

        return category_means, scene_name

    except Exception as e:
        print(f"Warning: Failed to load {scene_path}: {e}")
        return {}, scene_name


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def analyze_cross_scene_similarity_averaged(
    scenes: List[str],
    min_samples_per_class: int = 1,
) -> Dict[int, Dict]:
    """
    Analyze cross-scene similarity using averaged category features.

    Args:
        scenes: List of scene directory paths
        min_samples_per_class: Minimum samples required to compute mean

    Returns:
        Dictionary mapping category_id to statistics
    """
    # Load all scenes and compute mean features
    print("\nLoading scenes and computing mean features...")
    scene_category_means = []
    for scene_path in tqdm(scenes):
        category_means, scene_name = load_scene_category_features(
            scene_path, min_samples_per_class
        )
        if category_means:
            scene_category_means.append((category_means, scene_name))

    print(f"Successfully loaded {len(scene_category_means)} scenes")

    # Find all unique categories
    all_categories = set()
    for cat_means, _ in scene_category_means:
        all_categories.update(cat_means.keys())

    print(f"Found {len(all_categories)} unique categories across all scenes\n")

    # For each category, compute cross-scene similarities
    results = {}

    for cat_id in sorted(all_categories):
        # Find scenes that have this category
        scenes_with_cat = [
            (cat_means[cat_id], scene_name)
            for cat_means, scene_name in scene_category_means
            if cat_id in cat_means
        ]

        num_scenes = len(scenes_with_cat)

        if num_scenes < 2:
            results[cat_id] = {
                "category_id": cat_id,
                "category_name": NYU40_CLASS_NAMES[cat_id] if cat_id < len(NYU40_CLASS_NAMES) else f"class_{cat_id}",
                "num_scenes": num_scenes,
                "num_scene_pairs": 0,
                "mean_similarity": np.nan,
                "std_similarity": np.nan,
                "min_similarity": np.nan,
                "max_similarity": np.nan,
            }
            continue

        # Compute all pairwise similarities
        from itertools import combinations
        similarities = []

        for (mean_a, name_a), (mean_b, name_b) in combinations(scenes_with_cat, 2):
            sim = compute_cosine_similarity(mean_a, mean_b)
            similarities.append(sim)

        results[cat_id] = {
            "category_id": cat_id,
            "category_name": NYU40_CLASS_NAMES[cat_id] if cat_id < len(NYU40_CLASS_NAMES) else f"class_{cat_id}",
            "num_scenes": num_scenes,
            "num_scene_pairs": len(similarities),
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
        }

        print(f"Category {cat_id:3d} ({results[cat_id]['category_name']:20s}): "
              f"{num_scenes:4d} scenes, {len(similarities):4d} pairs, "
              f"mean_sim={results[cat_id]['mean_similarity']:.4f}")

    return results


def print_summary(results: Dict[int, Dict]):
    """Print summary of results."""
    print("\n" + "=" * 100)
    print("CROSS-SCENE FEATURE SIMILARITY SUMMARY (Averaged Features)")
    print("=" * 100)

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if not np.isnan(v["mean_similarity"])],
        key=lambda x: x[1]["mean_similarity"],
        reverse=True
    )

    print(f"\n{'Category ID':<12} {'Category Name':<25} {'Scenes':<8} {'Pairs':<10} {'Mean Sim':<10} {'Std':<10}")
    print("-" * 110)

    for cat_id, stats in sorted_results:
        print(f"{stats['category_id']:<12} {stats['category_name']:<25} "
              f"{stats['num_scenes']:<8} {stats['num_scene_pairs']:<10} "
              f"{stats['mean_similarity']:<10.4f} {stats['std_similarity']:<10.4f}")

    valid_results = [v for v in results.values() if not np.isnan(v["mean_similarity"])]
    if valid_results:
        overall_mean = np.mean([v["mean_similarity"] for v in valid_results])
        print("\n" + "=" * 100)
        print(f"Overall mean similarity across all categories: {overall_mean:.4f}")
        print(f"Total categories analyzed: {len(valid_results)}")
        print("=" * 100)


def save_results(results: Dict[int, Dict], output_path: str):
    """Save results to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "category_id", "category_name", "num_scenes", "num_scene_pairs",
            "mean_similarity", "std_similarity", "min_similarity", "max_similarity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cat_id in sorted(results.keys()):
            writer.writerow(results[cat_id])

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-scene similarity analysis using averaged features (fast, deterministic)"
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
        default="exp/cross_scene_similarity_averaged.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples required per class per scene"
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

    # Run analysis
    results = analyze_cross_scene_similarity_averaged(
        scenes=scenes,
        min_samples_per_class=args.min_samples,
    )

    # Print and save results
    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
