#!/usr/bin/env python
"""
Verify if SVD-compressed features are consistent across different datasets.

This provides INDIRECT evidence for whether a unified projection matrix would work:

If features from different datasets (trained separately) have similar distributions
and high cross-dataset similarity for same categories, it suggests that a unified
projection is feasible.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import csv

import numpy as np
from tqdm import tqdm


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


def load_scene_compressed_features(scene_path: str, min_samples_per_class: int = 10) -> Tuple[Dict[int, np.ndarray], str]:
    """Load SVD-compressed features and compute category means."""
    scene_name = Path(scene_path).name

    try:
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
                category_means[cat_id] = {
                    "mean": np.mean(cat_features, axis=0),
                    "std": np.std(cat_features, axis=0),
                    "count": len(cat_features),
                }

        return category_means, scene_name

    except Exception as e:
        return {}, scene_name


def analyze_feature_statistics_by_dataset(
    data_root: str,
    dataset_name: str,
    min_samples_per_class: int = 10,
    max_scenes: int = None,
) -> Dict:
    """Analyze feature distribution statistics for a dataset."""
    scenes = find_all_scenes(data_root)
    if max_scenes:
        scenes = scenes[:max_scenes]
        print(f"\n[{dataset_name}] Using {len(scenes)} scenes (limited from {len(find_all_scenes(data_root))} total)")
    else:
        print(f"\n[{dataset_name}] Found {len(scenes)} scenes")

    # Collect all features and category statistics
    all_features = []
    category_stats = {}

    for scene_path in tqdm(scenes, desc=f"Loading {dataset_name}"):
        cat_means, scene_name = load_scene_compressed_features(scene_path, min_samples_per_class)

        for cat_id, stats in cat_means.items():
            if cat_id not in category_stats:
                category_stats[cat_id] = []
            category_stats[cat_id].append({
                "scene": scene_name,
                "mean": stats["mean"],
                "std": stats["std"],
                "count": stats["count"],
            })
            all_features.append(stats["mean"])

    if all_features:
        all_features = np.vstack(all_features)  # [N, 16]

        # Compute global statistics
        global_mean = np.mean(all_features, axis=0)
        global_std = np.std(all_features, axis=0)
        global_norm = np.linalg.norm(all_features, axis=1).mean()

        # Per-dimension statistics
        dim_means = np.mean(all_features, axis=0)
        dim_stds = np.std(all_features, axis=0)

        print(f"\n[{dataset_name}] Feature Statistics:")
        print(f"  Total category instances: {len(all_features)}")
        print(f"  Global mean norm: {np.linalg.norm(global_mean):.4f}")
        print(f"  Avg feature norm: {global_norm:.4f}")
        print(f"  Dimension-wise mean: {dim_means[:4]} ... (showing first 4)")
        print(f"  Dimension-wise std: {dim_stds[:4]} ... (showing first 4)")

        return {
            "dataset_name": dataset_name,
            "num_scenes": len(scenes),
            "num_categories": len(category_stats),
            "total_instances": len(all_features),
            "global_mean": global_mean,
            "global_std": global_std,
            "dim_means": dim_means,
            "dim_stds": dim_stds,
            "category_stats": category_stats,
        }

    return None


def compute_cross_dataset_similarity(
    stats_a: Dict,
    stats_b: Dict,
    min_common_categories: int = 10,
) -> Dict:
    """
    Compute similarity between two datasets' compressed feature spaces.

    This tests if features from independently compressed datasets are compatible.
    """
    print(f"\nComputing cross-dataset similarity:")
    print(f"  Dataset A: {stats_a['dataset_name']} ({stats_a['num_scenes']} scenes)")
    print(f"  Dataset B: {stats_b['dataset_name']} ({stats_b['num_scenes']} scenes)")

    # Find common categories
    common_cats = set(stats_a["category_stats"].keys()) & set(stats_b["category_stats"].keys())

    print(f"  Common categories: {len(common_cats)}")

    if len(common_cats) < min_common_categories:
        print(f"  Not enough common categories for analysis")
        return {}

    # Compute similarities for common categories
    from itertools import product

    similarities = []
    category_similarities = {}

    for cat_id in sorted(common_cats):
        # Get means from both datasets
        means_a = [s["mean"] for s in stats_a["category_stats"][cat_id]]
        means_b = [s["mean"] for s in stats_b["category_stats"][cat_id]]

        # Compute cross-dataset similarity
        cat_sims = []
        for mean_a in means_a:
            for mean_b in means_b:
                sim = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b) + 1e-8)
                cat_sims.append(sim)
                similarities.append(sim)

        category_similarities[cat_id] = {
            "mean_similarity": float(np.mean(cat_sims)),
            "std_similarity": float(np.std(cat_sims)),
            "num_pairs": len(cat_sims),
        }

    # Overall statistics
    overall_mean = float(np.mean(similarities))
    overall_std = float(np.std(similarities))

    print(f"\n  Cross-dataset similarity results:")
    print(f"    Mean similarity: {overall_mean:.4f} ± {overall_std:.4f}")
    print(f"    Total comparisons: {len(similarities)}")
    print(f"    Categories analyzed: {len(category_similarities)}")

    # Interpretation
    print(f"\n  Interpretation:")
    if overall_mean > 0.95:
        print(f"    ✓ Similarity > 0.95: Features are CONSISTENT across datasets")
        print(f"      → Suggests unified projection matrix is VIABLE")
    elif overall_mean > 0.90:
        print(f"    ~ Similarity 0.90-0.95: Features are MODERATELY consistent")
        print(f"      → Unified projection might work with some degradation")
    else:
        print(f"    ✗ Similarity < 0.90: Features are INCONSISTENT")
        print(f"      → Unified projection matrix is NOT recommended")

    return {
        "dataset_a": stats_a["dataset_name"],
        "dataset_b": stats_b["dataset_name"],
        "mean_similarity": overall_mean,
        "std_similarity": overall_std,
        "num_comparisons": len(similarities),
        "num_categories": len(category_similarities),
        "category_similarities": category_similarities,
    }


def analyze_feature_distribution_alignment(stats_a: Dict, stats_b: Dict) -> Dict:
    """
    Analyze if the feature distributions are aligned across datasets.

    This checks the mean and variance of each dimension.
    """
    print(f"\nAnalyzing feature distribution alignment:")

    # Compare dimension-wise means
    mean_diff = np.abs(stats_a["dim_means"] - stats_b["dim_means"])
    mean_cosine = np.dot(stats_a["dim_means"], stats_b["dim_means"]) / (
        np.linalg.norm(stats_a["dim_means"]) * np.linalg.norm(stats_b["dim_means"]) + 1e-8
    )

    # Compare dimension-wise stds
    std_diff = np.abs(stats_a["dim_stds"] - stats_b["dim_stds"])

    print(f"  Dimension-wise mean difference: {np.mean(mean_diff):.4f} (max: {np.max(mean_diff):.4f})")
    print(f"  Dimension-wise mean cosine similarity: {mean_cosine:.4f}")
    print(f"  Dimension-wise std difference: {np.mean(std_diff):.4f} (max: {np.max(std_diff):.4f})")

    return {
        "mean_diff": float(np.mean(mean_diff)),
        "max_mean_diff": float(np.max(mean_diff)),
        "mean_cosine": float(mean_cosine),
        "std_diff": float(np.mean(std_diff)),
    }


def save_cross_dataset_results(results: Dict, output_path: str):
    """Save cross-dataset similarity results to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save per-category similarities
    with open(output_path, "w", newline="") as f:
        fieldnames = ["category_id", "mean_similarity", "std_similarity", "num_pairs"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cat_id, stats in results["category_similarities"].items():
            writer.writerow({
                "category_id": cat_id,
                **stats,
            })

    print(f"\nPer-category results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify SVD-compressed feature consistency across datasets"
    )
    parser.add_argument(
        "--data-root-a",
        type=str,
        default="/new_data/cyf/Datasets/SceneSplat7k/matterport3d/train_grid1.0cm_chunk6x6x4_stride4x4x4",
        help="Path to first dataset"
    )
    parser.add_argument(
        "--data-root-b",
        type=str,
        default="/new_data/cyf/Datasets/SceneSplat7k/matterport3d/val_grid1.0cm_chunk6x6x4_stride4x4x4",
        help="Path to second dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exp/cross_dataset_similarity.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples required per class per scene"
    )
    parser.add_argument(
        "--max-scenes-a",
        type=int,
        default=200,
        help="Maximum scenes to analyze from dataset A"
    )
    parser.add_argument(
        "--max-scenes-b",
        type=int,
        default=None,
        help="Maximum scenes to analyze from dataset B (None = all)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CROSS-DATASET FEATURE CONSISTENCY ANALYSIS")
    print("=" * 80)
    print("\nThis analysis tests if SVD-compressed features from different datasets")
    print("are consistent, which provides INDIRECT evidence for unified projection.")

    # Analyze each dataset
    stats_a = analyze_feature_statistics_by_dataset(
        args.data_root_a,
        "Dataset A (Train)",
        args.min_samples,
        args.max_scenes_a,
    )

    stats_b = analyze_feature_statistics_by_dataset(
        args.data_root_b,
        "Dataset B (Val)",
        args.min_samples,
        args.max_scenes_b,
    )

    if not stats_a or not stats_b:
        print("\nError: Could not load features from one or both datasets")
        return

    # Compare feature distributions
    alignment = analyze_feature_distribution_alignment(stats_a, stats_b)

    # Compute cross-dataset similarity
    cross_sim = compute_cross_dataset_similarity(stats_a, stats_b)

    if cross_sim:
        save_cross_dataset_results(cross_sim, args.output)

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("\nCurrent result (per-scene SVD, train split): 0.9896")
    print("\nIf cross-dataset similarity is also > 0.95:")
    print("  → Different datasets produce CONSISTENT compressed features")
    print("  → Suggests a UNIFIED projection matrix could work")
    print("\nHowever, this is INDIRECT evidence. To definitively prove:")
    print("  1. Need original 768-dim features")
    print("  2. Learn ONE projection matrix from training data")
    print("  3. Apply to test data and measure similarity")
    print("=" * 80)


if __name__ == "__main__":
    main()
