#!/usr/bin/env python
"""
Verify if a unified projection matrix can preserve cross-scene semantic similarity.

This script tests whether using a SINGLE projection matrix (learned from all scenes)
can preserve semantic information as well as per-scene SVD compression.

Key experiment:
1. Learn a UNIFIED projection matrix from all training scenes
2. Project features from different scenes using this SAME matrix
3. Compare cross-scene similarity of projected features vs per-scene SVD

Expected outcome:
- If unified projection works: cross-scene similarity should be comparable (>0.95)
- If unified projection fails: cross-scene similarity will drop significantly (<0.90)
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


def load_scene_original_features(scene_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load original 768-dim features and segment labels from a scene.

    Returns:
        (original_features, segment_labels, indices, scene_name)
    """
    scene_name = Path(scene_path).name

    try:
        # Try to load original features (uncompressed)
        # Check for pre-compressed features
        svd_file = Path(scene_path) / "lang_feat_grid_svd_r16.npz"
        if not svd_file.exists():
            return None, None, None, scene_name

        # Load segment labels
        segment_file = None
        for name in ["segment_nyu_160.npy", "segment.npy"]:
            candidate = Path(scene_path) / name
            if candidate.exists():
                segment_file = candidate
                break

        segment = np.load(segment_file)

        # Check if original features exist
        original_file = Path(scene_path) / "lang_feat.npy"
        if original_file.exists():
            original = np.load(original_file)  # [num_points, 768]
        else:
            # Original features not available, use SVD reconstruction as proxy
            # This is not ideal but allows some testing
            svd_data = np.load(svd_file)
            compressed = svd_data["compressed"]
            indices = svd_data["indices"]

            # Reconstruct original features from SVD (will be approximate)
            # For accurate testing, we need actual original features
            print(f"Warning: Original features not found for {scene_name}, using SVD reconstruction")
            # Try to load V matrix for reconstruction
            return None, None, None, scene_name

        indices = np.arange(len(original))  # Point indices
        return original, segment, indices, scene_name

    except Exception as e:
        print(f"Warning: Failed to load {scene_path}: {e}")
        return None, None, None, scene_name


def compute_unified_projection(scenes: List[str], max_scenes: int = 50, max_samples: int = 100000) -> np.ndarray:
    """
    Compute a unified projection matrix from all scenes using PCA/SVD.

    This learns a SINGLE projection matrix that can be applied to any scene.

    Args:
        scenes: List of scene paths
        max_scenes: Maximum scenes to use for learning projection
        max_samples: Maximum total samples to use

    Returns:
        Projection matrix P of shape [768, 16]
    """
    print("\nComputing unified projection matrix from all scenes...")

    # Collect samples from all scenes
    all_features = []
    samples_per_scene = max_samples // max_scenes

    for i, scene_path in enumerate(tqdm(scenes[:max_scenes])):
        original, segment, indices, scene_name = load_scene_original_features(scene_path)
        if original is None:
            continue

        # Sample features
        if len(original) > samples_per_scene:
            idx = np.random.choice(len(original), samples_per_scene, replace=False)
            sampled = original[idx]
        else:
            sampled = original

        all_features.append(sampled)

        if len(all_features) * samples_per_scene >= max_samples:
            break

    if not all_features:
        raise ValueError("No features collected for projection learning")

    # Stack all features
    all_features = np.vstack(all_features)  # [N, 768]
    print(f"Collected {all_features.shape[0]} samples for projection learning")

    # Compute SVD to get projection matrix
    # Use truncated SVD to get top 16 components
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=16, random_state=42)
    svd.fit(all_features)

    # Projection matrix: [768, 16]
    # V^T from sklearn is [16, 768], so we transpose
    projection_matrix = svd.components_.T  # [768, 16]

    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"Unified projection matrix: {projection_matrix.shape}")
    print(f"Explained variance: {explained_variance:.4f}")

    return projection_matrix


def analyze_cross_scene_similarity_with_unified_projection(
    scenes: List[str],
    projection_matrix: np.ndarray,
    min_samples_per_class: int = 10,
) -> Dict[int, Dict]:
    """
    Analyze cross-scene similarity using a UNIFIED projection matrix.

    This tests whether the same projection matrix works across all scenes.
    """
    print("\nAnalyzing cross-scene similarity with UNIFIED projection...")

    # Load and project features from all scenes
    scene_category_means = []

    for scene_path in tqdm(scenes):
        original, segment, indices, scene_name = load_scene_original_features(scene_path)
        if original is None:
            continue

        # Project using UNIFIED matrix
        # original: [N, 768], projection_matrix: [768, 16]
        projected = original @ projection_matrix  # [N, 16]

        # Map segment labels to point level (indices should align)
        # For each point, get its segment label

        # Compute mean feature per category
        category_means = {}
        for cat_id in np.unique(segment):
            if cat_id < 0:
                continue
            mask = segment == cat_id
            cat_features = projected[mask]
            if len(cat_features) >= min_samples_per_class:
                category_means[cat_id] = np.mean(cat_features, axis=0)

        if category_means:
            scene_category_means.append((category_means, scene_name))

    print(f"Successfully loaded {len(scene_category_means)} scenes")

    # Find all unique categories
    all_categories = set()
    for cat_means, _ in scene_category_means:
        all_categories.update(cat_means.keys())

    print(f"Found {len(all_categories)} unique categories\n")

    # Compute cross-scene similarities
    from itertools import combinations

    results = {}

    for cat_id in sorted(all_categories):
        scenes_with_cat = [
            (cat_means[cat_id], scene_name)
            for cat_means, scene_name in scene_category_means
            if cat_id in cat_means
        ]

        num_scenes = len(scenes_with_cat)

        if num_scenes < 2:
            results[cat_id] = {
                "category_id": cat_id,
                "num_scenes": num_scenes,
                "num_scene_pairs": 0,
                "mean_similarity": np.nan,
            }
            continue

        # Compute pairwise similarities
        similarities = []
        for (mean_a, name_a), (mean_b, name_b) in combinations(scenes_with_cat, 2):
            sim = compute_cosine_similarity(mean_a, mean_b)
            similarities.append(sim)

        results[cat_id] = {
            "category_id": cat_id,
            "num_scenes": num_scenes,
            "num_scene_pairs": len(similarities),
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
        }

        print(f"Category {cat_id:3d}: {num_scenes:4d} scenes, {len(similarities):4d} pairs, "
              f"mean_sim={results[cat_id]['mean_similarity']:.4f}")

    return results


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def print_summary(results: Dict[int, Dict], title: str = "CROSS-SCENE SIMILARITY"):
    """Print summary of results."""
    print("\n" + "=" * 90)
    print(f"{title} (UNIFIED PROJECTION)")
    print("=" * 90)

    valid_results = [(k, v) for k, v in results.items() if not np.isnan(v["mean_similarity"])]
    sorted_results = sorted(valid_results, key=lambda x: x[1]["mean_similarity"], reverse=True)

    if valid_results:
        overall_mean = np.mean([v["mean_similarity"] for _, v in valid_results])
        print(f"\nOverall mean similarity: {overall_mean:.4f}")
        print(f"Total categories analyzed: {len(valid_results)}")
        print("=" * 90)


def save_results(results: Dict[int, Dict], output_path: str):
    """Save results to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "category_id", "num_scenes", "num_scene_pairs",
            "mean_similarity", "std_similarity", "min_similarity", "max_similarity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cat_id in sorted(results.keys()):
            # Only save rows with data
            row = {k: results[cat_id].get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify unified projection matrix for cross-scene feature compression"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/new_data/cyf/Datasets/SceneSplat7k/matterport3d/train_grid1.0cm_chunk6x6x4_stride4x4x4",
        help="Path to Matterport3D data directory (for learning projection)"
    )
    parser.add_argument(
        "--test-root",
        type=str,
        default="/new_data/cyf/Datasets/SceneSplat7k/matterport3d/val_grid1.0cm_chunk6x6x4_stride4x4x4",
        help="Path to test data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exp/unified_projection_similarity.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--max-scenes-projection",
        type=int,
        default=50,
        help="Maximum scenes to use for learning projection matrix"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples required per class per scene"
    )

    args = parser.parse_args()

    # Find scenes
    print(f"Searching for training scenes in: {args.data_root}")
    train_scenes = find_all_scenes(args.data_root)
    print(f"Found {len(train_scenes)} training scenes")

    print(f"\nSearching for test scenes in: {args.test_root}")
    test_scenes = find_all_scenes(args.test_root)
    print(f"Found {len(test_scenes)} test scenes")

    # Step 1: Learn unified projection matrix from training scenes
    try:
        projection_matrix = compute_unified_projection(
            train_scenes,
            max_scenes=args.max_scenes_projection,
            max_samples=100000,
        )

        # Save projection matrix for later use
        projection_path = args.output.replace(".csv", "_projection_matrix.npz")
        np.savez(projection_path, projection=projection_matrix)
        print(f"\nProjection matrix saved to: {projection_path}")

    except ValueError as e:
        print(f"\nError: {e}")
        print("\nNOTE: This verification requires original 768-dim features (lang_feat.npy).")
        print("If your data only has SVD-compressed features, you cannot test unified projection.")
        print("\nTo enable this test, you need:")
        print("  1. Original lang_feat.npy files (768-dim)")
        print("  2. Or reconstruct original features from per-scene SVD + V matrices")
        return

    # Step 2: Analyze cross-scene similarity on test set using unified projection
    results = analyze_cross_scene_similarity_with_unified_projection(
        scenes=test_scenes,
        projection_matrix=projection_matrix,
        min_samples_per_class=args.min_samples,
    )

    # Print and save results
    print_summary(results)
    save_results(results, args.output)

    print("\n" + "=" * 90)
    print("INTERPRETATION:")
    print("  - If mean_similarity > 0.95: Unified projection WORKS well")
    print("  - If mean_similarity < 0.90: Unified projection FAILS")
    print("  - Compare with per-scene SVD results to see the gap")
    print("=" * 90)


if __name__ == "__main__":
    main()
