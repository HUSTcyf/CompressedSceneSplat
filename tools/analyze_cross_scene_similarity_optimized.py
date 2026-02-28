#!/usr/bin/env python
"""
Optimized version of cross-scene feature similarity analysis for Matterport3D dataset.

Key optimizations for large-scale processing:
1. Smart sampling strategies to control computational cost
2. Batch processing to reduce memory pressure
3. Vectorized similarity computation
4. Checkpoint support for resumable computation

Usage:
    # Fast analysis with sampling (recommended for 4000+ scenes)
    python tools/analyze_cross_scene_similarity_optimized.py \
        --max-pairs-per-category 100 \
        --output exp/similarity_fast.csv

    # Full analysis (very slow, not recommended)
    python tools/analyze_cross_scene_similarity_optimized.py \
        --max-pairs-per-category 0

Output:
    - CSV file with per-category cross-scene similarity statistics
    - Console summary with progress tracking
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import pickle

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


def load_scene_data(scene_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Load compressed features and segment labels from a scene."""
    scene_name = Path(scene_path).name

    try:
        svd_file = Path(scene_path) / "lang_feat_grid_svd_r16.npz"
        svd_data = np.load(svd_file)
        compressed = svd_data["compressed"]
        indices = svd_data["indices"]

        segment_file = None
        for name in ["segment_nyu_160.npy", "segment.npy"]:
            candidate = Path(scene_path) / name
            if candidate.exists():
                segment_file = candidate
                break

        segment = np.load(segment_file)

        # Map segment labels from point-level to grid-level
        num_grids = compressed.shape[0]
        grid_segment = np.full(num_grids, -1, dtype=np.int32)

        # Vectorized aggregation using bincount
        for grid_id in range(num_grids):
            point_indices = np.where(indices == grid_id)[0]
            if len(point_indices) > 0:
                segments = segment[point_indices]
                valid_segments = segments[segments >= 0]
                if len(valid_segments) > 0:
                    values, counts = np.unique(valid_segments, return_counts=True)
                    grid_segment[grid_id] = values[np.argmax(counts)]

        return compressed, grid_segment, scene_name
    except Exception as e:
        print(f"Warning: Failed to load {scene_path}: {e}")
        return None, None, scene_name


def compute_cosine_similarity_batch(
    features_a_list: List[np.ndarray],
    features_b_list: List[np.ndarray],
    max_samples_per_pair: int = 1000
) -> List[float]:
    """
    Compute cosine similarities for multiple pairs efficiently.

    Args:
        features_a_list: List of feature arrays
        features_b_list: List of feature arrays
        max_samples_per_pair: Maximum samples to use per pair

    Returns:
        List of similarity scores
    """
    similarities = []

    for feat_a, feat_b in zip(features_a_list, features_b_list):
        # Sample to balance sizes
        min_size = min(len(feat_a), len(feat_b))
        sample_size = min(min_size, max_samples_per_pair)

        if sample_size < len(feat_a):
            idx_a = np.random.choice(len(feat_a), sample_size, replace=False)
            feat_a = feat_a[idx_a]
        if sample_size < len(feat_b):
            idx_b = np.random.choice(len(feat_b), sample_size, replace=False)
            feat_b = feat_b[idx_b]

        # Normalize features
        norm_a = np.linalg.norm(feat_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(feat_b, axis=1, keepdims=True)
        norm_a = np.where(norm_a > 1e-8, norm_a, 1.0)
        norm_b = np.where(norm_b > 1e-8, norm_b, 1.0)

        feat_a_norm = feat_a / norm_a
        feat_b_norm = feat_b / norm_b

        # Compute all pairwise similarities and take mean
        sim_matrix = feat_a_norm @ feat_b_norm.T
        similarities.append(float(np.mean(sim_matrix)))

    return similarities


def compute_category_similarity(
    scene_features_list: List[np.ndarray],
    max_pairs: int = 0,
    samples_per_scene: int = 500,
    batch_size: int = 50
) -> Tuple[List[float], int]:
    """
    Compute cross-scene similarities for a category with smart sampling.

    Args:
        scene_features_list: List of feature arrays for this category
        max_pairs: Maximum number of scene pairs (0 = all pairs)
        samples_per_scene: Maximum samples to take from each scene
        batch_size: Batch size for similarity computation

    Returns:
        (similarities, num_pairs_used)
    """
    num_scenes = len(scene_features_list)
    if num_scenes < 2:
        return [], 0

    # Sample features from each scene
    sampled_features = []
    for features in scene_features_list:
        if len(features) > samples_per_scene:
            idx = np.random.choice(len(features), samples_per_scene, replace=False)
            sampled_features.append(features[idx])
        else:
            sampled_features.append(features)

    # Generate all scene pairs
    all_pairs = list(combinations(range(num_scenes), 2))

    # Limit pairs if needed
    if max_pairs > 0 and len(all_pairs) > max_pairs:
        np.random.seed(42)
        pair_indices = np.random.choice(len(all_pairs), max_pairs, replace=False)
        selected_pairs = [all_pairs[i] for i in pair_indices]
    else:
        selected_pairs = all_pairs

    # Compute similarities in batches
    similarities = []
    for i in range(0, len(selected_pairs), batch_size):
        batch_pairs = selected_pairs[i:i + batch_size]
        feat_a_batch = [sampled_features[idx_a] for idx_a, _ in batch_pairs]
        feat_b_batch = [sampled_features[idx_b] for _, idx_b in batch_pairs]

        batch_sims = compute_cosine_similarity_batch(feat_a_batch, feat_b_batch)
        similarities.extend(batch_sims)

    return similarities, len(selected_pairs)


def analyze_cross_scene_similarity_optimized(
    scenes: List[str],
    min_samples_per_class: int = 10,
    max_pairs_per_category: int = 100,
    samples_per_scene: int = 500,
    batch_size: int = 50,
    checkpoint_path: Optional[str] = None,
) -> Dict[int, Dict]:
    """
    Optimized cross-scene similarity analysis.

    Args:
        scenes: List of scene directory paths
        min_samples_per_class: Minimum samples required per class per scene
        max_pairs_per_category: Max scene pairs per category (0 = all)
        samples_per_scene: Max samples to use from each scene
        batch_size: Batch size for similarity computation
        checkpoint_path: Path to save/load checkpoint

    Returns:
        Dictionary mapping category_id to statistics
    """
    # Try to load from checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
            start_scene_idx = checkpoint.get("next_scene_idx", 0)
            scene_category_features = checkpoint.get("scene_category_features", [])
            results = checkpoint.get("results", {})
        print(f"Resuming from scene {start_scene_idx}/{len(scenes)}")
    else:
        start_scene_idx = 0
        scene_category_features = []
        results = {}

    # Load scene data with progress tracking
    if start_scene_idx < len(scenes):
        print(f"\nLoading scene data (starting from {start_scene_idx})...")

        for scene_path in tqdm(scenes[start_scene_idx:], initial=start_scene_idx, total=len(scenes)):
            features, segments, scene_name = load_scene_data(scene_path)

            if features is None or segments is None:
                continue

            # Group features by category
            category_features = {}
            for cat_id in np.unique(segments):
                if cat_id < 0:
                    continue
                mask = segments == cat_id
                cat_features = features[mask]
                if len(cat_features) >= min_samples_per_class:
                    category_features[cat_id] = cat_features

            scene_category_features.append((category_features, scene_name))

            # Save checkpoint periodically
            if checkpoint_path and len(scene_category_features) % 100 == 0:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump({
                        "scene_category_features": scene_category_features,
                        "results": results,
                        "next_scene_idx": len(scene_category_features),
                    }, f)

        print(f"Successfully loaded {len(scene_category_features)} scenes")

    # Find all unique categories
    all_categories = set()
    for cat_features, _ in scene_category_features:
        all_categories.update(cat_features.keys())

    print(f"\nFound {len(all_categories)} unique categories across {len(scene_category_features)} scenes")

    # Compute similarities for each category
    for cat_id in sorted(all_categories):
        if cat_id in results:
            print(f"Category {cat_id}: Already computed, skipping")
            continue

        # Find scenes with this category
        cat_features_list = [
            cat_features[cat_id]
            for cat_features, _ in scene_category_features
            if cat_id in cat_features
        ]

        num_scenes = len(cat_features_list)
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
                "num_comparisons": 0,
            }
            continue

        print(f"\nAnalyzing category {cat_id} ({num_scenes} scenes)...")

        # Compute similarities
        similarities, num_pairs = compute_category_similarity(
            cat_features_list,
            max_pairs=max_pairs_per_category,
            samples_per_scene=samples_per_scene,
            batch_size=batch_size,
        )

        if similarities:
            results[cat_id] = {
                "category_id": cat_id,
                "category_name": NYU40_CLASS_NAMES[cat_id] if cat_id < len(NYU40_CLASS_NAMES) else f"class_{cat_id}",
                "num_scenes": num_scenes,
                "num_scene_pairs": num_pairs,
                "mean_similarity": float(np.mean(similarities)),
                "std_similarity": float(np.std(similarities)),
                "min_similarity": float(np.min(similarities)),
                "max_similarity": float(np.max(similarities)),
                "num_comparisons": len(similarities),
            }
            print(f"  Mean similarity: {results[cat_id]['mean_similarity']:.4f} "
                  f"(std: {results[cat_id]['std_similarity']:.4f})")
        else:
            results[cat_id] = {
                "category_id": cat_id,
                "category_name": NYU40_CLASS_NAMES[cat_id] if cat_id < len(NYU40_CLASS_NAMES) else f"class_{cat_id}",
                "num_scenes": num_scenes,
                "num_scene_pairs": 0,
                "mean_similarity": np.nan,
                "std_similarity": np.nan,
                "min_similarity": np.nan,
                "max_similarity": np.nan,
                "num_comparisons": 0,
            }

        # Save checkpoint after each category
        if checkpoint_path:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "scene_category_features": scene_category_features,
                    "results": results,
                    "next_scene_idx": len(scene_category_features),
                }, f)

    # Clean up checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    return results


def print_summary(results: Dict[int, Dict]):
    """Print summary of results."""
    print("\n" + "=" * 90)
    print("CROSS-SCENE FEATURE SIMILARITY SUMMARY")
    print("=" * 90)

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if not np.isnan(v["mean_similarity"])],
        key=lambda x: x[1]["mean_similarity"],
        reverse=True
    )

    print(f"\n{'Category ID':<12} {'Category Name':<25} {'Scenes':<8} {'Pairs':<8} {'Mean Sim':<10} {'Std':<10}")
    print("-" * 100)

    for cat_id, stats in sorted_results:
        print(f"{stats['category_id']:<12} {stats['category_name']:<25} "
              f"{stats['num_scenes']:<8} {stats['num_scene_pairs']:<8} "
              f"{stats['mean_similarity']:<10.4f} {stats['std_similarity']:<10.4f}")

    valid_results = [v for v in results.values() if not np.isnan(v["mean_similarity"])]
    if valid_results:
        overall_mean = np.mean([v["mean_similarity"] for v in valid_results])
        print("\n" + "=" * 90)
        print(f"Overall mean similarity across all categories: {overall_mean:.4f}")
        print("=" * 90)


def save_results(results: Dict[int, Dict], output_path: str):
    """Save results to CSV."""
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
        description="Optimized cross-scene feature similarity analysis"
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
        default="exp/cross_scene_similarity_optimized.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples required per class per scene"
    )
    parser.add_argument(
        "--max-pairs-per-category",
        type=int,
        default=100,
        help="Maximum scene pairs per category (0 = all pairs, NOT recommended for large datasets)"
    )
    parser.add_argument(
        "--samples-per-scene",
        type=int,
        default=500,
        help="Maximum samples to use from each scene"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for similarity computation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file path for resumable computation"
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

    # Print computational estimate
    if args.max_pairs_per_category > 0:
        print(f"\nComputational settings:")
        print(f"  Max pairs per category: {args.max_pairs_per_category}")
        print(f"  Samples per scene: {args.samples_per_scene}")
        print(f"  Estimated total pairs: ~{len(set([i for s in scenes for i in range(50)])) * args.max_pairs_per_category}")
    else:
        print(f"\nWARNING: max_pairs_per_category=0 will compute ALL pairs.")
        print(f"This may take hours for large datasets!")

    # Run analysis
    results = analyze_cross_scene_similarity_optimized(
        scenes=scenes,
        min_samples_per_class=args.min_samples,
        max_pairs_per_category=args.max_pairs_per_category,
        samples_per_scene=args.samples_per_scene,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
    )

    # Print and save results
    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
