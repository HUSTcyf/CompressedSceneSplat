#!/usr/bin/env python
"""
Learn a global projection matrix from all training scenes.

This projects 768-dim text embeddings to 16-dim SVD space.
The projection matrix can be used at inference time for open-vocabulary classification.

Usage:
    python tools/learn_global_projection.py --rank 16 --output exp/global_projection_r16.npz
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm


def find_all_scenes(data_roots: List[str]) -> List[str]:
    """Find all scene directories containing SVD files."""
    scenes = []
    for data_root in data_roots:
        for scene_dir in Path(data_root).iterdir():
            if scene_dir.is_dir():
                svd_file = scene_dir / "lang_feat_grid_svd_r16.npz"
                # Check for original features
                original_file = scene_dir / "lang_feat.npy"
                if svd_file.exists() and original_file.exists():
                    scenes.append(str(scene_dir))
    return scenes


def load_scene_features(scene_path: str, max_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load original 768-dim features and corresponding 16-dim compressed features.

    Returns:
        (original_features, compressed_features)
    """
    try:
        # Load compressed features
        svd_file = Path(scene_path) / "lang_feat_grid_svd_r16.npz"
        if not svd_file.exists():
            return None, None

        svd_data = np.load(svd_file)
        compressed = svd_data["compressed"]  # [num_grids, 16]
        indices = svd_data["indices"]  # [num_points] - maps each point to a grid_id

        # Load original features
        original_file = Path(scene_path) / "lang_feat.npy"
        if not original_file.exists():
            return None, None

        original = np.load(original_file)  # [num_points, 768]

        # Match compressed grids to original points
        # indices[i] = grid_id for point i
        # compressed[grid_id] = 16-dim feature for that grid

        num_grids = compressed.shape[0]
        num_points = original.shape[0]

        if num_grids > max_samples:
            # Sample grids
            grid_idx = np.random.choice(num_grids, max_samples, replace=False)
            compressed_sample = compressed[grid_idx]

            # Find points belonging to sampled grids
            mask = np.isin(indices, grid_idx)
            original_sample = original[mask]
        else:
            compressed_sample = compressed
            original_sample = original

        # Match dimensions by aggregating points to grids
        # For each grid, aggregate original features from its points
        grid_original_features = []
        for grid_id in range(compressed_sample.shape[0]):
            # Find points in this grid
            if grid_id < len(compressed):
                # In compressed space, this grid_id exists
                # Find all points mapped to this grid
                point_mask = indices == grid_id
                if point_mask.sum() > 0:
                    grid_original_features.append(original[point_mask].mean(axis=0))
                else:
                    # No points for this grid, skip
                    continue

        if not grid_original_features:
            return None, None

        grid_original_features = np.array(grid_original_features)

        # Trim to match sizes
        min_size = min(len(compressed_sample), len(grid_original_features))
        if min_size == 0:
            return None, None

        return grid_original_features[:min_size], compressed_sample[:min_size]

    except Exception as e:
        print(f"Error loading {scene_path}: {e}")
        return None, None


def learn_global_projection(
    scenes: List[str],
    rank: int = 16,
    max_scenes: int = 100,
    max_samples_per_scene: int = 5000,
    total_samples: int = 100000,
) -> np.ndarray:
    """
    Learn a global projection matrix from all scenes.

    Method: Ridge Regression to find P that minimizes ||P^T @ X - Y||^2
    where X is 768-dim original features and Y is 16-dim compressed features.

    Returns:
        Projection matrix P of shape [768, rank]
    """
    print(f"\nLearning global projection matrix (rank={rank})...")
    print(f"Using up to {max_scenes} scenes, {total_samples} total samples")

    # Collect samples
    original_features = []
    compressed_features = []

    samples_per_scene = total_samples // min(max_scenes, len(scenes))

    for scene_path in tqdm(scenes[:max_scenes], desc="Loading scenes"):
        orig, comp = load_scene_features(scene_path, max_samples_per_scene)

        if orig is None or comp is None:
            continue

        # Sample to balance
        if len(orig) > samples_per_scene:
            idx = np.random.choice(len(orig), samples_per_scene, replace=False)
            orig = orig[idx]
            comp = comp[idx]

        original_features.append(orig)
        compressed_features.append(comp)

        if len(original_features) * samples_per_scene >= total_samples:
            break

    if not original_features:
        raise ValueError("No valid scenes found with original features!")

    # Stack all features
    X = np.vstack(original_features)  # [N, 768]
    Y = np.vstack(compressed_features)  # [N, rank]

    print(f"Collected {X.shape[0]} samples")
    print(f"  Original features: {X.shape}")
    print(f"  Compressed features: {Y.shape}")

    # Method 1: Ridge Regression (find P that minimizes ||P^T @ X - Y||^2)
    # Solution: P = X^T @ (X @ X^T + λI)^(-1) @ Y^T
    # Or equivalently: P^T = Y @ X^T @ (X @ X^T + λI)^(-1)

    print("\nSolving for projection matrix using Ridge Regression...")

    # Center the data
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Compute X @ X^T (N x N matrix)
    N = X.shape[0]
    if N > 50000:
        # Use incremental SVD for large datasets
        from sklearn.decomposition import TruncatedSVD

        # Reduce dimensionality first
        print("  Using incremental SVD for large dataset...")
        n_components = min(512, N // 10)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_reduced = svd.fit_transform(X_centered)  # [N, k]

        # Solve in reduced space
        # P_k = (X_k^T @ X_k)^(-1) @ X_k^T @ Y
        XXt_reduced = X_reduced.T @ X_reduced  # [k, k]
        XTY_reduced = X_reduced.T @ Y_centered  # [k, rank]

        # Add regularization
        lambda_reg = 0.1
        P_reduced = np.linalg.solve(XXt_reduced + lambda_reg * np.eye(XXt_reduced.shape[0]), XTY_reduced)

        # Project back to original space
        # P = V @ P_reduced, where V is from SVD of X
        P = svd.components_.T @ P_reduced  # [768, rank]

    else:
        # Direct solution using Ridge Regression normal equations
        # P = (X^T @ X + λI)^(-1) @ X^T @ Y
        XtX = X_centered.T @ X_centered  # [768, 768]
        XtY = X_centered.T @ Y_centered  # [768, rank]

        # Add regularization for numerical stability
        lambda_reg = 1.0
        P = np.linalg.solve(XtX + lambda_reg * np.eye(XtX.shape[0]), XtY)

    # Add bias term (mean adjustment)
    # Y_pred = X @ P + b, where b = Y_mean - X_mean @ P
    bias = Y_mean - X_mean @ P

    print(f"  Projection matrix shape: {P.shape}")
    print(f"  Bias shape: {bias.shape}")

    # Evaluate projection quality
    Y_pred = X @ P + bias
    reconstruction_error = np.mean((Y_pred - Y) ** 2)
    cosine_sim = np.mean([
        np.dot(y_pred, y_true) / (np.linalg.norm(y_pred) * np.linalg.norm(y_true) + 1e-8)
        for y_pred, y_true in zip(Y_pred, Y)
    ])

    print(f"\nProjection quality:")
    print(f"  MSE: {reconstruction_error:.6f}")
    print(f"  Mean cosine similarity: {cosine_sim:.4f}")

    return P, bias.squeeze()


def save_projection_matrix(P: np.ndarray, bias: np.ndarray, output_path: str, rank: int):
    """Save projection matrix to file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    np.savez(
        output_path,
        projection=P,  # [768, rank]
        bias=bias,  # [rank]
        rank=rank,
        shape=P.shape,
    )

    print(f"\nProjection matrix saved to: {output_path}")
    print(f"  Can be loaded with: np.load('{output_path}')")


def main():
    parser = argparse.ArgumentParser(
        description="Learn global projection matrix for text embeddings"
    )
    parser.add_argument(
        "--data-roots",
        type=str,
        nargs="+",
        default=[
            "/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train",
            "/new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS/train",
        ],
        help="Paths to training data directories"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="SVD rank (dimension of compressed features)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exp/global_projection_r16.npz",
        help="Output path for projection matrix"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=100,
        help="Maximum scenes to use for learning"
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=100000,
        help="Total samples to collect"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GLOBAL PROJECTION MATRIX LEARNING")
    print("=" * 80)
    print("\nThis learns a projection matrix P that transforms 768-dim text embeddings")
    print("to the 16-dim SVD-compressed space for open-vocabulary inference.")
    print("\nUsage at inference:")
    print("  text_emb_16d = text_emb_768d @ projection_matrix + bias")
    print("  similarity = model_output_16d @ text_emb_16d.T")

    # Find scenes
    scenes = find_all_scenes(args.data_roots)
    print(f"\nFound {len(scenes)} scenes with original features")

    if not scenes:
        print("\nERROR: No scenes found with original 768-dim features!")
        print("\nYour data only has SVD-compressed features.")
        print("\nAlternative solutions:")
        print("  1. Use a random orthogonal projection (not recommended)")
        print("  2. Use PCA on SVD-compressed features to estimate the space")
        print("  3. Retrain model to output 768-dim features instead of 16-dim")
        return

    # Learn projection
    P, bias = learn_global_projection(
        scenes=scenes,
        rank=args.rank,
        max_scenes=args.max_scenes,
        total_samples=args.total_samples,
    )

    # Save
    save_projection_matrix(P, bias, args.output, args.rank)

    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print(f"\nTo use at inference, load the projection matrix:")
    print(f"  data = np.load('{args.output}')")
    print(f"  projection = data['projection']  # [768, 16]")
    print(f"  bias = data['bias']  # [16]")
    print(f"  text_emb_16d = text_emb_768d @ projection + bias")
    print("=" * 80)


if __name__ == "__main__":
    main()
