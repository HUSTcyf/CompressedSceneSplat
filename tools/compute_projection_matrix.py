#!/usr/bin/env python3
"""
Compute Projection Matrix from 768-dim to 16-dim (Training-Free)

This script computes a projection matrix using existing paired data:
- Original 768-dim language features (from lang_feat.npy)
- Compressed 16-dim SVD features (from lang_feat_grid_svd_r16.npz)

The projection matrix allows text embeddings (768-dim) to be projected to the
compressed feature space (16-dim) for computing similarity with model predictions.

This is a ONE-TIME computation - the resulting matrix can be reused for all scenes.

Usage:
    # Compute projection matrix from one or multiple scenes
    python tools/compute_projection_matrix.py \
        --data_root /new_data/cyf/projects/SceneSplat/gaussian_train \
        --dataset 3DOVS \
        --split train \
        --output projection_matrix_768_to_16.npy

    # Use the computed matrix for text embedding projection
    import numpy as np
    W = np.load('projection_matrix_768_to_16.npy')  # [16, 768]
    text_embed_16d = text_embed_768 @ W.T  # [num_classes, 16]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


def find_scenes(data_root: str, dataset: str, split: str) -> List[str]:
    """Find all scene directories with required files."""
    dataset_path = os.path.join(data_root, dataset, split)
    if not os.path.exists(dataset_path):
        return []

    scenes = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            # Check for required files
            has_lang_feat = os.path.exists(os.path.join(item_path, 'lang_feat.npy'))
            has_svd = os.path.exists(os.path.join(item_path, 'lang_feat_grid_svd_r16.npz'))
            has_mask = os.path.exists(os.path.join(item_path, 'valid_feat_mask.npy'))
            if has_lang_feat and has_svd and has_mask:
                scenes.append(item_path)
    return sorted(scenes)


def load_paired_features(scene_path: str, sample_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load paired 768-dim and 16-dim features from a scene.

    Args:
        scene_path: Path to scene directory
        sample_size: Maximum number of samples to use (None = all)

    Returns:
        feat_768: [N, 768] - Original language features
        feat_16: [N, 16] - Compressed SVD features
    """
    # Load data
    lang_feat_orig = np.load(os.path.join(scene_path, 'lang_feat.npy'))  # [N_total, 768]
    valid_mask = np.load(os.path.join(scene_path, 'valid_feat_mask.npy'))
    svd_data = np.load(os.path.join(scene_path, 'lang_feat_grid_svd_r16.npz'))

    # Extract compressed features and indices
    compressed_grid = svd_data['compressed']  # [M, 16]
    indices = svd_data['indices']            # [N_valid] - indices mapping

    # Get paired data for valid points
    feat_768 = lang_feat_orig[valid_mask]  # [N_valid, 768]
    feat_16 = compressed_grid[indices]      # [N_valid, 16]

    # Sample if requested
    if sample_size is not None and len(feat_768) > sample_size:
        idx = np.random.choice(len(feat_768), sample_size, replace=False)
        feat_768 = feat_768[idx]
        feat_16 = feat_16[idx]

    return feat_768, feat_16


def compute_projection_matrix(
    scenes: List[str],
    max_samples_per_scene: int = 100000,
    method: str = 'ls',
) -> np.ndarray:
    """
    Compute projection matrix from 768-dim to 16-dim.

    Args:
        scenes: List of scene paths
        max_samples_per_scene: Maximum samples to use per scene
        method: 'ls' for least squares, 'pca' for PCA

    Returns:
        W: [16, 768] - Projection matrix (use: feat_16 = feat_768 @ W.T)
    """
    print(f"\nComputing projection matrix from {len(scenes)} scenes")

    if method == 'ls':
        # Method 1: Least Squares
        # Solve: X @ W^T = Y  => W = (X^T X)^-1 X^T Y
        all_X = []
        all_Y = []

        print("\nLoading features from scenes...")
        for scene_path in tqdm(scenes):
            try:
                X, Y = load_paired_features(scene_path, max_samples_per_scene)
                all_X.append(X)
                all_Y.append(Y)
            except Exception as e:
                print(f"  Warning: Failed to load {scene_path}: {e}")
                continue

        if not all_X:
            raise ValueError("No valid scenes found!")

        # Concatenate all samples
        X_all = np.concatenate(all_X, axis=0)  # [N_total, 768]
        Y_all = np.concatenate(all_Y, axis=0)  # [N_total, 16]

        print(f"\nTotal samples: {len(X_all):,}")
        print(f"  X shape: {X_all.shape}")
        print(f"  Y shape: {Y_all.shape}")

        # Use subset if too large (to avoid memory issues)
        if len(X_all) > 500000:
            print(f"  Subsampling to 500,000 for memory efficiency...")
            idx = np.random.choice(len(X_all), 500000, replace=False)
            X_all = X_all[idx]
            Y_all = Y_all[idx]

        print("\nComputing projection matrix (least squares)...")
        W, residuals, rank, s = np.linalg.lstsq(X_all, Y_all, rcond=None)
        W = W.T  # [16, 768]

        # Verify
        print("\nVerifying projection...")
        test_idx = np.random.choice(len(X_all), min(10000, len(X_all)), replace=False)
        X_test = X_all[test_idx]
        Y_true = Y_all[test_idx]
        Y_pred = X_test @ W.T

        mse = np.mean((Y_pred - Y_true) ** 2)
        cosine_sim = np.sum(Y_pred * Y_true, axis=1) / (
            np.linalg.norm(Y_pred, axis=1) * np.linalg.norm(Y_true, axis=1) + 1e-8
        )

        print(f"  MSE: {mse:.8f}")
        print(f"  Cosine similarity: {cosine_sim.mean():.4f} ± {cosine_sim.std():.4f}")
        print(f"  Rank: {rank}")
        print(f"  Projection matrix shape: {W.shape}")

        return W

    elif method == 'pca':
        # Method 2: PCA (alternative)
        print("\nComputing projection matrix (PCA)...")

        all_X = []
        for scene_path in tqdm(scenes):
            try:
                X, Y = load_paired_features(scene_path, max_samples_per_scene)
                all_X.append(X)
            except Exception as e:
                continue

        X_all = np.concatenate(all_X, axis=0)

        # Center the data
        X_mean = X_all.mean(axis=0, keepdims=True)
        X_centered = X_all - X_mean

        # Compute PCA to get 16 principal components
        # Then find projection that best matches Y
        # This is more complex, using least squares result for now
        raise NotImplementedError("PCA method not yet implemented")

    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute projection matrix from 768-dim text embeddings to 16-dim SVD space (training-free)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/new_data/cyf/projects/SceneSplat/gaussian_train",
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="3DOVS",
        choices=["3DOVS", "lerf_ovs"],
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Comma-separated list of specific scene names (default: all scenes)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100000,
        help="Maximum samples per scene",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="projection_matrix_768_to_16.npy",
        help="Output file path",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ls",
        choices=["ls", "pca"],
        help="Method: 'ls' for least squares, 'pca' for PCA",
    )

    args = parser.parse_args()

    # Find scenes
    if args.scenes:
        scene_names = [s.strip() for s in args.scenes.split(',')]
        scenes = [os.path.join(args.data_root, args.dataset, args.split, s) for s in scene_names]
    else:
        scenes = find_scenes(args.data_root, args.dataset, args.split)

    if not scenes:
        print(f"No scenes found!")
        return

    print(f"Found {len(scenes)} scenes")

    # Compute projection matrix
    W = compute_projection_matrix(scenes, args.max_samples, args.method)

    # Save projection matrix
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, W)
    print(f"\nProjection matrix saved to: {output_path}")
    print(f"  Shape: {W.shape}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")

    # Create example usage file
    example_file = output_path.parent / "projection_matrix_example.py"
    with open(example_file, 'w') as f:
        f.write(f'''# Example: Using Projection Matrix for Text Embedding

import numpy as np
import torch.nn.functional as F

# Load projection matrix
W = np.load("{output_path}")  # [16, 768]

# Project text embeddings from 768-dim to 16-dim
text_embed_768 = open_clip_model.encode_text(["chair", "table", "lamp"])  # [num_classes, 768]
text_embed_16d = text_embed_768 @ W.T  # [num_classes, 16]

# Compute similarity with model predictions
pred_16d = model.predict(points)  # [N, 16] - from LitePT

# Cosine similarity
similarity = F.cosine_similarity(
    pred_16d.unsqueeze(1),  # [N, 1, 16]
    torch.from_numpy(text_embed_16d).unsqueeze(0)  # [1, num_classes, 16]
).squeeze(1)  # [N, num_classes]

# Get predicted class for each point
predicted_class = similarity.argmax(dim=1)  # [N]
''')
    print(f"Example usage saved to: {example_file}")


if __name__ == "__main__":
    main()
