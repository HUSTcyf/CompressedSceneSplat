#!/usr/bin/env python3
"""
Matterport3D Projection Matrix Computation

This script handles the unique Matterport3D data structure:
- Train chunks: SVD-compressed features only, with lang_feat_index.npy
- Val chunks: 1152-dim features (multi-view fusion) + SVD-compressed features

For 768-dim text embeddings, we need to compute a projection matrix.
Since Matterport3D uses 1152-dim features, we have two options:

Option 1: Compute 1152→16 projection matrix, then adapt for 768-dim text
Option 2: Extract/use 768-dim subset from 1152-dim features

Usage:
    # Compute from val data (1152-dim -> 16-dim)
    python tools/compute_projection_matrix_matterport3d.py \\
        --data_root /new_data/cyf/Datasets/SceneSplat7k/matterport3d \\
        --split val \\
        --svd_rank 16 \\
        --output projection_matrix_1152_to_16.npy

    # Compute for 768-dim text embeddings (using subset)
    python tools/compute_projection_matrix_matterport3d.py \\
        --data_root /new_data/cyf/Datasets/SceneSplat7k/matterport3d \\
        --split val \\
        --svd_rank 16 \\
        --text_dim 768 \\
        --output projection_matrix_768_to_16.npy
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


def find_val_chunks_with_features(
    data_root: str,
    svd_rank: int = 16,
) -> List[str]:
    """Find val chunks with both lang_feat.npy and SVD files."""
    val_dir = os.path.join(data_root, 'val_grid1.0cm_chunk6x6x4_stride4x4x4')
    chunks = []

    for chunk_name in os.listdir(val_dir):
        chunk_path = os.path.join(val_dir, chunk_name)
        if not os.path.isdir(chunk_path):
            continue

        # Check required files
        lang_file = os.path.join(chunk_path, 'lang_feat.npy')
        svd_file = os.path.join(chunk_path, f'lang_feat_grid_svd_r{svd_rank}.npz')

        if os.path.exists(lang_file) and os.path.exists(svd_file):
            chunks.append(chunk_path)

    return sorted(chunks)


def compute_projection_matterport3d(
    chunks: List[str],
    svd_rank: int = 16,
    sample_per_chunk: int = None,  # None = use all data (no downsampling)
    text_dim: int = 1152,  # Full 1152-dim features from Matterport3D
    normalize: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Compute projection matrix from Matterport3D val chunks.

    Args:
        chunks: List of chunk paths
        svd_rank: SVD rank (16)
        sample_per_chunk: Samples per chunk (None = use all data)
        text_dim: Input feature dimension (1152 for full Matterport3D features)
        normalize: Whether to normalize features

    Returns:
        W: [svd_rank, text_dim] - Projection matrix [16, 1152]
        meta: Metadata dictionary
    """
    print(f"Processing {len(chunks)} val chunks...")
    print(f"Feature dim: {text_dim} (Matterport3D multi-view)")
    print(f"Target dim: {svd_rank}")

    if sample_per_chunk is None:
        print("No downsampling: using ALL data per chunk")

    # Accumulate for incremental least squares
    # Direct projection: 1152 -> 16
    XTX = np.zeros((text_dim, text_dim), dtype=np.float64)
    XTY = np.zeros((text_dim, svd_rank), dtype=np.float64)
    total_samples = 0
    chunk_idx = 0

    for chunk_path in tqdm(chunks):
        try:
            # Load 1152-dim features
            lang_feat = np.load(os.path.join(chunk_path, 'lang_feat.npy'))  # [N, 1152]
            svd_data = np.load(os.path.join(chunk_path, f'lang_feat_grid_svd_r{svd_rank}.npz'))
            Y_compressed = svd_data['compressed']  # [M, 16]
            indices = svd_data['indices']           # [N]

            # Map compressed features to points
            Y = Y_compressed[indices]  # [N, 16]

            # Use full 1152 dimensions
            X = lang_feat.astype(np.float64)  # [N, 1152]

            # Normalize
            if normalize:
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X = X / (norms + 1e-8)

            # Sample (only if sample_per_chunk is specified)
            if sample_per_chunk is not None and len(X) > sample_per_chunk:
                idx = np.random.choice(len(X), sample_per_chunk, replace=False)
                X = X[idx]
                Y = Y[idx]

            # Update accumulators (incremental update to avoid memory overflow)
            Y = Y.astype(np.float64)
            XTX += X.T @ X
            XTY += X.T @ Y
            total_samples += len(X)
            chunk_idx += 1

            # Progress update every 50 chunks
            if chunk_idx % 50 == 0:
                print(f"  Processed {chunk_idx}/{len(chunks)} chunks, {total_samples:,} samples so far")

        except Exception as e:
            print(f"Warning: Failed to process {chunk_path}: {e}")
            continue

    print(f"\nTotal samples: {total_samples:,}")

    # Solve for projection matrix
    print("Solving for projection matrix...")
    reg = 1e-6 * np.eye(text_dim)
    W_T = np.linalg.solve(XTX + reg, XTY)
    W = W_T.T  # [svd_rank, text_dim] = [16, 1152]

    # Normalize W
    W = W.astype(np.float32)

    return W, {
        'num_chunks': len(chunks),
        'total_samples': total_samples,
        'input_dim': text_dim,
        'output_dim': svd_rank,
        'feat_dim': 1152,
    }


def verify_projection(
    W: np.ndarray,
    chunks: List[str],
    svd_rank: int = 16,
    text_dim: int = 1152,
    num_test: int = 5,
) -> dict:
    """Verify projection matrix quality."""
    print("\n=== Verification ===")
    print(f"Projection matrix shape: {W.shape} [output_dim={svd_rank}, input_dim={text_dim}]")

    results = []
    for chunk_path in chunks[:num_test]:
        try:
            lang_feat = np.load(os.path.join(chunk_path, 'lang_feat.npy'))
            svd_data = np.load(os.path.join(chunk_path, f'lang_feat_grid_svd_r{svd_rank}.npz'))
            Y_compressed = svd_data['compressed']
            indices = svd_data['indices']

            # Use full 1152-dim features
            X_full = lang_feat[:, :text_dim]
            Y_true = Y_compressed[indices]

            # Normalize
            norms = np.linalg.norm(X_full, axis=1, keepdims=True)
            X_norm = X_full / (norms + 1e-8)

            # Predict
            Y_pred = X_norm @ W.T

            # Metrics
            mse = np.mean((Y_pred - Y_true) ** 2)
            cosine = np.mean(
                np.sum(Y_pred * Y_true, axis=1) /
                (np.linalg.norm(Y_pred, axis=1) * np.linalg.norm(Y_true, axis=1) + 1e-8)
            )

            results.append({'mse': mse, 'cosine': cosine})

            print(f"  {Path(chunk_path).name}: MSE={mse:.6f}, Cos={cosine:.4f}")

        except Exception as e:
            print(f"  Error: {e}")

    if results:
        avg_mse = np.mean([r['mse'] for r in results])
        avg_cos = np.mean([r['cosine'] for r in results])
        print(f"\nAverage: MSE={avg_mse:.6f}, Cos={avg_cos:.4f}")
        return {'mse': avg_mse, 'cosine': avg_cos}

    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--svd_rank', type=int, default=16)
    parser.add_argument('--sample_per_chunk', type=int, default=None,
                       help='Samples per chunk (None = use all data, no downsampling)')
    parser.add_argument('--text_dim', type=int, default=1152,
                       help='Input feature dimension (1152 for full Matterport3D features)')
    parser.add_argument('--output', type=str, default='projection_matrix_1152_to_16.npy')
    parser.add_argument('--verify', action='store_true', default=True)

    args = parser.parse_args()

    print("="*70)
    print("Matterport3D Projection Matrix Computation (1152 -> 16)")
    print("="*70)

    # Find chunks
    chunks = find_val_chunks_with_features(args.data_root, args.svd_rank)
    print(f"Found {len(chunks)} val chunks with features")

    if not chunks:
        print("No chunks found!")
        return

    # Compute projection matrix
    W, meta = compute_projection_matterport3d(
        chunks,
        svd_rank=args.svd_rank,
        sample_per_chunk=args.sample_per_chunk,
        text_dim=args.text_dim,
    )

    print(f"\nProjection matrix: {W.shape}")
    print(f"  Norm: {np.linalg.norm(W):.4f}")

    # Verify
    if args.verify:
        verify_projection(W, chunks, args.svd_rank, args.text_dim)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, W)

    # Save metadata
    meta_path = output_path.with_suffix('.meta.txt')
    with open(meta_path, 'w') as f:
        f.write("Matterport3D Projection Matrix (1152 -> 16)\n")
        f.write("="*50 + "\n")
        f.write(f"Shape: {W.shape}\n")
        f.write(f"Source chunks: {meta['num_chunks']}\n")
        f.write(f"Total samples: {meta['total_samples']}\n")
        f.write(f"Feature dim: {meta['feat_dim']} -> {meta['input_dim']} -> {meta['output_dim']}\n")
        f.write(f"No downsampling: {args.sample_per_chunk is None}\n")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
