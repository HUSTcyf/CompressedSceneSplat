#!/usr/bin/env python3
"""
ScanNet Projection Matrix Computation

This script computes projection matrices for ScanNet dataset with 768-dim features.

ScanNet data structure:
- Train/val scenes in data_root/{split}/scene_name/
- Each scene has lang_feat.npy (768-dim) and SVD compressed features
- Need to compute 768->16 projection matrix for text embeddings

Usage:
    # Compute from val data (768-dim -> 16-dim)
    python tools/compute_projection_matrix_scannet.py \\
        --data_root /path/to/scannet \\
        --split val \\
        --svd_rank 16 \\
        --output projection_matrix_768_to_16_scannet.npy

    # Compute from train data with sampling
    python tools/compute_projection_matrix_scannet.py \\
        --data_root /path/to/scannet \\
        --split train \\
        --svd_rank 16 \\
        --sample_per_scene 50000 \\
        --output projection_matrix_768_to_16_scannet.npy

    # Compute for 32-dim SVD
    python tools/compute_projection_matrix_scannet.py \\
        --data_root /path/to/scannet \\
        --split val \\
        --svd_rank 32 \\
        --output projection_matrix_768_to_32_scannet.npy
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


def find_scenes_with_features(
    data_root: str,
    split: str = 'val',
    svd_rank: int = 16,
) -> List[str]:
    """
    Find scenes with both lang_feat.npy and SVD files.

    Args:
        data_root: Root directory of ScanNet dataset
        split: 'train' or 'val'
        svd_rank: SVD rank (r8, r16, r32)

    Returns:
        List of scene paths
    """
    split_dir = os.path.join(data_root, split)
    scenes = []

    if not os.path.exists(split_dir):
        print(f"Warning: Split directory not found: {split_dir}")
        return scenes

    for scene_name in os.listdir(split_dir):
        scene_path = os.path.join(split_dir, scene_name)
        if not os.path.isdir(scene_path):
            continue

        # Check required files
        lang_file = os.path.join(scene_path, 'lang_feat.npy')
        svd_file = os.path.join(scene_path, f'lang_feat_grid_svd_r{svd_rank}.npz')

        if os.path.exists(lang_file) and os.path.exists(svd_file):
            scenes.append(scene_path)

    return sorted(scenes)


def compute_projection_scannet(
    scenes: List[str],
    svd_rank: int = 16,
    sample_per_scene: int = None,  # None = use all data (no downsampling)
    text_dim: int = 768,  # 768-dim features from ScanNet (CLIP/SigLIP)
    normalize: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Compute projection matrix from ScanNet scenes.

    Args:
        scenes: List of scene paths
        svd_rank: SVD rank (16)
        sample_per_scene: Samples per scene (None = use all data)
        text_dim: Input feature dimension (768 for ScanNet CLIP/SigLIP features)
        normalize: Whether to normalize features

    Returns:
        W: [svd_rank, text_dim] - Projection matrix [16, 768]
        meta: Metadata dictionary
    """
    print(f"Processing {len(scenes)} scenes...")
    print(f"Feature dim: {text_dim} (ScanNet CLIP/SigLIP)")
    print(f"Target dim: {svd_rank}")

    if sample_per_scene is None:
        print("No downsampling: using ALL data per scene")
    else:
        print(f"Sampling {sample_per_scene:,} points per scene")

    # Accumulate for incremental least squares
    # Direct projection: 768 -> 16
    XTX = np.zeros((text_dim, text_dim), dtype=np.float64)
    XTY = np.zeros((text_dim, svd_rank), dtype=np.float64)
    total_samples = 0
    scene_idx = 0

    for scene_path in tqdm(scenes, desc="Processing scenes"):
        try:
            # Load 768-dim features
            lang_feat = np.load(os.path.join(scene_path, 'lang_feat.npy'))  # [N, 768]
            svd_data = np.load(os.path.join(scene_path, f'lang_feat_grid_svd_r{svd_rank}.npz'))
            Y_compressed = svd_data['compressed']  # [M, 16]
            indices = svd_data['indices']           # [N]

            # Map compressed features to points
            # indices maps from point to compressed grid index
            # Y contains compressed features at grid level
            Y = Y_compressed[indices]  # [N, 16]

            # Use full 768 dimensions
            X = lang_feat.astype(np.float64)  # [N, 768]

            # Validate dimensions
            if X.shape[1] != text_dim:
                print(f"Warning: Unexpected feature dim {X.shape[1]} in {scene_path}, expected {text_dim}")
                # Truncate or pad to text_dim
                if X.shape[1] > text_dim:
                    X = X[:, :text_dim]
                else:
                    X_pad = np.zeros((X.shape[0], text_dim), dtype=np.float64)
                    X_pad[:, :X.shape[1]] = X
                    X = X_pad

            # Normalize
            if normalize:
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X = X / (norms + 1e-8)

            # Sample (only if sample_per_scene is specified)
            if sample_per_scene is not None and len(X) > sample_per_scene:
                idx = np.random.choice(len(X), sample_per_scene, replace=False)
                X = X[idx]
                Y = Y[idx]

            # Update accumulators (incremental update to avoid memory overflow)
            Y = Y.astype(np.float64)
            XTX += X.T @ X
            XTY += X.T @ Y
            total_samples += len(X)
            scene_idx += 1

            # Progress update every 50 scenes
            if scene_idx % 50 == 0:
                print(f"  Processed {scene_idx}/{len(scenes)} scenes, {total_samples:,} samples so far")

        except Exception as e:
            print(f"Warning: Failed to process {scene_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nTotal samples: {total_samples:,}")

    # Solve for projection matrix
    print("Solving for projection matrix...")
    reg = 1e-6 * np.eye(text_dim)
    W_T = np.linalg.solve(XTX + reg, XTY)
    W = W_T.T  # [svd_rank, text_dim] = [16, 768]

    # Normalize W
    W = W.astype(np.float32)

    return W, {
        'num_scenes': len(scenes),
        'total_samples': total_samples,
        'input_dim': text_dim,
        'output_dim': svd_rank,
        'feat_dim': 768,
    }


def verify_projection(
    W: np.ndarray,
    scenes: List[str],
    svd_rank: int = 16,
    text_dim: int = 768,
    num_test: int = 5,
) -> dict:
    """Verify projection matrix quality."""
    print("\n=== Verification ===")
    print(f"Projection matrix shape: {W.shape} [output_dim={svd_rank}, input_dim={text_dim}]")

    results = []
    for scene_path in scenes[:num_test]:
        try:
            lang_feat = np.load(os.path.join(scene_path, 'lang_feat.npy'))
            svd_data = np.load(os.path.join(scene_path, f'lang_feat_grid_svd_r{svd_rank}.npz'))
            Y_compressed = svd_data['compressed']
            indices = svd_data['indices']

            # Use 768-dim features
            X_full = lang_feat[:, :text_dim] if lang_feat.shape[1] >= text_dim else np.pad(
                lang_feat, ((0, 0), (0, text_dim - lang_feat.shape[1])), mode='constant'
            )
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

            print(f"  {Path(scene_path).name}: MSE={mse:.6f}, Cos={cosine:.4f}")

        except Exception as e:
            print(f"  Error: {e}")

    if results:
        avg_mse = np.mean([r['mse'] for r in results])
        avg_cos = np.mean([r['cosine'] for r in results])
        print(f"\nAverage: MSE={avg_mse:.6f}, Cos={avg_cos:.4f}")
        return {'mse': avg_mse, 'cosine': avg_cos}

    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Compute projection matrix for ScanNet dataset (768 -> 16)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of ScanNet dataset')
    parser.add_argument('--split', type=str, default='val',
                       help='Split to use (default: val)')
    parser.add_argument('--svd_rank', type=int, default=16,
                       help='SVD rank (default: 16)')
    parser.add_argument('--sample_per_scene', type=int, default=None,
                       help='Samples per scene (None = use all data, no downsampling)')
    parser.add_argument('--text_dim', type=int, default=768,
                       help='Input feature dimension (768 for ScanNet CLIP/SigLIP features)')
    parser.add_argument('--output', type=str, default='projection_matrix_768_to_16_scannet.npy',
                       help='Output path for projection matrix')
    parser.add_argument('--verify', action='store_true', default=True,
                       help='Verify projection matrix (default: True)')

    args = parser.parse_args()

    print("="*70)
    print("ScanNet Projection Matrix Computation (768 -> 16)")
    print("="*70)
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"SVD rank: {args.svd_rank}")
    print(f"Text dim: {args.text_dim}")
    print(f"Output: {args.output}")

    # Find scenes
    scenes = find_scenes_with_features(args.data_root, args.split, args.svd_rank)
    print(f"\nFound {len(scenes)} {args.split} scenes with features")

    if not scenes:
        print("No scenes found!")
        print(f"\nPlease check:")
        print(f"  1. Data root path: {args.data_root}")
        print(f"  2. Split directory exists: {os.path.join(args.data_root, args.split)}")
        print(f"  3. Scenes have lang_feat.npy and lang_feat_grid_svd_r{args.svd_rank}.npz")
        return

    # Compute projection matrix
    W, meta = compute_projection_scannet(
        scenes,
        svd_rank=args.svd_rank,
        sample_per_scene=args.sample_per_scene,
        text_dim=args.text_dim,
    )

    print(f"\nProjection matrix: {W.shape}")
    print(f"  Norm: {np.linalg.norm(W):.4f}")
    print(f"  Mean: {np.mean(np.abs(W)):.6f}")
    print(f"  Std: {np.std(W):.6f}")

    # Verify
    if args.verify:
        verify_projection(W, scenes, args.svd_rank, args.text_dim)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, W)

    # Save metadata
    meta_path = output_path.with_suffix('.meta.txt')
    with open(meta_path, 'w') as f:
        f.write("ScanNet Projection Matrix (768 -> 16)\n")
        f.write("="*50 + "\n")
        f.write(f"Shape: {W.shape}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Source scenes: {meta['num_scenes']}\n")
        f.write(f"Total samples: {meta['total_samples']}\n")
        f.write(f"Feature dim: {meta['feat_dim']} -> {meta['input_dim']} -> {meta['output_dim']}\n")
        f.write(f"Sample per scene: {args.sample_per_scene if args.sample_per_scene else 'All (no downsampling)'}\n")
        f.write(f"Matrix norm: {np.linalg.norm(W):.4f}\n")
        f.write(f"Matrix mean: {np.mean(np.abs(W)):.6f}\n")
        f.write(f"Matrix std: {np.std(W):.6f}\n")

    print(f"\nSaved to: {output_path}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
