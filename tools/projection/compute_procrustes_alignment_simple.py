#!/usr/bin/env python3
"""
Orthogonal Procrustes Q Matrix Computation - Per-Scene Version

This script computes the orthogonal Procrustes alignment matrix Q for each scene,
then computes the average Q across all scenes.

Implementation Details:
    - Text embeddings SVD reduction: Uses NumPy (efficient for small matrices)
    - Q matrix computation (Procrustes): Uses PyTorch GPU (faster for d×d matrices)

Mathematical Formulation:
    For each scene i:
        Find orthogonal Q_i (Q_i^T @ Q_i = I) that minimizes ||X_c_i @ Q_i - Y||_F^2
        Solution: Q_i = U @ V^T where M = X_c_i^T @ Y = U @ Σ @ V^T (SVD)

    Average Q:
        Q_avg = mean(Q_i) for all scenes
        Re-orthogonalize: Q_avg_final = U @ V^T where Q_avg = U @ Σ @ V^T

Typical Use Case:
    X_c_i: Grid-level SVD compressed features [M, 16] from scene i's lang_feat_grid_svd_r16.npz
    Y: Text embeddings after SVD reduction [N, 16] from scannet*_text_embeddings*.pt

Usage:
    # Compute Q for each scene and save separately
    python tools/compute_procrustes_alignment_simple.py \\
        --data_root /new_data/cyf/Datasets/SceneSplat7k/scannet/test_grid1.0cm_chunk6x6_stride3x3 \\
        --text_embed /new_data/cyf/projects/SceneSplat/pointcept/datasets/preprocessing/scannet/meta_data/scannet20_text_embeddings_siglip2.pt \\
        --svd_rank 16 \\
        --output_dir /path/to/output
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import torch
from tqdm import tqdm
import time


def perform_svd_reduction(
    matrix: np.ndarray,
    rank: int,
    normalize: bool = True,
    use_torch: bool = False,
    device: str = 'cuda',
    benchmark: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Perform SVD reduction on a matrix using PyTorch (GPU) or NumPy.

    Performance comparison (on RTX 4090):
    - Small matrices (< 100 rows): NumPy CPU is ~20x faster (less overhead)
    - Medium matrices (100-1000 rows): PyTorch GPU is ~7-9x faster
    - Large matrices (> 1000 rows): PyTorch GPU is ~17-26x faster

    Args:
        matrix: [N, D] input matrix
        rank: Target reduced dimension
        normalize: Whether to normalize features before SVD
        use_torch: False (default, use NumPy), True (use PyTorch GPU), or benchmark mode
        device: Device to use ('cuda' or 'cpu')
        benchmark: If True, run both implementations and return timing info

    Returns:
        reduced: [N, rank] reduced representation
        components: [D, rank] right singular vectors
        timing: (Optional) Dictionary with timing information if benchmark=True
    """
    N, D = matrix.shape
    timing = {}

    # Normalize if requested
    if normalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / (norms + 1e-8)
    else:
        matrix_norm = matrix

    if use_torch == 'benchmark':
        # Run both and compare
        use_torch = torch.cuda.is_available()

    if use_torch and torch.cuda.is_available():
        # PyTorch GPU implementation
        device_to_use = device if torch.cuda.is_available() else 'cpu'

        t_start = time.time()
        matrix_torch = torch.from_numpy(matrix_norm).to(device_to_use)
        U, S, Vt = torch.linalg.svd(matrix_torch, full_matrices=False)

        # Take top rank components
        Vt = Vt.cpu().numpy()
        components = Vt[:rank, :].T  # [D, rank]
        reduced = matrix @ components  # [N, rank]

        torch_time = time.time() - t_start
        timing['torch'] = torch_time
        timing['method'] = 'torch'

        if use_torch == 'benchmark':
            # Also run NumPy version for comparison
            t_start = time.time()
            _, _, Vt_np = np.linalg.svd(matrix_norm, full_matrices=False)
            components_np = Vt_np[:rank, :].T
            reduced_np = matrix @ components_np

            numpy_time = time.time() - t_start
            timing['numpy'] = numpy_time
            timing['speedup'] = numpy_time / torch_time if torch_time > 0 else float('inf')

            # Verify results are similar
            diff = np.abs(reduced - reduced_np).max()
            timing['max_difference'] = float(diff)

        return reduced, components, timing

    else:
        # NumPy implementation (standard SVD)
        t_start = time.time()
        _, _, Vt = np.linalg.svd(matrix_norm, full_matrices=False)
        components = Vt[:rank, :].T  # [D, rank]
        reduced = matrix @ components  # [N, rank]

        numpy_time = time.time() - t_start
        timing['numpy'] = numpy_time
        timing['method'] = 'numpy'

        return reduced, components, timing


def compute_procrustes_Q(
    X_c: np.ndarray,
    Y: np.ndarray,
    use_torch: bool = True,
    device: str = 'cuda'
) -> Tuple[np.ndarray, Dict]:
    """
    Compute orthogonal Procrustes alignment matrix Q using PyTorch.

    Minimizes ||X_c @ Q - Y||_F subject to Q^T @ Q = I
    Solution: Q = U @ V^T where M = X_c^T @ Y = U @ S @ V^T (SVD)

    Args:
        X_c: [N, d] source features (numpy array)
        Y: [N, d] target features (numpy array)
        use_torch: Whether to use PyTorch for SVD (default: True)
        device: Device for PyTorch computation ('cuda' or 'cpu')

    Returns:
        Q: [d, d] orthogonal alignment matrix (numpy array)
        metrics: Dictionary with alignment metrics
    """
    if X_c.shape != Y.shape:
        raise ValueError(
            f"Shape mismatch: X_c has shape {X_c.shape}, Y has shape {Y.shape}"
        )

    N, d = X_c.shape

    # Compute M = X_c^T @ Y
    M = X_c.T @ Y

    # Use PyTorch for SVD (faster for d x d matrices)
    if use_torch and torch.cuda.is_available():
        device_to_use = device if torch.cuda.is_available() else 'cpu'
        M_torch = torch.from_numpy(M).to(device_to_use)
        U, S, Vt = torch.linalg.svd(M_torch, full_matrices=False)
        U = U.cpu().numpy()
        S = S.cpu().numpy()
        Vt = Vt.cpu().numpy()
    else:
        # Fallback to NumPy
        U, S, Vt = np.linalg.svd(M, full_matrices=False)

    Q = U @ Vt

    # Ensure det(Q) = 1 (proper rotation)
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt

    # Compute metrics
    X_aligned = X_c @ Q
    residual = np.linalg.norm(X_aligned - Y, 'fro')
    relative_error = residual / (np.linalg.norm(Y, 'fro') + 1e-8)

    cosine_before = np.mean([
        np.dot(X_c[i], Y[i]) / (np.linalg.norm(X_c[i]) * np.linalg.norm(Y[i]) + 1e-8)
        for i in range(N)
    ])
    cosine_after = np.mean([
        np.dot(X_aligned[i], Y[i]) / (np.linalg.norm(X_aligned[i]) * np.linalg.norm(Y[i]) + 1e-8)
        for i in range(N)
    ])

    QtQ = Q.T @ Q
    orthogonality_error = np.linalg.norm(QtQ - np.eye(d), 'fro')

    metrics = {
        'N': N,
        'd': d,
        'residual_norm': float(residual),
        'relative_error': float(relative_error),
        'cosine_before': float(cosine_before),
        'cosine_after': float(cosine_after),
        'cosine_improvement': float(cosine_after - cosine_before),
        'singular_values': S.tolist(),
        'orthogonality_error': float(orthogonality_error),
        'det_Q': float(np.linalg.det(Q)),
    }

    return Q.astype(np.float32), metrics


def compute_procrustes_Q_cuda_with_labels(
    X_c: torch.Tensor,
    Y: torch.Tensor,
    labels: torch.Tensor
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute orthogonal Procrustes alignment matrix Q using label-based aggregation.

    This method is used when X_c contains many duplicate rows (corresponding to
    different points with the same semantic class), and Y contains the unique
    rows (class embeddings).

    Instead of sampling, we efficiently compute M = X_c^T @ Y_reconstructed by
    aggregating points with the same label:
        M = Σ_{j=1}^{M} (Σ_{i: labels[i]=j} X_c[i])^T @ Y[j]

    Args:
        X_c: [N, d] source features (CUDA tensor)
        Y: [M, d] unique target features (CUDA tensor, M << N typically)
        labels: [N] label indices, labels[i] ∈ {0, ..., M-1}

    Returns:
        Q: [d, d] orthogonal alignment matrix (CUDA tensor)
        metrics: Dictionary with alignment metrics
    """
    N, d = X_c.shape
    M, d_y = Y.shape

    if d != d_y:
        raise ValueError(
            f"Feature dimension mismatch: X_c has d={d}, Y has d={d_y}"
        )

    if labels.shape[0] != N:
        raise ValueError(
            f"Labels length mismatch: X_c has N={N}, labels has {labels.shape[0]}"
        )

    # Efficient computation of M = X_c^T @ Y_reconstructed
    # by aggregating points with the same label (without explicit large matrix)
    #
    # M = Σ_{i=1}^{N} X_c[i]^T ⊗ Y[labels[i]]
    #   = Σ_{j=1}^{M} (Σ_{i: labels[i]=j} X_c[i])^T ⊗ Y[j]
    #
    # Steps:
    # 1. For each label j, compute sum_j = Σ_{i: labels[i]=j} X_c[i]
    # 2. Compute M = Σ_{j=1}^{M} sum_j^T @ Y[j]

    # Memory-efficient method using index_add_
    # sum_j[j] = Σ_{i: labels[i]=j} X_c[i]  for all j in [0, M-1]
    # This avoids creating the [N, M] one-hot matrix
    sum_j = torch.zeros(M, d, device=X_c.device, dtype=X_c.dtype)

    # Ensure labels is LongTensor (required by index_add_)
    if labels.dtype != torch.long:
        labels = labels.long()

    # Use index_add_ for efficient aggregation
    # sum_j[labels[i]] += X_c[i] for all i
    # Memory: only O(M*d) for sum_j, no large intermediate matrices
    sum_j.index_add_(0, labels, X_c)  # [M, d]

    # Compute M = sum_j^T @ Y: [d, M] @ [M, d] = [d, d]
    M_matrix = torch.mm(sum_j.t(), Y)  # [d, d]

    # SVD on GPU
    U, S, Vt = torch.linalg.svd(M_matrix, full_matrices=False)

    # Q = U @ V^T
    Q = torch.mm(U, Vt)

    # Ensure det(Q) = 1 (proper rotation)
    det_Q = torch.det(Q)
    if det_Q < 0:
        U[:, -1] *= -1
        Q = torch.mm(U, Vt)
        det_Q = torch.det(Q)

    # Compute metrics using all data (no sampling)
    # Y_reconstructed[i] = Y[labels[i]]
    Y_reconstructed = Y[labels]
    X_aligned = torch.mm(X_c, Q)
    residual = torch.norm(X_aligned - Y_reconstructed, 'fro')
    relative_error = residual / (torch.norm(Y_reconstructed, 'fro') + 1e-8)

    # Cosine similarity (vectorized on GPU)
    X_c_norm = torch.norm(X_c, dim=1, keepdim=True) + 1e-8
    Y_norm = torch.norm(Y_reconstructed, dim=1, keepdim=True) + 1e-8
    cosine_before = torch.mean(
        torch.sum(X_c * Y_reconstructed, dim=1) / (X_c_norm.squeeze() * Y_norm.squeeze())
    )

    X_aligned_norm = torch.norm(X_aligned, dim=1, keepdim=True) + 1e-8
    cosine_after = torch.mean(
        torch.sum(X_aligned * Y_reconstructed, dim=1) / (X_aligned_norm.squeeze() * Y_norm.squeeze())
    )

    # Orthogonality error
    QtQ = torch.mm(Q.t(), Q)
    I = torch.eye(d, device=Q.device)
    orthogonality_error = torch.norm(QtQ - I, 'fro')

    metrics = {
        'N': N,
        'M': M,
        'd': d,
        'unique_labels': M,
        'residual_norm': float(residual.cpu()),
        'relative_error': float(relative_error.cpu()),
        'cosine_before': float(cosine_before.cpu()),
        'cosine_after': float(cosine_after.cpu()),
        'cosine_improvement': float((cosine_after - cosine_before).cpu()),
        'singular_values': S.cpu().tolist(),
        'orthogonality_error': float(orthogonality_error.cpu()),
        'det_Q': float(det_Q.cpu()),
    }

    return Q, metrics


def load_grid_svd_features(svd_file: str) -> np.ndarray:
    """
    Load grid-level SVD compressed features and reconstruct to point-level.

    The npz file contains:
        - compressed: [M, rank] grid-level compressed features (M unique grid cells)
        - indices: [N] point-to-grid mapping indices

    Returns:
        point_lang_feat: [N, rank] point-level reconstructed features
    """
    svd_data = np.load(svd_file)
    compressed = svd_data['compressed'].astype(np.float32)  # [M, rank]
    indices = svd_data['indices']  # [N]

    # Reconstruct point-level features: compressed[indices] -> [N, rank]
    point_lang_feat = compressed[indices].astype(np.float32)
    return point_lang_feat


def get_grid_svd_shapes(svd_file: str) -> Tuple[Tuple[int, int], int]:
    """
    Get shapes from SVD file without loading all data.

    Returns:
        grid_shape: (M, rank) shape of compressed grid features
        num_points: N number of points (length of indices)
    """
    svd_data = np.load(svd_file)
    compressed_shape = svd_data['compressed'].shape  # (M, rank)
    num_points = svd_data['indices'].shape[0]  # N
    return compressed_shape, num_points


def load_text_embeddings(
    embed_file: str,
    svd_rank: int,
    normalize: bool = True,
    use_torch: bool = False,
    benchmark: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Load and SVD-reduce text embeddings using NumPy.

    Uses NumPy for SVD (efficient for small text embedding matrices).
    For Q matrix computation, PyTorch GPU is used instead.

    Args:
        embed_file: Path to text embeddings (.pt or .npy)
        svd_rank: Target SVD rank
        normalize: Whether to normalize features
        use_torch: Whether to use PyTorch for SVD (default: False, use NumPy)
        benchmark: If True, return timing info

    Returns:
        embeddings: [N, svd_rank] SVD-reduced embeddings
        timing: Dictionary with timing information
    """
    if embed_file.endswith('.pt'):
        embeddings = torch.load(embed_file, weights_only=False).numpy()
    elif embed_file.endswith('.npy'):
        embeddings = np.load(embed_file)
    else:
        raise ValueError(f"Unsupported file format: {embed_file}")

    embeddings = embeddings.astype(np.float32)

    timing = {}
    if embeddings.shape[1] > svd_rank:
        embeddings, components, timing = perform_svd_reduction(
            embeddings, svd_rank, normalize, use_torch, 'cuda', benchmark
        )
    else:
        components = None

    return embeddings, timing


def compute_procrustes_per_scene(
    data_root: str,
    text_embed_file: str,
    svd_rank: int = 16,
    normalize: bool = True,
    save: bool = False,
    save_avg: bool = False,
    device: str = 'cuda',
    benchmark: bool = False,
    label_filename: str = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute Procrustes Q for each scene and compute average Q.

    Implementation:
        - Text embeddings SVD reduction: Uses NumPy (efficient for small matrices)
        - Q matrix computation (Procrustes): Uses PyTorch GPU with label aggregation

    For each scene:
        1. Load grid features X_c, text embeddings Y, and labels
        2. Compute Q_i via orthogonal Procrustes using compute_procrustes_Q_cuda_with_labels
        3. (Optional) Save Q_i to same directory as SVD file

    Then:
        4. Average all Q_i and re-orthogonalize
        5. (Optional) Save Q_avg to data_root

    Args:
        data_root: Root directory containing scene subdirectories
        text_embed_file: Path to text embeddings
        svd_rank: SVD rank
        normalize: Whether to normalize features
        save: Whether to save individual Q matrices (default: False)
        save_avg: Whether to save average Q matrix (default: False)
        device: Device for PyTorch computation ('cuda' or 'cpu')
        benchmark: If True, run SVD benchmark for text embeddings
        label_filename: Name of label file (e.g., 'segment_nyu_160.npy', 'segment.npy')

    Returns:
        Q_avg: [d, d] average alignment matrix
        metadata: Dictionary with metadata
    """
    # Extract benchmark name from text_embed_file
    # e.g., "scannet20_text_embeddings_siglip2.pt" -> "scannet20"
    text_embed_basename = Path(text_embed_file).stem
    if '_text_embeddings' in text_embed_basename:
        benchmark_name = text_embed_basename.split('_text_embeddings')[0]
    else:
        benchmark_name = text_embed_basename

    # Load text embeddings once (using NumPy for SVD reduction)
    print(f"Loading text embeddings from {text_embed_file}")
    Y, svd_timing = load_text_embeddings(
        text_embed_file, svd_rank, normalize, use_torch=False, benchmark=benchmark
    )
    print(f"  Y shape: {Y.shape}")
    print(f"  Benchmark: {benchmark_name}")

    if benchmark and svd_timing.get('method') == 'numpy':
        print(f"\n{'='*70}")
        print("Text Embeddings SVD (NumPy)")
        print(f"{'='*70}")
        print(f"  NumPy time:  {svd_timing.get('numpy', 0)*1000:.2f} ms")
        print(f"{'='*70}\n")

    # Find scenes
    scenes = []
    for scene_name in sorted(os.listdir(data_root)):
        scene_path = os.path.join(data_root, scene_name)
        if os.path.isdir(scene_path):
            svd_file = os.path.join(scene_path, f'lang_feat_grid_svd_r{svd_rank}.npz')
            if os.path.exists(svd_file):
                scenes.append((scene_name, svd_file))

    print(f"\nFound {len(scenes)} scenes with grid SVD files")

    if not scenes:
        raise ValueError("No scenes found with grid SVD files!")

    # Compute Q for each scene (using PyTorch for Procrustes)
    Q_list = []
    scene_metadata = []

    print(f"\nComputing Q for each scene (using PyTorch for Procrustes)...")
    if save:
        print(f"Q matrices will be saved alongside SVD files (benchmark: {benchmark_name})")

    for scene_name, svd_file in tqdm(scenes, desc="Processing scenes"):
        try:
            # Clear CUDA cache before processing each scene
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

            # Load grid features
            X_c = load_grid_svd_features(svd_file)

            # Load labels if label_filename is specified
            if label_filename is not None:
                scene_dir = Path(svd_file).parent
                label_path = scene_dir / label_filename
                if not label_path.exists():
                    print(f"  Warning: Label file not found: {label_path}")
                    continue

                labels = np.load(label_path).astype(np.int64)

                # Check if labels need filtering using valid_feat_mask
                # SVD files only contain valid points, so we need to filter labels
                if labels.shape[0] != X_c.shape[0]:
                    # Try to load valid_feat_mask to filter labels
                    valid_mask_path = scene_dir / 'valid_feat_mask.npy'
                    if valid_mask_path.exists():
                        valid_mask = np.load(valid_mask_path).astype(bool)
                        labels_filtered = labels[valid_mask]

                        if labels_filtered.shape[0] == X_c.shape[0]:
                            labels = labels_filtered
                        else:
                            print(f"  Warning: After filtering, label count {labels_filtered.shape[0]} != X_c count {X_c.shape[0]}, skipping scene")
                            continue
                    else:
                        print(f"  Warning: Label count {labels.shape[0]} != X_c count {X_c.shape[0]}, skipping scene")
                        continue

                # Also need to filter out ignore_index labels to avoid CUDA indexing errors
                # Y[labels] will fail if labels contains -1 or other invalid indices
                valid_label_mask = (labels >= 0) & (labels < Y.shape[0])
                if valid_label_mask.sum() < labels.shape[0]:
                    # print(f"  Info: Filtering out {(~valid_label_mask).sum()} invalid labels (ignore_index or out of range)")
                    labels = labels[valid_label_mask]
                    X_c = X_c[valid_label_mask]

                if labels.shape[0] == 0:
                    print(f"  Warning: No valid labels remaining after filtering, skipping scene")
                    continue

                # Convert to CUDA tensors
                X_c_tensor = torch.from_numpy(X_c).cuda()
                Y_tensor = torch.from_numpy(Y).cuda()
                labels_tensor = torch.from_numpy(labels).cuda()

                # Use compute_procrustes_Q_cuda_with_labels (no sampling, efficient)
                Q_cuda, metrics_cuda = compute_procrustes_Q_cuda_with_labels(
                    X_c_tensor, Y_tensor, labels_tensor
                )
                Q = Q_cuda.cpu().numpy()
                metrics = metrics_cuda
            else:
                # No labels, use old method with sampling
                print(f"  Warning: No label file specified, using sampling (not recommended)")
                # Sample to match Y dimensions
                if X_c.shape[0] > Y.shape[0]:
                    idx = np.random.choice(X_c.shape[0], Y.shape[0], replace=False)
                    X_c_sampled = X_c[idx]
                else:
                    X_c_sampled = X_c

                # Compute Q using old method
                Q, metrics = compute_procrustes_Q(X_c_sampled, Y, use_torch=True, device=device)
            Q_list.append(Q)

            # Get grid shape info without reloading all data
            grid_feat_shape, num_points = get_grid_svd_shapes(svd_file)
            scene_info = {
                'scene_name': scene_name,
                'svd_file': svd_file,
                'grid_feat_shape': grid_feat_shape,
                'num_points': num_points,
                'X_c_shape': X_c.shape,
                'Y_shape': Y.shape,
                **metrics
            }
            scene_metadata.append(scene_info)

            # Save individual Q file (same directory as SVD file)
            if save:
                scene_dir = Path(svd_file).parent
                Q_file = scene_dir / f"Q_procrustes_{benchmark_name}_r{svd_rank}.npz"
                np.savez_compressed(
                    Q_file,
                    Q=Q,
                    metadata=scene_info
                )

        except Exception as e:
            print(f"  Warning: Failed to process {scene_name}: {e}")
            continue

    if not Q_list:
        raise ValueError("No valid scenes processed!")

    # Compute average Q
    print(f"\nComputing average Q from {len(Q_list)} scenes...")
    Q_avg_raw = np.mean(Q_list, axis=0)

    # Re-orthogonalize using SVD (polar decomposition)
    U, _, Vt = np.linalg.svd(Q_avg_raw, full_matrices=False)
    Q_avg = U @ Vt

    # Ensure proper rotation
    if np.linalg.det(Q_avg) < 0:
        U[:, -1] *= -1
        Q_avg = U @ Vt

    Q_avg = Q_avg.astype(np.float32)

    # Compute metrics for average Q
    print("\nComputing alignment metrics for average Q...")
    cosine_before_list = []
    cosine_after_list = []

    for scene_name, svd_file in scenes[:min(len(scenes), 10)]:
        try:
            X_c = load_grid_svd_features(svd_file)
            if X_c.shape[0] > Y.shape[0]:
                idx = np.random.choice(X_c.shape[0], Y.shape[0], replace=False)
                X_c = X_c[idx]

            X_aligned = X_c @ Q_avg
            cos_before = np.mean([
                np.dot(X_c[i], Y[i]) / (np.linalg.norm(X_c[i]) * np.linalg.norm(Y[i]) + 1e-8)
                for i in range(Y.shape[0])
            ])
            cos_after = np.mean([
                np.dot(X_aligned[i], Y[i]) / (np.linalg.norm(X_aligned[i]) * np.linalg.norm(Y[i]) + 1e-8)
                for i in range(Y.shape[0])
            ])
            cosine_before_list.append(cos_before)
            cosine_after_list.append(cos_after)
        except:
            continue

    avg_cosine_before = np.mean(cosine_before_list) if cosine_before_list else 0
    avg_cosine_after = np.mean(cosine_after_list) if cosine_after_list else 0

    # Save average Q (to data_root)
    Q_avg_file = None
    avg_metadata = {
        'num_scenes': len(Q_list),
        'svd_rank': svd_rank,
        'benchmark': benchmark_name,
        'Y_shape': Y.shape,
        'text_embed_file': text_embed_file,
        'orthogonality_error': float(np.linalg.norm(Q_avg.T @ Q_avg - np.eye(svd_rank), 'fro')),
        'det_Q': float(np.linalg.det(Q_avg)),
        'cosine_before_avg': float(avg_cosine_before),
        'cosine_after_avg': float(avg_cosine_after),
        'cosine_improvement': float(avg_cosine_after - avg_cosine_before),
        'scene_metadata': scene_metadata,
    }

    if save_avg:
        data_root_path = Path(data_root)
        Q_avg_file = data_root_path / f"Q_procrustes_{benchmark_name}_average_r{svd_rank}.npz"
        np.savez_compressed(
            Q_avg_file,
            Q=Q_avg,
            metadata=avg_metadata
        )

    print(f"\n{'='*70}")
    print(f"Results")
    print(f"{'='*70}")
    print(f"  Processed {len(Q_list)} scenes")
    print(f"  Q_avg shape: {Q_avg.shape}")
    print(f"  det(Q_avg): {avg_metadata['det_Q']:.6f}")
    print(f"  Orthogonality error: {avg_metadata['orthogonality_error']:.8e}")
    print(f"  Cosine similarity (before): {avg_metadata['cosine_before_avg']:.4f}")
    print(f"  Cosine similarity (after): {avg_metadata['cosine_after_avg']:.4f}")
    print(f"  Improvement: {avg_metadata['cosine_improvement']:+.4f}")

    # Per-scene statistics from scene_metadata
    if scene_metadata:
        # Extract cosine_before, cosine_after from each scene
        cosine_before_per_scene = [m.get('cosine_before', 0) for m in scene_metadata if 'cosine_before' in m]
        cosine_after_per_scene = [m.get('cosine_after', 0) for m in scene_metadata if 'cosine_after' in m]
        cosine_improvement_per_scene = [m.get('cosine_improvement', 0) for m in scene_metadata if 'cosine_improvement' in m]

        if cosine_before_per_scene and cosine_after_per_scene:
            avg_cosine_before_scene = np.mean(cosine_before_per_scene)
            avg_cosine_after_scene = np.mean(cosine_after_per_scene)
            avg_improvement_scene = np.mean(cosine_improvement_per_scene)

            print(f"\n{'='*70}")
            print(f"Per-Scene Statistics (from {len(scene_metadata)} processed scenes)")
            print(f"{'='*70}")
            print(f"  Cosine similarity (before): {avg_cosine_before_scene:.4f}")
            print(f"  Cosine similarity (after):  {avg_cosine_after_scene:.4f}")
            print(f"  Improvement:                {avg_improvement_scene:+.4f}")

    if save:
        print(f"\n  Individual Q files: <scene_dir>/Q_procrustes_{benchmark_name}_r{svd_rank}.npz")
    if save_avg:
        print(f"  Average Q file: {Q_avg_file}")
    if not save and not save_avg:
        print(f"\n  Q matrices not saved (use --save and/or --save_avg)")
    elif save and not save_avg:
        print(f"  Average Q not saved (use --save_avg to save)")

    return Q_avg, avg_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Compute Procrustes Q matrix per scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing scene subdirectories')
    parser.add_argument('--text_embed', type=str, required=True,
                       help='Path to text embeddings file (.pt or .npy)')
    parser.add_argument('--svd_rank', type=int, default=16,
                       help='SVD rank (default: 16)')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize features (default: True)')
    parser.add_argument('--save', action='store_true', default=False,
                       help='Save individual Q matrices to scene directories (default: False)')
    parser.add_argument('--save_avg', action='store_true', default=False,
                       help='Save average Q matrix to data_root (default: False)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for PyTorch computation (default: cuda)')
    parser.add_argument('--benchmark', action='store_true', default=False,
                       help='Run SVD benchmark for text embeddings')
    parser.add_argument('--label_file', type=str, default=None,
                       help='Name of label file in scene directories (e.g., "segment_nyu_160.npy"). '
                            'If specified, uses compute_procrustes_Q_cuda_with_labels for efficient aggregation')

    args = parser.parse_args()

    print("="*70)
    print("Per-Scene Procrustes Q Matrix Computation")
    print("="*70)
    print(f"Data root: {args.data_root}")
    print(f"Text embeddings: {args.text_embed}")
    print(f"SVD rank: {args.svd_rank}")
    print(f"Save individual Q: {args.save}")
    print(f"Save average Q: {args.save_avg}")
    print(f"Device: {args.device}")
    print(f"Label file: {args.label_file if args.label_file else 'None (using sampling method)'}")
    print(f"\nImplementation:")
    print(f"  - Text embeddings SVD: NumPy (efficient for small matrices)")
    print(f"  - Q matrix computation: PyTorch {args.device} (faster for d×d matrices)")
    if torch.cuda.is_available() and args.device == 'cuda':
        print(f"  CUDA available: True")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  CUDA available: False")

    Q_avg, metadata = compute_procrustes_per_scene(
        args.data_root,
        args.text_embed,
        args.svd_rank,
        args.normalize,
        args.save,
        args.save_avg,
        args.device,
        args.benchmark,
        args.label_file
    )


if __name__ == "__main__":
    main()
