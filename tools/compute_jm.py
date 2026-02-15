#!/usr/bin/env python3
"""
Compute Average Jeffries-Matusita (JM) Distance Between Different Classes

This script calculates the average JM distance between different semantic classes
using either Gaussian distribution assumption or histogram-based estimation.

Higher JM distance = Better separability between classes
Lower JM distance = Classes overlap more in the feature space

Usage:
    python tools/compute_jm.py --data-root /path/to/data --rank 16

Formula:
    Gaussian: JM = sqrt(2 * (1 - exp(-DB)))
    Histogram: JM = sqrt(2 * (1 - sum(sqrt(p_i * q_i))))
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

import numpy as np
from scipy.linalg import inv, det
from itertools import combinations
from sklearn.neighbors import KernelDensity
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
reg = 1e-10
eps = 1e-10

def compute_gaussian_params(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance matrix for a set of features.

    Args:
        features: Feature array, shape (N, D)

    Returns:
        (mean, covariance) where mean is (D,) and covariance is (D, D)
    """
    mu = np.mean(features, axis=0)

    # Compute covariance with regularization for numerical stability
    sigma = np.cov(features, rowvar=False) + np.eye(features.shape[1]) * reg

    return mu, sigma


def compute_jm_gaussian(mu1: np.ndarray, sigma1: np.ndarray,
                        mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """
    Compute Jeffries-Matusita (JM) Distance between two Gaussian distributions.

    Analytical formula for multivariate Gaussians:
        DB = (1/8)*(μ1-μ2)^T * Σ^(-1) * (μ1-μ2)
        where Σ = (Σ1 + Σ2) / 2

    JM = sqrt(2 * (1 - exp(-DB)))

    Args:
        mu1: Mean of distribution 1, shape (D,)
        sigma1: Covariance of distribution 1, shape (D, D)
        mu2: Mean of distribution 2, shape (D,)
        sigma2: Covariance of distribution 2, shape (D, D)

    Returns:
        JM distance (0 to sqrt(2))
    """
    d = len(mu1)

    # Average covariance
    sigma = (sigma1 + sigma2) / 2

    # Difference in means
    diff = mu1 - mu2

    # Add regularization for numerical stability
    sigma_reg = sigma + np.eye(d) * reg

    try:
        # Inverse of sigma
        sigma_inv = inv(sigma_reg)

        # First term: (1/8) * (μ1-μ2)^T * Σ^(-1) * (μ1-μ2)
        term1 = 0.125 * diff @ sigma_inv @ diff

        # Second term: (1/2) * ln(det(Σ) / sqrt(det(Σ1)*det(Σ2)))
        det_sigma = det(sigma_reg)
        det_sigma1 = det(sigma1 + np.eye(d) * reg)
        det_sigma2 = det(sigma2 + np.eye(d) * reg)

        # Avoid log of zero
        det_sigma = max(det_sigma, 1e-10)
        det_sigma1 = max(det_sigma1, 1e-10)
        det_sigma2 = max(det_sigma2, 1e-10)

        ratio = det_sigma / np.sqrt(det_sigma1 * det_sigma2)
        term2 = 0.5 * np.log(ratio)

        # Bhattacharyya Distance
        DB = term1 + term2

        # JM distance
        JM = np.sqrt(2 * (1 - np.exp(-DB)))

    except np.linalg.LinAlgError:
        # Fallback for singular matrices
        term1 = np.linalg.norm(diff) / (np.sqrt(np.trace(sigma1)) * np.sqrt(np.trace(sigma2)))
        term2 = 0.5 * np.log((np.trace(sigma)) / np.sqrt(np.trace(sigma1) * np.trace(sigma2)))
        DB = term1 + term2
        JM = np.sqrt(2 * (1 - np.exp(-DB)))

    return JM


def compute_jm_histogram(feat1: np.ndarray, feat2: np.ndarray,
                      n_bins: int = 50) -> float:
    """
    Compute JM distance using histogram-based probability estimation.

    Projects features to 1D using L2 norm, then computes JM on histograms.

    Args:
        feat1: Features from distribution 1, shape (N1, D1)
        feat2: Features from distribution 2, shape (N2, D2)
        n_bins: Number of histogram bins

    Returns:
        JM distance
    """
    # Compute L2 norm (magnitude) for each feature vector
    norm1 = np.linalg.norm(feat1, axis=1)
    norm2 = np.linalg.norm(feat2, axis=1)

    # Create common bin range based on percentiles
    all_values = np.concatenate([norm1, norm2])
    bins = np.linspace(
        np.percentile(all_values, 0.1),
        np.percentile(all_values, 99.9),
        n_bins
    )

    # Compute histograms (probability distributions)
    hist1, _ = np.histogram(norm1, bins=bins, density=True)
    hist2, _ = np.histogram(norm2, bins=bins, density=True)

    # Add small epsilon to avoid numerical issues
    hist1 = np.maximum(hist1, 0) + eps
    hist2 = np.maximum(hist2, 0) + eps

    # Normalize to ensure sum = 1
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()

    # Compute Bhattacharyya coefficient: BC = sum(sqrt(p_i * q_i))
    BC = np.sum(np.sqrt(hist1 * hist2))

    # JM distance: JM = sqrt(2 * (1 - BC))
    JM = np.sqrt(2 * (1 - BC))

    return JM


def compute_jm_multidim_gpu(feat1: np.ndarray, feat2: np.ndarray,
                            n_bins: int = 30,
                            aggregation: str = 'mean',
                            device: str = 'cuda') -> float:
    """
    Compute JM distance using GPU-accelerated multi-dimensional histograms.

    This method computes 1D histograms for each dimension in parallel on GPU,
    then combines them to estimate the multi-dimensional JM distance.

    Args:
        feat1: Features from distribution 1, shape (N1, D)
        feat2: Features from distribution 2, shape (N2, D)
        n_bins: Number of histogram bins per dimension (default: 30)
        aggregation: How to combine per-dimension BC values (default: 'mean'):
            - 'product': Product over dimensions (ASSUMES INDEPENDENCE - NOT RECOMMENDED for high-D)
            - 'mean': Arithmetic mean (RECOMMENDED for high-dimensional features)
            - 'geometric': Geometric mean (intermediate option)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        JM distance (0 to sqrt(2) ≈ 1.414)

    Aggregation Method Comparison:
        For high-dimensional features (e.g., 768-dim) with bc_per_dim = 0.986:

        - 'product': bc_joint = 0.986^768 ≈ 0.00003, JM ≈ 1.414 (MAXIMUM - unreliable)
        - 'mean':   bc_joint = 0.986,             JM ≈ 0.16  (reasonable)
        - 'geometric': bc_joint = 0.986,           JM ≈ 0.16  (same as mean for uniform dims)

    RECOMMENDATION: Use aggregation='mean' for high-dimensional features to avoid
    dimension independence assumption issues.

    Note:
        - 'mean' aggregation computes average JM across dimensions
        - 'geometric' uses exp(mean(log(bc_per_dim))) which is more robust to outliers
    """
    # Check if CUDA is available
    assert torch.cuda.is_available()

    # Convert to torch tensors with half precision to save memory
    # Use float16 for features, convert to float32 for quantile computation
    feat1_tensor = torch.from_numpy(feat1)
    feat2_tensor = torch.from_numpy(feat2)

    # Stack features for combined bin edge computation (use float32 for quantile)
    all_feat = torch.cat([feat1_tensor, feat2_tensor], dim=0).float()  # (N1+N2, D)

    # Compute bin edges for each dimension (percentile-based)
    # Using quantile for robustness
    lower = torch.quantile(all_feat, 0.001, dim=0, keepdim=True).to(device)  # (1, D)
    upper = torch.quantile(all_feat, 0.999, dim=0, keepdim=True).to(device)  # (1, D)

    # Free memory after computing bin edges
    del all_feat

    feat1_tensor = feat1_tensor.to(dtype=torch.float16, device=device)
    feat2_tensor = feat2_tensor.to(dtype=torch.float16, device=device)

    # Create bin edges for each dimension (use half precision)
    # Shape: (n_bins + 1, D)
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=device).unsqueeze(1)  # (n_bins+1, 1)
    bin_edges = (lower + bin_edges * (upper - lower)).to(dtype=torch.float16)  # (n_bins+1, D)

    # Transpose bin_edges for comparison: (n_bins+1, D) -> (D, n_bins+1)
    bin_edges_t = bin_edges.T  # (D, n_bins+1)

    # Compute histogram counts using digitize-like operation
    # For each dimension, find which bin each sample belongs to
    def compute_histogram(features, bin_edges_t):
        """
        Compute histogram counts for all dimensions in parallel.

        Uses searchsorted to avoid large (N, D, n_bins) intermediate tensors.

        Args:
            features: (N, D) tensor
            bin_edges_t: (D, n_bins+1) tensor, bin edges per dimension

        Returns:
            hist: (D, n_bins) tensor, histogram counts per dimension
        """
        _, D = features.shape

        # Pre-allocate hist tensor to avoid list append operations
        hist = torch.zeros((D, n_bins), dtype=torch.float32, device=device)

        # Process each dimension independently to avoid large intermediate tensors
        for d in range(D):
            feat_d = features[:, d].contiguous()  # (N,)
            edges_d = bin_edges_t[d, :].contiguous()  # (n_bins+1,)

            # Use searchsorted to find bin indices (memory efficient)
            # This returns (N,) tensor of bin indices
            bin_indices = torch.searchsorted(edges_d, feat_d, right=False)

            # Clip to valid range [0, n_bins-1]
            bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)

            # Count occurrences using bincount (much more efficient)
            hist[d, :] = torch.bincount(bin_indices, minlength=n_bins).float()

        return hist

    # Compute histograms for both distributions
    hist1 = compute_histogram(feat1_tensor, bin_edges_t)  # (D, n_bins)
    hist2 = compute_histogram(feat2_tensor, bin_edges_t)  # (D, n_bins)

    # Convert to probability distributions
    hist1 = hist1 / (hist1.sum(dim=1, keepdim=True) + eps)  # (D, n_bins)
    hist2 = hist2 / (hist2.sum(dim=1, keepdim=True) + eps)  # (D, n_bins)

    # Add epsilon to avoid numerical issues
    hist1 = torch.clamp(hist1, min=eps)
    hist2 = torch.clamp(hist2, min=eps)

    # Compute Bhattacharyya coefficient for each dimension
    # BC_dim = sum(sqrt(p_i * q_i)) over bins
    bc_per_dim = torch.sum(torch.sqrt(hist1 * hist2), dim=1)  # (D,)

    # ============================================================================
    # Combine per-dimension BC values using specified aggregation method
    # ============================================================================
    # The key question is how to combine bc_per_dim into bc_joint

    if aggregation == 'product':
        # Original method: ASSUMES INDEPENDENCE across dimensions
        # bc_joint = product(bc_per_dim[d] for d in D)
        # This causes "dimension penalty" for high-dimensional features
        log_bc_joint = torch.sum(torch.log(bc_per_dim))
        bc_joint = torch.exp(log_bc_joint)

    elif aggregation == 'mean':
        # Arithmetic mean: NO independence assumption (RECOMMENDED for high-D)
        # bc_joint = mean(bc_per_dim)
        # This gives equal weight to each dimension, avoiding dimension penalty
        bc_joint = torch.mean(bc_per_dim)

    elif aggregation == 'geometric':
        # Geometric mean: intermediate option
        # bc_joint = exp(mean(log(bc_per_dim)))
        # More robust to outliers than arithmetic mean
        log_bc_joint = torch.mean(torch.log(bc_per_dim))
        bc_joint = torch.exp(log_bc_joint)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}. Use 'product', 'mean', or 'geometric'")

    # Clamp to valid range
    bc_joint = torch.clamp(bc_joint, 0.0, 1.0)

    # JM distance: JM = sqrt(2 * (1 - BC))
    jm = torch.sqrt(2 * (1 - bc_joint))

    return jm.item()


def compute_jm_kde(feat1: np.ndarray, feat2: np.ndarray,
                   n_samples: int = 2000,
                   bandwidth: str = 'silverman') -> float:
    """
    Compute JM distance using Kernel Density Estimation (KDE).

    This method estimates the probability density function non-parametrically
    using KDE and computes the Bhattacharyya coefficient via Monte Carlo sampling.

    Args:
        feat1: Features from distribution 1, shape (N1, D)
        feat2: Features from distribution 2, shape (N2, D)
        n_samples: Number of samples for KDE and MC integration (default: 2000)
        bandwidth: Bandwidth selection method ('silverman' or 'scott')

    Returns:
        JM distance
    """
    from sklearn.neighbors import KernelDensity

    # Subsample if too many samples for computational efficiency
    if len(feat1) > n_samples:
        idx1 = np.random.choice(len(feat1), n_samples, replace=False)
        feat1_sub = feat1[idx1]
    else:
        feat1_sub = feat1

    if len(feat2) > n_samples:
        idx2 = np.random.choice(len(feat2), n_samples, replace=False)
        feat2_sub = feat2[idx2]
    else:
        feat2_sub = feat2

    # Fit KDE for each distribution
    kde1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde1.fit(feat1_sub)

    kde2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde2.fit(feat2_sub)

    # Monte Carlo integration for Bhattacharyya coefficient
    # Sample from combined distribution
    combined = np.vstack([feat1_sub, feat2_sub])
    mc_samples = combined[np.random.choice(len(combined), min(n_samples, len(combined)), replace=False)]

    # Compute log-densities
    log_p1 = kde1.score_samples(mc_samples)
    log_p2 = kde2.score_samples(mc_samples)

    # sqrt(p(x) * q(x)) = exp(0.5 * (log(p(x)) + log(q(x))))
    sqrt_pq = np.exp(0.5 * (log_p1 + log_p2))

    # Estimate BC with normalization correction
    BC = np.mean(sqrt_pq)
    BC = BC / (np.sqrt(np.mean(np.exp(log_p1))) * np.sqrt(np.mean(np.exp(log_p2))))
    BC = np.clip(BC, 0, 1)

    # JM distance
    JM = np.sqrt(2 * (1 - BC))

    return JM


def compute_jm_random_projection(feat1: np.ndarray, feat2: np.ndarray,
                                   n_projections: int = 50,
                                   n_bins: int = 30,
                                   random_state: int = 42,
                                   device: str = 'cuda') -> float:
    """
    Compute JM distance using random projection aggregation.

    Projects features onto random 1D directions, computes 1D JM distance
    for each projection, and aggregates results.

    This is more efficient than full multi-dimensional histograms while
    capturing more information than just L2 norm.

    Args:
        feat1: Features from distribution 1, shape (N1, D)
        feat2: Features from distribution 2, shape (N2, D)
        n_projections: Number of random projections (default: 50)
        n_bins: Number of histogram bins per projection (default: 30)
        random_state: Random seed for reproducibility
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Average JM distance across projections
    """
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    D = feat1.shape[1]

    # Convert to torch tensors with half precision to save memory
    feat1_tensor = torch.from_numpy(feat1).to(dtype=torch.float16, device=device)
    feat2_tensor = torch.from_numpy(feat2).to(dtype=torch.float16, device=device)

    # Generate random projection directions (half precision)
    directions = torch.randn(n_projections, D, dtype=torch.float16, device=device)
    directions = directions / directions.norm(dim=1, keepdim=True)

    # Project features onto all directions at once (result is float16)
    # proj1: (n_projections, N1)
    proj1 = (directions @ feat1_tensor.T)  # (n_projections, D) @ (D, N1) = (n_projections, N1)
    proj2 = (directions @ feat2_tensor.T)  # (n_projections, N2)

    # Convert to numpy for histogram computation (could be done on GPU too)
    proj1_np = proj1.cpu().numpy()
    proj2_np = proj2.cpu().numpy()

    jm_values = []

    for i in range(n_projections):
        p1 = proj1_np[i]
        p2 = proj2_np[i]

        # Create common bin range
        all_values = np.concatenate([p1, p2])
        bins = np.linspace(
            np.percentile(all_values, 0.1),
            np.percentile(all_values, 99.9),
            n_bins
        )

        # Compute histograms
        hist1, _ = np.histogram(p1, bins=bins, density=True)
        hist2, _ = np.histogram(p2, bins=bins, density=True)

        # Add epsilon and normalize
        hist1 = np.maximum(hist1, 0) + eps
        hist2 = np.maximum(hist2, 0) + eps
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        # Compute BC and JM for this projection
        BC = np.sum(np.sqrt(hist1 * hist2))
        JM = np.sqrt(2 * (1 - BC))
        jm_values.append(JM)

    # Return mean JM across projections
    return np.mean(jm_values)


def compute_pairwise_jm_distances(features: np.ndarray, labels: np.ndarray,
                                  method: str = 'gaussian',
                                  n_bins: int = 50,
                                  aggregation: str = 'mean',
                                  device: str = 'cuda') -> Tuple[Dict[Tuple[int, int], float], int]:
    """
    Compute pairwise JM distances between all classes.

    Args:
        features: Feature array, shape (N, D)
        labels: Class labels, shape (N,)
        method: JM distance computation method:
            - 'gaussian': Analytical Gaussian (default)
            - 'histogram': 1D histogram (L2 norm based)
            - 'multidim_gpu': Multi-dimensional histogram with GPU (uses mean aggregation by default)
            - 'kde': Kernel Density Estimation
            - 'random_proj': Random projection aggregation
        n_bins: Number of histogram bins (for histogram-based methods)
        aggregation: Aggregation method for multidim_gpu (default: 'mean'):
            - 'product': Product over dimensions (ASSUMES INDEPENDENCE - NOT recommended for high-D)
            - 'mean': Arithmetic mean (RECOMMENDED for high-dimensional features)
            - 'geometric': Geometric mean (intermediate option)
        device: Device for GPU-accelerated methods ('cuda' or 'cpu')

    Returns:
        (pairwise_distances, n_pairs) where:
        - pairwise_distances: Dictionary mapping (class_i, class_j) to JM distance
        - n_pairs: Number of valid class pairs computed
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]  # Ignore invalid label -1

    class_params = {}

    # Compute parameters for each class
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 10:  # Need at least 10 samples
            class_features = features[mask]
            mu = np.mean(class_features, axis=0)

            if method == 'gaussian':
                sigma = np.cov(class_features, rowvar=False) + np.eye(class_features.shape[1]) * reg
                class_params[int(label)] = (mu, sigma)
            else:  # For histogram-based methods, just store label for lookup
                class_params[int(label)] = (mu, None)

    # Compute pairwise JM distances
    pairwise_distances = {}
    n_pairs = 0

    for i, j in combinations(sorted(unique_labels), 2):
        if i in class_params and j in class_params:
            # Extract features for each class
            mask_i = labels == i
            mask_j = labels == j
            feat_i = features[mask_i]
            feat_j = features[mask_j]

            # Compute JM based on method
            if method == 'gaussian':
                mu_i, sigma_i = class_params[i]
                mu_j, sigma_j = class_params[j]
                JM = compute_jm_gaussian(mu_i, sigma_i, mu_j, sigma_j)
            elif method == 'histogram':
                JM = compute_jm_histogram(feat_i, feat_j, n_bins=n_bins)
            elif method == 'multidim_gpu':
                JM = compute_jm_multidim_gpu(feat_i, feat_j, n_bins=n_bins, aggregation=aggregation, device=device)
            elif method == 'kde':
                JM = compute_jm_kde(feat_i, feat_j)
            elif method == 'random_proj':
                JM = compute_jm_random_projection(feat_i, feat_j, n_bins=n_bins, device=device)
            else:
                raise ValueError(f"Unknown method: {method}")

            pairwise_distances[(i, j)] = JM
            n_pairs += 1

    return pairwise_distances, n_pairs


def compute_average_jm_distance(pairwise_distances: Dict[Tuple[int, int], float]) -> float:
    """
    Compute average JM distance from pairwise distances.

    Args:
        pairwise_distances: Dictionary of (class_i, class_j) -> JM distance

    Returns:
        Average JM distance across all class pairs
    """
    if not pairwise_distances:
        return 0.0

    jm_values = list(pairwise_distances.values())
    return np.mean(jm_values)


def find_scene_data(data_root: str) -> Dict[str, Dict[str, Path]]:
    """Find all scenes with both lang_feat.npy and lang_feat_svd.npz files."""
    data_root = Path(data_root)
    scenes = {}

    for lang_feat_path in data_root.glob("*/train/*/lang_feat.npy"):
        scene_dir = lang_feat_path.parent
        scene_name = scene_dir.name

        svd_path = scene_dir / "lang_feat_svd.npz"
        if svd_path.exists():
            dataset_name = scene_dir.parent.parent.name
            key = f"{dataset_name}/{scene_name}"
            scenes[key] = {
                "original": lang_feat_path,
                "svd": svd_path,
                "scene_name": scene_name,
                "dataset": dataset_name,
                "scene_dir": scene_dir
            }

    return scenes


def load_features(scene_info: Dict, rank: int = 16,
                  max_samples: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load original features, compressed features, and labels for a scene.

    Returns:
        (original_features, compressed_features, labels)
    """
    scene_dir = scene_info["original"].parent

    # Load features
    original = np.load(scene_info["original"])
    svd_data = np.load(scene_info["svd"])
    U = svd_data["U"]
    S = svd_data["S"]

    # Load labels
    label_path = scene_dir / "lang_label.npy"
    label_mask_path = scene_dir / "lang_label_mask.npy"

    if label_path.exists():
        labels = np.load(label_path)
    else:
        print(f"  Warning: {scene_info['scene_name']} has no lang_label.npy")
        labels = np.full(len(original), -1, dtype=int)
    
    if label_mask_path.exists():
        label_mask = np.load(label_mask_path).astype(bool)
    else:
        print(f"  Warning: {scene_info['scene_name']} has no lang_label_mask.npy")
        label_mask = np.full(len(original), False, dtype=bool)

    # Load valid_feat_mask
    valid_feat_mask_path = scene_dir / "valid_feat_mask.npy"

    if valid_feat_mask_path.exists():
        valid_feat_mask = np.load(valid_feat_mask_path).astype(bool)
    else:
        print(f"  Warning: {scene_info['scene_name']} has no valid_feat_mask.npy")
        valid_feat_mask = np.ones(len(original), dtype=bool)

    # 1. Apply valid_feat_mask to label_mask to get filtered_label_mask
    if valid_feat_mask_path.exists():
        filtered_label_mask = label_mask & valid_feat_mask if label_mask_path.exists() else valid_feat_mask
    else:
        filtered_label_mask = label_mask

    # 2. Apply filtered_label_mask to U (U shape depends on valid_feat_mask)
    if valid_feat_mask_path.exists():
        if label_mask_path.exists():
            # lang_label_mask = np.load(label_mask_path).astype(bool)
            # Intersection: both valid_feat AND lang_label
            mask_for_u = label_mask[valid_feat_mask]
            U = U[mask_for_u]
        else:
            # No lang_label_mask file
            print(f"  Warning: No lang_label_mask found, using U as is")
    else:
        # No valid_feat_mask file
        print(f"  Warning: No valid_feat_mask found, using U as is")

    # 3. Apply same filtered_label_mask to original and labels
    original = original[filtered_label_mask]
    labels = labels[filtered_label_mask]

    # Reconstruct compressed features
    U_r = U[:, :rank]
    S_r = S[:rank]
    compressed = U_r * S_r

    # Subsample if needed
    if max_samples is not None and len(original) > max_samples:
        indices = np.random.choice(len(original), max_samples, replace=False)
        original = original[indices]
        compressed = compressed[indices]
        labels = labels[indices]

    return original, compressed, labels


def main():
    parser = argparse.ArgumentParser(
        description='Compute Average JM Distance Between Classes'
    )
    parser.add_argument('--data-root', type=str,
                       default='/new_data/cyf/projects/SceneSplat/gaussian_train',
                       help='Path to training data directory')
    parser.add_argument('--rank', type=int, default=16,
                       help='SVD rank for compression (default: 16)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per scene (default: use all)')
    parser.add_argument('--method', type=str, default='gaussian',
                       choices=['gaussian', 'histogram', 'multidim_gpu', 'kde', 'random_proj'],
                       help='JM distance computation method (default: gaussian):\n'
                            '  gaussian     - Analytical Gaussian (fast, assumes normality)\n'
                            '  histogram    - 1D histogram (L2 norm based)\n'
                            '  multidim_gpu - Multi-dimensional histogram with GPU (uses mean aggregation by default)\n'
                            '  kde          - Kernel Density Estimation (non-parametric)\n'
                            '  random_proj  - Random projection aggregation\n'
                            'Recommended for high-D features: gaussian or multidim_gpu')
    parser.add_argument('--aggregation', type=str, default='mean',
                       choices=['product', 'mean', 'geometric'],
                       help='Aggregation method for multidim_gpu (default: mean):\n'
                            '  product  - Product over dimensions (ASSUMES INDEPENDENCE - NOT recommended for high-D)\n'
                            '  mean     - Arithmetic mean (RECOMMENDED for high-dimensional features)\n'
                            '  geometric - Geometric mean (intermediate option)')
    parser.add_argument('--bins', type=int, default=30,
                       help='Number of histogram bins (for histogram-based methods, default: 30)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    parser.add_argument('--per-scene', action='store_true',
                       help='Show per-scene results')
    parser.add_argument('--per-class-pair', action='store_true',
                       help='Show per-class-pair JM distances')

    args = parser.parse_args()

    print("=" * 70)
    print("Average JM Distance Calculator")
    print("=" * 70)
    print(f"Data root:      {args.data_root}")
    print(f"SVD rank:       {args.rank}")
    print(f"Max samples:    {args.max_samples}")
    print(f"Per-scene:      {args.per_scene}")
    print(f"Per-class-pair: {args.per_class_pair}")
    print(f"Method:         {args.method}")
    if args.method in ['histogram', 'multidim_gpu', 'random_proj']:
        print(f"Histogram bins:  {args.bins}")

    # Find all scene data
    print(f"\nSearching for scenes in {args.data_root}...")
    scenes = find_scene_data(args.data_root)

    if not scenes:
        print("No scenes found with both lang_feat.npy and lang_feat_svd.npz files!")
        return

    print(f"Found {len(scenes)} scenes")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Compute results
    results = {}
    original_jm_values = []
    compressed_jm_values = []

    for scene_key in sorted(scenes.keys()):
        scene_info = scenes[scene_key]
        print(f"\n[{list(scenes.keys()).index(scene_key) + 1}/{len(scenes)}] Processing {scene_key}...")

        try:
            original, compressed, labels = load_features(
                scene_info, rank=args.rank, max_samples=args.max_samples
            )

            n_valid = np.sum(labels >= 0)
            n_classes = len(np.unique(labels[labels >= 0]))

            print(f"  Samples: {len(original)} (valid: {n_valid}, classes: {n_classes})")

            # Skip if no valid labels
            if n_classes < 2:
                print(f"  Skipped: need at least 2 valid classes")
                continue

            # Compute pairwise JM distances for original features
            pairwise_orig, n_pairs_orig = compute_pairwise_jm_distances(
                original, labels, method=args.method, n_bins=args.bins, aggregation=args.aggregation, device='cuda'
            )
            jm_original = compute_average_jm_distance(pairwise_orig)
            original_jm_values.append(jm_original)

            print(f"  Original features:   Avg JM = {jm_original:.4f} ({n_pairs_orig} pairs)")

            # Compute pairwise JM distances for compressed features
            pairwise_comp, n_pairs_comp = compute_pairwise_jm_distances(
                compressed, labels, method=args.method, n_bins=args.bins, aggregation=args.aggregation, device='cuda'
            )
            jm_compressed = compute_average_jm_distance(pairwise_comp)
            compressed_jm_values.append(jm_compressed)

            print(f"  Compressed features: Avg JM = {jm_compressed:.4f} ({n_pairs_comp} pairs)")

            # Store results
            results[scene_key] = {
                "original_jm": float(jm_original),
                "compressed_jm": float(jm_compressed),
                "n_classes": int(n_classes),
                "n_samples": int(len(original)),
                "n_pairs_original": int(n_pairs_orig),
                "n_pairs_compressed": int(n_pairs_comp)
            }

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute overall statistics
    print(f"\n{'=' * 70}")
    print("Overall Results Summary")
    print(f"{'=' * 70}")

    n_scenes = len(original_jm_values)

    if n_scenes == 0:
        print("No valid results!")
        return

    # Original features statistics
    jm_orig_mean = np.mean(original_jm_values)
    jm_orig_std = np.std(original_jm_values)
    jm_orig_min = np.min(original_jm_values)
    jm_orig_max = np.max(original_jm_values)

    print(f"\n--- Original Features (768-dim) ---")
    print(f"Scenes:        {n_scenes}")
    print(f"Mean JM:       {jm_orig_mean:.4f} ± {jm_orig_std:.4f}")
    print(f"Median JM:     {np.median(original_jm_values):.4f}")
    print(f"Min JM:        {jm_orig_min:.4f}")
    print(f"Max JM:        {jm_orig_max:.4f}")

    # Compressed features statistics
    jm_comp_mean = np.mean(compressed_jm_values)
    jm_comp_std = np.std(compressed_jm_values)
    jm_comp_min = np.min(compressed_jm_values)
    jm_comp_max = np.max(compressed_jm_values)

    print(f"\n--- Compressed Features ({args.rank}-dim SVD) ---")
    print(f"Scenes:        {n_scenes}")
    print(f"Mean JM:       {jm_comp_mean:.4f} ± {jm_comp_std:.4f}")
    print(f"Median JM:     {np.median(compressed_jm_values):.4f}")
    print(f"Min JM:        {jm_comp_min:.4f}")
    print(f"Max JM:        {jm_comp_max:.4f}")

    # Comparison
    jm_change = ((jm_comp_mean - jm_orig_mean) / jm_orig_mean) * 100
    print(f"\n--- Comparison ({args.method} method) ---")
    print(f"JM distance change: {jm_change:+.1f}%")
    print(f"(Positive = better separability, Negative = worse separability)")

    if args.per_scene:
        print(f"\nPer-scene JM distances:")
        print(f"{'Scene':35s} | {'Dataset':12s} | {'Original JM':>12s} | {'Compressed JM':>12s}")
        print(f"{'-'*70}")
        for scene_key in sorted(results.keys()):
            r = results[scene_key]
            dataset = scene_key.split('/')[0]
            scene = scene_key.split('/')[1]
            print(f"{scene:35s} | {dataset:12s} | {r['original_jm']:9.4f} | {r['compressed_jm']:9.4f}")

    if args.per_class_pair:
        # Show one example of pairwise distances
        if results:
            first_scene = list(results.keys())[0]
            print(f"\nExample pairwise JM distances for {first_scene} ({args.method} method):")
            scene_info = scenes[first_scene]
            try:
                original, _, labels = load_features(scene_info, rank=args.rank, max_samples=1000)
                pairwise_orig, _ = compute_pairwise_jm_distances(
                    original, labels, method=args.method, n_bins=args.bins, aggregation=args.aggregation, device='cuda'
                )

                # Get unique labels
                unique_labels = sorted([l for l in np.unique(labels) if l >= 0])

                print(f"Class labels: {unique_labels}")
                print(f"Class pair JM distances:")
                print(f"{'Class 1':>12s} | {'Class 2':>12s} | {'JM Distance':>12s}")
                for (i, j), jm in sorted(pairwise_orig.items(), key=lambda x: x[0]*1000 + x[1])[:15]:
                    label1_name = f"class_{i}" if i < 10 else f"class_{i}"
                    label2_name = f"class_{j}" if j < 10 else f"class_{j}"
                    print(f"{label1_name:>12s} | {label2_name:>12s} | {jm:.4f}")
            except:
                pass

    # Save results if output path provided
    if args.output:
        output_data = {
            "original_features": {
                "mean_jm_distance": float(jm_orig_mean),
                "std_jm_distance": float(jm_orig_std),
                "median_jm_distance": float(np.median(original_jm_values)),
                "min_jm_distance": float(jm_orig_min),
                "max_jm_distance": float(jm_orig_max),
                "n_scenes": n_scenes
            },
            "compressed_features": {
                "mean_jm_distance": float(jm_comp_mean),
                "std_jm_distance": float(jm_comp_std),
                "median_jm_distance": float(np.median(compressed_jm_values)),
                "min_jm_distance": float(jm_comp_min),
                "max_jm_distance": float(jm_comp_max),
                "n_scenes": n_scenes
            },
            "comparison": {
                "jm_change_percent": float(jm_change),
                "method_used": args.method
            },
            "per_scene": results,
            "config": {
                "svd_rank": args.rank,
                "max_samples": args.max_samples,
                "method": args.method,
                "histogram_bins": args.bins if args.method in ['histogram', 'multidim_gpu', 'random_proj'] else None
            }
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
