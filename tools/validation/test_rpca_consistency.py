#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Script to Verify RPCA Implementation Consistency

Compares pyrpca library implementation with our custom scipy implementation
to ensure they produce equivalent results.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import pyrpca
try:
    from pyrpca import rpca_pcp_ialm
    PYRPCA_AVAILABLE = True
except ImportError:
    PYRPCA_AVAILABLE = False
    print("WARNING: pyrpca not available. Install with: pip install pyrpca")

# Import our implementations
from rpca_utils import RPCA_CPU


def create_test_matrix(m: int = 100, n: int = 50, rank: int = 5,
                       sparse_ratio: float = 0.05, noise_level: float = 0.01,
                       seed: int = 42) -> np.ndarray:
    """
    Create a synthetic test matrix with low-rank + sparse + noise structure.

    Args:
        m: Number of rows
        n: Number of columns
        rank: Rank of low-rank component
        sparse_ratio: Ratio of sparse outliers
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed

    Returns:
        Synthetic data matrix D = L + S + noise
    """
    rng = np.random.default_rng(seed)

    # Generate low-rank component L
    U = rng.standard_normal((m, rank))
    V = rng.standard_normal((rank, n))
    L = U @ V / np.sqrt(rank)

    # Generate sparse component S
    S = np.zeros((m, n))
    n_sparse = int(m * n * sparse_ratio)
    sparse_indices = rng.choice(m * n, n_sparse, replace=False)
    sparse_values = rng.uniform(-5, 5, n_sparse)  # Larger values for outliers
    S.flat[sparse_indices] = sparse_values

    # Add Gaussian noise
    noise = rng.normal(0, noise_level, (m, n))

    D = L + S + noise
    return D, L, S


def compare_results(pyrpca_L: np.ndarray, pyrpca_S: np.ndarray,
                    scipy_L: np.ndarray, scipy_S: np.ndarray,
                    name: str = "Comparison") -> dict:
    """
    Compare results from two RPCA implementations.

    Returns:
        Dictionary with comparison metrics
    """
    # Check shapes match
    assert pyrpca_L.shape == scipy_L.shape, f"Shape mismatch: {pyrpca_L.shape} vs {scipy_L.shape}"
    assert pyrpca_S.shape == scipy_S.shape, f"Shape mismatch: {pyrpca_S.shape} vs {scipy_S.shape}"

    # Compute absolute differences
    L_diff = np.abs(pyrpca_L - scipy_L)
    S_diff = np.abs(pyrpca_S - scipy_S)

    # Compute relative differences (avoid division by zero)
    L_rel_diff = L_diff / (np.abs(pyrpca_L) + 1e-10)
    S_rel_diff = S_diff / (np.abs(pyrpca_S) + 1e-10)

    metrics = {
        'name': name,
        'L_max_abs_diff': float(np.max(L_diff)),
        'L_mean_abs_diff': float(np.mean(L_diff)),
        'L_max_rel_diff': float(np.max(L_rel_diff)),
        'L_mean_rel_diff': float(np.mean(L_rel_diff)),
        'S_max_abs_diff': float(np.max(S_diff)),
        'S_mean_abs_diff': float(np.mean(S_diff)),
        'S_max_rel_diff': float(np.max(S_rel_diff)),
        'S_mean_rel_diff': float(np.mean(S_rel_diff)),
        'L_correlation': float(np.corrcoef(pyrpca_L.ravel(), scipy_L.ravel())[0, 1]),
        'S_correlation': float(np.corrcoef(pyrpca_S.ravel(), scipy_S.ravel())[0, 1]),
    }

    return metrics


def print_comparison(metrics: dict):
    """Print comparison results."""
    print(f"\n{'='*60}")
    print(f"Test: {metrics['name']}")
    print(f"{'='*60}")
    print(f"Low-rank component (L):")
    print(f"  Max absolute difference: {metrics['L_max_abs_diff']:.6e}")
    print(f"  Mean absolute difference: {metrics['L_mean_abs_diff']:.6e}")
    print(f"  Max relative difference: {metrics['L_max_rel_diff']:.6e}")
    print(f"  Mean relative difference: {metrics['L_mean_rel_diff']:.6e}")
    print(f"  Correlation: {metrics['L_correlation']:.6f}")
    print(f"\nSparse component (S):")
    print(f"  Max absolute difference: {metrics['S_max_abs_diff']:.6e}")
    print(f"  Mean absolute difference: {metrics['S_mean_abs_diff']:.6e}")
    print(f"  Max relative difference: {metrics['S_max_rel_diff']:.6e}")
    print(f"  Mean relative difference: {metrics['S_mean_rel_diff']:.6e}")
    print(f"  Correlation: {metrics['S_correlation']:.6f}")

    # Overall assessment
    L_good = metrics['L_mean_rel_diff'] < 0.1 and metrics['L_correlation'] > 0.95
    S_good = metrics['S_mean_rel_diff'] < 0.1 and metrics['S_correlation'] > 0.95

    if L_good and S_good:
        print(f"\n✓ PASS: Results are consistent")
    else:
        print(f"\n✗ FAIL: Results differ significantly")
        print(f"  Expected: mean_rel_diff < 0.1 and correlation > 0.95")


def test_consistency():
    """Run consistency tests between pyrpca and scipy implementations."""
    print("="*60)
    print("RPCA Implementation Consistency Test")
    print("="*60)

    if not PYRPCA_AVAILABLE:
        print("\nERROR: pyrpca library not available!")
        print("Install with: pip install pyrpca")
        return False

    test_cases = [
        {"m": 100, "n": 50, "rank": 5, "sparse_ratio": 0.05, "seed": 42, "name": "Small matrix (100x50)"},
        {"m": 200, "n": 100, "rank": 10, "sparse_ratio": 0.1, "seed": 123, "name": "Medium matrix (200x100)"},
        {"m": 50, "n": 200, "rank": 3, "sparse_ratio": 0.03, "seed": 456, "name": "Wide matrix (50x200)"},
    ]

    all_results = []

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"Test case: {tc['name']}")
        print(f"{'='*60}")

        # Create test data
        D, L_true, S_true = create_test_matrix(
            m=tc['m'], n=tc['n'], rank=tc['rank'],
            sparse_ratio=tc['sparse_ratio'], seed=tc['seed']
        )

        # Common parameters
        sparsity_factor = 1 / np.sqrt(max(D.shape))  # Same as our lmbda
        max_iter = 1000
        tol = 1e-7

        # Run pyrpca
        print("\n--- Running pyrpca ---")
        pyrpca_L, pyrpca_S = rpca_pcp_ialm(
            observations=D,
            sparsity_factor=sparsity_factor,
            max_iter=max_iter,
            tol=tol,
            verbose=False
        )

        # Run our scipy implementation
        print("--- Running scipy RPCA_CPU ---")
        rpca = RPCA_CPU(D, lmbda=sparsity_factor, n_threads=4)
        scipy_L, scipy_S = rpca.fit(max_iter=max_iter, tol=tol, iter_print=1000)

        # Compare results
        metrics = compare_results(pyrpca_L, pyrpca_S, scipy_L, scipy_S, name=tc['name'])
        all_results.append(metrics)
        print_comparison(metrics)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for m in all_results
                 if m['L_mean_rel_diff'] < 0.1 and m['L_correlation'] > 0.95
                 and m['S_mean_rel_diff'] < 0.1 and m['S_correlation'] > 0.95)
    print(f"Passed: {passed}/{len(all_results)} tests")

    return passed == len(all_results)


def analyze_algorithmic_differences():
    """Analyze and document algorithmic differences between implementations."""
    print(f"\n{'='*60}")
    print("ALGORITHMIC DIFFERENCES ANALYSIS")
    print(f"{'='*60}")

    print("""
Key Differences Found:

1. MU INITIALIZATION:
   - pyrpca:    mu = 1.25 / spectral_norm(D)
   - scipy:     mu = prod(D.shape) / (4 * ||D||_F^2)

2. DUAL VARIABLE INITIALIZATION:
   - pyrpca:    dual = D / max(||D||_2, ||D||_inf / lambda)
   - scipy:     Y = zeros_like(D)

3. MU UPDATE STRATEGY:
   - pyrpca:    mu = min(mu * 1.5, mu * 1e7)  [ADAPTIVE]
   - scipy:     mu = constant                 [FIXED]

4. TOLERANCE:
   - pyrpca:    tol = relative error (||D-L-S||_F / ||D||_F)
   - scipy:     tol = absolute (with different default)

These differences explain why results may differ!
The pyrpca implementation uses adaptive mu which can converge differently.
""")

    return True


if __name__ == "__main__":
    analyze_algorithmic_differences()
    success = test_consistency()
    sys.exit(0 if success else 1)
