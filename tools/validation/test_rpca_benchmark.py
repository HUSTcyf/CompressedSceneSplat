#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RPCA Performance Benchmark: pyrpca vs scipy

Compares the performance of pyrpca library with our scipy-based implementation.
"""

import numpy as np
import sys
import time
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


def create_test_matrix(m: int = 1000, n: int = 500, rank: int = 10,
                       sparse_ratio: float = 0.05, noise_level: float = 0.01,
                       seed: int = 42) -> np.ndarray:
    """Create a synthetic test matrix with low-rank + sparse + noise structure."""
    rng = np.random.default_rng(seed)

    # Generate low-rank component L
    U = rng.standard_normal((m, rank))
    V = rng.standard_normal((rank, n))
    L = U @ V / np.sqrt(rank)

    # Generate sparse component S
    S = np.zeros((m, n))
    n_sparse = int(m * n * sparse_ratio)
    sparse_indices = rng.choice(m * n, n_sparse, replace=False)
    sparse_values = rng.uniform(-5, 5, n_sparse)
    S.flat[sparse_indices] = sparse_values

    # Add Gaussian noise
    noise = rng.normal(0, noise_level, (m, n))

    D = L + S + noise
    return D


def benchmark_pyrpca(D: np.ndarray, max_iter: int = 1000, tol: float = 1e-7,
                     n_runs: int = 3) -> dict:
    """Benchmark pyrpca implementation."""
    if not PYRPCA_AVAILABLE:
        return None

    sparsity_factor = 1 / np.sqrt(max(D.shape))
    results = {
        'times': [],
        'iterations': [],
        'final_errors': [],
    }

    for run in range(n_runs):
        start = time.time()
        L, S = rpca_pcp_ialm(
            observations=D,
            sparsity_factor=sparsity_factor,
            max_iter=max_iter,
            tol=tol,
            verbose=False
        )
        elapsed = time.time() - start
        results['times'].append(elapsed)

        # Get final error
        error = np.linalg.norm(D - L - S, ord='fro') / np.linalg.norm(D, ord='fro')
        results['final_errors'].append(error)

    results['mean_time'] = np.mean(results['times'])
    results['std_time'] = np.std(results['times'])
    results['mean_error'] = np.mean(results['final_errors'])

    return results


def benchmark_scipy(D: np.ndarray, max_iter: int = 1000, tol: float = 1e-7,
                    n_threads: int = 1, n_runs: int = 3) -> dict:
    """Benchmark our scipy-based RPCA_CPU implementation."""
    results = {
        'times': [],
        'iterations': [],
        'final_errors': [],
    }

    for run in range(n_runs):
        rpca = RPCA_CPU(D, lmbda=1 / np.sqrt(max(D.shape)), n_threads=n_threads)
        start = time.time()
        L, S = rpca.fit(max_iter=max_iter, tol=tol, iter_print=10000)
        elapsed = time.time() - start
        results['times'].append(elapsed)

        # Get final error
        error = np.linalg.norm(D - L - S, ord='fro') / np.linalg.norm(D, ord='fro')
        results['final_errors'].append(error)

    results['mean_time'] = np.mean(results['times'])
    results['std_time'] = np.std(results['times'])
    results['mean_error'] = np.mean(results['final_errors'])

    return results


def print_comparison(name: str, pyrpca_results: dict, scipy_results: dict,
                    scipy_threads: int):
    """Print comparison results."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"{'='*70}")

    if pyrpca_results is None:
        print("pyrpca not available - only scipy results shown")
        print(f"\nscipy ({scipy_threads} threads):")
        print(f"  Mean time: {scipy_results['mean_time']:.4f} ± {scipy_results['std_time']:.4f} seconds")
        print(f"  Mean final error: {scipy_results['mean_error']:.6e}")
        return

    print(f"\npyrpca (library):")
    print(f"  Mean time: {pyrpca_results['mean_time']:.4f} ± {pyrpca_results['std_time']:.4f} seconds")
    print(f"  Mean final error: {pyrpca_results['mean_error']:.6e}")

    print(f"\nscipy ({scipy_threads} threads):")
    print(f"  Mean time: {scipy_results['mean_time']:.4f} ± {scipy_results['std_time']:.4f} seconds")
    print(f"  Mean final error: {scipy_results['mean_error']:.6e}")

    speedup = pyrpca_results['mean_time'] / scipy_results['mean_time']
    if speedup > 1:
        print(f"\n  scipy is {speedup:.2f}x FASTER than pyrpca")
    else:
        print(f"\n  scipy is {1/speedup:.2f}x SLOWER than pyrpca")


def run_all_benchmarks():
    """Run all benchmarks."""
    print("="*70)
    print("RPCA Performance Benchmark: pyrpca vs scipy")
    print("="*70)

    if not PYRPCA_AVAILABLE:
        print("\nWARNING: pyrpca library not available!")
        print("Install with: pip install pyrpca")
        print("\nWill only benchmark scipy implementation...")
        print()

    test_cases = [
        {"m": 500, "n": 250, "rank": 5, "name": "Small (500x250)", "threads": 1},
        {"m": 1000, "n": 500, "rank": 10, "name": "Medium (1000x500)", "threads": 1},
        {"m": 2000, "n": 1000, "rank": 15, "name": "Large (2000x1000)", "threads": 1},
        {"m": 1000, "n": 500, "rank": 10, "name": "Medium multi-threaded (1000x500)", "threads": 4},
        {"m": 1000, "n": 500, "rank": 10, "name": "Medium 8 threads (1000x500)", "threads": 8},
    ]

    summary = []

    for tc in test_cases:
        # Create test data
        D = create_test_matrix(m=tc['m'], n=tc['n'], rank=tc['rank'], seed=42)

        # Benchmark pyrpca
        pyrpca_results = benchmark_pyrpca(D, max_iter=500, tol=1e-6, n_runs=3)

        # Benchmark scipy
        scipy_results = benchmark_scipy(D, max_iter=500, tol=1e-6,
                                       n_threads=tc['threads'], n_runs=3)

        # Print comparison
        print_comparison(tc['name'], pyrpca_results, scipy_results, tc['threads'])

        summary.append({
            'name': tc['name'],
            'threads': tc['threads'],
            'pyrpca_time': pyrpca_results['mean_time'] if pyrpca_results else None,
            'scipy_time': scipy_results['mean_time'],
        })

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Test':<30} {'Threads':<10} {'pyrpca (s)':<12} {'scipy (s)':<12} {'Speedup':<10}")
    print("-"*70)

    for s in summary:
        pyrpca_time_str = f"{s['pyrpca_time']:.4f}" if s['pyrpca_time'] else "N/A"
        speedup_str = ""
        if s['pyrpca_time']:
            speedup = s['pyrpca_time'] / s['scipy_time']
            if speedup > 1:
                speedup_str = f"scipy +{speedup:.2f}x"
            else:
                speedup_str = f"scipy {1/speedup:.2f}x"

        print(f"{s['name']:<30} {s['threads']:<10} {pyrpca_time_str:<12} {s['scipy_time']:<12.4f} {speedup_str:<10}")

    print("\nConclusion:")
    print("- If scipy is faster with multiple threads, it demonstrates the benefit")
    print("  of multi-threaded BLAS for SVD operations.")
    print("- Single-threaded comparison shows pure algorithm efficiency.")


if __name__ == "__main__":
    run_all_benchmarks()
