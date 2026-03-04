#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPCA Performance Benchmark on LERF Dataset

Compares three RPCA variants:
1. rpca-cpu: CPU-based using pyrpca library
2. rpca-gpu: GPU-accelerated (SVD and Shrinkage on GPU, data on CPU)
3. structure-rpca: Structured RPCA with weighted SVD (all GPU)

Measures:
- Per-iteration latency
- Final error after 30 iterations
- Total execution time
- Memory usage

Author: SceneSplat Team
"""

import numpy as np
import torch
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import gc

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import RPCA utilities
from rpca_utils import RPCA_CPU, RPCA_GPU, StructuredRPCA_GPU, CUDA_AVAILABLE, RPCA_CPU_AVAILABLE


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single run."""
    method: str
    iterations: List[float]  # Per-iteration times
    total_time: float
    final_error: float
    converged: bool
    final_iterations: int
    memory_mb: float


@dataclass
class BenchmarkSummary:
    """Summary of all benchmarks."""
    results: List[BenchmarkResult]
    data_shape: Tuple[int, int]
    data_size_mb: float


def get_gpu_memory() -> float:
    """Get current GPU memory usage in MB."""
    if CUDA_AVAILABLE:
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def clear_gpu_cache():
    """Clear GPU cache."""
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        gc.collect()


def load_lerf_features(scene_name: str = "figurines",
                       data_root: str = "/new_data/cyf/projects/SceneSplat/output_features") -> np.ndarray:
    """
    Load LERF features from checkpoint.

    Args:
        scene_name: Name of the scene (figurines, ramen, teatime, waldo_kitchen)
        data_root: Root directory for features

    Returns:
        Feature matrix [N, D]
    """
    ckpt_path = Path(data_root) / scene_name / "checkpoint_with_features.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading LERF features from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Extract language features (item 0[7]: [N, 16])
    item0 = ckpt[0]
    lang_feat = item0[7] if isinstance(item0[7], np.ndarray) else item0[7].numpy()
    valid_mask = item0[8] if isinstance(item0[8], np.ndarray) else item0[8].numpy()
    segment_labels = item0[13]

    # Filter valid features and labels together
    valid_indices = valid_mask > 0
    lang_feat_valid = lang_feat[valid_indices]
    labels_valid = segment_labels[valid_indices]

    # Filter out background/invalid labels
    valid_label_mask = labels_valid >= 0
    lang_feat_valid = lang_feat_valid[valid_label_mask]

    print(f"  Loaded shape: {lang_feat_valid.shape}")
    print(f"  Data size: {lang_feat_valid.nbytes / (1024**2):.2f} MB")

    return lang_feat_valid.astype(np.float32)


def prepare_structured_rpca_data(features: np.ndarray,
                                  grid_size: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for StructuredRPCA_GPU.

    Simulates a grid structure where features are assigned to grid cells.
    Rows with the same grid cell are duplicated and should use weighted SVD.

    Args:
        features: Input feature matrix [N, D]
        grid_size: Size of the grid (grid_size x grid_size)

    Returns:
        A_u: Upper triangular matrix [r, n] where r <= n
        indices: Index mapping array [n]
        d: Repetition count array [r]
    """
    N, D = features.shape

    # Assign each feature to a grid cell
    n_grid_cells = grid_size * grid_size
    grid_indices = np.random.randint(0, n_grid_cells, N)

    # For each unique grid cell, store its features
    unique_cells = np.unique(grid_indices)
    r = len(unique_cells)

    # Build upper triangular matrix
    A_u = np.zeros((r, D), dtype=np.float32)
    d = np.zeros(r, dtype=np.float32)

    for i, cell_idx in enumerate(unique_cells):
        mask = grid_indices == cell_idx
        # Average features in this cell
        A_u[i] = np.mean(features[mask], axis=0)
        d[i] = np.sum(mask)

    # Build index mapping (A[i] = A_u[indices[i]])
    indices = np.zeros(N, dtype=np.int64)
    for i, cell_idx in enumerate(unique_cells):
        mask = grid_indices == cell_idx
        indices[mask] = i

    print(f"Structured RPCA data prepared:")
    print(f"  Original: {features.shape}")
    print(f"  Upper triangular: {A_u.shape}")
    print(f"  Grid cells: {r}")
    print(f"  Avg repetitions per cell: {np.mean(d):.2f}")

    return A_u, indices, d


def benchmark_rpca_cpu(features: np.ndarray,
                       max_iter: int = 30,
                       tol: float = 1e-7) -> BenchmarkResult:
    """Benchmark RPCA_CPU (pyrpca library)."""
    if not RPCA_CPU_AVAILABLE:
        raise RuntimeError("pyrpca not available. Install with: pip install pyrpca")

    print("\n" + "="*70)
    print("BENCHMARK: rpca-cpu (pyrpca library)")
    print("="*70)

    clear_gpu_cache()
    start_memory = get_gpu_memory()

    iteration_times = []
    total_start = time.time()

    rpca = RPCA_CPU(features, lmbda=1 / np.sqrt(max(features.shape)))

    # Manually instrument iteration timing
    # Note: pyrpca doesn't expose per-iteration timing, so we estimate
    D = features
    lmbda = rpca.lmbda
    norm_fro_D = np.linalg.norm(D, ord='fro')

    from pyrpca import rpca_pcp_ialm

    fit_start = time.time()
    L, S = rpca_pcp_ialm(
        observations=D,
        sparsity_factor=lmbda,
        max_iter=max_iter,
        tol=tol,
        verbose=True
    )
    total_time = time.time() - fit_start

    # Compute final error
    error = np.linalg.norm(D - L - S, ord='fro') / norm_fro_D

    end_memory = get_gpu_memory()

    result = BenchmarkResult(
        method="rpca-cpu",
        iterations=iteration_times,  # Not available from pyrpca
        total_time=total_time,
        final_error=error,
        converged=error < tol,
        final_iterations=max_iter,  # pyrpca doesn't return actual iterations
        memory_mb=end_memory - start_memory
    )

    print(f"\nResult:")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Final error: {error:.6e}")
    print(f"  Memory: {result.memory_mb:.2f} MB")

    return result


def benchmark_rpca_gpu(features: np.ndarray,
                       max_iter: int = 30,
                       tol: float = 1e-7,
                       device: str = 'cuda:0') -> BenchmarkResult:
    """Benchmark RPCA_GPU (GPU for SVD and Shrinkage)."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")

    print("\n" + "="*70)
    print("BENCHMARK: rpca-gpu (GPU for SVD and Shrinkage)")
    print("="*70)

    clear_gpu_cache()
    start_memory = get_gpu_memory()

    iteration_times = []

    # Patch the fit method to record iteration times
    original_fit = RPCA_GPU.fit

    def timed_fit(self, tol=None, max_iter=10, iter_print=1, enable_timing=True, **kwargs):
        _tol = tol or 1e-7

        D = self.D
        lmbda = self.lmbda
        rho = self.rho
        mu_upper_bound = self.mu_upper_bound

        spectral_norm = np.linalg.norm(D, ord=2)
        inf_norm = np.linalg.norm(D, ord=np.inf)
        Y = D / max(spectral_norm, inf_norm / lmbda)
        S = np.zeros_like(D)
        norm_fro_D = np.linalg.norm(D, ord='fro')

        i, err = 0, np.inf
        mu = self.mu

        while err > _tol and i < max_iter:
            i += 1
            iter_start = time.time()

            # Update L
            M = D - S + (1.0 / mu) * Y
            L = self._svt_on_gpu(M, 1.0 / mu)

            # Update S
            residual_for_sparse = D - L + (1.0 / mu) * Y
            S = self._shrink(residual_for_sparse, lmbda / mu)

            # Compute error
            residual = D - L - S
            err = np.linalg.norm(residual, ord='fro') / norm_fro_D

            # Update dual
            Y = Y + mu * residual
            mu = min(mu * rho, mu_upper_bound)

            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)

            if (i % iter_print) == 0 or i == 1:
                print(f'  Iteration: {i:4d}; Error: {err:0.4e}; Time: {iter_time:.4f}s')

            if err < _tol:
                print(f'  Converged at iteration {i}')
                break

        if i >= max_iter:
            print(f'  Max iterations reached.')

        self.L = L
        self.S = S

        return L, S

    # Temporarily replace fit method
    RPCA_GPU.fit = timed_fit

    try:
        total_start = time.time()
        rpca = RPCA_GPU(features, device=device)
        L, S = rpca.fit(max_iter=max_iter, tol=tol, iter_print=5)
        total_time = time.time() - total_start

        error = np.linalg.norm(features - L - S, ord='fro') / np.linalg.norm(features, ord='fro')
    finally:
        # Restore original fit method
        RPCA_GPU.fit = original_fit

    end_memory = get_gpu_memory()

    result = BenchmarkResult(
        method="rpca-gpu",
        iterations=iteration_times,
        total_time=total_time,
        final_error=error,
        converged=error < tol,
        final_iterations=len(iteration_times),
        memory_mb=end_memory - start_memory
    )

    print(f"\nResult:")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Final error: {error:.6e}")
    print(f"  Avg iter time: {np.mean(iteration_times):.4f}s")
    print(f"  Memory: {result.memory_mb:.2f} MB")

    return result


def benchmark_structure_rpca(features: np.ndarray,
                              max_iter: int = 30,
                              tol: float = 1e-7,
                              device: str = 'cuda:0',
                              grid_size: int = 100) -> BenchmarkResult:
    """Benchmark StructuredRPCA_GPU (weighted SVD, all GPU)."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available for StructuredRPCA_GPU")

    print("\n" + "="*70)
    print("BENCHMARK: structure-rpca (weighted SVD, all GPU)")
    print("="*70)

    # Prepare structured data
    A_u, indices, d = prepare_structured_rpca_data(features, grid_size=grid_size)

    clear_gpu_cache()
    start_memory = get_gpu_memory()

    iteration_times = []

    # Patch the fit method to record iteration times
    original_fit = StructuredRPCA_GPU.fit

    def timed_fit(self, tol=None, max_iter=10, iter_print=1, enable_timing=True, **kwargs):
        _tol = tol or 1e-7

        L_u = self.A_u.clone()
        E_u = torch.zeros_like(self.A_u)
        Y = torch.zeros_like(self.A_u)
        mu = self.mu

        d_sqrt = torch.sqrt(self.d)
        d_inv_sqrt = 1.0 / d_sqrt

        torch.cuda.synchronize()

        i, err = 0, float('inf')
        hybrid_mode = False
        restart_attempted = False

        while True:
            try:
                if not restart_attempted:
                    L_u = torch.zeros_like(self.A_u)
                    E_u = torch.zeros_like(self.A_u)
                    Y = torch.zeros_like(self.A_u)
                    i, err = 0, float('inf')
                    print(f"\nStructured RPCA iterations (full GPU mode):")
                else:
                    # Hybrid mode fallback
                    self.A_u = self.A_u.cpu()
                    self.d = self.d.cpu()
                    d_sqrt = d_sqrt.cpu()
                    d_inv_sqrt = d_inv_sqrt.cpu()
                    L_u = torch.zeros_like(self.A_u)
                    E_u = torch.zeros_like(self.A_u)
                    Y = torch.zeros_like(self.A_u)
                    i, err = 0, float('inf')
                    hybrid_mode = True
                    print(f"  [Restart] Hybrid mode")

                while err > _tol and i < max_iter:
                    i += 1
                    iter_start = time.time()

                    # Update L_u
                    M = L_u - E_u + Y / mu
                    L_u_origin = L_u
                    M_weighted = M * d_sqrt.unsqueeze(-1)

                    if hybrid_mode:
                        try:
                            L_u_weighted = self._svt_gpu(M_weighted, self.lambda_structured / mu)
                        except RuntimeError:
                            L_u_weighted = self._svt_cpu(M_weighted, self.lambda_structured / mu)
                        L_u = L_u_weighted * d_inv_sqrt.unsqueeze(-1)
                    else:
                        L_u_weighted = self._svt_gpu(M_weighted, self.lambda_structured / mu)
                        L_u = L_u_weighted * d_inv_sqrt.unsqueeze(-1)

                    s_u = mu * (L_u - L_u_origin)

                    if not hybrid_mode:
                        torch.cuda.synchronize()

                    # Update E_u
                    M = self.A_u - L_u + Y / mu
                    thresholds = self.lambda_structured * self.d / mu
                    E_u = torch.sign(M) * torch.clamp(torch.abs(M) - thresholds.unsqueeze(-1), min=0)

                    if not hybrid_mode:
                        torch.cuda.synchronize()

                    # Compute error
                    residual = self.A_u - L_u - E_u
                    err = torch.linalg.norm(residual, ord='fro') / torch.linalg.norm(self.A_u, ord='fro')
                    err = err.item()

                    if not hybrid_mode:
                        torch.cuda.synchronize()

                    # Update dual
                    Y = Y + mu * residual
                    mu = self.update_mu_safely(residual.abs().mean(), s_u.abs().mean(), mu)

                    if not hybrid_mode:
                        torch.cuda.synchronize()

                    iter_time = time.time() - iter_start
                    iteration_times.append(iter_time)

                    if (i % iter_print) == 0 or i == 1:
                        print(f'  Iteration: {i:4d}; Error: {err:0.4e}; Time: {iter_time:.4f}s')

                    if err < _tol:
                        print(f'  Converged at iteration {i}')
                        break

                if i >= max_iter:
                    print(f'  Max iterations reached.')

                if hybrid_mode:
                    self.L = L_u.to(self.device)
                    self.E = E_u.to(self.device)
                else:
                    self.L = L_u
                    self.E = E_u

                return self.L, self.E

            except RuntimeError as e:
                is_oom = ('out of memory' in str(e) or 'CUDA driver error' in str(e))
                if is_oom and not restart_attempted:
                    try:
                        del L_u, E_u, Y, M, M_weighted, L_u_weighted
                    except NameError:
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()
                    restart_attempted = True
                    continue
                else:
                    raise

    # Temporarily replace fit method
    StructuredRPCA_GPU.fit = timed_fit

    try:
        total_start = time.time()
        rpca = StructuredRPCA_GPU(A_u, indices, d, device=device)
        L, E = rpca.fit(max_iter=max_iter, tol=tol, iter_print=5)
        total_time = time.time() - total_start

        # Map back to full space and compute error
        L_full, E_full, rel_error = rpca.map_to_full(torch.from_numpy(features).to(device))

        # For comparison, compute reconstruction error
        reconstructed = L_full + E_full
        error = torch.norm(reconstructed - torch.from_numpy(features).to(device), p='fro').item()
        error /= torch.norm(torch.from_numpy(features).to(device), p='fro').item()

    finally:
        # Restore original fit method
        StructuredRPCA_GPU.fit = original_fit

    end_memory = get_gpu_memory()

    result = BenchmarkResult(
        method="structure-rpca",
        iterations=iteration_times,
        total_time=total_time,
        final_error=error,
        converged=error < tol,
        final_iterations=len(iteration_times),
        memory_mb=end_memory - start_memory
    )

    print(f"\nResult:")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Final error: {error:.6e}")
    print(f"  Avg iter time: {np.mean(iteration_times):.4f}s")
    print(f"  Memory: {result.memory_mb:.2f} MB")

    return result


def run_benchmark(scene_name: str = "figurines",
                  max_iter: int = 30,
                  tol: float = 1e-7,
                  output_dir: Optional[str] = None) -> BenchmarkSummary:
    """
    Run complete benchmark comparing all three RPCA variants.

    Args:
        scene_name: LERF scene to test
        max_iter: Maximum iterations per method
        tol: Convergence tolerance
        output_dir: Directory to save results

    Returns:
        BenchmarkSummary with all results
    """
    print("="*70)
    print("RPCA BENCHMARK ON LERF DATASET")
    print("="*70)
    print(f"Scene: {scene_name}")
    print(f"Max iterations: {max_iter}")
    print(f"Tolerance: {tol}")
    print(f"CUDA available: {CUDA_AVAILABLE}")
    print(f"pyrpca available: {RPCA_CPU_AVAILABLE}")

    # Load features
    features = load_lerf_features(scene_name)
    N, D = features.shape
    data_size_mb = features.nbytes / (1024 ** 2)

    results = []

    # 1. Benchmark rpca-cpu
    try:
        result_cpu = benchmark_rpca_cpu(features, max_iter=max_iter, tol=tol)
        results.append(result_cpu)
    except Exception as e:
        print(f"ERROR in rpca-cpu: {e}")
        # Add placeholder result
        results.append(BenchmarkResult(
            method="rpca-cpu", iterations=[], total_time=0, final_error=-1,
            converged=False, final_iterations=0, memory_mb=0
        ))

    # Clear cache between runs
    clear_gpu_cache()
    time.sleep(1)

    # 2. Benchmark rpca-gpu
    try:
        result_gpu = benchmark_rpca_gpu(features, max_iter=max_iter, tol=tol, device='cuda:0')
        results.append(result_gpu)
    except Exception as e:
        print(f"ERROR in rpca-gpu: {e}")
        results.append(BenchmarkResult(
            method="rpca-gpu", iterations=[], total_time=0, final_error=-1,
            converged=False, final_iterations=0, memory_mb=0
        ))

    # Clear cache between runs
    clear_gpu_cache()
    time.sleep(1)

    # 3. Benchmark structure-rpca
    try:
        result_struct = benchmark_structure_rpca(features, max_iter=max_iter, tol=tol,
                                                  device='cuda:0', grid_size=100)
        results.append(result_struct)
    except Exception as e:
        print(f"ERROR in structure-rpca: {e}")
        results.append(BenchmarkResult(
            method="structure-rpca", iterations=[], total_time=0, final_error=-1,
            converged=False, final_iterations=0, memory_mb=0
        ))

    summary = BenchmarkSummary(
        results=results,
        data_shape=(N, D),
        data_size_mb=data_size_mb
    )

    # Print comparison
    print_comparison(summary)

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        try:
            save_results(summary, output_path / f"rpca_benchmark_{scene_name}.json")
        except Exception as e:
            print(f"Warning: Could not save JSON results: {e}")

    return summary


def print_comparison(summary: BenchmarkSummary):
    """Print comparison table."""
    print("\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)
    print(f"Data shape: {summary.data_shape}")
    print(f"Data size: {summary.data_size_mb:.2f} MB")
    print()

    # Table header
    print(f"{'Method':<20} {'Total (s)':<12} {'Avg Iter (s)':<14} {'Final Error':<15} {'Memory (MB)':<12}")
    print("-" * 100)

    for r in summary.results:
        if r.iterations:
            avg_iter = np.mean(r.iterations)
        elif r.total_time > 0 and r.final_iterations > 0:
            avg_iter = r.total_time / r.final_iterations
        else:
            avg_iter = 0

        if r.final_error >= 0:
            error_str = f"{r.final_error:.6e}"
            time_str = f"{r.total_time:.4f}"
            memory_str = f"{r.memory_mb:.2f}"
        else:
            error_str = "ERROR"
            time_str = "N/A"
            memory_str = "N/A"
            avg_iter = 0

        print(f"{r.method:<20} {time_str:<12} {avg_iter:<14.4f} {error_str:<15} {memory_str:<12}")

    print()

    # Speedup comparison
    gpu_result = next((r for r in summary.results if r.method == "rpca-gpu" and r.iterations), None)
    struct_result = next((r for r in summary.results if r.method == "structure-rpca" and r.iterations), None)

    if gpu_result and struct_result:
        speedup = gpu_result.total_time / struct_result.total_time
        if speedup > 1:
            print(f"structure-rpca is {speedup:.2f}x FASTER than rpca-gpu")
        else:
            print(f"structure-rpca is {1/speedup:.2f}x SLOWER than rpca-gpu")

    print()


def save_results(summary: BenchmarkSummary, output_path: Path):
    """Save results to JSON file."""
    data = {
        'data_shape': list(summary.data_shape),
        'data_size_mb': float(summary.data_size_mb),
        'results': []
    }

    for r in summary.results:
        result_dict = asdict(r)
        # Convert numpy and non-serializable types to native types
        for key, value in result_dict.items():
            if isinstance(value, np.ndarray):
                result_dict[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                result_dict[key] = float(value)
            elif isinstance(value, (np.bool_, bool)):
                result_dict[key] = bool(value)
            elif isinstance(value, list):
                result_dict[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in value]
        data['results'].append(result_dict)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='RPCA Benchmark on LERF Dataset')
    parser.add_argument('--scene', type=str, default='figurines',
                        choices=['figurines', 'ramen', 'teatime', 'waldo_kitchen'],
                        help='LERF scene to test')
    parser.add_argument('--max_iter', type=int, default=30,
                        help='Maximum iterations')
    parser.add_argument('--tol', type=float, default=1e-7,
                        help='Convergence tolerance')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU device to use')

    args = parser.parse_args()

    run_benchmark(
        scene_name=args.scene,
        max_iter=args.max_iter,
        tol=args.tol,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
