#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RPCA Utilities

Robust PCA implementations with CPU (pyrpca library) and GPU (PyTorch/CUDA) support.

The CPU implementation (RPCA_CPU) is a wrapper around the pyrpca library:
- Uses pyrpca.pcp_ialm for optimized performance
- Default max_iter=10 for fast processing
- Supports multi-threading via scipy BLAS

Based on:
- https://github.com/surgura/PyRPCA (pyrpca library)
- https://github.com/dganguli/robust-pca (Original robust-pca)
- https://gist.github.com/jcreinhold/ebf27f997f4c93c2f637c3c900d6388f (GPU version)

References:
[1] Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011).
    Robust principal component analysis?. Journal of the ACM (JACM), 58(3), 11.

[2] Cai, J. F., Candès, E. J., & Shen, Z. (2010).
    A singular value thresholding algorithm for matrix completion.
    SIAM Journal on Optimization, 20(4), 1956-1982.

Author: SceneSplat Team
"""

__all__ = ['RPCA_GPU', 'RPCA_CPU', 'StructuredRPCA_GPU', 'svt_gpu', 'RPCA_AVAILABLE', 'CUDA_AVAILABLE']

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from pathlib import Path
import os

# Check availability
try:
    from pyrpca import rpca_pcp_ialm
    PYRPCA_AVAILABLE = True
except ImportError:
    PYRPCA_AVAILABLE = False

RPCA_CPU_AVAILABLE = PYRPCA_AVAILABLE  # CPU RPCA requires pyrpca library
CUDA_AVAILABLE = torch.cuda.is_available()
RPCA_AVAILABLE = CUDA_AVAILABLE or RPCA_CPU_AVAILABLE


def _get_device(device: Optional[str] = None) -> str:
    """Get the device to use for computation."""
    if device is not None:
        return device
    return 'cuda' if CUDA_AVAILABLE else 'cpu'


class RPCA_GPU:
    """
    GPU-accelerated Robust PCA that matches pyrpca implementation.

    This implementation uses GPU for SVD and Shrinkage operations:
    - All data (D, L, S, Y) stored as numpy arrays on CPU
    - SVD and Shrinkage computation offloaded to GPU
    - GPU memory cleaned up after each operation
    - Significantly reduces GPU memory usage while accelerating computation
    - Good balance between speed and memory efficiency

    Example:
        >>> import numpy as np
        >>> from rpca_utils import RPCA_GPU
        >>> X = np.random.randn(1000, 100)
        >>> rpca = RPCA_GPU(X)
        >>> L, S = rpca.fit(max_iter=10, tol=1e-7)
    """

    def __init__(self, D: np.ndarray, lmbda: Optional[float] = None,
                 device: Optional[str] = None, mu: Optional[float] = None):
        """
        Initialize RPCA_GPU (pyrpca-compatible, GPU for SVD and Shrinkage).

        Args:
            D: Input data matrix as numpy array [m, n]
            lmbda: Sparsity parameter (default: 1/sqrt(max(m, n)))
            device: Device to use for SVD and Shrinkage ('cuda' or 'cpu', default: auto-detect)
            mu: Initial penalty parameter (default: auto-computed as 1.25/spectral_norm)
        """
        device = _get_device(device)
        self.device = device

        # Store data as numpy array on CPU (use float32 for memory efficiency)
        self.D = D.astype(np.float32) if D.dtype != np.float32 else D.copy()

        self.lmbda = lmbda or 1 / np.sqrt(np.max(self.D.shape))

        # Compute initial mu (matching pyrpca: 1.25 / spectral_norm)
        if mu is None:
            spectral_norm = np.linalg.norm(self.D, ord=2)
            self.mu = float(1.25 / spectral_norm) if spectral_norm > 0 else 1.0
        else:
            self.mu = mu
        self.mu_upper_bound = self.mu * 1e7
        self.rho = 1.5  # Mu growth factor (matching pyrpca)

        print(f"RPCA_GPU initialized (pyrpca-compatible, GPU for SVD and Shrinkage):")
        print(f"  Data shape: {self.D.shape}")
        print(f"  SVD device: {device}")
        print(f"  Shrinkage device: {device}")
        print(f"  lambda: {self.lmbda:.6f}")
        print(f"  mu: {self.mu:.6e}")

        # Show memory info if using CUDA
        if device.startswith('cuda') and CUDA_AVAILABLE:
            total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            print(f"  GPU Memory: {total:.2f} GB total (data kept on CPU)")

    def _shrink(self, M: np.ndarray, tau: float) -> np.ndarray:
        """
        Shrinkage operator for soft thresholding (GPU version with memory cleanup).

        shrink(M, tau) = sign(M) * max(|M| - tau, 0)

        Args:
            M: Input matrix [m, n] as numpy array
            tau: Threshold parameter

        Returns:
            S: Shrunk matrix as numpy array
        """
        m, n = M.shape
        matrix_size_gb = m * n * 4 / (1024**3)  # float32

        # Try GPU first, fall back to CPU on OOM
        if self.device.startswith('cuda') and CUDA_AVAILABLE:
            try:
                return self._shrink_on_gpu(M, tau)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  [Memory] GPU OOM on {matrix_size_gb:.2f} GB matrix for shrinkage, falling back to CPU")
                    torch.cuda.empty_cache()
                    return self._shrink_on_cpu(M, tau)
                else:
                    # Print detailed error information before re-raising
                    import traceback
                    print(f"  [Error] RuntimeError during GPU shrinkage (not OOM):")
                    print(f"    Error type: {type(e).__name__}")
                    print(f"    Error message: {str(e)}")
                    print(f"    Traceback:")
                    traceback.print_exc()
                    raise
        else:
            return self._shrink_on_cpu(M, tau)

    def _shrink_on_gpu(self, M: np.ndarray, tau: float) -> np.ndarray:
        """
        GPU-based shrinkage operator with automatic memory cleanup.

        Performs: sign(M) * max(|M| - tau, 0)
        """
        # Transfer to GPU
        M_gpu = torch.from_numpy(M).to(self.device)

        # Compute shrinkage on GPU
        S_gpu = torch.sign(M_gpu) * torch.clamp(torch.abs(M_gpu) - tau, min=0)

        # Convert back to numpy
        S = S_gpu.cpu().numpy()

        # Explicitly cleanup GPU tensors
        del M_gpu, S_gpu
        torch.cuda.empty_cache()

        return S

    def _shrink_on_cpu(self, M: np.ndarray, tau: float) -> np.ndarray:
        """
        CPU-based shrinkage operator (numpy version).

        Performs: sign(M) * max(|M| - tau, 0)
        """
        return np.sign(M) * np.maximum(np.abs(M) - tau, 0)

    def _svt_on_gpu(self, M: np.ndarray, tau: float) -> np.ndarray:
        """
        Perform Singular Value Thresholding (SVT) with automatic GPU/CPU selection.

        SVT(M, tau) = U * diag(shrink(s, tau)) * V^T
        where shrink(s, tau) = max(s - tau, 0)

        Automatically falls back to CPU if GPU doesn't have enough memory.

        Args:
            M: Input matrix [m, n] as numpy array
            tau: Threshold parameter for singular values

        Returns:
            L: Thresholded low-rank matrix as numpy array
        """
        m, n = M.shape
        matrix_size_gb = m * n * 4 / (1024**3)  # float32

        # Try GPU first, fall back to CPU on OOM
        if self.device.startswith('cuda') and CUDA_AVAILABLE:
            return self._svt_on_gpu_fast(M, tau)
        else:
            return self._svt_on_cpu(M, tau)

    def _svt_on_gpu_fast(self, M: np.ndarray, tau: float) -> np.ndarray:
        """
        Fast GPU SVT: SVD on GPU, matrix multiplication on GPU/CPU based on memory.

        Strategy:
        1. Always do SVD on GPU (fast)
        2. Try matrix multiplication on GPU first
        3. Fall back to CPU if GPU OOM during multiplication
        """
        # Step 1: SVD on GPU (always - this is the expensive part)
        M_gpu = torch.from_numpy(M).to(self.device)
        U, s, Vt = torch.linalg.svd(M_gpu, full_matrices=False)

        # Step 2: Threshold singular values (tiny, no memory issue)
        s_thresholded = torch.clamp(s - tau, min=0)

        # Step 3: Try matrix multiplication on GPU, fall back to CPU if OOM
        try:
            # Attempt full GPU computation
            L_gpu = U @ torch.diag(s_thresholded) @ Vt
            return L_gpu.cpu().numpy()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # GPU OOM during matrix multiply
                print(f"  [Memory] GPU OOM during matrix multiply, using CPU for reconstruction")
                torch.cuda.empty_cache()

                # Move U to CPU for matrix multiplication
                # Note: s_thresholded and Vt are small, keep them on GPU for now
                U_cpu = U.cpu().numpy()
                s_cpu = s_thresholded.cpu().numpy()
                Vt_cpu = Vt.cpu().numpy()

                # Matrix multiplication on CPU using broadcasting (memory-efficient)
                # L = U_cpu @ np.diag(s_cpu) @ Vt_cpu -> L = U_cpu * s_cpu[np.newaxis, :] @ Vt_cpu
                L = U_cpu * s_cpu[np.newaxis, :] @ Vt_cpu

                # Cleanup GPU tensors
                del U, s, Vt, s_thresholded, M_gpu
                torch.cuda.empty_cache()

                return L
            else:
                # Print detailed error information before re-raising
                import traceback
                print(f"  [Error] RuntimeError during GPU SVT (not OOM):")
                print(f"    Error type: {type(e).__name__}")
                print(f"    Error message: {str(e)}")
                print(f"    Traceback:")
                traceback.print_exc()
                raise

    def _svt_on_cpu(self, M: np.ndarray, tau: float) -> np.ndarray:
        """CPU-based SVT using scipy/numpy."""
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        s_thresholded = np.maximum(s - tau, 0)
        # Use broadcasting instead of np.diag to save memory: L = U * s_thresholded[np.newaxis, :] @ Vt
        L = U * s_thresholded[np.newaxis, :] @ Vt
        return L

    def fit(self, tol: Optional[float] = None, max_iter: int = 10,
            iter_print: int = 1, enable_timing: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit RPCA model using pyrpca-compatible IALM algorithm.

        Data stored on CPU, with SVD and Shrinkage operations offloaded to GPU
        for accelerated computation. GPU memory is cleaned up after each operation.

        Args:
            tol: Convergence tolerance (default: 1e-7)
            max_iter: Maximum iterations (default: 10)
            iter_print: Print progress every N iterations (default: 1)
            enable_timing: Enable detailed timing analysis (default: True)

        Returns:
            L: Low-rank matrix (numpy array)
            S: Sparse matrix (numpy array)
        """
        import time

        _tol = tol or 1e-7

        print(f"\nStarting RPCA fitting (GPU for SVD and Shrinkage)...")
        print(f"  Max iterations: {max_iter}")
        print(f"  Tolerance: {_tol}")
        print(f"  Timing analysis: {'enabled' if enable_timing else 'disabled'}")

        # Timing dictionaries
        timings = {
            'init': 0.0,
            'svt': 0.0,
            'shrink': 0.0,
            'error': 0.0,
            'dual_update': 0.0,
        }

        # All data on CPU as numpy arrays
        D = self.D
        lmbda = self.lmbda
        rho = self.rho
        mu_upper_bound = self.mu_upper_bound

        # Initialize dual variable (matching pyrpca: non-zero initialization)
        t_start = time.time() if enable_timing else 0
        spectral_norm = np.linalg.norm(D, ord=2)
        inf_norm = np.linalg.norm(D, ord=np.inf)
        Y = D / max(spectral_norm, inf_norm / lmbda)

        # Initialize sparse component
        S = np.zeros_like(D)

        # Normalize observations for error computation
        norm_fro_D = np.linalg.norm(D, ord='fro')
        if enable_timing:
            timings['init'] = time.time() - t_start

        # IALM algorithm with adaptive mu (matching pyrpca)
        i, err = 0, np.inf
        mu = self.mu

        while err > _tol and i < max_iter:
            i += 1

            # Update L (low-rank component) - SVT on GPU (lines 191-195)
            # L = SVT(D - S + Y/mu, 1/mu)
            t_svt = time.time() if enable_timing else 0
            M = D - S + (1.0 / mu) * Y
            L = self._svt_on_gpu(M, 1.0 / mu)
            if enable_timing:
                iter_svt = time.time() - t_svt
                timings['svt'] += iter_svt

            # Update S (sparse component) - on GPU
            t_shrink = time.time() if enable_timing else 0
            residual_for_sparse = D - L + (1.0 / mu) * Y
            S = self._shrink(residual_for_sparse, lmbda / mu)
            if enable_timing:
                iter_shrink = time.time() - t_shrink
                timings['shrink'] += iter_shrink

            # Compute error - on CPU
            t_error = time.time() if enable_timing else 0
            residual = D - L - S
            err = np.linalg.norm(residual, ord='fro') / norm_fro_D
            if enable_timing:
                iter_error = time.time() - t_error
                timings['error'] += iter_error

            # Update dual variable and mu (matching pyrpca) - on CPU
            t_dual = time.time() if enable_timing else 0
            Y = Y + mu * residual
            mu = min(mu * rho, mu_upper_bound)
            if enable_timing:
                iter_dual = time.time() - t_dual
                timings['dual_update'] += iter_dual

            # Print progress with per-iteration timing
            if (i % iter_print) == 0 or i == 1 or err <= _tol:
                timing_info = ""
                if enable_timing:
                    iter_total = iter_svt + iter_shrink + iter_error + iter_dual
                    timing_info = f" | SVT: {iter_svt:.4f}s | Shrink: {iter_shrink:.4f}s | Error: {iter_error:.4f}s | Dual: {iter_dual:.4f}s | Total: {iter_total:.4f}s"
                print(f'  Iteration: {i:4d}; Error: {err:0.4e}; mu: {mu:.6e}{timing_info}')

            # Check convergence
            if err < _tol:
                print(f'  Finished optimization. Error smaller than tolerance.')
                break

        if i >= max_iter:
            print(f'  Finished optimization. Max iterations reached.')

        self.L = L
        self.S = S

        print(f"\nRPCA fitting completed:")
        print(f"  Final iterations: {i}")
        print(f"  Final error: {err:.6e}")

        # Print timing breakdown
        if enable_timing:
            total_time = sum(timings.values())
            device_shrink = "GPU" if self.device.startswith('cuda') and CUDA_AVAILABLE else "CPU"
            print(f"\nTiming breakdown ({i} iterations):")
            print(f"  Initialization:  {timings['init']:.4f}s ({timings['init']/total_time*100:.1f}%)")
            print(f"  SVT (GPU):       {timings['svt']:.4f}s ({timings['svt']/total_time*100:.1f}%)")
            print(f"  Shrinkage ({device_shrink}): {timings['shrink']:.4f}s ({timings['shrink']/total_time*100:.1f}%)")
            print(f"  Error comp:      {timings['error']:.4f}s ({timings['error']/total_time*100:.1f}%)")
            print(f"  Dual update:     {timings['dual_update']:.4f}s ({timings['dual_update']/total_time*100:.1f}%)")
            print(f"  Total:           {total_time:.4f}s")

        # Clean up and clear GPU cache
        del Y, residual, residual_for_sparse
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()

        return L, S

    def get_results(self) -> Dict[str, np.ndarray]:
        """
        Get results as numpy arrays.

        Returns:
            Dictionary with 'L' (low-rank) and 'S' (sparse) matrices
        """
        if not hasattr(self, 'L'):
            raise RuntimeError("Must call fit() before getting results")

        results = {
            'L': self.L,
            'S': self.S,
        }

        return results

    def get_rank(self, threshold: float = 0.01) -> int:
        """
        Estimate rank of low-rank component (numpy-based).

        Args:
            threshold: Threshold as fraction of max singular value (default: 0.01)

        Returns:
            Estimated rank
        """
        if not hasattr(self, 'L'):
            raise RuntimeError("Must call fit() before getting rank")

        # Compute SVD of low-rank matrix using numpy (compute_uv=False -> only return S)
        s = np.linalg.svd(self.L, compute_uv=False)
        max_sv = s[0]
        rank = (s > threshold * max_sv).sum()

        return rank

    def __del__(self):
        """Cleanup GPU memory on deletion."""
        if hasattr(self, 'device') and self.device.startswith('cuda'):
            torch.cuda.empty_cache()


class RPCA_CPU:
    """
    CPU-based Robust PCA using pyrpca library.

    This is a wrapper around the pyrpca library for RPCA decomposition.
    Uses the optimized pyrpca.pcp_ialm implementation for best performance.

    Example:
        >>> import numpy as np
        >>> from rpca_utils import RPCA_CPU
        >>> X = np.random.randn(1000, 100)
        >>> rpca = RPCA_CPU(X)
        >>> L, S = rpca.fit(max_iter=10, tol=1e-7)
    """

    def __init__(self, D: np.ndarray, lmbda: Optional[float] = None,
                 n_threads: Optional[int] = None, mu: Optional[float] = None):
        """
        Initialize RPCA_CPU (wrapper around pyrpca library).

        Args:
            D: Input data matrix as numpy array [m, n]
            lmbda: Sparsity parameter (default: 1/sqrt(max(m, n)))
            n_threads: Number of threads (for scipy BLAS, set via env vars)
            mu: Initial penalty parameter (passed to pyrpca)
        """
        import os

        # Configure thread count for scipy BLAS
        if n_threads is None:
            n_threads = os.cpu_count() or 1
        self.n_threads = max(1, min(n_threads, os.cpu_count() or 1))

        # Set environment variables for scipy BLAS threading
        os.environ['OMP_NUM_THREADS'] = str(self.n_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.n_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.n_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.n_threads)

        # Use float32 for memory efficiency (consistent with RPCA_GPU)
        self.D = D.astype(np.float32)
        self.lmbda = lmbda or 1 / np.sqrt(np.max(self.D.shape))
        self.mu = mu

        print(f"RPCA_CPU initialized (using pyrpca library):")
        print(f"  Data shape: {self.D.shape}")
        print(f"  lambda: {self.lmbda:.6f}")
        print(f"  Threads: {self.n_threads}")

    def fit(self, tol: Optional[float] = None, max_iter: int = 10,
            iter_print: int = 1, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit RPCA model using pyrpca library (optimized implementation).

        Args:
            tol: Convergence tolerance (default: 1e-7)
            max_iter: Maximum iterations (default: 10)
            iter_print: Print progress every N iterations (default: 1)
            **kwargs: Additional arguments (unused, for compatibility)

        Returns:
            L: Low-rank matrix (numpy array)
            S: Sparse matrix (numpy array)
        """
        # Import pyrpca
        try:
            from pyrpca import rpca_pcp_ialm
        except ImportError:
            raise ImportError("pyrpca is required. Install with: pip install pyrpca")

        _tol = tol or 1e-7

        print(f"\nStarting RPCA fitting (using pyrpca library)...")
        print(f"  Max iterations: {max_iter}")
        print(f"  Tolerance: {_tol}")

        # Call pyrpca
        L, S = rpca_pcp_ialm(
            observations=self.D,
            sparsity_factor=self.lmbda,
            max_iter=max_iter,
            mu=self.mu,
            tol=_tol,
            verbose=(iter_print > 0)
        )

        self.L = L
        self.S = S

        # Compute final error
        err = np.linalg.norm(self.D - L - S, ord='fro') / np.linalg.norm(self.D, ord='fro')

        print(f"\nRPCA fitting completed:")
        print(f"  Final error: {err:.6e}")

        return L, S

    def get_results(self) -> Dict[str, np.ndarray]:
        """Get results as dictionary."""
        if not hasattr(self, 'L'):
            raise RuntimeError("Must call fit() before getting results")

        return {'L': self.L, 'S': self.S}

    def get_rank(self, threshold: float = 0.01) -> int:
        """Estimate rank of low-rank component."""
        if not hasattr(self, 'L'):
            raise RuntimeError("Must call fit() before getting rank")

        s = np.linalg.svd(self.L, compute_uv=False)
        max_sv = s[0]
        rank = (s > threshold * max_sv).sum()

        return rank


def svt_gpu(X: np.ndarray, mask: np.ndarray, tau: Optional[float] = None,
            delta: Optional[float] = None, eps: float = 1e-2,
            max_iter: int = 1000, iter_print: int = 5,
            device: Optional[str] = None) -> np.ndarray:
    """
    Matrix completion via Singular Value Thresholding (SVT) with GPU support.

    Reference:
        Cai, J. F., Candès, E. J., & Shen, Z. (2010).
        A singular value thresholding algorithm for matrix completion.

    Args:
        X: Input matrix with missing values [m, n]
        mask: Binary mask where 1 indicates observed values [m, n]
        tau: Singular value threshold parameter
        delta: Step size for gradient descent
        eps: Convergence tolerance
        max_iter: Maximum iterations
        iter_print: Print progress every N iterations
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Completed matrix
    """
    device = _get_device(device)

    # Convert to torch tensors
    if isinstance(X, np.ndarray):
        X_torch = torch.from_numpy(X).float().to(device)
        mask_torch = torch.from_numpy(mask).float().to(device)
    else:
        X_torch = X.float().to(device)
        mask_torch = mask.float().to(device)

    # Initialize
    Z = torch.zeros_like(X_torch)
    tau = tau or (5 * np.sum(X_torch.shape) / 2)
    delta = delta or (1.2 * np.prod(X_torch.shape) / torch.sum(mask_torch)).item()

    print(f"SVT initialized:")
    print(f"  Device: {device}")
    print(f"  tau: {tau:.2f}")
    print(f"  delta: {delta:.6f}")

    for i in range(max_iter):
        # SVD thresholding
        U, s, V = torch.linalg.svd(Z, full_matrices=False)
        s_shrunk = F.relu(s - tau)
        A = torch.mm(U, torch.mm(torch.diag(s_shrunk), V.t()))

        # Gradient update
        Z += delta * mask_torch * (X_torch - A)

        # Compute error
        error = (torch.norm(mask_torch * (X_torch - A)) /
                 torch.norm(mask_torch * X_torch)).item()

        if i % iter_print == 0:
            print(f'  Iteration: {i:4d}; Error: {error:.4e}')

        if error < eps:
            print(f'  Converged at iteration {i}')
            break

    return A.cpu().numpy()


class StructuredRPCA_GPU:
    """
    Structured Robust PCA with weighted SVD - ALL GPU implementation.

    This variant handles structured sparse input with row repetition weights.
    All matrix operations are performed on PyTorch GPU for maximum speed.

    Input:
        A_u: Upper triangular matrix (r x n) representing sparse data
        indices: Mapping array A[i] = A_u[indices[i]]
        d: Repetition count array d[i] = count of unique row i

    Algorithm (all on GPU):
    1. Update L_u via weighted SVT (uses D_sqrt and D_inv_sqrt weights on GPU)
    2. Update E_u via weighted soft thresholding (per-row lambda thresholds on GPU)
    3. Update Lagrange multiplier Y (on GPU)
    4. Check convergence (on GPU)
    5. Map back from upper triangular to full matrices (on GPU)

    Example:
        >>> import numpy as np
        >>> from rpca_utils import StructuredRPCA_GPU
        >>> A_u = np.random.randn(100, 1000)  # Upper triangular (r x n)
        >>> indices = np.random.randint(0, 100, 1000)
        >>> d = np.random.randint(1, 10, 100)
        >>> rpca = StructuredRPCA_GPU(A_u, indices, d)
        >>> L, E = rpca.fit(max_iter=10, tol=1e-7)
    """

    def __init__(self, A_u: np.ndarray, indices: np.ndarray, d: np.ndarray,
                 lambda_val: Optional[float] = None, device: Optional[str] = None,
                 mu: Optional[float] = None):
        """
        Initialize StructuredRPCA_GPU with all-GPU operations.

        Args:
            A_u: Upper triangular matrix [r, n] - sparse/structured input
            indices: Index mapping array [n] - A[i] = A_u[indices[i]]
            d: Repetition count array [r] - d[i] = count of unique row i
            lambda_val: Sparsity parameter (default: 1/sqrt(max(r, n)))
            device: Device to use (default: cuda:0)
            mu: Initial penalty parameter (default: auto-computed)
        """
        device = _get_device(device)
        self.device = device

        if not device.startswith('cuda'):
            raise ValueError(f"StructuredRPCA_GPU requires CUDA device, got: {device}")
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available for StructuredRPCA_GPU")

        # Store data as PyTorch tensors on GPU
        self.A_u = torch.from_numpy(A_u.astype(np.float32)).to(device)
        self.indices = torch.from_numpy(indices.astype(np.int64)).to(device)
        self.d = torch.from_numpy(d.astype(np.float32)).to(device)

        r, n = A_u.shape
        self.r, self.n = r, n

        self.lambda_val = lambda_val or 1.0 / np.sqrt(max(r, n))
        self.lambda_structured = self.lambda_val / np.sqrt(np.mean(d))
        self.sparse_threshold = 1e-6 * torch.max(torch.abs(self.A_u))

        # Compute initial mu on GPU
        if mu is None:
            spectral_norm = torch.linalg.norm(self.A_u*torch.sqrt(self.d).unsqueeze(-1), ord=2)
            mu_val = float(1.25 / spectral_norm.item()) if spectral_norm > 0 else 1.0
        else:
            mu_val = mu
        self.mu = mu_val
        self.mu_upper_bound = self.mu * 1e4
        self.mu_lower_bound = self.mu * 1e-2
        self.rho = 1.2

        print(f"StructuredRPCA_GPU initialized (ALL GPU operations):")
        print(f"  A_u shape: {self.A_u.shape}")
        print(f"  Indices shape: {self.indices.shape}")
        print(f"  d shape: {self.d.shape}")
        print(f"  Device: {device}")
        print(f"  lambda: {self.lambda_structured:.6f}")
        print(f"  mu: {self.mu:.6e}")

        # Show memory info
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"  GPU Memory: {total:.2f} GB total")
    
    def update_mu_safely(self, primal_residual, dual_residual, mu, rt=10):
        # 计算残差比例
        ratio = primal_residual / (dual_residual + 1e-10)
        
        # 平衡更新
        if ratio > rt:  # 原始残差远大于对偶残差
            mu_new = min(mu * self.rho, self.mu_upper_bound)  # 缓慢增加，并有上限
        elif ratio < 1/rt:  # 对偶残差远大于原始残差
            mu_new = max(mu / self.rho, self.mu_lower_bound)   # 缓慢减少，并有下限
        else:
            mu_new = mu  # 保持稳定
        
        return mu_new
    
    def adaptive_lambda_structured_sparsity(self, E, target_sparsity=1e-6, adjustment_rate=0.1):
        print_reason = True
        device = self.device
        E = E.to(device)
        d = self.d

        current_lambda = self.lambda_structured
        
        # 计算加权E的绝对值
        # d[:, None] 将权重向量扩展为 [r, 1]，然后广播到 [r, n]
        weighted_E = d.unsqueeze(-1) * torch.abs(E)
        
        # 计算加权非零元素数量
        non_zero_mask = weighted_E > self.sparse_threshold
        non_zero_weighted = torch.sum(non_zero_mask.float())
        
        # 计算总权重：sum(d) * n
        total_weight = torch.sum(d) * E.shape[1]
        current_sparsity = (non_zero_weighted / total_weight).item()
        
        # 计算稀疏性比例
        if current_sparsity > 0:
            sparsity_ratio = current_sparsity / target_sparsity
        else:
            sparsity_ratio = 0.0  # 如果E全零，认为稀疏性很高
        
        # 根据稀疏性调整λ
        if sparsity_ratio > 1.5:  # 过于稠密
            new_lambda = current_lambda * (1.0 + adjustment_rate)
            if print_reason: print("weighted_too_dense")
        elif sparsity_ratio < 0.5:  # 过于稀疏
            new_lambda = current_lambda * (1.0 - adjustment_rate)
            if print_reason: print("weighted_too_sparse")
        else:  # 在合理范围内
            new_lambda = current_lambda
            if print_reason: print("weighted_optimal")
        
        self.lambda_structured = new_lambda
        
        # 返回结果
        info = {
            "current_sparsity": current_sparsity,
            "target_sparsity": target_sparsity,
            "sparsity_ratio": sparsity_ratio,
            "non_zero_count": non_zero_weighted.item()
        }
        
        return info

    def fit(self, tol: Optional[float] = None, max_iter: int = 10,
            iter_print: int = 1, enable_timing: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit structured RPCA model using all-GPU weighted SVD algorithm.

        Args:
            tol: Convergence tolerance (default: 1e-7)
            max_iter: Maximum iterations (default: 10)
            iter_print: Print progress every N iterations (default: 1)
            enable_timing: Enable detailed timing analysis (default: True)

        Returns:
            L: Low-rank matrix [m, n] as numpy array
            E: Sparse matrix [m, n] as numpy array
        """
        import time

        _tol = tol or 1e-7

        print(f"\nStarting Structured RPCA fitting (ALL GPU operations)...")
        print(f"  Max iterations: {max_iter}")
        print(f"  Tolerance: {_tol}")
        print(f"  Timing analysis: {'enabled' if enable_timing else 'disabled'}")

        # Timing dictionaries
        timings = {
            'init': 0.0,
            'svt': 0.0,
            'shrink': 0.0,
            'error': 0.0,
            'dual_update': 0.0,
            'final_map': 0.0,
        }

        # Initialize upper triangular variables on GPU
        t_start = time.time() if enable_timing else 0
        L_u = self.A_u.clone()
        E_u = torch.zeros_like(self.A_u)
        Y = torch.zeros_like(self.A_u)
        mu = self.mu

        # Construct weight vectors for row-wise multiplication on GPU (memory-efficient)
        # D_sqrt: D^{1/2}, D_inv_sqrt: D^{-1/2}
        # Use broadcasting instead of diag to avoid OOM: [M, N] * [M, 1] instead of [M, M] @ [M, N]
        d_sqrt = torch.sqrt(self.d)  # [M] - sqrt of grid point counts
        d_inv_sqrt = 1.0 / d_sqrt  # [M] - inverse sqrt of grid point counts

        if enable_timing:
            timings['init'] = time.time() - t_start

        # Synchronize GPU before timing
        torch.cuda.synchronize()

        # IALM algorithm
        i, err = 0, float('inf')

        print(f"\nStructured RPCA iterations:")
        while err > _tol and i < max_iter:
            i += 1

            # Step 1: Update L_u using weighted SVT (all on GPU)
            t_svt = time.time() if enable_timing else 0

            M = L_u - E_u + Y / mu
            L_u_origin = L_u

            # Apply weights using broadcasting: M_weighted = M * d_sqrt.unsqueeze(1)
            # This is memory-efficient: [M, N] * [M, 1] instead of [M, M] @ [M, N]
            M_weighted = M * d_sqrt.unsqueeze(-1)

            # Perform SVT on weighted matrix (GPU)
            L_u_weighted = self._svt_gpu(M_weighted, self.lambda_structured / mu)

            # Map back using broadcasting: L_u = L_u_weighted * d_inv_sqrt.unsqueeze(1)
            # This is memory-efficient: [M, N] * [M, 1] instead of [M, M] @ [M, N]
            L_u = L_u_weighted * d_inv_sqrt.unsqueeze(-1)
            s_u = mu * (L_u - L_u_origin) # Dual Residual
            # print("L_u", float(L_u.min()), float(L_u.max()))
            # print("s_u", float(s_u.min()), float(s_u.max()))

            torch.cuda.synchronize()
            if enable_timing:
                iter_svt = time.time() - t_svt
                timings['svt'] += iter_svt

            # Step 2: Update E_u using weighted soft thresholding (all on GPU)
            t_shrink = time.time() if enable_timing else 0
            M = self.A_u - L_u + Y / mu

            # Apply different threshold per row: lambda * d[i] / mu
            thresholds = self.lambda_structured * self.d / mu

            # Apply thresholding using vectorized operations (optimized: no for loop)
            # E_u = sign(M) * clamp(abs(M) - thresholds, min=0) for each row
            # Tensor shapes: M [r, n], thresholds [r], thresholds.unsqueeze(-1) [r, 1], E_u [r, n]
            E_u = torch.sign(M) * torch.clamp(torch.abs(M) - thresholds.unsqueeze(-1), min=0)

            torch.cuda.synchronize()
            if enable_timing:
                iter_shrink = time.time() - t_shrink
                timings['shrink'] += iter_shrink

            # Step 3: Compute error (on GPU)
            t_error = time.time() if enable_timing else 0
            residual = self.A_u - L_u - E_u
            err = torch.linalg.norm(residual, ord='fro') / torch.linalg.norm(self.A_u, ord='fro')
            err = err.item()

            torch.cuda.synchronize()
            if enable_timing:
                iter_error = time.time() - t_error
                timings['error'] += iter_error

            # Step 4: Update Lagrange multiplier (on GPU)
            t_dual = time.time() if enable_timing else 0
            Y = Y + mu * residual
            # mu = min(mu * self.rho, self.mu_upper_bound)
            mu = self.update_mu_safely(residual.abs().mean(), s_u.abs().mean(), mu)

            torch.cuda.synchronize()
            if enable_timing:
                iter_dual = time.time() - t_dual
                timings['dual_update'] += iter_dual

            # Print progress
            if (i % iter_print) == 0 or i == 1 or err <= _tol:
                timing_info = ""
                if enable_timing:
                    iter_total = iter_svt + iter_shrink + iter_error + iter_dual
                    timing_info = f" | SVT: {iter_svt:.4f}s | Shrink: {iter_shrink:.4f}s | Error: {iter_error:.4f}s | Dual: {iter_dual:.4f}s | Total: {iter_total:.4f}s"
                print(f'  Iteration: {i:4d}; Error: {err:0.4e}; mu: {mu:0.6e}{timing_info}')

            # Check convergence
            if err < _tol:
                print(f'  Finished optimization. Error smaller than tolerance.')
                break

        if i >= max_iter:
            print(f'  Finished optimization. Max iterations reached.')

        # Keep L and E as tensors on GPU for efficient subsequent operations
        self.L = L_u
        self.E = E_u

        print(f"\nStructured RPCA fitting completed:")
        print(f"  Final iterations: {i}")
        print(f"  Final error: {err:.6e}")
        print(f"  Output L shape: {self.L.shape}")
        print(f"  Output E shape: {self.E.shape}")

        # Print timing breakdown
        if enable_timing:
            total_time = sum(timings.values())
            print(f"\nTiming breakdown ({i} iterations):")
            print(f"  Initialization:  {timings['init']:.4f}s ({timings['init']/total_time*100:.1f}%)")
            print(f"  SVT (GPU):       {timings['svt']:.4f}s ({timings['svt']/total_time*100:.1f}%)")
            print(f"  Shrinkage (GPU): {timings['shrink']:.4f}s ({timings['shrink']/total_time*100:.1f}%)")
            print(f"  Error comp (GPU): {timings['error']:.4f}s ({timings['error']/total_time*100:.1f}%)")
            print(f"  Dual update (GPU): {timings['dual_update']:.4f}s ({timings['dual_update']/total_time*100:.1f}%)")
            print(f"  Total:           {total_time:.4f}s")

        # Clean up GPU tensors
        del L_u, E_u, Y, M, M_weighted, L_u_weighted, residual
        torch.cuda.empty_cache()

        return self.L, self.E

    def _svt_gpu(self, M: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Perform SVT entirely on GPU.

        SVT(M, tau) = U * diag(shrink(s, tau)) * V^T
        where shrink(s, tau) = max(s - tau, 0)
        """
        # SVD on GPU
        U, s, Vt = torch.linalg.svd(M, full_matrices=False)

        # Threshold singular values
        s_thresholded = torch.clamp(s - tau, min=0)

        # Reconstruction on GPU
        L = U @ torch.diag(s_thresholded) @ Vt

        return L

    def map_to_full(self, mat_origin: torch.Tensor):
        if not isinstance(self.indices, torch.Tensor):
            indices_tensor = torch.tensor(self.indices, device=self.device)
        else:
            indices_tensor = self.indices.to(self.device)

        L = self.L[indices_tensor]  # (m, n)
        E = self.E[indices_tensor]  # (m, n)
        
        # 计算重建矩阵
        reconstructed = L + E
        
        # 计算误差（多种误差指标）
        if mat_origin.device != self.device:
            mat_origin = mat_origin.to(self.device)
        
        # 均方误差 (MSE)
        # mse = torch.mean((reconstructed - mat_origin) ** 2)
        
        # 平均绝对误差 (MAE/L1)
        # mae = torch.mean(torch.abs(reconstructed - mat_origin))
        
        # 相对误差
        # relative_error = torch.mean(torch.abs(reconstructed - mat_origin) / (torch.abs(mat_origin) + 1e-8))
        
        # 最大绝对误差
        # max_abs_error = torch.max(torch.abs(reconstructed - mat_origin))
        
        # Frobenius范数相对误差
        fro_norm_original = torch.norm(mat_origin, p='fro')
        fro_norm_diff = torch.norm(reconstructed - mat_origin, p='fro')
        relative_fro_error = fro_norm_diff / (fro_norm_original + 1e-8)
        print(f'relative_fro_error: {relative_fro_error}')

        return L, E, relative_fro_error

    def get_results(self, return_tensors: bool = True) -> Dict[str, np.ndarray | torch.Tensor]:
        """
        Get results as numpy arrays or GPU tensors.

        Args:
            return_tensors: If True, return GPU tensors; if False, return numpy arrays

        Returns:
            Dictionary with 'L' (low-rank) and 'S'/'E' (sparse) matrices
        """
        if not hasattr(self, 'L'):
            raise RuntimeError("Must call fit() before getting results")

        if return_tensors:
            # Return GPU tensors directly
            return {
                'L': self.L,
                'S': self.E,
            }
        else:
            # Return numpy arrays (convert from GPU tensors)
            return {
                'L': self.L.cpu().numpy(),
                'S': self.E.cpu().numpy(),
            }

    def get_rank(self, threshold: float = 0.01) -> int:
        """Estimate rank of low-rank component."""
        if not hasattr(self, 'L'):
            raise RuntimeError("Must call fit() before getting rank")

        # Get only singular values (S) from SVD
        L_tensor = self.L if isinstance(self.L, torch.Tensor) else torch.from_numpy(self.L)
        _, s, _ = torch.linalg.svd(L_tensor)
        s = s.cpu().numpy()
        max_sv = s[0]
        rank = (s > threshold * max_sv).sum()

        return rank

    def __del__(self):
        """Cleanup GPU memory on deletion."""
        if hasattr(self, 'device') and self.device.startswith('cuda'):
            torch.cuda.empty_cache()


def auto_rpca(D: np.ndarray, use_gpu: Optional[bool] = None,
              device: Optional[str] = None, n_threads: Optional[int] = None,
              structured: bool = False, indices: Optional[np.ndarray] = None,
              d: Optional[np.ndarray] = None, return_tensors: bool = False,
              **kwargs) -> Dict[str, np.ndarray | torch.Tensor]:
    """
    Automatically choose GPU or CPU RPCA based on availability.
    Supports both standard and structured RPCA variants.

    Args:
        D: Input data matrix [m, n]
        use_gpu: Force GPU (True) or CPU (False), None for auto-detect
        device: Specific device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')
        n_threads: Number of threads for CPU RPCA (None for auto-detect)
        structured: Use StructuredRPCA_GPU for weighted SVD (requires CUDA)
        indices: Index mapping array for structured RPCA (required if structured=True)
        d: Repetition count array for structured RPCA (required if structured=True)
        return_tensors: If True, return GPU tensors (only for structured GPU); if False, return numpy arrays
        **kwargs: Additional arguments (max_iter, tol for fit; mu, lmbda for __init__)

    Returns:
        Dictionary with 'L', 'S', 'rank' keys (as tensors or numpy arrays)

    Structured RPCA:
        Uses weighted SVD with row repetition counts.
        Requires: A_u (upper triangular), indices, d (repetition counts)
        Example use:
            auto_rpca(A_u, structured=True, indices=indices, d=d, device='cuda:0', return_tensors=True)
    """
    # Structured RPCA variant
    if structured:
        if not CUDA_AVAILABLE:
            raise RuntimeError("StructuredRPCA_GPU requires CUDA")
        if indices is None or d is None:
            raise ValueError("Structured RPCA requires 'indices' and 'd' parameters")

        print("Using StructuredRPCA_GPU (weighted SVD with row repetition)")

        # Separate init kwargs from fit kwargs
        init_kwargs = {}
        fit_kwargs = {}

        # Valid init parameters for StructuredRPCA_GPU
        struct_init_params = {'mu', 'lambda_val'}
        # Valid fit parameters for StructuredRPCA_GPU
        struct_fit_params = {'max_iter', 'tol', 'iter_print', 'enable_timing'}

        for k, v in kwargs.items():
            if k in struct_fit_params:
                fit_kwargs[k] = v
            elif k in struct_init_params:
                init_kwargs[k] = v
            else:
                fit_kwargs[k] = v

        # Handle device parameter
        if device is not None:
            if device.startswith('cuda:'):
                use_gpu = True
            elif device == 'cpu':
                raise ValueError("StructuredRPCA_GPU requires CUDA device")

        if use_gpu and CUDA_AVAILABLE:
            if device is not None:
                init_kwargs['device'] = device
            rpca = StructuredRPCA_GPU(D, indices, d, **init_kwargs)
        else:
            raise RuntimeError("StructuredRPCA_GPU requires CUDA")

        # Fit and get results
        rpca.fit(**fit_kwargs)
        results = rpca.get_results(return_tensors=return_tensors)
        # results['rank'] = rpca.get_rank()

        return results

    # Standard RPCA variant (original code)
    if use_gpu is None:
        use_gpu = CUDA_AVAILABLE

    # Separate init kwargs from fit kwargs
    init_kwargs = {}
    fit_kwargs = {}

    # Valid init parameters for RPCA_GPU
    gpu_init_params = {'mu', 'lmbda'}
    # Valid init parameters for RPCA_CPU (now includes mu for pyrpca compatibility)
    cpu_init_params = {'lmbda', 'n_threads', 'mu'}
    # Valid fit parameters for both
    fit_params = {'max_iter', 'tol', 'iter_print'}

    for k, v in kwargs.items():
        if k in fit_params:
            fit_kwargs[k] = v
        elif k in gpu_init_params or k in cpu_init_params:
            init_kwargs[k] = v
        else:
            # Pass other kwargs to fit() (they might be fit params with different names)
            fit_kwargs[k] = v

    # Handle device parameter
    if device is not None:
        if device.startswith('cuda:'):
            use_gpu = True
        elif device == 'cpu':
            use_gpu = False

    if use_gpu and CUDA_AVAILABLE:
        print("Using GPU-accelerated RPCA")
        # Only pass valid init params to GPU constructor, plus device if specified
        valid_gpu_init = {k: v for k, v in init_kwargs.items() if k in gpu_init_params}
        if device is not None:
            valid_gpu_init['device'] = device
        rpca = RPCA_GPU(D, **valid_gpu_init)
    elif use_gpu and not CUDA_AVAILABLE:
        print("GPU requested but CUDA not available, using multi-threaded CPU")
        if not RPCA_CPU_AVAILABLE:
            raise RuntimeError("pyrpca is not available for CPU RPCA. Install with: pip install pyrpca")
        # Only pass valid init params to CPU constructor
        valid_cpu_init = {k: v for k, v in init_kwargs.items() if k in cpu_init_params}
        # Add n_threads if specified
        if n_threads is not None:
            valid_cpu_init['n_threads'] = n_threads
        rpca = RPCA_CPU(D, **valid_cpu_init)
    else:
        print("Using multi-threaded CPU RPCA")
        if not RPCA_CPU_AVAILABLE:
            raise ImportError("pyrpca is not available for CPU RPCA. Install with: pip install pyrpca")
        # Only pass valid init params to CPU constructor
        valid_cpu_init = {k: v for k, v in init_kwargs.items() if k in cpu_init_params}
        # Add n_threads if specified
        if n_threads is not None:
            valid_cpu_init['n_threads'] = n_threads
        rpca = RPCA_CPU(D, **valid_cpu_init)

    # Fit and get results
    rpca.fit(**fit_kwargs)
    results = rpca.get_results()
    # results['rank'] = rpca.get_rank()

    return results


# Convenience function for backward compatibility
def apply_rpca(features: np.ndarray, max_iter: int = 10, tol: float = 1e-7,
               use_gpu: Optional[bool] = None, device: Optional[str] = None,
               n_threads: Optional[int] = None, structured: bool = False, indices: Optional[np.ndarray] = None,
               d: Optional[np.ndarray] = None, return_tensors: bool = False,
               **kwargs) -> Dict[str, np.ndarray | torch.Tensor]:
    """
    Apply Robust PCA (RPCA) to decompose features into low-rank and sparse components.

    This is a convenience function that automatically selects GPU or CPU implementation.

    Args:
        features: Feature matrix [N, D]
        max_iter: Maximum iterations for RPCA (default: 10)
        tol: Tolerance for convergence
        use_gpu: Use GPU if available (None for auto-detect)
        device: Specific device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')
        n_threads: Number of threads for CPU RPCA (None for auto-detect)
        structured: Use StructuredRPCA_GPU for weighted SVD (requires CUDA)
        indices: Index mapping array for structured RPCA
        d: Repetition count array for structured RPCA
        return_tensors: If True, return GPU tensors (only for structured GPU); if False, return numpy arrays
        **kwargs: Additional arguments (passed through)

    Returns:
        Dictionary with:
            - 'L': Low-rank matrix [N, D]
            - 'S': Sparse matrix [N, D]
            - 'rank': Estimated rank of L (if computed)
    """
    result = auto_rpca(features, use_gpu=use_gpu, device=device, n_threads=n_threads, structured=structured,
                       indices=indices, d=d, return_tensors=return_tensors, max_iter=max_iter, tol=tol, **kwargs)
    return result


if __name__ == "__main__":
    # Simple test
    print("RPCA Utilities Module")
    print("=" * 50)
    print(f"RPCA (CPU) available: {RPCA_CPU_AVAILABLE}")
    print(f"CUDA available: {CUDA_AVAILABLE}")

    # Test with random data
    if CUDA_AVAILABLE:
        print("\nTesting GPU RPCA with random data...")
        np.random.seed(42)
        X = np.random.randn(1000, 100).astype(np.float32)

        rpca = RPCA_GPU(X)
        L, S = rpca.fit(max_iter=100, iter_print=50)
        results = rpca.get_results()
        rank = rpca.get_rank()

        print(f"\nResults:")
        print(f"  Low-rank L shape: {results['L'].shape}")
        print(f"  Sparse S shape: {results['S'].shape}")
        print(f"  Estimated rank: {rank}")
