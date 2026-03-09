"""
Per-Dimension Correlation Monitoring for Training

This module provides utilities to monitor per-dimension correlation between
predicted and ground-truth features during training.

Key insights:
- Overall loss convergence doesn't guarantee per-dimension learning
- Minor dimensions (dims 1-15) may fail to learn even if total loss decreases
- Monitoring per-dim correlation helps detect trivial solution early
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class PerDimMonitor:
    """
    Monitor per-dimension correlation between predictions and ground truth.

    This helps detect if the model is learning meaningful features or just
    converging to a trivial solution (e.g., predicting constant values).

    Usage:
        monitor = PerDimonitor(num_dims=16, log_freq=100)

        # During training
        metrics = monitor.update(pred, gt, valid_mask, iteration)

        # Check if training is successful
        if monitor.is_learning_minor_components():
            print("Minor components are being learned!")
    """

    def __init__(
        self,
        num_dims: int = 16,
        log_freq: int = 100,
        target_dim0_corr: float = 0.90,
        target_minor_corr: float = 0.30,
        warmup_iters: int = 200,
    ):
        """
        Args:
            num_dims: Number of feature dimensions to monitor
            log_freq: How often to log correlations (in iterations)
            target_dim0_corr: Target correlation for dimension 0 (principal component)
            target_minor_corr: Target correlation for dimensions 1-15 (minor components)
            warmup_iters: Number of iterations before checking correlation targets
        """
        self.num_dims = num_dims
        self.log_freq = log_freq
        self.target_dim0_corr = target_dim0_corr
        self.target_minor_corr = target_minor_corr
        self.warmup_iters = warmup_iters

        # Tracking history
        self.history = {
            'iterations': [],
            'dim0_corr': [],
            'minor_corr_mean': [],
            'minor_corr_std': [],
            'all_dims': [],
        }

        # Statistics
        self.best_dim0_corr = 0.0
        self.best_minor_corr = 0.0
        self.converged_dim0 = False
        self.converged_minor = False

    def compute_per_dim_correlation(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute Pearson correlation coefficient for each dimension.

        Args:
            pred: Predicted features [N, D]
            gt: Ground truth features [N, D]
            valid_mask: Optional binary mask [N] for valid features

        Returns:
            correlations: Array of shape [D] with correlation coefficients
        """
        # Apply valid mask if provided
        if valid_mask is not None and valid_mask.any():
            pred = pred[valid_mask]
            gt = gt[valid_mask]

        N, D = pred.shape

        if N < 10:  # Not enough samples
            return np.zeros(D)

        correlations = []
        for dim in range(D):
            pred_dim = pred[:, dim].detach().cpu().numpy()
            gt_dim = gt[:, dim].detach().cpu().numpy()

            # Skip if no variance
            if pred_dim.std() < 1e-8 or gt_dim.std() < 1e-8:
                correlations.append(0.0)
                continue

            # Pearson correlation coefficient
            pred_centered = pred_dim - pred_dim.mean()
            gt_centered = gt_dim - gt_dim.mean()

            numerator = (pred_centered * gt_centered).sum()
            denominator = np.sqrt((pred_centered ** 2).sum()) * np.sqrt((gt_centered ** 2).sum())

            if denominator > 1e-8:
                corr = numerator / denominator
                # Clamp to [-1, 1]
                corr = max(-1.0, min(1.0, corr))
            else:
                corr = 0.0

            correlations.append(corr)

        return np.array(correlations)

    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        iteration: int = 0,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """
        Update monitoring metrics.

        Args:
            pred: Predicted features [N, D]
            gt: Ground truth features [N, D]
            valid_mask: Optional binary mask [N] for valid features
            iteration: Current iteration number
            epoch: Current epoch number

        Returns:
            metrics: Dictionary with current metrics
        """
        # Compute correlations
        correlations = self.compute_per_dim_correlation(pred, gt, valid_mask)

        dim0_corr = correlations[0]
        minor_corrs = correlations[1:] if len(correlations) > 1 else np.array([])
        minor_corr_mean = minor_corrs.mean() if len(minor_corrs) > 0 else 0.0
        minor_corr_std = minor_corrs.std() if len(minor_corrs) > 0 else 0.0

        # Update best scores
        self.best_dim0_corr = max(self.best_dim0_corr, dim0_corr)
        self.best_minor_corr = max(self.best_minor_corr, minor_corr_mean)

        # Check convergence
        if iteration > self.warmup_iters:
            self.converged_dim0 = dim0_corr >= self.target_dim0_corr
            self.converged_minor = minor_corr_mean >= self.target_minor_corr

        # Log if needed
        if iteration % self.log_freq == 0:
            self._log_correlations(iteration, epoch, correlations, dim0_corr, minor_corr_mean, minor_corr_std)

        # Update history
        self.history['iterations'].append(iteration)
        self.history['dim0_corr'].append(dim0_corr)
        self.history['minor_corr_mean'].append(minor_corr_mean)
        self.history['minor_corr_std'].append(minor_corr_std)
        self.history['all_dims'].append(correlations.copy())

        return {
            'dim0_corr': dim0_corr,
            'minor_corr_mean': minor_corr_mean,
            'minor_corr_std': minor_corr_std,
            'best_dim0_corr': self.best_dim0_corr,
            'best_minor_corr': self.best_minor_corr,
        }

    def _log_correlations(
        self,
        iteration: int,
        epoch: int,
        correlations: np.ndarray,
        dim0_corr: float,
        minor_corr_mean: float,
        minor_corr_std: float,
    ):
        """Log correlation metrics."""
        # Status
        status = []
        if self.converged_dim0:
            status.append("✓ Dim0")
        else:
            status.append("✗ Dim0")

        if self.converged_minor:
            status.append("✓ Minor")
        else:
            status.append("✗ Minor")

        status_str = " | ".join(status)
        print(f"[PerDim] Dim0={dim0_corr:.4f} (best:{self.best_dim0_corr:.4f}) | Minor={minor_corr_mean:.4f}±{minor_corr_std:.4f} (best:{self.best_minor_corr:.4f}) | {status_str}")

    def is_learning_principal_component(self) -> bool:
        """Check if principal component (dim 0) is being learned."""
        return self.converged_dim0

    def is_learning_minor_components(self) -> bool:
        """Check if minor components (dims 1-15) are being learned."""
        return self.converged_minor

    def is_trivial_solution(self) -> bool:
        """
        Detect if model has converged to trivial solution.

        Trivial solution: All dimensions have near-zero correlation,
        indicating the model is predicting constant values.
        """
        if len(self.history['dim0_corr']) < 10:
            return False

        recent_dim0 = np.mean(self.history['dim0_corr'][-10:])
        recent_minor = np.mean(self.history['minor_corr_mean'][-10:])

        # Trivial solution: both principal and minor components have low correlation
        return recent_dim0 < 0.2 and recent_minor < 0.1

    def get_summary(self) -> Dict:
        """Get summary of monitoring results."""
        if len(self.history['iterations']) == 0:
            return {
                'total_iterations': 0,
                'final_dim0_corr': 0.0,
                'final_minor_corr': 0.0,
                'best_dim0_corr': 0.0,
                'best_minor_corr': 0.0,
                'converged_dim0': False,
                'converged_minor': False,
                'is_trivial': False,
            }

        return {
            'total_iterations': self.history['iterations'][-1],
            'final_dim0_corr': self.history['dim0_corr'][-1],
            'final_minor_corr': self.history['minor_corr_mean'][-1],
            'best_dim0_corr': self.best_dim0_corr,
            'best_minor_corr': self.best_minor_corr,
            'converged_dim0': self.converged_dim0,
            'converged_minor': self.converged_minor,
            'is_trivial': self.is_trivial_solution(),
        }

    def save_history(self, path: str):
        """Save monitoring history to file."""
        import json

        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {
            k: [float(x) if not isinstance(x, np.ndarray) else x.tolist() for x in v]
            for k, v in self.history.items()
        }

        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)

        print(f"Per-dim correlation history saved to {path}")


def compute_per_dim_correlation(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Standalone function to compute per-dimension correlation.

    Args:
        pred: Predicted features [N, D]
        gt: Ground truth features [N, D]
        valid_mask: Optional binary mask [N] for valid features

    Returns:
        correlations: Array of shape [D] with correlation coefficients
    """
    monitor = PerDimMonitor(num_dims=pred.shape[1])
    return monitor.compute_per_dim_correlation(pred, gt, valid_mask)
