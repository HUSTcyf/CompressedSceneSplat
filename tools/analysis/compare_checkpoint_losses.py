#!/usr/bin/env python3
"""
Compare checkpoint losses with training losses.

This script calculates the actual losses between predicted features
(checkpoint_with_features_s.pth) and GT features (checkpoint_with_features.pth),
then compares them with the training losses from the loss curves.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import json
import numpy as np


def load_loss_data(exp_dir, scene):
    """Load training loss data from JSON file."""
    loss_file = os.path.join(exp_dir, "loss_curves", f"{scene}_loss_data.json")
    with open(loss_file, 'r') as f:
        data = json.load(f)
    return data


def get_final_losses(loss_data):
    """Get final training losses."""
    return {
        'iteration': loss_data['iterations'][-1],
        'total': loss_data['total_loss'][-1],
        'l1': loss_data['l1_loss'][-1],
        'cosine': loss_data['cos_loss'][-1],
        'contrast': loss_data['contrast_loss'][-1],
    }


def load_checkpoints(scene_path, device='cpu'):
    """Load GT and predicted checkpoints."""
    output_dir = PROJECT_ROOT / "output_features" / scene_path

    # Load GT features
    gt_ckpt = torch.load(output_dir / "checkpoint_with_features.pth", map_location=device)
    gt_feat = gt_ckpt[0][7]  # [N, 16]

    # Load predicted features
    pred_ckpt = torch.load(output_dir / "checkpoint_with_features_s.pth", map_location=device)
    pred_feat = pred_ckpt[0][7]  # [N, 16]

    # Get coords
    coords = gt_ckpt[0][1]  # [N, 3]

    return {
        'gt': gt_feat,
        'pred': pred_feat,
        'coords': coords,
    }


def filter_valid_features(gt_feat, pred_feat, coords):
    """Filter to valid (non-zero) features."""
    gt_nonzero = (gt_feat.abs().sum(dim=1) > 1e-6)
    pred_nonzero = (pred_feat.abs().sum(dim=1) > 1e-6)
    combined_mask = gt_nonzero & pred_nonzero

    return {
        'gt': gt_feat[combined_mask],
        'pred': pred_feat[combined_mask],
        'coords': coords[combined_mask],
        'mask': combined_mask,
    }


def compute_l1_loss(pred, gt, weights=None):
    """Compute weighted L1 loss."""
    # Per-dimension L1 loss
    loss_per_dim = torch.abs(pred - gt).mean(dim=0)  # [16]

    if weights is not None:
        # Apply weights
        weighted_loss = (loss_per_dim * weights).sum()
    else:
        weighted_loss = loss_per_dim.mean()

    return weighted_loss.item(), loss_per_dim


def compute_cosine_loss(pred, gt, weights=None):
    """Compute weighted cosine loss."""
    # Compute per-sample cosine similarity (correct method)
    # Normalize along feature dimension (dim=1), not batch dimension
    cos_sim_per_sample = F.cosine_similarity(pred, gt, dim=1)

    # Convert to loss: 1 - cosine similarity
    loss_per_sample = 1 - cos_sim_per_sample

    # Mean loss
    mean_loss = loss_per_sample.mean()

    # For per-dimension loss (for analysis), compute differently
    # We need to compute cosine similarity per dimension
    pred_norm = F.normalize(pred, dim=0)  # Normalize each dimension independently
    gt_norm = F.normalize(gt, dim=0)
    cos_sim_per_dim = (pred_norm * gt_norm).mean(dim=0)  # [16]
    loss_per_dim = 1 - cos_sim_per_dim

    if weights is not None:
        # Apply weights to per-dimension loss
        weighted_loss = (loss_per_dim * weights).sum()
    else:
        weighted_loss = mean_loss

    return weighted_loss.item(), loss_per_dim


def compute_contrastive_loss(pred, gt, coords, temperature=0.05, k=10):
    """
    Compute contrastive loss.

    This encourages features of neighboring points to be similar.
    """
    # Simplified contrastive loss for efficiency
    # Use k-nearest neighbors based on coordinates

    # Convert to numpy for sklearn
    coords_np = coords.cpu().numpy()
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()

    # Sample fewer points for efficiency
    sample_size = min(50000, len(coords_np))
    indices = np.random.choice(len(coords_np), sample_size, replace=False)

    coords_sample = coords_np[indices]
    pred_sample = pred_np[indices]
    gt_sample = gt_np[indices]

    # For each point, find its neighbors
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords_sample)
    _, neighbor_indices = nbrs.kneighbors(coords_sample)

    # Compute contrastive loss
    # Positive pairs: neighboring points should have similar features
    contrast_loss = 0
    count = 0

    for i in range(len(coords_sample)):
        # Get neighbors (excluding self)
        neighbors = neighbor_indices[i][1:]  # Skip self

        # Positive pairs: pred should be close to gt for neighbors
        anchor_pred = torch.from_numpy(pred_sample[i:i+1]).float()
        positive_gt = torch.from_numpy(gt_sample[neighbors]).float()

        # Compute similarity
        sim = F.cosine_similarity(
            anchor_pred.unsqueeze(0),
            positive_gt,
            dim=1
        )

        # Contrastive loss: 1 - similarity
        contrast_loss += (1 - sim).mean().item()
        count += 1

    contrast_loss = contrast_loss / count if count > 0 else 0

    return contrast_loss


def compute_total_loss(l1_loss, cosine_loss, contrast_loss, l1_weight=1.0,
                       cosine_weight=1.0, contrast_weight=0.02):
    """Compute total loss with weights."""
    total = (
        l1_weight * l1_loss +
        cosine_weight * cosine_loss +
        contrast_weight * contrast_loss
    )
    return total


def main():
    print("\n" + "="*80)
    print("CHECKPOINT LOSS COMPARISON WITH TRAINING LOSSES")
    print("="*80)

    scene = "bed"
    exp_dir = PROJECT_ROOT / "exp" / "lite-16-gridsvd"

    # Load training losses
    print("\nLoading training loss data...")
    loss_data = load_loss_data(exp_dir, scene)
    training_losses = get_final_losses(loss_data)

    print(f"\nTraining Losses (iteration {training_losses['iteration']}):")
    print(f"  Total:     {training_losses['total']:.6f}")
    print(f"  L1:        {training_losses['l1']:.6f}")
    print(f"  Cosine:    {training_losses['cosine']:.6f}")
    print(f"  Contrast:  {training_losses['contrast']:.6f}")

    # Load checkpoints
    print("\nLoading checkpoints...")
    checkpoint_data = load_checkpoints(scene, device='cpu')

    # Filter valid features
    print("Filtering valid features...")
    valid_data = filter_valid_features(
        checkpoint_data['gt'],
        checkpoint_data['pred'],
        checkpoint_data['coords']
    )

    print(f"Valid features: {valid_data['mask'].sum()}/{len(valid_data['mask'])}")

    gt_feat = valid_data['gt']
    pred_feat = valid_data['pred']
    coords = valid_data['coords']

    # Compute variance-based weights
    gt_var = gt_feat.var(dim=0)
    weights = gt_var / gt_var.sum()

    # Compute checkpoint losses
    print("\nComputing checkpoint losses...")

    # L1 loss (unweighted and weighted)
    l1_loss_unweighted, l1_per_dim = compute_l1_loss(pred_feat, gt_feat, weights=None)
    l1_loss_weighted, _ = compute_l1_loss(pred_feat, gt_feat, weights=weights)

    # Cosine loss (unweighted and weighted)
    cosine_loss_unweighted, cos_per_dim = compute_cosine_loss(pred_feat, gt_feat, weights=None)
    cosine_loss_weighted, _ = compute_cosine_loss(pred_feat, gt_feat, weights=weights)

    # Contrastive loss (approximate)
    print("Computing contrastive loss (this may take a moment)...")
    contrast_loss = compute_contrastive_loss(pred_feat, gt_feat, coords, temperature=0.05, k=10)

    # Total loss
    total_loss_unweighted = compute_total_loss(
        l1_loss_unweighted,
        cosine_loss_unweighted,
        contrast_loss,
        l1_weight=1.0,
        cosine_weight=1.0,
        contrast_weight=0.02
    )

    total_loss_weighted = compute_total_loss(
        l1_loss_weighted,
        cosine_loss_weighted,
        contrast_loss,
        l1_weight=1.0,
        cosine_weight=1.0,
        contrast_weight=0.02
    )

    print("\n" + "="*80)
    print("CHECKPOINT LOSSES (Computed from features)")
    print("="*80)

    print("\nUnweighted:")
    print(f"  Total:     {total_loss_unweighted:.6f}")
    print(f"  L1:        {l1_loss_unweighted:.6f}")
    print(f"  Cosine:    {cosine_loss_unweighted:.6f}")
    print(f"  Contrast:  {contrast_loss:.6f}")

    print("\nVariance-Weighted:")
    print(f"  Total:     {total_loss_weighted:.6f}")
    print(f"  L1:        {l1_loss_weighted:.6f}")
    print(f"  Cosine:    {cosine_loss_weighted:.6f}")
    print(f"  Contrast:  {contrast_loss:.6f}")

    print("\n" + "="*80)
    print("COMPARISON: Training vs Checkpoint")
    print("="*80)

    print("\n| Loss Type | Training | Checkpoint (Unwtd) | Checkpoint (Wtd) | Difference |")
    print("|-----------|----------|--------------------|------------------|------------|")

    diff_l1_unwtd = l1_loss_unweighted - training_losses['l1']
    diff_l1_wtd = l1_loss_weighted - training_losses['l1']
    print(f"| L1        | {training_losses['l1']:8.6f} | {l1_loss_unweighted:19.6f} | {l1_loss_weighted:16.6f} | {diff_l1_unwtd:+10.6f} |")

    diff_cos_unwtd = cosine_loss_unweighted - training_losses['cosine']
    diff_cos_wtd = cosine_loss_weighted - training_losses['cosine']
    print(f"| Cosine    | {training_losses['cosine']:8.6f} | {cosine_loss_unweighted:19.6f} | {cosine_loss_weighted:16.6f} | {diff_cos_unwtd:+10.6f} |")

    diff_contrast = contrast_loss - training_losses['contrast']
    print(f"| Contrast  | {training_losses['contrast']:8.6f} | {contrast_loss:19.6f} | {contrast_loss:16.6f} | {diff_contrast:+10.6f} |")

    diff_total_unwtd = total_loss_unweighted - training_losses['total']
    diff_total_wtd = total_loss_weighted - training_losses['total']
    print(f"| Total     | {training_losses['total']:8.6f} | {total_loss_unweighted:19.6f} | {total_loss_weighted:16.6f} | {diff_total_unwtd:+10.6f} |")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    print("\n1. Loss Comparison:")
    if abs(diff_total_unwtd) < 0.01:
        print("   ✓ Checkpoint losses closely match training losses")
        print("   → The checkpoint represents the final trained state")
    elif diff_total_unwtd > 0:
        print(f"   ✗ Checkpoint losses are {diff_total_unwtd/training_losses['total']*100:.1f}% HIGHER than training")
        print("   → The checkpoint may be from an earlier iteration")
    else:
        print(f"   ⚠ Checkpoint losses are {abs(diff_total_unwtd)/training_losses['total']*100:.1f}% LOWER than training")
        print("   → The checkpoint may be from a later iteration or different run")

    print("\n2. Component Breakdown:")

    if abs(diff_l1_unwtd / training_losses['l1']) < 0.1:
        print("   ✓ L1 loss matches training")
    else:
        direction = "HIGHER" if diff_l1_unwtd > 0 else "LOWER"
        print(f"   ⚠ L1 loss is {abs(diff_l1_unwtd/training_losses['l1']*100):.1f}% {direction} than training")

    if abs(diff_cos_unwtd / training_losses['cosine']) < 0.1:
        print("   ✓ Cosine loss matches training")
    else:
        direction = "HIGHER" if diff_cos_unwtd > 0 else "LOWER"
        print(f"   ⚠ Cosine loss is {abs(diff_cos_unwtd/training_losses['cosine']*100):.1f}% {direction} than training")

    if abs(diff_contrast / training_losses['contrast']) < 0.1:
        print("   ✓ Contrast loss matches training")
    else:
        direction = "HIGHER" if diff_contrast > 0 else "LOWER"
        print(f"   ⚠ Contrast loss is {abs(diff_contrast/training_losses['contrast']*100):.1f}% {direction} than training")

    print("\n3. Per-Dimension L1 Loss:")
    print("\n   Dim | Training (est) | Checkpoint | GT Variance | Weight")
    print("   -----|----------------|------------|-------------|--------")
    for dim in range(16):
        var = gt_var[dim].item()
        w = weights[dim].item()
        l1 = l1_per_dim[dim].item()
        print(f"   {dim:4d} |              - |   {l1:8.6f} |    {var:8.6f} | {w:6.4f}")

    print("\n4. Key Findings:")

    # Check if training actually converged
    if training_losses['iteration'] < 30000:
        print(f"   ⚠ Training stopped at iteration {training_losses['iteration']}, not 30000!")
        print("   → This may explain poor performance")
        print("   → Model may not have converged")

    # Check if losses are high
    if l1_loss_unweighted > 0.1:
        print(f"   ✗ L1 loss is very high ({l1_loss_unweighted:.4f})")
        print("   → Model predictions are far from GT")

    if cosine_loss_unweighted > 0.1:
        print(f"   ✗ Cosine loss is high ({cosine_loss_unweighted:.4f})")
        print("   → Model predictions have wrong direction")

    return {
        'training': training_losses,
        'checkpoint_unweighted': {
            'total': total_loss_unweighted,
            'l1': l1_loss_unweighted,
            'cosine': cosine_loss_unweighted,
            'contrast': contrast_loss,
        },
        'checkpoint_weighted': {
            'total': total_loss_weighted,
            'l1': l1_loss_weighted,
            'cosine': cosine_loss_weighted,
            'contrast': contrast_loss,
        },
        'differences': {
            'total_unweighted': diff_total_unwtd,
            'total_weighted': diff_total_wtd,
            'l1_unweighted': diff_l1_unwtd,
            'l1_weighted': diff_l1_wtd,
            'cosine_unweighted': diff_cos_unwtd,
            'cosine_weighted': diff_cos_wtd,
            'contrast': diff_contrast,
        }
    }


if __name__ == "__main__":
    results = main()
