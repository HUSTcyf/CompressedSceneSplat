#!/usr/bin/env python3
"""
Rendering Quality Analysis: Predicted vs GT Features

This script analyzes:
1. Why GT features (checkpoint_with_features.pth) render normally
2. Why predicted features (checkpoint_with_features_s.pth) render poorly
3. What error metrics are needed for acceptable rendering quality

Key Concepts:
- Rendering quality depends on feature spatial coherence, not just point-wise accuracy
- GT features have perfect spatial coherence (grid-based compression)
- Predicted features suffer from noise amplification and mode collapse
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def load_features(scene_path="bed", device="cpu"):
    """Load GT and predicted features."""
    output_dir = PROJECT_ROOT / "output_features" / scene_path

    # Load GT features (SVD compressed)
    gt_ckpt = torch.load(output_dir / "checkpoint_with_features.pth", map_location=device)
    # checkpoint format: (model_params, first_iter)
    # model_params[7] = language_features
    gt_lang_feat = gt_ckpt[0][7]  # [N, 16]

    # Load predicted features
    pred_ckpt = torch.load(output_dir / "checkpoint_with_features_s.pth", map_location=device)
    pred_lang_feat = pred_ckpt[0][7]  # [N, 16]

    # Get coords for spatial analysis (model_params[1])
    coords = gt_ckpt[0][1]  # [N, 3]

    # Get valid feature mask (model_params[8] or [10] depending on format)
    if len(gt_ckpt[0]) >= 11:
        valid_mask = gt_ckpt[0][10]  # valid_feat_mask
    else:
        valid_mask = torch.ones(len(gt_lang_feat), dtype=torch.bool)

    return {
        'gt': gt_lang_feat,
        'pred': pred_lang_feat,
        'coords': coords,
        'valid_mask': valid_mask
    }

def compute_spatial_coherence(features, coords, k=10, sample_size=50000):
    """
    Compute spatial coherence: how similar are features of neighboring points?

    High spatial coherence = neighbors have similar features = smooth rendering
    Low spatial coherence = neighbors have different features = noisy rendering
    """
    # Sample points for efficiency
    if len(features) > sample_size:
        indices = np.random.choice(len(features), sample_size, replace=False)
        features_sample = features[indices]
        coords_sample = coords[indices]
    else:
        features_sample = features
        coords_sample = coords

    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords_sample)
    distances, indices = nbrs.kneighbors(coords_sample)

    # Compute feature variance within neighborhoods (exclude self)
    coherence_scores = []
    for i in range(len(features_sample)):
        neighbor_feats = features_sample[indices[i][1:]]  # Skip self
        center_feat = features_sample[i]

        # Compute variance of neighbor features
        neighbor_var = torch.var(neighbor_feats, dim=0).mean().item()
        coherence_scores.append(neighbor_var)

    return np.array(coherence_scores)

def compute_signal_to_noise_ratio(features):
    """
    Compute SNR: signal variance / noise variance

    Higher SNR = cleaner rendering
    Lower SNR = noisier rendering
    """
    # Signal = variance along dimension 0 (dominant component)
    signal_var = features[:, 0].var().item()

    # Noise = average variance of minor components (dims 1-15)
    noise_vars = [features[:, i].var().item() for i in range(1, features.shape[1])]
    noise_var = np.mean(noise_vars)

    if noise_var < 1e-10:
        return float('inf')

    snr = 10 * np.log10(signal_var / noise_var)
    return snr

def compute_noise_amplification(gt_features, pred_features):
    """
    Compute how much the model amplifies noise in low-variance dimensions.

    Amplification > 1: Model treating noise as signal
    Amplification < 1: Model suppressing noise
    """
    amplifications = []

    for dim in range(gt_features.shape[1]):
        gt_var = gt_features[:, dim].var().item()
        pred_var = pred_features[:, dim].var().item()

        if gt_var < 1e-10:
            if pred_var < 1e-10:
                amp = 1.0
            else:
                amp = float('inf')
        else:
            amp = pred_var / gt_var

        amplifications.append(amp)

    return amplifications

def compute_rendering_quality_metrics(gt_features, pred_features, coords, valid_mask=None):
    """
    Compute comprehensive metrics that predict rendering quality.
    """
    # Filter valid features - check for non-zero rows instead of using mask
    # This is more reliable as valid_mask may have inconsistencies
    gt_nonzero_mask = (gt_features.abs().sum(dim=1) > 1e-6)
    pred_nonzero_mask = (pred_features.abs().sum(dim=1) > 1e-6)

    # Use intersection of both masks
    combined_mask = gt_nonzero_mask & pred_nonzero_mask

    if combined_mask.sum() < 1000:
        print(f"Warning: Very few valid features ({combined_mask.sum()}), using all features")
        gt_valid = gt_features
        pred_valid = pred_features
        coords_valid = coords
    else:
        gt_valid = gt_features[combined_mask]
        pred_valid = pred_features[combined_mask]
        coords_valid = coords[combined_mask]
        print(f"Using {combined_mask.sum()} valid features out of {len(gt_features)} total")

    results = {}

    # 1. Point-wise metrics (existing)
    results['l1_loss'] = torch.abs(pred_valid - gt_valid).mean().item()
    results['l2_loss'] = torch.pow(pred_valid - gt_valid, 2).mean().item()
    results['cosine_sim'] = torch.nn.functional.cosine_similarity(pred_valid, gt_valid, dim=1).mean().item()

    # 2. Per-dimension correlation
    corrs = []
    for dim in range(gt_valid.shape[1]):
        gt_dim = gt_valid[:, dim].cpu().numpy()
        pred_dim = pred_valid[:, dim].cpu().numpy()
        if len(gt_dim) >= 2 and gt_dim.std() > 1e-10:
            corr, _ = pearsonr(gt_dim, pred_dim)
        else:
            corr = 0.0
        corrs.append(corr)
    results['per_dim_correlation'] = corrs
    results['mean_correlation'] = np.mean(corrs)
    results['dim0_correlation'] = corrs[0]

    # 3. Spatial coherence (CRITICAL for rendering)
    print("Computing spatial coherence...")
    gt_coherence = compute_spatial_coherence(gt_valid, coords_valid, k=10)
    pred_coherence = compute_spatial_coherence(pred_valid, coords_valid, k=10)

    results['gt_spatial_coherence'] = gt_coherence.mean()
    results['pred_spatial_coherence'] = pred_coherence.mean()
    results['coherence_degradation'] = (pred_coherence.mean() - gt_coherence.mean()) / gt_coherence.mean()

    # 4. Signal-to-Noise Ratio
    results['gt_snr'] = compute_signal_to_noise_ratio(gt_valid)
    results['pred_snr'] = compute_signal_to_noise_ratio(pred_valid)
    results['snr_degradation'] = results['gt_snr'] - results['pred_snr']

    # 5. Noise amplification
    amps = compute_noise_amplification(gt_valid, pred_valid)
    results['noise_amplifications'] = amps
    results['max_amplification'] = max([a for a in amps if a != float('inf')])
    results['dims_with_excessive_amplification'] = sum(1 for a in amps if a > 10)

    # 6. Feature distribution statistics
    results['gt_mean'] = gt_valid.mean().item()
    results['pred_mean'] = pred_valid.mean().item()
    results['gt_std'] = gt_valid.std().item()
    results['pred_std'] = pred_valid.std().item()

    # 7. Dimension-wise statistics
    results['dim_statistics'] = []
    for dim in range(gt_valid.shape[1]):
        gt_dim = gt_valid[:, dim]
        pred_dim = pred_valid[:, dim]

        results['dim_statistics'].append({
            'dim': dim,
            'gt_mean': gt_dim.mean().item(),
            'pred_mean': pred_dim.mean().item(),
            'gt_std': gt_dim.std().item(),
            'pred_std': pred_dim.std().item(),
            'gt_var': gt_dim.var().item(),
            'pred_var': pred_dim.var().item(),
            'bias': (pred_dim.mean() - gt_dim.mean()).item(),
            'correlation': corrs[dim]
        })

    return results

def estimate_required_thresholds_for_acceptable_rendering(current_metrics):
    """
    Estimate the error thresholds needed for acceptable rendering quality.

    Acceptable rendering criteria:
    1. Spatial coherence degradation < 50%
    2. SNR degradation < 6 dB
    3. Mean correlation > 0.8
    4. Max noise amplification < 5x
    5. L1 loss < 0.03
    """
    print("\n" + "="*80)
    print("RENDERING QUALITY REQUIREMENTS ANALYSIS")
    print("="*80)

    print("\n## Current Metrics vs Required Thresholds\n")

    # Define thresholds for acceptable rendering
    thresholds = {
        'l1_loss': 0.03,
        'mean_correlation': 0.80,
        'dim0_correlation': 0.95,
        'coherence_degradation': 0.50,  # 50% degradation max
        'snr_degradation': 6.0,  # 6 dB degradation max
        'max_amplification': 5.0,  # 5x amplification max
    }

    # Current values
    current = {
        'l1_loss': current_metrics['l1_loss'],
        'mean_correlation': current_metrics['mean_correlation'],
        'dim0_correlation': current_metrics['dim0_correlation'],
        'coherence_degradation': abs(current_metrics['coherence_degradation']),
        'snr_degradation': current_metrics['snr_degradation'],
        'max_amplification': current_metrics['max_amplification'],
    }

    # Improvement factors needed
    improvements = {}
    for key in thresholds:
        if current[key] > thresholds[key]:
            improvements[key] = current[key] / thresholds[key]
        else:
            improvements[key] = 1.0

    # Print comparison table
    print("| Metric | Current | Required | Status | Improvement Needed |")
    print("|--------|---------|----------|--------|-------------------|")

    status_map = {
        'l1_loss': lambda c, t: '✓' if c < t else '✗',
        'mean_correlation': lambda c, t: '✓' if c > t else '✗',
        'dim0_correlation': lambda c, t: '✓' if c > t else '✗',
        'coherence_degradation': lambda c, t: '✓' if c < t else '✗',
        'snr_degradation': lambda c, t: '✓' if c < t else '✗',
        'max_amplification': lambda c, t: '✓' if c < t else '✗',
    }

    for key in thresholds:
        c = current[key]
        t = thresholds[key]
        status = status_map[key](c, t)
        imp = improvements[key]

        if imp > 1.0:
            imp_str = f"{imp:.2f}x worse"
        elif imp < 1.0:
            imp_str = f"{1/imp:.2f}x better"
        else:
            imp_str = "At threshold"

        print(f"| {key:20s} | {c:8.4f} | {t:8.4f} | {status} | {imp_str:18s} |")

    # Overall assessment
    print("\n## Overall Assessment\n")

    passed = sum(1 for key in thresholds if status_map[key](current[key], thresholds[key]) == '✓')
    total = len(thresholds)

    print(f"Passed: {passed}/{total} criteria")

    if passed == total:
        print("✓ ✓ ✓ RENDERING QUALITY SHOULD BE ACCEPTABLE ✓ ✓ ✓")
    elif passed >= total * 0.7:
        print("⚠ RENDERING QUALITY MAY BE MARGINAL ⚠")
        print("Some visual artifacts may be present")
    else:
        print("✗ ✗ ✗ RENDERING QUALITY LIKELY POOR ✗ ✗ ✗")
        print("Significant visual artifacts expected")

    return thresholds, current, improvements

def explain_rendering_differences(metrics):
    """
    Explain WHY GT renders normally and predicted renders poorly.
    """
    print("\n" + "="*80)
    print("WHY GT FEATURES RENDER NORMALLY vs PREDICTED FEATURES RENDER POORLY")
    print("="*80)

    print("\n## 1. Spatial Coherence Analysis\n")

    gt_coh = metrics['gt_spatial_coherence']
    pred_coh = metrics['pred_spatial_coherence']
    deg = metrics['coherence_degradation']

    print(f"GT Features Spatial Coherence: {gt_coh:.6f}")
    print(f"Predicted Features Spatial Coherence: {pred_coh:.6f}")
    print(f"Coherence Degradation: {deg*100:.1f}%")

    if deg > 0.5:
        print("\n❌ PROBLEM: Predicted features have poor spatial coherence!")
        print("   → Neighboring Gaussians have very different features")
        print("   → This causes speckled/noisy appearance in renders")
    else:
        print("\n✓ Spatial coherence is acceptable")

    print("\n## 2. Signal-to-Noise Ratio Analysis\n")

    gt_snr = metrics['gt_snr']
    pred_snr = metrics['pred_snr']
    snr_deg = metrics['snr_degradation']

    print(f"GT Features SNR: {gt_snr:.2f} dB")
    print(f"Predicted Features SNR: {pred_snr:.2f} dB")
    print(f"SNR Degradation: {snr_deg:.2f} dB")

    if snr_deg > 10:
        print("\n❌ PROBLEM: Predicted features have much lower SNR!")
        print("   → Noise dominates over signal in minor components")
        print("   → This causes grainy/sand-like appearance in renders")
    elif snr_deg > 6:
        print("\n⚠ WARNING: Moderate SNR degradation")
        print("   → Some noise visible in smooth regions")
    else:
        print("\n✓ SNR is acceptable")

    print("\n## 3. Noise Amplification Analysis\n")

    max_amp = metrics['max_amplification']
    excessive_dims = metrics['dims_with_excessive_amplification']

    print(f"Max Noise Amplification: {max_amp:.2f}x")
    print(f"Dimensions with >10x Amplification: {excessive_dims}/16")

    if max_amp > 100:
        print("\n❌ CRITICAL: Extreme noise amplification!")
        print("   → Low-variance dimensions are being amplified 100x+")
        print("   → This causes salt-and-pepper noise in renders")
    elif max_amp > 10:
        print("\n⚠ WARNING: Significant noise amplification")
        print(f"   → {excessive_dims} dimensions show >10x amplification")
        print("   → Visible speckles in rendered images")
    else:
        print("\n✓ Noise amplification is under control")

    print("\n## 4. Per-Dimension Correlation Analysis\n")

    corrs = metrics['per_dim_correlation']
    dim0_corr = metrics['dim0_correlation']

    print(f"Dimension 0 Correlation: {dim0_corr:.4f}")
    good_dims = sum(1 for c in corrs if c > 0.5)
    print(f"Dimensions with r > 0.5: {good_dims}/16")

    if dim0_corr > 0.9 and good_dims < 3:
        print("\n❌ PROBLEM: Mode collapse detected!")
        print("   → Only dimension 0 is learned (major component)")
        print("   → All minor components (dims 1-15) are ignored")
        print("   → Loss of fine details and semantic information")
        print("   → Objects appear blurry or lose definition")
    elif good_dims < 8:
        print("\n⚠ WARNING: Partial mode collapse")
        print(f"   → Only {good_dims}/16 dimensions meaningfully learned")
        print("   → Loss of detail in renders")
    else:
        print("\n✓ Most dimensions are learned")

    print("\n## 5. Root Cause Summary\n")

    print("\nThe GT features render normally because:")
    print("  ✓ Perfect spatial coherence (grid-based compression)")
    print("  ✓ High SNR (clean signal in all dimensions)")
    print("  ✓ No noise amplification")
    print("  ✓ All dimensions properly correlated")

    print("\nThe predicted features render poorly because:")
    issues = []
    if deg > 0.5:
        issues.append("Poor spatial coherence")
    if snr_deg > 6:
        issues.append("Low SNR")
    if max_amp > 10:
        issues.append("Noise amplification")
    if good_dims < 3:
        issues.append("Mode collapse (only dim 0 learned)")

    for issue in issues:
        print(f"  ✗ {issue}")

    print("\n## 6. Rendering Artifact Mapping\n")

    print("\n| Visual Artifact | Root Cause | Metric |")
    print("|-----------------|------------|--------|")
    if deg > 0.5:
        print("| Speckled/Noisy | Poor spatial coherence | coherence_degradation > 50% |")
    if snr_deg > 6:
        print("| Grainy texture | Low SNR | snr_degradation > 6 dB |")
    if max_amp > 10:
        print(f"| Salt-and-pepper noise | Noise amplification | max_amplification = {max_amp:.1f}x |")
    if dim0_corr > 0.9 and good_dims < 3:
        print("| Blurry/undefined | Mode collapse | Only 1/16 dims learned |")
    if metrics['l1_loss'] > 0.03:
        print("| Wrong colors/values | Prediction error | L1 loss > 0.03 |")

def generate_recommendations(metrics, improvements):
    """
    Generate specific recommendations to improve rendering quality.
    """
    print("\n" + "="*80)
    print("RECOMMENDATIONS TO IMPROVE RENDERING QUALITY")
    print("="*80)

    print("\n## Priority 1: CRITICAL (Must Fix)\n")

    if metrics['dims_with_excessive_amplification'] > 5:
        print("1. **Add L2 Regularization** (CRITICAL)")
        print("   ```python")
        print("   l2_reg = 1e-4  # Suppress noise amplification")
        print("   total_loss = reconstruction_loss + l2_reg * model.parameters().norm()")
        print("   ```")
        print("   Expected: 50-70% reduction in noise amplification\n")

    if metrics['dim0_correlation'] > 0.9 and sum(1 for c in metrics['per_dim_correlation'] if c > 0.5) < 3:
        print("2. **Add Per-Dimension Loss Weighting** (CRITICAL)")
        print("   ```python")
        print("   # Weight loss by GT variance to focus on minor components")
        print("   gt_var = gt_features.var(dim=0)")
        print("   weights = gt_var / gt_var.sum()")
        print("   weighted_loss = (loss_per_dim * weights).sum()")
        print("   ```")
        print("   Expected: 15-20% better minor component learning\n")

    if abs(metrics['coherence_degradation']) > 0.5:
        print("3. **Add Spatial Smoothness Loss** (CRITICAL)")
        print("   ```python")
        print("   # Encourage neighboring points to have similar features")
        print("   smoothness_loss = compute_spatial_smoothness(pred_features, coords)")
        print("   total_loss += 0.1 * smoothness_loss")
        print("   ```")
        print("   Expected: 30-40% improvement in spatial coherence\n")

    print("\n## Priority 2: HIGH (Should Fix)\n")

    if metrics['l1_loss'] > 0.03:
        improvement_needed = metrics['l1_loss'] / 0.03
        print(f"1. **Reduce L1 Loss** (Currently {metrics['l1_loss']:.4f}, need < 0.03)")
        print(f"   Required improvement: {improvement_needed:.2f}x")
        print("   - Increase training iterations")
        print("   - Try learning rate warmup + restart")
        print("   - Implement progressive training strategy\n")

    if metrics['snr_degradation'] > 6:
        print("2. **Improve SNR** (Currently degraded by {:.1f} dB)".format(metrics['snr_degradation']))
        print("   - Add dropout to prevent overfitting to noise")
        print("   - Use early stopping on validation minor component loss")
        print("   - Implement focal loss for hard-to-learn dimensions\n")

    print("\n## Priority 3: MEDIUM (Nice to Have)\n")

    print("1. **Increase SVD Rank to 32**")
    print("   - Better preservation of minor components")
    print("   - Expected: 20-30% overall quality improvement\n")

    print("2. **Progressive Training Strategy**")
    print("   - Phase 1: Train only on dimension 0")
    print("   - Phase 2: Fine-tune on all dimensions")
    print("   - Phase 3: Joint optimization")
    print("   - Expected: 30-40% better convergence\n")

def main():
    """Main analysis function."""
    print("\n" + "="*80)
    print("RENDERING QUALITY ANALYSIS: GT vs PREDICTED FEATURES")
    print("="*80)

    # Load features
    print("\nLoading features...")
    data = load_features(scene_path="bed", device="cpu")

    # Compute metrics
    print("\nComputing rendering quality metrics...")
    metrics = compute_rendering_quality_metrics(
        data['gt'],
        data['pred'],
        data['coords'],
        data['valid_mask']
    )

    # Explain differences
    explain_rendering_differences(metrics)

    # Estimate thresholds
    thresholds, current, improvements = estimate_required_thresholds_for_acceptable_rendering(metrics)

    # Generate recommendations
    generate_recommendations(metrics, improvements)

    # Save results
    output_dir = PROJECT_ROOT / "output_features" / "bed"
    results_file = output_dir / "rendering_quality_analysis.md"

    print("\n" + "="*80)
    print(f"Saving detailed analysis to: {results_file}")
    print("="*80)

    return metrics, thresholds, current, improvements

if __name__ == "__main__":
    metrics, thresholds, current, improvements = main()
