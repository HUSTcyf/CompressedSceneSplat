#!/usr/bin/env python3
"""
Diagnostic script to analyze mode collapse and contrastive loss issues.

This script analyzes:
1. Feature variance and distribution across classes
2. SVD compression quality
3. Contrastive pair quality
4. Mode collapse detection
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json


def analyze_svd_compression_quality(
    compressed_path: str,
    original_feat: torch.Tensor,
    svd_rank: int = 16
) -> Dict:
    """
    Analyze how much information is preserved after SVD compression.
    """
    data = np.load(compressed_path)
    compressed = torch.from_numpy(data['compressed']).float()

    print(f"\n{'='*60}")
    print(f"SVD-{svd_rank} Compression Analysis")
    print(f"{'='*60}")
    print(f"Compressed features shape: {compressed.shape}")
    print(f"Mean: {compressed.mean():.6f}, Std: {compressed.std():.6f}")
    print(f"Min: {compressed.min():.6f}, Max: {compressed.max():.6f}")

    # Analyze variance distribution
    variance_per_dim = compressed.var(dim=0)
    print(f"\nVariance per dimension (first {min(10, svd_rank)} dims):")
    for i in range(min(10, svd_rank)):
        print(f"  Dim {i}: {variance_per_dim[i]:.6f}")

    # Check if features are normalized
    norms = torch.norm(compressed, p=2, dim=1)
    print(f"\nFeature norms - Mean: {norms.mean():.6f}, Std: {norms.std():.6f}")

    # Compute pairwise cosine similarity
    compressed_norm = torch.nn.functional.normalize(compressed, p=2, dim=1)
    cosine_sim = torch.mm(compressed_norm[:1000], compressed_norm[:1000].T)
    # Remove diagonal
    mask = ~torch.eye(cosine_sim.shape[0], dtype=torch.bool, device=cosine_sim.device)
    off_diagonal = cosine_sim[mask]

    print(f"\nPairwise cosine similarity (sample of 1000):")
    print(f"  Mean: {off_diagonal.mean():.6f}")
    print(f"  Std: {off_diagonal.std():.6f}")
    print(f"  Min: {off_diagonal.min():.6f}")
    print(f"  Max: {off_diagonal.max():.6f}")
    print(f"  High similarity ratio (>0.9): {(off_diagonal > 0.9).float().mean():.2%}")

    return {
        "mean": compressed.mean().item(),
        "std": compressed.std().item(),
        "variance_per_dim": variance_per_dim.tolist(),
        "cosine_sim_mean": off_diagonal.mean().item(),
        "cosine_sim_std": off_diagonal.std().item(),
        "high_sim_ratio": (off_diagonal > 0.9).float().mean().item(),
    }


def analyze_contrastive_loss_quality(
    features: torch.Tensor,
    labels: torch.Tensor,
    min_samples: int = 20,
    temperature: float = 0.2
) -> Dict:
    """
    Analyze the quality of contrastive pairs.
    """
    print(f"\n{'='*60}")
    print(f"Contrastive Loss Quality Analysis")
    print(f"{'='*60}")

    unique_labels = torch.unique(labels)
    valid_classes = 0
    total_pairs = 0
    collapsed_pairs = 0

    class_stats = {}

    for lab in unique_labels:
        indices = (labels == lab).nonzero(as_tuple=True)[0]
        if indices.numel() < min_samples:
            continue

        valid_classes += 1

        # Split into two halves
        perm = indices[torch.randperm(indices.size(0))]
        split = perm.size(0) // 2

        if split == 0 or (perm.size(0) - split) == 0:
            continue

        group_a = features[perm[:split]]
        group_b = features[perm[split:]]

        # Aggregate (using sum as in current implementation)
        agg_a = group_a.sum(dim=0)
        agg_b = group_b.sum(dim=0)

        # Normalize
        agg_a_norm = torch.nn.functional.normalize(agg_a.unsqueeze(0), p=2, dim=1).squeeze(0)
        agg_b_norm = torch.nn.functional.normalize(agg_b.unsqueeze(0), p=2, dim=1).squeeze(0)

        # Compute similarity
        similarity = torch.dot(agg_a_norm, agg_b_norm).item()

        # Check for collapse (high similarity between halves)
        is_collapsed = similarity > 0.95
        if is_collapsed:
            collapsed_pairs += 1
        total_pairs += 1

        class_stats[int(lab)] = {
            "num_samples": indices.numel(),
            "similarity": similarity,
            "collapsed": is_collapsed
        }

    print(f"Valid classes (>= {min_samples} samples): {valid_classes}")
    print(f"Total pairs: {total_pairs}")
    print(f"Collapsed pairs (sim > 0.95): {collapsed_pairs} ({collapsed_pairs/total_pairs*100:.1f}%)")

    return {
        "valid_classes": valid_classes,
        "total_pairs": total_pairs,
        "collapsed_pairs": collapsed_pairs,
        "collapse_ratio": collapsed_pairs / total_pairs if total_pairs > 0 else 0,
        "class_stats": class_stats
    }


def detect_mode_collapse(
    features: torch.Tensor,
    labels: torch.Tensor
) -> Dict:
    """
    Detect if mode collapse is occurring.
    """
    print(f"\n{'='*60}")
    print(f"Mode Collapse Detection")
    print(f"{'='*60}")

    # Normalize features
    features_norm = torch.nn.functional.normalize(features, p=2, dim=1)

    # Global statistics
    global_mean = features_norm.mean(dim=0)
    global_std = features_norm.std(dim=0)

    print(f"Global feature mean norm: {torch.norm(global_mean):.6f}")
    print(f"Global feature std mean: {global_std.mean():.6f}")

    # Per-class statistics
    unique_labels = torch.unique(labels)
    class_means = []
    class_stds = []

    for lab in unique_labels:
        mask = labels == lab
        if mask.sum() < 5:
            continue
        class_feat = features_norm[mask]
        class_means.append(class_feat.mean(dim=0))
        class_stds.append(class_feat.std(dim=0))

    class_means = torch.stack(class_means)
    class_stds = torch.stack(class_stds)

    print(f"\nPer-class statistics ({len(class_means)} classes with >=5 samples):")
    print(f"  Mean of class means norm: {class_means.norm(dim=1).mean():.6f}")
    print(f"  Std of class means norm: {class_means.norm(dim=1).std():.6f}")
    print(f"  Mean of class stds: {class_stds.mean():.6f}")

    # Inter-class similarity (are classes distinct?)
    class_means_norm = torch.nn.functional.normalize(class_means, p=2, dim=1)
    inter_class_sim = torch.mm(class_means_norm, class_means_norm.T)
    mask = ~torch.eye(inter_class_sim.shape[0], dtype=torch.bool, device=inter_class_sim.device)
    off_diagonal = inter_class_sim[mask]

    print(f"\nInter-class cosine similarity:")
    print(f"  Mean: {off_diagonal.mean():.6f}")
    print(f"  Std: {off_diagonal.std():.6f}")
    print(f"  Max: {off_diagonal.max():.6f}")

    # Collapse criteria
    collapse_detected = False
    reasons = []

    if global_std.mean() < 0.1:
        collapse_detected = True
        reasons.append("Global feature std < 0.1 (features are too similar)")

    if off_diagonal.mean() > 0.9:
        collapse_detected = True
        reasons.append("Mean inter-class similarity > 0.9 (classes are not distinct)")

    if class_stds.mean() < 0.05:
        collapse_detected = True
        reasons.append("Mean within-class std < 0.05 (no intra-class diversity)")

    print(f"\nMode collapse detected: {collapse_detected}")
    if collapse_detected:
        print("  Reasons:")
        for reason in reasons:
            print(f"    - {reason}")

    return {
        "collapse_detected": collapse_detected,
        "reasons": reasons,
        "global_std_mean": global_std.mean().item(),
        "inter_class_sim_mean": off_diagonal.mean().item(),
        "class_std_mean": class_stds.mean().item(),
    }


def analyze_current_model(
    checkpoint_path: str,
    data_path: str
):
    """
    Analyze the current model state to diagnose mode collapse.
    """
    print(f"\n{'='*60}")
    print(f"Current Model Analysis")
    print(f"{'='*60}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model state
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Find backbone weights
    backbone_keys = [k for k in state_dict.keys() if "backbone" in k]
    print(f"Found {len(backbone_keys)} backbone parameters")

    # Analyze final layer weights (decoder output)
    decoder_keys = [k for k in backbone_keys if "dec" in k and "weight" in k]
    for key in decoder_keys:
        weight = state_dict[key]
        print(f"\n{key}:")
        print(f"  Shape: {weight.shape}")
        print(f"  Mean: {weight.mean():.6f}")
        print(f"  Std: {weight.std():.6f}")
        print(f"  Min: {weight.min():.6f}")
        print(f"  Max: {weight.max():.6f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose mode collapse and contrastive loss issues")
    parser.add_argument("--scene-path", type=str, required=True,
                        help="Path to scene directory with SVD files")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (optional)")
    parser.add_argument("--svd-rank", type=int, default=16,
                        help="SVD rank to analyze (default: 16)")
    parser.add_argument("--min-samples", type=int, default=20,
                        help="Minimum samples per class for contrastive loss (default: 20)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for contrastive loss (default: 0.2)")

    args = parser.parse_args()

    # Load SVD file
    svd_file = Path(args.scene_path) / f"lang_feat_grid_svd_r{args.svd_rank}.npz"
    if not svd_file.exists():
        print(f"Error: SVD file not found: {svd_file}")
        return

    print(f"Analyzing SVD file: {svd_file}")

    # Load segment labels if available
    segment_file = Path(args.scene_path) / "segment.npy"
    if segment_file.exists():
        segments = torch.from_numpy(np.load(segment_file)).long()
        print(f"Loaded segment labels: {segments.shape}")
    else:
        print("Warning: segment.npy not found, skipping class-based analysis")
        segments = None

    # Load compressed features
    data = np.load(svd_file)
    compressed = torch.from_numpy(data['compressed']).float()
    print(f"Loaded compressed features: {compressed.shape}")

    # Run analyses
    svd_stats = analyze_svd_compression_quality(
        str(svd_file), compressed, args.svd_rank
    )

    if segments is not None:
        # Match segments to grid features (this is approximate)
        # In practice, you'd need the point_to_grid mapping
        print("\nNote: Segment-to-grid mapping requires point_to_grid data.")
        print("Skipping class-based contrastive loss analysis.")

    # Analyze mode collapse on compressed features themselves
    print("\n" + "="*60)
    print("Analyzing SVD-Compressed Target Features for Mode Collapse")
    print("="*60)

    # Create dummy labels (each grid as a "class") to check feature diversity
    dummy_labels = torch.arange(compressed.shape[0])

    collapse_stats = detect_mode_collapse(compressed, dummy_labels)

    # Save results
    results = {
        "svd_rank": args.svd_rank,
        "svd_compression": svd_stats,
        "mode_collapse": collapse_stats,
    }

    output_file = Path(args.scene_path) / f"diagnosis_r{args.svd_rank}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Diagnosis complete. Results saved to: {output_file}")
    print(f"{'='*60}")

    # Print recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if collapse_stats["collapse_detected"]:
        print("\nMode collapse DETECTED! Recommended actions:")
        print("1. Increase SVD rank from 16 to 32 or 64")
        print("2. Reduce L2 loss weight to 0.01")
        print("3. Increase contrastive loss weight to 0.5")
        print("4. Change temperature from 0.2 to 0.5")
        print("5. Change sum aggregation to mean aggregation")
    else:
        print("\nNo severe mode collapse detected.")
        print("If contrastive loss still doesn't decrease, try:")
        print("1. Increase temperature to 0.5")
        print("2. Reduce min_samples threshold to 10")
        print("3. Increase contrastive loss weight")


if __name__ == "__main__":
    main()
