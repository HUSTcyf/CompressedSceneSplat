#!/usr/bin/env python3
"""
Compare language features between checkpoint_with_features_p.pth and checkpoint_with_features.pth

Computes L1 loss and cosine similarity loss for common non-zero rows.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_checkpoint_language_features(checkpoint_path: str) -> torch.Tensor:
    """
    Load language features from checkpoint file.

    Checkpoint format: ((13-element tuple), iteration)
    Language features are at index 7 (8th element)
    """
    print(f"Loading: {checkpoint_path}")
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle tuple format: ((13-element tuple), iteration)
    if isinstance(ckpt_data, tuple) and len(ckpt_data) == 2:
        checkpoint_tuple, iteration = ckpt_data
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt_data)}")

    # Language features are at index 7
    language_features = checkpoint_tuple[7]  # [N, feat_dim]
    print(f"  Language features shape: {language_features.shape}, dtype: {language_features.dtype}")

    return language_features


def find_common_nonzero_rows(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    threshold: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find rows that are non-zero in both feature matrices.

    Args:
        feat_a: [N_a, D] feature matrix
        feat_b: [N_b, D] feature matrix
        threshold: Threshold for considering a value as non-zero

    Returns:
        (nonzero_a, nonzero_b): Filtered feature matrices with only common non-zero rows
    """
    # Check if both have same number of rows
    if feat_a.shape[0] != feat_b.shape[0]:
        print(f"Warning: Different row counts - feat_a: {feat_a.shape[0]}, feat_b: {feat_b.shape[0]}")
        min_rows = min(feat_a.shape[0], feat_b.shape[0])
        feat_a = feat_a[:min_rows]
        feat_b = feat_b[:min_rows]

    # Find non-zero rows in each matrix
    # A row is non-zero if its L2 norm is above threshold
    norm_a = torch.norm(feat_a, p=2, dim=1)  # [N]
    norm_b = torch.norm(feat_b, p=2, dim=1)  # [N]

    nonzero_mask_a = norm_a > threshold
    nonzero_mask_b = norm_b > threshold

    # Common non-zero rows
    common_mask = nonzero_mask_a & nonzero_mask_b

    print(f"  Non-zero rows in feat_a: {nonzero_mask_a.sum()}/{len(nonzero_mask_a)}")
    print(f"  Non-zero rows in feat_b: {nonzero_mask_b.sum()}/{len(nonzero_mask_b)}")
    print(f"  Common non-zero rows: {common_mask.sum()}/{len(common_mask)}")

    return feat_a[common_mask], feat_b[common_mask]


def compute_l1_loss(feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
    """Compute L1 loss (mean absolute error) between two feature matrices."""
    return torch.abs(feat_a - feat_b).mean().item()


def compute_cosine_similarity_loss(feat_a: torch.Tensor, feat_b: torch.Tensor) -> Tuple[float, float]:
    """
    Compute cosine similarity and cosine similarity loss (1 - cosine similarity).

    Returns:
        (cosine_similarity, cosine_loss): Mean cosine similarity and its corresponding loss
    """
    # Compute cosine similarity for each row
    # cos_sim = (a · b) / (||a|| * ||b||)
    dot_product = (feat_a * feat_b).sum(dim=1)  # [N]
    norm_a = torch.norm(feat_a, p=2, dim=1)  # [N]
    norm_b = torch.norm(feat_b, p=2, dim=1)  # [N]

    # Avoid division by zero
    cosine_sim = dot_product / (norm_a * norm_b + 1e-8)

    # Mean cosine similarity
    mean_cosine_sim = cosine_sim.mean().item()
    cosine_loss = 1.0 - mean_cosine_sim

    return mean_cosine_sim, cosine_loss


def compare_scene_checkpoints(
    checkpoint_p_path: str,
    checkpoint_path: str,
    scene_name: str
) -> Dict:
    """Compare two checkpoint files for a scene."""
    print(f"\n{'='*60}")
    print(f"Scene: {scene_name}")
    print(f"{'='*60}")

    # Load language features
    feat_p = load_checkpoint_language_features(checkpoint_p_path)
    feat = load_checkpoint_language_features(checkpoint_path)

    # Find common non-zero rows
    feat_p_common, feat_common = find_common_nonzero_rows(feat_p, feat)

    if feat_p_common.shape[0] == 0:
        print("  No common non-zero rows found!")
        return {
            "scene": scene_name,
            "n_common_rows": 0,
            "l1_loss": None,
            "cosine_similarity": None,
            "cosine_loss": None,
        }

    # Compute losses
    l1_loss = compute_l1_loss(feat_p_common, feat_common)
    cosine_sim, cosine_loss = compute_cosine_similarity_loss(feat_p_common, feat_common)

    print(f"\nResults:")
    print(f"  Common non-zero rows: {feat_p_common.shape[0]}")
    print(f"  Feature dimension: {feat_p_common.shape[1]}")
    print(f"  L1 Loss: {l1_loss:.6f}")
    print(f"  Cosine Similarity: {cosine_sim:.6f}")
    print(f"  Cosine Loss (1 - cos): {cosine_loss:.6f}")

    return {
        "scene": scene_name,
        "n_common_rows": feat_p_common.shape[0],
        "feature_dim": feat_p_common.shape[1],
        "l1_loss": l1_loss,
        "cosine_similarity": cosine_sim,
        "cosine_loss": cosine_loss,
    }


def main():
    output_dir = Path("/new_data/cyf/projects/SceneSplat/output_features")

    # Find all scenes with both checkpoint types
    scenes = []
    for scene_dir in output_dir.iterdir():
        if not scene_dir.is_dir():
            continue

        checkpoint_p = scene_dir / "checkpoint_with_features_p.pth"
        checkpoint = scene_dir / "checkpoint_with_features.pth"

        if checkpoint_p.exists() and checkpoint.exists():
            scenes.append((scene_dir.name, checkpoint_p, checkpoint))

    print(f"Found {len(scenes)} scenes with both checkpoint types")

    if not scenes:
        print("No matching scenes found!")
        return

    results = []
    for scene_name, checkpoint_p, checkpoint in scenes:
        try:
            result = compare_scene_checkpoints(
                str(checkpoint_p),
                str(checkpoint),
                scene_name
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Scene':<20} {'Rows':<10} {'L1 Loss':<12} {'Cosine Sim':<12} {'Cosine Loss':<12}")
    print("-" * 66)

    for r in results:
        if r["n_common_rows"] > 0:
            print(f"{r['scene']:<20} {r['n_common_rows']:<10} "
                  f"{r['l1_loss']:<12.6f} {r['cosine_similarity']:<12.6f} {r['cosine_loss']:<12.6f}")
        else:
            print(f"{r['scene']:<20} {r['n_common_rows']:<10} "
                  f"{'N/A':<12} {'N/A':<12} {'N/A':<12}")

    # Calculate averages
    valid_results = [r for r in results if r["n_common_rows"] > 0]
    if valid_results:
        avg_l1 = np.mean([r["l1_loss"] for r in valid_results])
        avg_cosine_sim = np.mean([r["cosine_similarity"] for r in valid_results])
        avg_cosine_loss = np.mean([r["cosine_loss"] for r in valid_results])

        print("-" * 66)
        print(f"{'Average':<20} {'':<10} {avg_l1:<12.6f} {avg_cosine_sim:<12.6f} {avg_cosine_loss:<12.6f}")


if __name__ == "__main__":
    main()
