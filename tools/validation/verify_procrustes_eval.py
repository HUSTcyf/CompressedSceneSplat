#!/usr/bin/env python3
"""
Verify Procrustes alignment and evaluation logic without model inference.

This script loads pre-computed SVD features directly and tests the evaluation
pipeline to ensure the Procrustes alignment is working correctly.

Usage:
    python tools/verify_procrustes_eval.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Import PROJECT_ROOT - handle both script and module execution
try:
    from .. import PROJECT_ROOT  # Relative import when run as module
except ImportError:
    # Fallback when run as script: add parent dir to sys.path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from compute_procrustes_alignment_simple import compute_procrustes_Q_cuda_with_labels


def load_svd_features(svd_file):
    """Load grid-level SVD compressed features and reconstruct to point-level."""
    svd_data = np.load(svd_file)
    compressed = svd_data['compressed'].astype(np.float32)  # [M, rank]
    indices = svd_data['indices']  # [N]
    point_lang_feat = compressed[indices].astype(np.float32)
    return point_lang_feat, indices, compressed


def main():
    print("="*70)
    print("Procrustes Evaluation Verification")
    print("="*70)

    # Paths
    scene_path = Path("/new_data/cyf/Datasets/SceneSplat7k/matterport3d/val_grid1.0cm_chunk6x6x4_stride4x4x4/2azQ1b91cZZ_0")
    repo_root = Path("/new_data/cyf/projects/SceneSplat")

    # Check scene exists
    if not scene_path.exists():
        print(f"Error: Scene path not found: {scene_path}")
        return

    print(f"\nScene: {scene_path.name}")
    print(f"Path: {scene_path}")

    # Load SVD features
    svd_file = scene_path / "lang_feat_grid_svd_r16.npz"
    if not svd_file.exists():
        print(f"Error: SVD file not found: {svd_file}")
        return

    print(f"\nLoading SVD features from: {svd_file}")
    point_feat, indices, compressed = load_svd_features(svd_file)
    print(f"  point_feat shape: {point_feat.shape}")
    print(f"  unique grid cells: {compressed.shape[0]}")
    print(f"  total points: {point_feat.shape[0]}")

    # Load GT labels (try different naming conventions)
    segment_file = None
    for name in ["segment.npy", "segment_nyu_160.npy"]:
        potential_file = scene_path / name
        if potential_file.exists():
            segment_file = potential_file
            break

    if segment_file is None:
        print(f"Error: Segment file not found (tried segment.npy, segment_nyu_160.npy)")
        return

    print(f"\nLoading GT labels from: {segment_file}")
    segment = np.load(segment_file)
    print(f"  segment shape: {segment.shape}")

    # Load valid mask
    valid_mask_file = scene_path / "valid_feat_mask.npy"
    if valid_mask_file.exists():
        valid_mask = np.load(valid_mask_file)
        print(f"  valid_feat_mask shape: {valid_mask.shape}")
        print(f"  valid points: {valid_mask.sum()}/{len(valid_mask)}")

        # SVD features are already filtered (point_feat has 144192 points)
        # So we need to filter segment to match
        # Convert to boolean mask if needed (in case it's stored as int)
        if valid_mask.dtype != bool:
            valid_mask = valid_mask.astype(bool)
        segment = segment[valid_mask]
        print(f"  After filtering segment: {segment.shape[0]} points")

        # Verify shapes match
        if segment.shape[0] != point_feat.shape[0]:
            print(f"  ERROR: Shape mismatch! point_feat={point_feat.shape[0]}, segment={segment.shape[0]}")
            return

    # Load text embeddings
    text_embed_path = repo_root / "pointcept/datasets/preprocessing/matterport3d/meta_data/matterport-nyu160_text_embeddings_siglip2.pt"
    if not text_embed_path.exists():
        print(f"Error: Text embeddings not found: {text_embed_path}")
        return

    print(f"\nLoading text embeddings from: {text_embed_path}")
    text_embeddings = torch.load(text_embed_path, weights_only=False)
    print(f"  text_embeddings shape: {text_embeddings.shape}")

    # Get number of classes
    num_classes = text_embeddings.shape[0]
    feat_dim = text_embeddings.shape[1]
    svd_rank = point_feat.shape[1]
    print(f"  num_classes: {num_classes}")
    print(f"  text embedding dim: {feat_dim}")
    print(f"  SVD compressed dim: {svd_rank}")

    # Apply SVD reduction to text embeddings (same as evaluator does)
    if feat_dim != svd_rank:
        print(f"\nApplying SVD reduction to text embeddings: [{num_classes}, {feat_dim}] -> [{num_classes}, {svd_rank}]")
        text_emb_np = text_embeddings.numpy() if isinstance(text_embeddings, torch.Tensor) else text_embeddings

        # Normalize features
        norms = np.linalg.norm(text_emb_np, axis=1, keepdims=True)
        text_emb_norm = text_emb_np / (norms + 1e-8)

        # Compute SVD and take top rank components
        _, _, Vt = np.linalg.svd(text_emb_norm, full_matrices=False)
        components = Vt[:svd_rank, :].T  # [D, rank]
        text_emb_reduced = text_emb_np @ components  # [N, rank]

        text_embeddings = F.normalize(torch.from_numpy(text_emb_reduced), p=2, dim=1)
        print(f"  Reduced text_embeddings shape: {text_embeddings.shape}")
        print(f"  SVD components shape: {components.shape}")
    else:
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    # Convert to CUDA tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    point_feat_tensor = torch.from_numpy(point_feat).to(device)
    segment_tensor = torch.from_numpy(segment).long().to(device)
    text_embeddings_tensor = text_embeddings.to(device)

    # Filter valid labels for Procrustes
    valid_label_mask = (segment_tensor >= 0) & (segment_tensor < num_classes)
    print(f"\nValid labels for Procrustes: {valid_label_mask.sum()}/{len(segment_tensor)}")

    if valid_label_mask.sum() == 0:
        print("Error: No valid labels for Procrustes computation!")
        return

    point_feat_valid = point_feat_tensor[valid_label_mask]
    segment_valid = segment_tensor[valid_label_mask]

    # === STEP 1: Compute Procrustes Q matrix ===
    print(f"\n{'='*70}")
    print("STEP 1: Computing Procrustes Q Matrix")
    print(f"{'='*70}")
    print(f"  point_feat shape: {point_feat_valid.shape}")
    print(f"  segment shape: {segment_valid.shape}")
    print(f"  text_embeddings shape: {text_embeddings_tensor.shape}")

    Q, metrics = compute_procrustes_Q_cuda_with_labels(
        point_feat_valid,
        text_embeddings_tensor,
        segment_valid
    )

    print(f"\nProcrustes Results:")
    print(f"  Q shape: {Q.shape}")
    print(f"  det(Q): {metrics['det_Q']:.6f}")
    print(f"  Cosine similarity (before): {metrics['cosine_before']:.4f}")
    print(f"  Cosine similarity (after): {metrics['cosine_after']:.4f}")
    print(f"  Improvement: {metrics['cosine_improvement']:+.4f}")
    print(f"  Orthogonality error: {metrics['orthogonality_error']:.8e}")

    # === STEP 2: Apply Procrustes to text embeddings ===
    print(f"\n{'='*70}")
    print("STEP 2: Applying Procrustes Alignment")
    print(f"{'='*70}")

    text_embeddings_aligned = torch.mm(text_embeddings_tensor, Q.T)
    print(f"  text_embeddings_aligned shape: {text_embeddings_aligned.shape}")

    # === STEP 3: Compute predictions ===
    print(f"\n{'='*70}")
    print("STEP 3: Computing Predictions")
    print(f"{'='*70}")

    # Normalize point features
    point_feat_norm = F.normalize(point_feat_tensor, p=2, dim=1)
    print(f"  point_feat_norm shape: {point_feat_norm.shape}")

    # Compute cosine similarity
    similarity = torch.mm(point_feat_norm, text_embeddings_aligned.t())
    print(f"  similarity shape: {similarity.shape}")
    print(f"  similarity range: [{similarity.min():.4f}, {similarity.max():.4f}]")

    # Get predictions
    max_probs, pred_labels = torch.max(similarity, dim=1)
    pred_labels = pred_labels.cpu().numpy()
    max_probs = max_probs.cpu().numpy()

    print(f"  pred_labels shape: {pred_labels.shape}")
    print(f"  max_probs range: [{max_probs.min():.4f}, {max_probs.max():.4f}]")

    # Apply confidence threshold (shifted from [-1,1] to [0,1])
    confidence_threshold = 0.1
    max_probs_shifted = (max_probs + 1) / 2
    below_threshold = (max_probs_shifted < confidence_threshold).sum()
    pred_labels[max_probs_shifted < confidence_threshold] = -1  # ignore_index

    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Predictions below threshold: {below_threshold}/{len(pred_labels)}")

    # === STEP 4: Compute IoU metrics ===
    print(f"\n{'='*70}")
    print("STEP 4: Computing IoU Metrics")
    print(f"{'='*70}")

    # Filter to valid GT labels
    valid_mask = (segment >= 0) & (segment < num_classes)
    valid_pred = pred_labels[valid_mask]
    valid_gt = segment[valid_mask]

    print(f"  Valid samples: {len(valid_gt)}")

    # Compute confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    fn_ignore = np.zeros(num_classes, dtype=np.int64)

    for gt, pred in zip(valid_gt, valid_pred):
        if pred == -1:
            fn_ignore[gt] += 1
        else:
            confusion[gt, pred] += 1

    # Compute per-class IoU
    ious = []
    present_classes = []
    missing_classes = []

    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp + fn_ignore[c]
        denom = tp + fp + fn

        if confusion[c, :].sum() + fn_ignore[c] > 0:
            present_classes.append(c)
            iou = tp / denom if denom > 0 else 0.0
            ious.append(iou)
        else:
            missing_classes.append(c)

    # Compute metrics
    mIoU = np.mean(ious) if ious else 0.0
    global_acc = np.diag(confusion).sum() / confusion.sum() if confusion.sum() > 0 else 0.0

    # Mean class accuracy
    class_accs = []
    for c in present_classes:
        tp = confusion[c, c]
        fn = confusion[c, :].sum() - tp + fn_ignore[c]
        denom = tp + fn
        if denom > 0:
            class_accs.append(tp / denom)

    mean_class_acc = np.mean(class_accs) if class_accs else 0.0

    print(f"\nResults:")
    print(f"  Present classes: {len(present_classes)}/{num_classes}")
    print(f"  Missing classes: {len(missing_classes)}")
    print(f"  Global Accuracy: {global_acc:.4f}")
    print(f"  Mean Class Accuracy: {mean_class_acc:.4f}")
    print(f"  Mean IoU (mIoU): {mIoU:.4f}")

    # Show top 10 classes by IoU
    print(f"\nTop 10 classes by IoU:")
    class_ious = [(c, ious[i]) for i, c in enumerate(present_classes)]
    class_ious.sort(key=lambda x: x[1], reverse=True)

    for i, (cls, iou) in enumerate(class_ious[:10]):
        print(f"  Class {cls:3d}: IoU = {iou:.4f}")

    # Show bottom 10 classes by IoU
    print(f"\nBottom 10 classes by IoU:")
    for i, (cls, iou) in enumerate(class_ious[-10:]):
        print(f"  Class {cls:3d}: IoU = {iou:.4f}")

    # Check if results are reasonable
    print(f"\n{'='*70}")
    print("Validation Summary")
    print(f"{'='*70}")

    if mIoU > 0.01:
        print("✓ PASS: mIoU > 0.01 (evaluation is working)")
    else:
        print("✗ FAIL: mIoU <= 0.01 (evaluation may have issues)")

    if global_acc > 0.01:
        print("✓ PASS: Global Acc > 0.01")
    else:
        print("✗ FAIL: Global Acc <= 0.01")

    if metrics['cosine_improvement'] > 0:
        print(f"✓ PASS: Procrustes improved cosine similarity by {metrics['cosine_improvement']:.4f}")
    else:
        print(f"✗ WARNING: Procrustes did not improve cosine similarity")

    print(f"\n{'='*70}")
    print("Verification Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
