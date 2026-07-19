#!/usr/bin/env python3
"""
Instance segmentation evaluation for SceneSplat on Replica dataset.

This script evaluates 3D Gaussian Splatting predictions against Replica
ground truth semantic instance masks. It supports:

1. 2D evaluation: Project 3D predictions to 2D renders and compare with GT masks
2. 3D evaluation: Compare 3D point labels directly (if GT 3D labels available)

The evaluation computes:
- Mean IoU (mIoU)
- Precision@0.5 (percentage of predictions with IoU > 0.5)
- Recall@0.5 (percentage of GT instances with IoU > 0.5)

Usage:
    # 2D evaluation with rendered predictions
    python tools/eval_replica_instance.py \\
        --gt-masks /path/to/replica/office_0/semantic_instance \\
        --pred-masks /path/to/predicted/masks \\
        --dataset-type replica

    # 3D evaluation with point cloud labels
    python tools/eval_replica_instance.py \\
        --gt-labels /path/to/gt/segment.npy \\
        --pred-labels /path/to/pred/segment.npy \\
        --mode 3d

Reference:
    Based on /new_data/cyf/projects/Gaga/eval.py
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from typing import Optional, Tuple, List


def calculate_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """Calculate IoU between two binary masks.

    Args:
        mask1: Binary mask [H, W] or [N]
        mask2: Binary mask [H, W] or [N]

    Returns:
        IoU score (0-1)
    """
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)

    if torch.sum(union) == 0:
        return 0.0

    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score


def get_iou_for_label_pair(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_label_idx: int,
    pred_label_idx: int
) -> torch.Tensor:
    """Calculate IoU for a pair of predicted and GT labels across all images.

    Args:
        pred_masks: Predicted masks [B, H, W]
        gt_masks: Ground truth masks [B, H, W]
        gt_label_idx: GT label index
        pred_label_idx: Predicted label index

    Returns:
        Mean IoU across all images
    """
    assert pred_masks.shape == gt_masks.shape, \
        "Predicted and ground truth masks must have the same shape"

    all_image_iou = []
    for i in range(len(gt_masks)):
        gt_mask_binary = gt_masks[i] == gt_label_idx
        pred_mask_binary = pred_masks[i] == pred_label_idx

        if torch.sum(gt_mask_binary) == 0:
            continue

        iou = calculate_iou(pred_mask_binary, gt_mask_binary)
        all_image_iou.append(iou)

    if len(all_image_iou) == 0:
        return torch.tensor(0.0)

    return torch.tensor(all_image_iou).mean()


def get_linear_sum_assignment(iou_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve linear sum assignment using Hungarian algorithm.

    Args:
        iou_matrix: IoU matrix [num_gt, num_pred]

    Returns:
        row_ind, col_ind: Assignment indices
    """
    row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
    return row_ind, col_ind


def load_2d_masks(gt_dir: str, pred_dir: str, dataset_type: str = "replica") -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load 2D ground truth and predicted masks.

    Args:
        gt_dir: Ground truth mask directory
        pred_dir: Predicted mask directory
        dataset_type: Dataset type (replica, scannet, etc.)

    Returns:
        gt_masks: List of GT mask arrays
        pred_masks: List of predicted mask arrays
    """
    pred_mask_names = sorted(os.listdir(pred_dir))
    pred_masks = []
    gt_masks = []

    print(f"Found {len(pred_mask_names)} predicted masks")

    for mask_name in tqdm(pred_mask_names, desc="Loading masks"):
        # Load predicted mask
        pred_path = os.path.join(pred_dir, mask_name)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        if pred_mask is None:
            print(f"Warning: Could not load {pred_path}")
            continue
        pred_masks.append(pred_mask)

        # Load corresponding GT mask
        if dataset_type == "replica":
            # Replica: test_rgb_XXXX.png -> test_semantic_instance_XXXX.png
            if mask_name.startswith("test_rgb"):
                gt_mask_name = mask_name.replace("test_rgb", "test_semantic_instance")
            elif mask_name.startswith("train_rgb"):
                gt_mask_name = mask_name.replace("train_rgb", "train_semantic_instance")
            else:
                gt_mask_name = mask_name
        else:
            gt_mask_name = mask_name

        gt_path = os.path.join(gt_dir, gt_mask_name)
        if not os.path.exists(gt_path):
            print(f"Warning: GT mask not found: {gt_path}")
            # Create dummy mask
            gt_mask = np.zeros_like(pred_mask)
        else:
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        gt_masks.append(gt_mask)

    return gt_masks, pred_masks


def load_3d_labels(gt_path: str, pred_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load 3D point cloud labels.

    Args:
        gt_path: Path to GT label file (.npy)
        pred_path: Path to predicted label file (.npy)

    Returns:
        gt_labels: GT label array [N]
        pred_labels: Predicted label array [N]
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT labels not found: {gt_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predicted labels not found: {pred_path}")

    gt_labels = np.load(gt_path)
    pred_labels = np.load(pred_path)

    print(f"Loaded GT labels: {gt_path}, shape={gt_labels.shape}")
    print(f"Loaded pred labels: {pred_path}, shape={pred_labels.shape}")

    if gt_labels.shape != pred_labels.shape:
        raise ValueError(f"Shape mismatch: GT {gt_labels.shape} vs pred {pred_labels.shape}")

    return gt_labels, pred_labels


def evaluate_2d(
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    device: torch.device
) -> dict:
    """Evaluate 2D instance segmentation.

    Args:
        gt_masks: List of GT masks
        pred_masks: List of predicted masks
        device: Torch device

    Returns:
        Dictionary with metrics
    """
    # Convert to tensors
    gt_masks_tensor = torch.from_numpy(np.array(gt_masks, dtype=np.int64)).to(device)
    pred_masks_tensor = torch.from_numpy(np.array(pred_masks, dtype=np.int64)).to(device)

    num_gt_masks, h, w = gt_masks_tensor.shape
    num_pred_masks, h_pred, w_pred = pred_masks_tensor.shape

    assert num_gt_masks == num_pred_masks, \
        f"Number of GT masks ({num_gt_masks}) != number of pred masks ({num_pred_masks})"

    # Resize if needed
    if h != h_pred or w != w_pred:
        print(f"Resizing predictions from ({h_pred}, {w_pred}) to ({h}, {w})")
        pred_masks_tensor = torch.nn.functional.interpolate(
            pred_masks_tensor.unsqueeze(0).float(),
            size=(h, w),
            mode="nearest"
        ).long().squeeze(0)

    # Get unique labels
    gt_label_idx = torch.unique(gt_masks_tensor)
    pred_label_idx = torch.unique(pred_masks_tensor)

    num_gt_instances = len(gt_label_idx)
    num_pred_instances = len(pred_label_idx)

    print(f"Number of GT instances: {num_gt_instances}")
    print(f"Number of predicted instances: {num_pred_instances}")

    # Build IoU matrix
    iou_matrix = torch.zeros((num_gt_instances, max(num_gt_instances, num_pred_instances))).to(device)

    print("Computing IoU matrix...")
    for i in tqdm(range(num_gt_instances), desc="GT instances"):
        for j in range(num_pred_instances):
            iou_matrix[i, j] = get_iou_for_label_pair(
                pred_masks_tensor, gt_masks_tensor,
                gt_label_idx[i], pred_label_idx[j]
            )

    # Hungarian algorithm for optimal matching
    row_ind, col_ind = get_linear_sum_assignment(iou_matrix.cpu().numpy())

    # Compute metrics
    paired_iou = iou_matrix[row_ind, col_ind]
    mean_iou = paired_iou.mean().item()

    num_hit_05 = torch.sum(paired_iou > 0.5).item()
    precision_05 = num_hit_05 / num_pred_instances
    recall_05 = num_hit_05 / num_gt_instances

    results = {
        "mean_iou": mean_iou,
        "precision_05": precision_05,
        "recall_05": recall_05,
        "num_gt_instances": num_gt_instances,
        "num_pred_instances": num_pred_instances,
        "num_hit_05": num_hit_05,
    }

    return results


def evaluate_3d(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    device: torch.device
) -> dict:
    """Evaluate 3D point cloud instance segmentation.

    Args:
        gt_labels: GT labels [N]
        pred_labels: Predicted labels [N]
        device: Torch device

    Returns:
        Dictionary with metrics
    """
    # Convert to tensors
    gt_labels_tensor = torch.from_numpy(gt_labels).to(device)
    pred_labels_tensor = torch.from_numpy(pred_labels).to(device)

    # Get unique labels (exclude background/invalid)
    gt_label_idx = torch.unique(gt_labels_tensor)
    pred_label_idx = torch.unique(pred_labels_tensor)

    # Filter out invalid labels (assuming -1 or 0 is background)
    gt_label_idx = gt_label_idx[gt_label_idx >= 0]
    pred_label_idx = pred_label_idx[pred_label_idx >= 0]

    num_gt_instances = len(gt_label_idx)
    num_pred_instances = len(pred_label_idx)

    print(f"Number of GT instances: {num_gt_instances}")
    print(f"Number of predicted instances: {num_pred_instances}")

    # Build IoU matrix for 3D points
    iou_matrix = torch.zeros((num_gt_instances, max(num_gt_instances, num_pred_instances))).to(device)

    print("Computing IoU matrix...")
    for i, gt_label in enumerate(tqdm(gt_label_idx, desc="GT instances")):
        gt_mask = gt_labels_tensor == gt_label
        gt_count = gt_mask.sum().item()

        if gt_count == 0:
            continue

        for j, pred_label in enumerate(pred_label_idx):
            pred_mask = pred_labels_tensor == pred_label
            pred_count = pred_mask.sum().item()

            if pred_count == 0:
                continue

            # Compute IoU
            intersection = torch.logical_and(gt_mask, pred_mask).sum().item()
            union = torch.logical_or(gt_mask, pred_mask).sum().item()

            if union > 0:
                iou_matrix[i, j] = intersection / union

    # Hungarian algorithm
    row_ind, col_ind = get_linear_sum_assignment(iou_matrix.cpu().numpy())

    # Compute metrics
    paired_iou = iou_matrix[row_ind, col_ind]
    mean_iou = paired_iou.mean().item()

    num_hit_05 = torch.sum(paired_iou > 0.5).item()
    precision_05 = num_hit_05 / num_pred_instances if num_pred_instances > 0 else 0
    recall_05 = num_hit_05 / num_gt_instances if num_gt_instances > 0 else 0

    results = {
        "mean_iou": mean_iou,
        "precision_05": precision_05,
        "recall_05": recall_05,
        "num_gt_instances": num_gt_instances,
        "num_pred_instances": num_pred_instances,
        "num_hit_05": num_hit_05,
    }

    return results


def print_results(results: dict, mode: str = "2d"):
    """Print evaluation results.

    Args:
        results: Dictionary with metrics
        mode: Evaluation mode ("2d" or "3d")
    """
    print("\n" + "=" * 70)
    print(f"Instance Segmentation Evaluation Results ({mode.upper()})")
    print("=" * 70)

    print(f"\nMean IoU: {results['mean_iou']:.4f}")
    print(f"Precision (IoU > 0.5): {results['precision_05']:.4f}")
    print(f"Recall (IoU > 0.5): {results['recall_05']:.4f}")

    print(f"\nGT instances: {results['num_gt_instances']}")
    print(f"Predicted instances: {results['num_pred_instances']}")
    print(f"Detections@0.5: {results['num_hit_05']}")

    # Calculate F1 score
    precision = results['precision_05']
    recall = results['recall_05']
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f"F1 Score (IoU > 0.5): {f1:.4f}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Instance segmentation evaluation for SceneSplat on Replica dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2D evaluation with Replica dataset
  python tools/eval_replica_instance.py \\
      --mode 2d \\
      --gt-masks /new_data/cyf/projects/Gaga/datasets/replica/office_0/semantic_instance \\
      --pred-masks /path/to/predicted/masks \\
      --dataset-type replica

  # 3D evaluation with point cloud labels
  python tools/eval_replica_instance.py \\
      --mode 3d \\
      --gt-labels /path/to/gt_segment.npy \\
      --pred-labels /path/to/pred_segment.npy
        """
    )

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["2d", "3d"], default="2d",
                        help="Evaluation mode: 2d (rendered masks) or 3d (point cloud)")

    # 2D mode arguments
    parser.add_argument("--gt-masks", type=str, default=None,
                        help="Path to ground truth mask directory (2D mode)")
    parser.add_argument("--pred-masks", type=str, default=None,
                        help="Path to predicted mask directory (2D mode)")
    parser.add_argument("--dataset-type", type=str, default="replica",
                        choices=["replica", "scannet", "custom"],
                        help="Dataset type for naming conventions (2D mode)")

    # 3D mode arguments
    parser.add_argument("--gt-labels", type=str, default=None,
                        help="Path to ground truth label file (.npy, 3D mode)")
    parser.add_argument("--pred-labels", type=str, default=None,
                        help="Path to predicted label file (.npy, 3D mode)")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output file to save results (JSON format)")

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "2d":
        if args.gt_masks is None or args.pred_masks is None:
            parser.error("--gt-masks and --pred-masks are required for 2D mode")
        if not os.path.exists(args.gt_masks):
            parser.error(f"GT masks directory not found: {args.gt_masks}")
        if not os.path.exists(args.pred_masks):
            parser.error(f"Predicted masks directory not found: {args.pred_masks}")
    else:  # 3d mode
        if args.gt_labels is None or args.pred_labels is None:
            parser.error("--gt-labels and --pred-labels are required for 3D mode")
        if not os.path.exists(args.gt_labels):
            parser.error(f"GT labels file not found: {args.gt_labels}")
        if not os.path.exists(args.pred_labels):
            parser.error(f"Predicted labels file not found: {args.pred_labels}")

    # Set device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU")
        device = torch.device("cpu")

    # Run evaluation
    if args.mode == "2d":
        print(f"\nRunning 2D evaluation...")
        print(f"  GT masks: {args.gt_masks}")
        print(f"  Pred masks: {args.pred_masks}")
        print(f"  Dataset type: {args.dataset_type}")

        gt_masks, pred_masks = load_2d_masks(args.gt_masks, args.pred_masks, args.dataset_type)
        results = evaluate_2d(gt_masks, pred_masks, device)
    else:
        print(f"\nRunning 3D evaluation...")
        print(f"  GT labels: {args.gt_labels}")
        print(f"  Pred labels: {args.pred_labels}")

        gt_labels, pred_labels = load_3d_labels(args.gt_labels, args.pred_labels)
        results = evaluate_3d(gt_labels, pred_labels, device)

    # Print results
    print_results(results, args.mode)

    # Save results
    if args.output:
        import json
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
