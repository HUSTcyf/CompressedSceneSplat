#!/usr/bin/env python3
#
# Replica dataset instance segmentation evaluation
#
# Direct port of /new_data/cyf/projects/Gaga/eval.py with adaptations for SceneSplat
#
# Usage:
#   python tools/eval_replica.py \\
#       --gt_masks /new_data/cyf/projects/Gaga/datasets/replica/office_0/semantic_instance \\
#       --pred_masks /path/to/predicted/masks
#

import os
import numpy as np
from argparse import ArgumentParser
import cv2
import torch
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


def calculate_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """Helper function to calculate IoU between two masks."""
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    assert torch.sum(union) > 0, "The union of the two masks must be non-zero"
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score


def get_iou_for_label_pair(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_label_idx: int,
    pred_label_idx: int
) -> torch.Tensor:
    """Calculate the IoU score for a pair of predicted and ground truth labels."""
    assert pred_masks.shape == gt_masks.shape, \
        "Predicted and ground truth masks must have the same shape"

    all_image_iou = []
    for i in range(len(gt_masks)):
        gt_masks_binary = gt_masks[i] == gt_label_idx
        pred_masks_binary = pred_masks[i] == pred_label_idx
        if torch.sum(gt_masks_binary) == 0:
            continue
        iou = calculate_iou(pred_masks_binary, gt_masks_binary)
        all_image_iou.append(iou)

    return torch.tensor(all_image_iou).mean()


def get_linear_sum_assignment(iou_matrix: np.ndarray) -> np.ndarray:
    """Solve the linear sum assignment problem using the Hungarian algorithm."""
    row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
    return row_ind, col_ind


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


if __name__ == "__main__":
    # Parse arguments
    args = ArgumentParser()
    args.add_argument("--gt_masks", type=str, required=True,
                      help="Path to the ground truth masks")
    args.add_argument("--pred_masks", type=str, required=True,
                      help="Path to the predicted masks")
    args.add_argument("--dataset", type=str, default="replica",
                      choices=["replica", "scannet", "other"],
                      help="Dataset type (affects naming conventions)")
    args = args.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load predicted masks
    pred_mask_names = sorted(os.listdir(args.pred_masks))
    pred_masks = []
    gt_masks = []

    print(f"Loading {len(pred_mask_names)} predicted masks from: {args.pred_masks}")

    for mask_name in tqdm(pred_mask_names, desc="Loading masks"):
        # Load predicted mask
        pred_path = os.path.join(args.pred_masks, mask_name)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        if pred_mask is None:
            print(f"Warning: Could not load {pred_path}")
            continue
        pred_masks.append(pred_mask)

        # Load corresponding GT mask based on dataset type
        if args.dataset == "replica":
            # Replica naming: test_rgb_XXXX.png -> test_semantic_instance_XXXX.png
            assert mask_name.startswith("test_rgb"), \
                "For Replica dataset, image names must start with 'test_rgb'"
            gt_mask_name = mask_name.replace("test_rgb", "test_semantic_instance")
        else:
            gt_mask_name = mask_name

        gt_path = os.path.join(args.gt_masks, gt_mask_name)
        if not os.path.exists(gt_path):
            print(f"Warning: GT mask not found: {gt_path}, using zeros")
            gt_mask = np.zeros_like(pred_mask)
        else:
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        gt_masks.append(gt_mask)

    # Convert to tensors
    pred_masks = np.array(pred_masks, dtype=np.int64)
    gt_masks = np.array(gt_masks, dtype=np.int64)
    pred_masks = torch.tensor(pred_masks).to(device)
    gt_masks = torch.tensor(gt_masks).to(device)

    num_gt_mask, h, w = gt_masks.shape
    num_pred_mask, h_pred, w_pred = pred_masks.shape

    assert num_gt_mask == num_pred_mask, \
        "The number of ground truth masks must be equal to the number of predicted masks"

    # Resize predictions if needed
    if h != h_pred or w != w_pred:
        print(f"Resizing predictions from ({h_pred}, {w_pred}) to ({h}, {w})")
        pred_masks = torch.nn.functional.interpolate(
            pred_masks.unsqueeze(0).float(),
            size=(h, w),
            mode="nearest"
        ).long().squeeze(0)

    # Get unique labels
    gt_label_idx = torch.unique(gt_masks)
    num_gt_mask = len(gt_label_idx)

    pred_label_idx = torch.unique(pred_masks)
    num_pred_mask = len(pred_label_idx)

    print(f"\nNumber of ground truth instances: {num_gt_mask}")
    print(f"Number of predicted instances: {num_pred_mask}")

    # Build IoU matrix
    iou_matrix = torch.zeros((num_gt_mask, max(num_gt_mask, num_pred_mask))).to(device)

    print("\nBuilding IoU matrix...")
    for i in tqdm(range(num_gt_mask), desc="GT instances"):
        for j in range(num_pred_mask):
            iou_matrix[i, j] = get_iou_for_label_pair(
                pred_masks, gt_masks, gt_label_idx[i], pred_label_idx[j]
            )

    # Solve the linear sum assignment problem
    row_ind, col_ind = get_linear_sum_assignment(iou_matrix.cpu().numpy())

    # Get mean IoU, precision, and recall
    paired_iou = iou_matrix[row_ind, col_ind]
    mean_iou = paired_iou.mean()

    num_hit_05 = torch.sum(paired_iou > 0.5)
    precision_05 = num_hit_05 / num_pred_mask
    recall_05 = num_hit_05 / num_gt_mask

    # Calculate F1 score
    f1_05 = calculate_f1(precision_05, recall_05)

    # Print results
    print("\n" + "=" * 70)
    print("Instance Segmentation Evaluation Results")
    print("=" * 70)
    print(f"\nMean IoU: {mean_iou:.4f}")
    print(f"Precision (IoU > 0.5): {precision_05:.4f}")
    print(f"Recall (IoU > 0.5): {recall_05:.4f}")
    print(f"F1 Score (IoU > 0.5): {f1_05:.4f}")
    print(f"\nGT instances: {num_gt_mask}")
    print(f"Predicted instances: {num_pred_mask}")
    print(f"Detections@0.5: {num_hit_05}")
    print("=" * 70 + "\n")
