#!/usr/bin/env python3
"""
Test validation on ScanNet test data using SVD-decomposed text embeddings with Procrustes alignment.

This script tests the validation flow with:
- ScanNet test data at /new_data/cyf/Datasets/SceneSplat7k/scannet/test_grid1.0cm_chunk6x6_stride3x3
- 768-dim text embeddings SVD-reduced to 16-dim
- Procrustes alignment matrix Q for feature alignment
- SVD-16 model outputs

Approach:
1. Load 768-dim text embeddings
2. Perform SVD decomposition to reduce to 16-dim
3. For each scene, compute Procrustes Q matrix to align predicted features with text embeddings
4. Apply Q alignment before computing similarity metrics

Usage:
    python tools/test_validation_with_compressed_embeddings.py \
        --config configs/custom/lang-pretrain-litept-scannet.py \
        --weight exp/lite-16-gridsvd/model_best.pth \
        --text_embeddings pointcept/datasets/preprocessing/scannet/meta_data/scannet200_text_embeddings_siglip2.pt \
        --svd_rank 16
"""

import os
import sys
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.logger import get_root_logger

# Import SVD and Procrustes functions from compute_procrustes_alignment_simple.py
from tools.compute_procrustes_alignment_simple import (
    perform_svd_reduction,
    compute_procrustes_Q,
)


def svd_reduce_text_embeddings(
    text_embeddings: np.ndarray,
    svd_rank: int,
    sample_size: int = None,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    SVD-reduce text embeddings and optionally sample.

    Args:
        text_embeddings: [C, D] original text embeddings
        svd_rank: Target SVD rank
        sample_size: Number of text embeddings to sample for Q computation
        normalize: Whether to normalize features

    Returns:
        text_embeddings_reduced: [C, svd_rank] SVD-reduced embeddings
        text_embeddings_sampled: [sample_size, svd_rank] sampled embeddings for Q computation
        metadata: Dictionary with metadata
    """
    C, D = text_embeddings.shape

    # SVD reduction
    print(f"  SVD reduction: [{C}, {D}] -> [{C}, {svd_rank}]")
    text_embeddings_reduced, _, _ = perform_svd_reduction(
        text_embeddings, svd_rank, normalize
    )

    # Sample for Q computation (if specified)
    if sample_size is not None and sample_size < text_embeddings_reduced.shape[0]:
        indices = np.random.choice(text_embeddings_reduced.shape[0], sample_size, replace=False)
        text_embeddings_sampled = text_embeddings_reduced[indices]
    else:
        text_embeddings_sampled = text_embeddings_reduced

    metadata = {
        'original_shape': (C, D),
        'reduced_shape': text_embeddings_reduced.shape,
        'svd_rank': svd_rank,
    }

    return text_embeddings_reduced, text_embeddings_sampled, metadata


def validate_with_compressed_embeddings(
    config_path: str,
    weight_path: str,
    text_embeddings_path: str,
    data_root: str = "/new_data/cyf/Datasets/SceneSplat7k/scannet",
    split: str = "test_grid1.0cm_chunk6x6_stride3x3",
    device: str = "cuda",
    num_scenes: int = 10,
    svd_rank: int = 16,
    use_procrustes: bool = True,
    procrustes_sample_size: int = None,
):
    """
    Validate model with SVD-reduced text embeddings and Procrustes alignment.

    New approach:
    1. Load 768-dim text embeddings
    2. Perform SVD reduction to svd_rank dimensions
    3. For each scene, compute Procrustes Q matrix to align features
    4. Apply Q alignment before computing similarity metrics

    Args:
        config_path: Path to config file
        weight_path: Path to model checkpoint
        text_embeddings_path: Path to 768-dim text embeddings (will be SVD-reduced)
        data_root: Path to test data root
        split: Test split name
        device: Device to use
        num_scenes: Number of scenes to validate (None for all)
        svd_rank: SVD rank for text embedding reduction
        use_procrustes: Whether to use Procrustes alignment
        procrustes_sample_size: Sample size for Procrustes Q computation
    """
    print("=" * 70)
    print("Validation with SVD-Reduced Text Embeddings + Procrustes Alignment")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Weight: {weight_path}")
    print(f"Text embeddings: {text_embeddings_path}")
    print(f"Data: {data_root}/{split}")
    print(f"SVD rank: {svd_rank}")
    print(f"Use Procrustes: {use_procrustes}")

    # Load config
    cfg = Config.fromfile(config_path)

    # Override test data path
    cfg.data.test.data_root = data_root
    cfg.data.test.split = split

    # Build model
    print("\nBuilding model...")
    model = build_model(cfg.model)
    model = model.to(device)
    model.eval()

    # Load checkpoint
    if weight_path and os.path.exists(weight_path):
        print(f"Loading checkpoint from {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded")

    # Load 768-dim text embeddings and perform SVD reduction
    print(f"\nLoading text embeddings from {text_embeddings_path}")
    text_embeddings_768 = torch.load(text_embeddings_path, weights_only=True)
    if isinstance(text_embeddings_768, torch.Tensor):
        text_embeddings_768 = text_embeddings_768.cpu().numpy()
    print(f"  Original text embeddings shape: {text_embeddings_768.shape}")

    num_classes = text_embeddings_768.shape[0]

    # Perform SVD reduction to svd_rank dimensions
    print(f"\nPerforming SVD reduction on text embeddings...")
    text_embeddings_reduced, text_embeddings_sampled, svd_metadata = svd_reduce_text_embeddings(
        text_embeddings_768, svd_rank, sample_size=procrustes_sample_size, normalize=True
    )
    print(f"  Reduced text embeddings shape: {text_embeddings_reduced.shape}")
    print(f"  Sampled text embeddings shape: {text_embeddings_sampled.shape} (for Procrustes)")

    text_embeddings_reduced_tensor = torch.from_numpy(text_embeddings_reduced).to(device)

    # Build test dataset
    print("\nBuilding test dataset...")
    try:
        dataset = build_dataset(cfg.data.test)
    except Exception as e:
        print(f"Error building dataset: {e}")
        print("Trying with test_mode=True...")
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)

    print(f"Dataset size: {len(dataset)}")

    # Validate on first few scenes
    if num_scenes is None:
        num_scenes = len(dataset)

    print(f"\nValidating on {min(num_scenes, len(dataset))} scenes...")

    # Collect metrics
    all_intersection = np.zeros(num_classes, dtype=np.float64)
    all_union = np.zeros(num_classes, dtype=np.float64)
    all_target = np.zeros(num_classes, dtype=np.float64)

    scene_count = 0
    for idx in range(min(num_scenes, len(dataset))):
        try:
            # Get sample
            sample = dataset[idx]

            # Move to device
            input_dict = {}
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    input_dict[key] = torch.from_numpy(value).to(device)
                elif isinstance(value, torch.Tensor):
                    input_dict[key] = value.to(device)
                elif isinstance(value, str) or isinstance(value, bool):
                    input_dict[key] = value
                else:
                    input_dict[key] = value

            print(f"\nScene {idx+1}: {input_dict.get('name', 'unknown')}")

            # If feat is missing, manually construct it from scene files
            if 'feat' not in input_dict:
                print(f"  'feat' not in dataset output, constructing manually...")
                # Try to get scene_path from input_dict, or construct from data_root + name
                scene_path = input_dict.get('scene_path')
                if scene_path is None:
                    # Construct from data_root and name
                    name = input_dict.get('name', '')
                    if not name:
                        print(f"  ERROR: 'name' not found, cannot construct scene path")
                        continue
                    # Use data_root from config or construct from available info
                    # For val split, scenes are in data_root/val/
                    scene_path = Path(data_root) / "val" / name

                scene_path = Path(scene_path)

                # Check required files
                required_files = ['color.npy', 'opacity.npy', 'quat.npy', 'scale.npy']
                missing_files = [f for f in required_files if not (scene_path / f).exists()]
                if missing_files:
                    print(f"  ERROR: Missing files: {missing_files}")
                    continue

                # Load Gaussian parameters
                color = np.load(scene_path / 'color.npy').astype(np.float32) / 255.0
                opacity = np.load(scene_path / 'opacity.npy').astype(np.float32).clip(0.001).reshape(-1, 1)
                quat = np.load(scene_path / 'quat.npy').astype(np.float32)
                scale = np.load(scene_path / 'scale.npy').astype(np.float32).clip(1e-4, 1.0)

                # Stack features: [N_full, 11]
                feat = np.stack([
                    color[:, 0], color[:, 1], color[:, 2],  # color (3)
                    opacity[:, 0],  # opacity (1)
                    quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3],  # quat (4)
                    scale[:, 0], scale[:, 1], scale[:, 2],  # scale (3)
                ], axis=1)  # [N_full, 11]

                # Check if we need to subsample using inverse mapping
                if 'inverse' in input_dict:
                    # Use inverse mapping to subsample features to match coord
                    inverse = input_dict['inverse']
                    print(f"  inverse shape: {inverse.shape}, min: {inverse.min()}, max: {inverse.max()}")
                    print(f"  feat before subsample: {feat.shape}")
                    # inverse[i] maps from full-resolution point i to subsampled index
                    # We need to reverse this: find which full point maps to each subsampled index
                    # Use unique with return_index to find first occurrence of each subsampled index
                    inverse_np = inverse.cpu().numpy()
                    unique_subsampled, first_full_indices = np.unique(inverse_np, return_index=True)
                    print(f"  unique_subsampled shape: {unique_subsampled.shape}, first_full_indices shape: {first_full_indices.shape}")
                    # Select features at the first full-resolution index for each subsampled point
                    feat = feat[first_full_indices]  # [N_sub, 11]
                    print(f"  Subsampled feat using inverse mapping: {feat.shape}")
                else:
                    print(f"  WARNING: No 'inverse' mapping, feat size mismatch may occur!")

                input_dict['feat'] = torch.from_numpy(feat).to(device)
                print(f"  Final feat shape: {input_dict['feat'].shape}")

            # If grid_coord is missing, create it from coord
            if 'grid_coord' not in input_dict:
                coord = input_dict['coord']
                coord_min = coord.min(0, keepdim=True)[0]
                grid_coord = torch.round((coord - coord_min) / 0.01).long()
                input_dict['grid_coord'] = grid_coord

            # If batch is missing, create it
            if 'batch' not in input_dict and 'offset' not in input_dict:
                N = input_dict['coord'].shape[0]
                input_dict['batch'] = torch.zeros(N, dtype=torch.long, device=device)

            print(f"  Input keys: {list(input_dict.keys())}")
            print(f"  coord shape: {input_dict['coord'].shape}")
            print(f"  feat shape: {input_dict['feat'].shape}")
            print(f"  grid_coord shape: {input_dict['grid_coord'].shape}")
            print(f"  batch shape: {input_dict['batch'].shape}")

            # Model forward pass
            with torch.no_grad():
                # Check if model returns dict with point_feat
                if hasattr(model, 'lang_feat_dim'):
                    # SVD-16 model, outputs 16-dim features
                    chunk_size = min(600000, input_dict['coord'].shape[0])
                    output = model(input_dict, chunk_size=chunk_size)
                else:
                    output = model(input_dict)

            # Get point features
            if isinstance(output, dict) and 'point_feat' in output:
                point_feat = output['point_feat']['feat']  # [N, 16]
            elif isinstance(output, dict) and 'feat' in output:
                point_feat = output['feat']
            else:
                print(f"  Unexpected output format: {output.keys() if isinstance(output, dict) else type(output)}")
                continue

            print(f"  Model output shape: {point_feat.shape}")

            # Convert point_feat to numpy for Procrustes alignment
            point_feat_np = point_feat.cpu().numpy()

            if use_procrustes:
                # Sample point features for Q computation
                point_feat_sample = point_feat_np
                text_emb_for_Q = text_embeddings_sampled

                if text_embeddings_sampled.shape[0] < point_feat_np.shape[0]:
                    # Sample from point features to match text embeddings size
                    indices = np.random.choice(point_feat_np.shape[0], text_embeddings_sampled.shape[0], replace=False)
                    point_feat_sample = point_feat_np[indices]
                elif point_feat_np.shape[0] < text_embeddings_sampled.shape[0]:
                    # Sample from text embeddings to match point features size
                    text_emb_for_Q = text_embeddings_sampled[:point_feat_np.shape[0]]

                # Ensure shapes match
                min_size = min(point_feat_sample.shape[0], text_emb_for_Q.shape[0])
                point_feat_sample = point_feat_sample[:min_size]
                text_emb_for_Q = text_emb_for_Q[:min_size]

                # Compute Procrustes Q matrix
                Q, procrustes_metrics = compute_procrustes_Q(
                    point_feat_sample, text_emb_for_Q, use_torch=True, device=device
                )
                print(f"  Procrustes Q: det={procrustes_metrics['det_Q']:.4f}, "
                      f"ortho_err={procrustes_metrics['orthogonality_error']:.2e}")
                print(f"  Cosine similarity: {procrustes_metrics['cosine_before']:.4f} -> {procrustes_metrics['cosine_after']:.4f}")

                # Apply Q alignment to all point features
                point_feat_aligned_np = point_feat_np @ Q
                point_feat = torch.from_numpy(point_feat_aligned_np).to(device)
            else:
                # No alignment, use original point features
                point_feat = point_feat

            # Compute logits with aligned 16-dim features and SVD-reduced text embeddings
            # point_feat: [N, svd_rank], text_embeddings_reduced: [C, svd_rank]
            logits = torch.mm(point_feat, text_embeddings_reduced_tensor.t())  # [N, C]
            probs = torch.sigmoid(logits)
            pred = probs.argmax(dim=1)

            print(f"  Logits shape: {logits.shape}, Pred shape: {pred.shape}")

            # Get ground truth labels
            if 'segment' in input_dict:
                segment = input_dict['segment']
            elif 'origin_segment' in input_dict:
                segment = input_dict['origin_segment']
            else:
                print("  No ground truth labels found, skipping metrics")
                scene_count += 1
                continue

            print(f"  segment shape before check: {segment.shape}")

            # Check if segment needs to be subsampled
            if segment.shape[0] != pred.shape[0] and 'inverse' in input_dict:
                print(f"  Subsampling segment using inverse mapping...")
                inverse = input_dict['inverse']
                inverse_np = inverse.cpu().numpy()
                unique_subsampled, first_full_indices = np.unique(inverse_np, return_index=True)
                segment = segment[first_full_indices]
                print(f"  Subsampled segment shape: {segment.shape}")

            # Verify shapes match now
            if segment.shape[0] != pred.shape[0]:
                print(f"  ERROR: segment {segment.shape[0]} != pred {pred.shape[0]}, cannot compute metrics")
                continue

            # Calculate metrics
            ignore_index = cfg.data.get('ignore_index', -1)
            for cls_idx in range(num_classes):
                mask = (segment == cls_idx) & (segment != ignore_index)
                if mask.sum() == 0:
                    continue

                pred_mask = (pred == cls_idx) & mask
                gt_mask = mask

                intersection = (pred_mask & gt_mask).sum().cpu().item()
                union = (pred_mask | gt_mask).sum().cpu().item()
                target = gt_mask.sum().cpu().item()

                all_intersection[cls_idx] += intersection
                all_union[cls_idx] += union
                all_target[cls_idx] += target

            scene_count += 1

        except Exception as e:
            print(f"  Error processing scene {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Calculate overall metrics
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    # Calculate IoU and accuracy
    iou_class = all_intersection / (all_union + 1e-10)
    accuracy_class = all_intersection / (all_target + 1e-10)

    # Mean metrics (only for classes with samples)
    valid_classes = all_target > 0
    m_iou = np.mean(iou_class[valid_classes])
    m_acc = np.mean(accuracy_class[valid_classes])
    all_acc = all_intersection.sum() / (all_target.sum() + 1e-10)

    print(f"Validated scenes: {scene_count}")
    print(f"mIoU: {m_iou:.4f}")
    print(f"mAcc: {m_acc:.4f}")
    print(f"allAcc: {all_acc:.4f}")

    print("\n" + "=" * 60)

    return {
        'm_iou': m_iou,
        'm_acc': m_acc,
        'all_acc': all_acc,
        'scenes_validated': scene_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Test validation with SVD-reduced text embeddings + Procrustes alignment")
    parser.add_argument("--config", type=str, default="configs/custom/lang-pretrain-litept-scannet.py",
                       help="Path to config file")
    parser.add_argument("--weight", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--text_embeddings", type=str, required=True,
                       help="Path to 768-dim text embeddings (will be SVD-reduced)")
    parser.add_argument("--data_root", type=str, default="/new_data/cyf/Datasets/SceneSplat7k/scannet",
                       help="Path to test data root")
    parser.add_argument("--split", type=str, default="test_grid1.0cm_chunk6x6_stride3x3",
                       help="Test split name")
    parser.add_argument("--num_scenes", type=int, default=10,
                       help="Number of scenes to validate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--svd_rank", type=int, default=16,
                       help="SVD rank for text embedding reduction (default: 16)")
    parser.add_argument("--no_procrustes", action="store_true",
                       help="Disable Procrustes alignment (use SVD-reduced text embeddings directly)")
    parser.add_argument("--procrustes_sample_size", type=int, default=None,
                       help="Sample size for Procrustes Q computation (default: all classes)")

    args = parser.parse_args()

    # Run validation
    results = validate_with_compressed_embeddings(
        config_path=args.config,
        weight_path=args.weight,
        text_embeddings_path=args.text_embeddings,
        data_root=args.data_root,
        split=args.split,
        device=args.device,
        num_scenes=args.num_scenes,
        svd_rank=args.svd_rank,
        use_procrustes=not args.no_procrustes,
        procrustes_sample_size=args.procrustes_sample_size,
    )

    print("\nValidation complete!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
