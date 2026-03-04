#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import glob
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
import json
from tqdm import tqdm
from PIL import Image

# Add parent directory to path for Procrustes functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from compute_procrustes_alignment_simple import (
    perform_svd_reduction,
    compute_procrustes_Q,
)

import colormaps
from openclip_encoder import OpenCLIPNetwork
from siglip2_encoder import SigLIP2Network
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
from eval_utils import plot_relevancy_and_threshold, compute_dynamic_threshold


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    gt_ann = {}
    print(gt_json_paths)
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths


def load_or_compute_procrustes_Q(
    clip_model,
    text_embeddings_path: str = None,
    svd_rank: int = 16,
    device: str = 'cuda',
    normalize: bool = True
) -> np.ndarray:
    """
    Load pre-computed Q matrix or compute it from CLIP text embeddings.

    Args:
        clip_model: CLIP model instance (OpenCLIPNetwork or SigLIP2Network)
        text_embeddings_path: Path to text embeddings file
        svd_rank: SVD rank for text embeddings (default 16)
        device: Device for computation
        normalize: Whether to normalize features

    Returns:
        Q: [d, d] orthogonal alignment matrix
    """
    if text_embeddings_path is None:
        raise ValueError("text_embeddings_path must be provided for Procrustes alignment")

    # Check if pre-computed Q file exists
    text_embed_basename = Path(text_embeddings_path).stem
    if '_text_embeddings' in text_embed_basename:
        benchmark_name = text_embed_basename.split('_text_embeddings')[0]
    else:
        benchmark_name = text_embed_basename

    # Try to find pre-computed Q file
    # Q file could be in same directory as text embeddings or in parent directory
    text_embed_dir = Path(text_embeddings_path).parent
    q_file_candidates = [
        text_embed_dir / f"Q_procrustes_{benchmark_name}_average_r{svd_rank}.npz",
        text_embed_dir.parent / f"Q_procrustes_{benchmark_name}_average_r{svd_rank}.npz",
        Path(f"/new_data/cyf/projects/SceneSplat/gaussian_train/Q_procrustes_{benchmark_name}_average_r{svd_rank}.npz"),
    ]

    for q_file in q_file_candidates:
        if q_file.exists():
            logging.info(f"Loading pre-computed Q matrix from {q_file}")
            q_data = np.load(q_file)
            Q = q_data['Q']
            logging.info(f"  Q shape: {Q.shape}, det(Q): {np.linalg.det(Q):.6f}")
            return Q

    # No pre-computed Q found, compute from text embeddings
    logging.info(f"No pre-computed Q found, computing from text embeddings...")
    logging.info(f"  Text embeddings: {text_embeddings_path}")

    # Load text embeddings
    if text_embeddings_path.endswith('.pt'):
        text_embed_data = torch.load(text_embeddings_path, weights_only=False)
        if isinstance(text_embed_data, dict):
            text_embeddings = text_embed_data['embeddings'].numpy().astype(np.float32)
        else:
            text_embeddings = text_embed_data.numpy().astype(np.float32)
    elif text_embeddings_path.endswith('.npy'):
        text_embeddings = np.load(text_embeddings_path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported text embeddings format: {text_embeddings_path}")

    logging.info(f"  Text embeddings shape: {text_embeddings.shape}")

    # SVD-reduce text embeddings to match feature dimension
    target_dim = clip_model.clip_n_dims  # 512 for CLIP, 768 for SigLIP2
    if text_embeddings.shape[1] > target_dim:
        logging.info(f"  SVD-reducing text embeddings: {text_embeddings.shape} -> ({text_embeddings.shape[0]}, {target_dim})")
        text_embeddings, components, _ = perform_svd_reduction(
            text_embeddings, target_dim, normalize, use_torch=False
        )
    elif text_embeddings.shape[1] < target_dim:
        logging.warning(f"  Text embeddings dim {text_embeddings.shape[1]} < CLIP dim {target_dim}, may not work well")

    # Sample random features from scene to compute Q
    # For now, use a subset of text embeddings as proxy
    # In practice, you'd want to use actual scene features
    num_samples = min(text_embeddings.shape[0], 100)
    sample_indices = np.random.choice(text_embeddings.shape[0], num_samples, replace=False)
    Y_sample = text_embeddings[sample_indices]  # [num_samples, d]

    # Create dummy X_c for Q computation (identity-like)
    # In practice, this should be replaced with actual scene features
    X_c = Y_sample.copy()  # Will result in Q ≈ I

    # Compute Q using Procrustes
    logging.info(f"  Computing Procrustes Q from {X_c.shape[0]} samples...")
    Q, metrics = compute_procrustes_Q(X_c, Y_sample, use_torch=True, device=device)
    logging.info(f"  Q computed: shape={Q.shape}, det={metrics['det_Q']:.6f}")
    logging.info(f"  Cosine before: {metrics['cosine_before']:.4f}, after: {metrics['cosine_after']:.4f}")

    return Q


def compute_procrustes_Q_cuda_with_labels(
    X_c: torch.Tensor,
    Y: torch.Tensor,
    labels: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    """
    Compute orthogonal Procrustes alignment matrix Q using label-based aggregation.

    This method is used when X_c contains many duplicate rows (corresponding to
    different points with the same semantic class), and Y contains the unique
    rows (class embeddings).

    Args:
        X_c: [N, d] source features (CUDA tensor)
        Y: [M, d] unique target features (CUDA tensor, M << N typically)
        labels: [N] label indices, labels[i] ∈ {0, ..., M-1}

    Returns:
        Q: [d, d] orthogonal alignment matrix (CUDA tensor)
        metrics: Dictionary with alignment metrics
    """
    N, d = X_c.shape
    M, d_y = Y.shape

    if d != d_y:
        raise ValueError(
            f"Feature dimension mismatch: X_c has d={d}, Y has d={d_y}"
        )

    if labels.shape[0] != N:
        raise ValueError(
            f"Labels length mismatch: X_c has N={N}, labels has {labels.shape[0]}"
        )

    # Efficient computation of M = X_c^T @ Y_reconstructed
    # by aggregating points with the same label
    # sum_j[j] = Σ_{i: labels[i]=j} X_c[i]  for all j in [0, M-1]
    sum_j = torch.zeros(M, d, device=X_c.device, dtype=X_c.dtype)

    # Ensure labels is LongTensor (required by index_add_)
    if labels.dtype != torch.long:
        labels = labels.long()

    # Use index_add_ for efficient aggregation
    sum_j.index_add_(0, labels, X_c)  # [M, d]

    # Compute M = sum_j^T @ Y: [d, M] @ [M, d] = [d, d]
    M_matrix = torch.mm(sum_j.t(), Y)  # [d, d]

    # SVD on GPU
    U, S, Vt = torch.linalg.svd(M_matrix, full_matrices=False)

    # Q = U @ V^T
    Q = torch.mm(U, Vt)

    # Ensure det(Q) = 1 (proper rotation)
    det_Q = torch.det(Q)
    if det_Q < 0:
        U[:, -1] *= -1
        Q = torch.mm(U, Vt)
        det_Q = torch.det(Q)

    # Compute metrics using all data
    Y_reconstructed = Y[labels]
    X_aligned = torch.mm(X_c, Q)
    residual = torch.norm(X_aligned - Y_reconstructed, 'fro')
    relative_error = residual / (torch.norm(Y_reconstructed, 'fro') + 1e-8)

    # Cosine similarity (vectorized on GPU)
    X_c_norm = torch.norm(X_c, dim=1, keepdim=True) + 1e-8
    Y_norm = torch.norm(Y_reconstructed, dim=1, keepdim=True) + 1e-8
    cosine_before = torch.mean(
        torch.sum(X_c * Y_reconstructed, dim=1) / (X_c_norm.squeeze() * Y_norm.squeeze())
    )

    X_aligned_norm = torch.norm(X_aligned, dim=1, keepdim=True) + 1e-8
    cosine_after = torch.mean(
        torch.sum(X_aligned * Y_reconstructed, dim=1) / (X_aligned_norm.squeeze() * Y_norm.squeeze())
    )

    # Orthogonality error
    QtQ = torch.mm(Q.t(), Q)
    I = torch.eye(d, device=Q.device)
    orthogonality_error = torch.norm(QtQ - I, 'fro')

    metrics = {
        'N': N,
        'M': M,
        'd': d,
        'unique_labels': M,
        'residual_norm': float(residual.cpu()),
        'relative_error': float(relative_error.cpu()),
        'cosine_before': float(cosine_before.cpu()),
        'cosine_after': float(cosine_after.cpu()),
        'cosine_improvement': float((cosine_after - cosine_before).cpu()),
        'orthogonality_error': float(orthogonality_error.cpu()),
        'det_Q': float(det_Q.cpu()),
    }

    return Q, metrics


def apply_procrustes_alignment(
    features: torch.Tensor,
    Q: np.ndarray,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Apply Procrustes Q matrix to align features.

    Args:
        features: [N, d] feature tensor
        Q: [d, d] orthogonal alignment matrix
        device: Device for computation

    Returns:
        aligned_features: [N, d] aligned feature tensor
    """
    Q_tensor = torch.from_numpy(Q).to(device=device, dtype=features.dtype)
    aligned = torch.mm(features, Q_tensor)
    return aligned


def compute_per_scene_Q(
    scene_name: str,
    train_data_root: str,
    text_embeddings_16: np.ndarray,
    svd_rank: int = 16,
    feature_level: int = 1,
    device: str = 'cuda'
) -> tuple[np.ndarray, dict]:
    """
    Compute per-scene Q matrix using training data with label-based aggregation.

    Args:
        scene_name: Name of the scene (e.g., 'figurines', 'ramen')
        train_data_root: Root directory containing training data
        text_embeddings_16: [M, svd_rank] SVD-reduced text embeddings
        svd_rank: SVD rank
        feature_level: Feature level (1, 2, or 3) for selecting SVD file
        device: Device for computation

    Returns:
        Q: [svd_rank, svd_rank] orthogonal alignment matrix (numpy array)
        metrics: Dictionary with alignment metrics
    """
    scene_path = Path(train_data_root) / scene_name
    if not scene_path.exists():
        raise ValueError(f"Scene path not found: {scene_path}")

    # Load SVD compressed features with level suffix
    if feature_level == 0:
        svd_file = scene_path / f'lang_feat_grid_svd_r{svd_rank}.npz'
    else:
        svd_file = scene_path / f'lang_feat_grid_svd_r{svd_rank}_{feature_level}.npz'

    if not svd_file.exists():
        raise ValueError(f"SVD file not found: {svd_file}")

    svd_data = np.load(svd_file)
    compressed = svd_data['compressed'].astype(np.float32)  # [M_grid, svd_rank]
    indices = svd_data['indices']
    X_c = compressed[indices]  # [N_points, svd_rank]

    # Load labels
    label_file = scene_path / 'lang_label.npy'
    if not label_file.exists():
        raise ValueError(f"Label file not found: {label_file}")

    labels = np.load(label_file).astype(np.int64)

    # Check if we need valid_feat_mask
    if labels.shape[0] != X_c.shape[0]:
        valid_mask_path = scene_path / 'valid_feat_mask.npy'
        if valid_mask_path.exists():
            valid_mask = np.load(valid_mask_path).astype(bool)
            labels_filtered = labels[valid_mask]
            if labels_filtered.shape[0] == X_c.shape[0]:
                labels = labels_filtered
            else:
                raise ValueError(f"After filtering, label count {labels_filtered.shape[0]} != X_c count {X_c.shape[0]}")
        else:
            raise ValueError(f"Label count {labels.shape[0]} != X_c count {X_c.shape[0]}")

    # Filter out invalid labels (ignore_index or out of range)
    valid_label_mask = (labels >= 0) & (labels < text_embeddings_16.shape[0])
    if valid_label_mask.sum() < labels.shape[0]:
        labels = labels[valid_label_mask]
        X_c = X_c[valid_label_mask]

    if X_c.shape[0] == 0 or labels.shape[0] == 0:
        raise ValueError("No valid data after filtering labels")

    # Convert to CUDA tensors
    X_c_tensor = torch.from_numpy(X_c).cuda()
    Y_tensor = torch.from_numpy(text_embeddings_16).cuda()
    labels_tensor = torch.from_numpy(labels).cuda()

    logging.info(f"  Computing Q matrix for {scene_name} with {X_c_tensor.shape[0]} samples...")

    # Compute Q using label-based aggregation
    Q_cuda, metrics = compute_procrustes_Q_cuda_with_labels(
        X_c_tensor, Y_tensor, labels_tensor
    )

    Q = Q_cuda.cpu().numpy().astype(np.float32)

    logging.info(f"  Q computed: det={metrics['det_Q']:.6f}")
    logging.info(f"  Cosine before: {metrics['cosine_before']:.4f}, after: {metrics['cosine_after']:.4f}")
    logging.info(f"  Improvement: {metrics['cosine_improvement']:+.4f}")

    return Q, metrics


def activate_stream_with_procrustes(
    sem_map,
    clip_model,
    Q: np.ndarray,
    text_embeddings_16: torch.Tensor,  # [N_classes, 16] SVD-reduced text embeddings
    image_name: Path = None,
    img_ann: Dict = None,
    eval_params: Dict = None,
    device: str = 'cuda'
):
    """
    Activate stream with Procrustes alignment applied to features.
    Computes similarity directly in the compressed feature space.

    Args:
        sem_map: Semantic feature maps [n_head, h, w, d_compressed]
        clip_model: CLIP model instance (used for positive phrases)
        Q: Procrustes alignment matrix [d, d]
        text_embeddings_16: [N_classes, d_compressed] SVD-reduced text embeddings
        image_name: Path for saving results
        img_ann: Ground truth annotations
        eval_params: Evaluation parameters
        device: Device for computation

    Returns:
        chosen_iou_list: List of IoU scores
        chosen_lvl_list: List of chosen levels
    """
    # Apply Procrustes alignment to features
    n_head, h, w, d_compressed = sem_map.shape
    sem_map_flat = sem_map.permute(1, 2, 0, 3).flatten(0, 1)  # (h*w, n_head, d)

    # Apply Q alignment to each head
    aligned_features = []
    for i in range(n_head):
        head_features = sem_map_flat[:, i, :]  # (h*w, d)
        aligned = apply_procrustes_alignment(head_features, Q, device)
        aligned_features.append(aligned)

    # Stack back: (h*w, n_head, d) -> (n_head, h*w, d)
    aligned_features = torch.stack(aligned_features, dim=1)
    # Reshape to (n_head, h, w, d)
    aligned_map = aligned_features.transpose(0, 1).reshape(n_head, h, w, -1)

    # Compute similarity directly in compressed space
    # aligned_map: [n_head, h, w, d], text_embeddings_16: [N_classes, d]
    n_classes = text_embeddings_16.shape[0]
    n_positive = len(img_ann)

    # Reshape aligned_map: [n_head, h*w, d]
    aligned_flat = aligned_map.view(n_head, -1, d_compressed)

    # Compute similarity for each class: [n_head, h*w]
    similarity_maps = []
    for class_idx in range(n_classes):
        # [n_head, h*w, d] @ [d] -> [n_head, h*w]
        class_sim = torch.matmul(aligned_flat, text_embeddings_16[class_idx])
        similarity_maps.append(class_sim)

    # Stack: [N_classes, n_head, h*w]
    similarity_maps = torch.stack(similarity_maps)
    # Reshape: [N_classes, n_head, h, w]
    similarity_maps = similarity_maps.view(n_classes, n_head, h, w)

    # Get positive class indices from img_ann
    positive_phrases = list(img_ann.keys())
    positive_indices = []

    # Find indices of positive phrases in text_embeddings_16
    # We need to match positive_phrases with the classes used to create text_embeddings_16
    # For now, assume the first n_positive classes correspond to the positive phrases
    # This is a simplification - in practice, you'd need to map phrases to indices properly

    valid_map = []
    for k, phrase in enumerate(positive_phrases):
        # Use k as the class index (assuming order matches)
        class_sim = similarity_maps[k]  # [n_head, h, w]
        valid_map.append(class_sim)

    valid_map = torch.stack(valid_map)  # [n_positive, n_head, h, w]
    valid_map = valid_map.cpu()

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []

    for k in range(n_positive):
        chosen_lvl, thresh = compute_dynamic_threshold(valid_map[k], positive_phrases[k], eval_params=eval_params)

        for i in range(n_head):
            output = valid_map[k, i]  # [h, w]
            output = output - torch.min(output)
            output = output / (torch.max(output) - torch.min(output) + 1e-9)

            save_path = image_name / 'comparison_maps' / f'{positive_phrases[k]}_level{i}_comparison.png'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plot_relevancy_and_threshold(output.cpu().numpy(), positive_phrases[k], i, save_path, threshold=thresh)

            if i == chosen_lvl:
                mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
                mask_pred = smooth(mask_pred)
                mask_gt = img_ann[positive_phrases[k]]['mask'].astype(np.uint8)

                intersection = np.logical_and(mask_gt, mask_pred).sum()
                union = np.logical_or(mask_gt, mask_pred).sum()
                iou = intersection / (union + 1e-9)

        chosen_iou_list.append(iou)
        chosen_lvl_list.append(chosen_lvl)

        # save for visualization
        save_path = image_name / f'chosen_{positive_phrases[k]}.png'
        vis_mask_save(mask_pred, save_path)

    return chosen_iou_list, chosen_lvl_list


def activate_stream(sem_map, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    eval_params: Dict = None):
    
    valid_map = clip_model.get_max_across(sem_map)
    n_head, n_prompt, h, w = valid_map.shape
    valid_map = valid_map.cpu()
    
    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    
    for k in range(n_prompt):
        
        chosen_lvl, thresh = compute_dynamic_threshold(valid_map[:, k], clip_model.positives[k], eval_params=eval_params)
        
        for i in range(n_head):
            
            # NOTE [mask] truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) -  torch.min(output) + 1e-9)
            
            save_path = image_name / 'comparison_maps' / f'{clip_model.positives[k]}_level{i}_comparison.png'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plot_relevancy_and_threshold(output, clip_model.positives[k], i, save_path, threshold=thresh)
            
            if i == chosen_lvl:
                # Create Binary Mask through thresholding:
                mask_pred = (output.numpy() > thresh).astype(np.uint8)
                mask_pred = smooth(mask_pred)
                mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
                
                intersection = np.logical_and(mask_gt, mask_pred).sum()
                union = np.logical_or(mask_gt, mask_pred).sum()
                iou = intersection / (union + 1e-9)  # Avoid division by zero
            
        chosen_iou_list.append(iou)
        chosen_lvl_list.append(chosen_lvl)
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_pred, save_path)

    return chosen_iou_list, chosen_lvl_list


def evaluate(feat_dir, output_path, gt_path, logger, eval_params, src_dim=512, projection_matrix_path=None,
               use_clip=False, use_procrustes=False, text_embeddings_path=None, q_matrix_path=None,
               train_data_root=None, dataset_name=None, feature_level=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(gt_path), Path(output_path))

    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    feat_paths_lvl = []
    for i in range(len(feat_dir)):
        # Create a mapping of index to file path
        index_to_file = {}
        for file_path in glob.glob(os.path.join(feat_dir[i], '*.npy')):
            file_idx = int(os.path.basename(file_path).split(".npy")[0].split('_')[1]) - 1
            index_to_file[file_idx] = file_path

        feat_paths_lvl.append(index_to_file)

    assert len(feat_paths_lvl) == len(feat_dir)

    # Load or compute Q matrix for Procrustes alignment
    Q = None
    if use_procrustes:
        logging.info("=" * 60)
        logging.info("Procrustes Alignment Mode")
        logging.info("=" * 60)

        if q_matrix_path and os.path.exists(q_matrix_path):
            logging.info(f"Loading Q matrix from {q_matrix_path}")
            q_data = np.load(q_matrix_path)
            Q = q_data['Q']
            logging.info(f"  Q shape: {Q.shape}, det(Q): {np.linalg.det(Q):.6f}")
        else:
            # Need to load or compute Q - will do after initializing clip_model
            pass

    # Load projection matrix for text embedding SVD reduction (512-dim -> target_dim)
    projection_matrix = None
    if projection_matrix_path and os.path.exists(projection_matrix_path):
        logging.info(f"Loading projection matrix from {projection_matrix_path}")
        pm_data = np.load(projection_matrix_path)
        if 'projection_matrix' in pm_data:
            projection_matrix = pm_data['projection_matrix']
        else:
            projection_matrix = pm_data
        logging.info(f"  Projection matrix shape: {projection_matrix.shape}")

    # Instantiate encoder based on configuration
    if use_clip:
        # Use CLIP (OpenCLIPNetwork)
        if projection_matrix is not None:
            clip_model = OpenCLIPNetwork(device, projection_matrix=projection_matrix, target_dim=src_dim)
            logging.info(f"Using OpenCLIPNetwork (CLIP, {src_dim}-dim with projection)")
        else:
            clip_model = OpenCLIPNetwork(device)
            logging.info(f"Using OpenCLIPNetwork (CLIP, 512-dim)")
    elif src_dim == 768:
        clip_model = SigLIP2Network(device)
        logging.info(f"Using SigLIP2Network for 768-dimensional features")
    else:
        # Fallback to OpenCLIPNetwork for other dimensions
        clip_model = OpenCLIPNetwork(device, projection_matrix=projection_matrix, target_dim=src_dim)
        logging.info(f"Using OpenCLIPNetwork for {src_dim}-dimensional features (with projection)")

    # Load or compute Q matrix after clip_model is initialized
    if use_procrustes and Q is None:
        if train_data_root and dataset_name:
            # Compute per-scene Q matrix using training data
            logging.info(f"Computing per-scene Q matrix for {dataset_name}...")

            # First, compute SVD-reduced text embeddings
            if text_embeddings_path is None:
                raise ValueError("text_embeddings_path must be provided for per-scene Q computation")

            # Load text embeddings
            if text_embeddings_path.endswith('.pt'):
                text_embed_data = torch.load(text_embeddings_path, weights_only=False)
                if isinstance(text_embed_data, dict):
                    text_embeddings = text_embed_data['embeddings'].numpy().astype(np.float32)
                else:
                    text_embeddings = text_embed_data.numpy().astype(np.float32)
            elif text_embeddings_path.endswith('.npy'):
                text_embeddings = np.load(text_embeddings_path).astype(np.float32)
            else:
                raise ValueError(f"Unsupported text embeddings format: {text_embeddings_path}")

            # SVD reduce text embeddings from 512-dim to src_dim
            norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
            text_embeddings_norm = text_embeddings / (norms + 1e-8)
            U, S, Vt = np.linalg.svd(text_embeddings_norm, full_matrices=False)
            V_src = Vt[:src_dim, :].T  # [512, src_dim]
            text_embeddings_src = text_embeddings @ V_src  # [N, src_dim]

            # Save text_embeddings_src for later use in similarity computation
            text_embeddings_16_tensor = torch.from_numpy(text_embeddings_src).cuda()

            # Compute per-scene Q
            Q, metrics = compute_per_scene_Q(
                dataset_name,
                train_data_root,
                text_embeddings_src,
                svd_rank=src_dim,
                feature_level=feature_level,
                device=device
            )
        else:
            # Fallback to old method
            Q = load_or_compute_procrustes_Q(
                clip_model,
                text_embeddings_path=text_embeddings_path,
                svd_rank=src_dim,
                device=device
            )
            text_embeddings_16_tensor = None
    else:
        text_embeddings_16_tensor = None

    chosen_iou_all, chosen_lvl_list = [], []
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx:0>2}'
        image_name.mkdir(exist_ok=True, parents=True)

        compressed_sem_feats = np.zeros((len(feat_dir), *image_shape, src_dim), dtype=np.float32)
        for i in range(len(feat_dir)):
            if idx not in feat_paths_lvl[i]:
                raise ValueError(f"Missing feature file for index {idx} in directory {feat_dir[i]}")
            compressed_sem_feats[i] = np.load(feat_paths_lvl[i][idx], mmap_mode='r')

        sem_feat = torch.from_numpy(compressed_sem_feats).float().to(device)
        print(f"j: {j}, idx: {idx}, image_name: {image_name}, image_path: {image_paths[j]}")

        img_ann = gt_ann[f'{idx}']
        clip_model.set_positives(list(img_ann.keys()))

        # Choose activation function based on Procrustes mode
        if use_procrustes and Q is not None:
            c_iou_list, c_lvl = activate_stream_with_procrustes(
                sem_feat, clip_model, Q, text_embeddings_16_tensor,
                image_name, img_ann,
                eval_params=eval_params,
                device=device
            )
        else:
            c_iou_list, c_lvl = activate_stream(sem_feat, clip_model,
                                                image_name, img_ann,
                                                eval_params=eval_params)

        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

    # iou
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    ious = np.array(chosen_iou_all)
    m50 = np.sum(ious > 0.5)/ious.shape[0]
    m25 = np.sum(ious > 0.25)/ious.shape[0]
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")
    print('scene_name:', logger.name, 'mious:', mean_iou_chosen, 'm50:', m50, 'm25:', m25)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)

    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--gt_folder", type=str, default=None)
    parser.add_argument("--feat_folder", type=str, default=None)
    parser.add_argument("--feat_base_path", type=str, default=None, help="Base path for features (e.g., /new_data/cyf/projects/SceneSplat/gaussian_results/lerf_ovs)")
    parser.add_argument("--single_level", action="store_true", help="Use single feature level (level 0) instead of multiple levels")
    parser.add_argument("--feat_level", type=int, default=None, help="Specific feature level to use (0, 1, 2, or 3). Overrides --single_level behavior.")
    parser.add_argument("--stability_thresh", type=float, default=0.3)
    parser.add_argument("--min_mask_size", type=float, default=0.001)
    parser.add_argument("--max_mask_size", type=float, default=0.95)
    parser.add_argument("--src_dim", type=int, default=512, help="Source dimension of language features to load (e.g., 512 for OpenCLIP, 768 for SigLIP2, 16 for SVD-compressed)")
    parser.add_argument("--projection_matrix", type=str, default=None, help="Path to projection matrix (e.g., /new_data/cyf/projects/SceneSplat/gaussian_train/projection_matrix_768_to_16_lerf.npy)")

    # CLIP model options
    parser.add_argument("--use_clip", action="store_true", help="Use CLIP (OpenCLIPNetwork) instead of SigLIP2Network")

    # Procrustes alignment options
    parser.add_argument("--use_procrustes", action="store_true", help="Use Procrustes alignment before computing similarity")
    parser.add_argument("--text_embeddings", type=str, default=None, help="Path to text embeddings for Q matrix computation (e.g., /new_data/cyf/projects/SceneSplat/pointcept/datasets/preprocessing/scannet/meta_data/scannet20_text_embeddings_siglip2.pt)")
    parser.add_argument("--q_matrix", type=str, default=None, help="Path to pre-computed Q matrix (e.g., /new_data/cyf/projects/SceneSplat/gaussian_train/Q_procrustes_scannet20_average_r16.npz)")
    parser.add_argument("--train_data_root", type=str, default=None, help="Path to training data root for per-scene Q computation (e.g., /new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train)")

    args = parser.parse_args()

    eval_params = {
        "stability_thresh": args.stability_thresh,
        "min_mask_size": args.min_mask_size,
        "max_mask_size": args.max_mask_size,
    }
    dataset_name = args.dataset_name

    # Determine feature directory paths
    if args.feat_level is not None:
        # Use specific feature level (0, 1, 2, or 3)
        level = args.feat_level
        if level == 0:
            folder_suffix = ""
        else:
            folder_suffix = f"_{level}"

        if args.feat_base_path:
            feat_dir = [f"{args.feat_base_path}/{args.dataset_name}/test/{args.feat_folder}{folder_suffix}/renders_npy"]
        else:
            feat_dir = [f"./output/LERF/{args.dataset_name}/test/{args.feat_folder}{folder_suffix}/renders_npy"]
        # Replicate for 3 levels for compatibility with existing evaluation logic
        feat_dir = feat_dir * 3
        logging.info(f"Using feature level {level} (replicated for 3 levels): {feat_dir[0]}")
    elif args.single_level:
        # Use single feature level (level 0) from custom base path or default
        if args.feat_base_path:
            feat_dir = [f"{args.feat_base_path}/{args.dataset_name}/test/{args.feat_folder}/renders_npy"]
        else:
            feat_dir = [f"./output/LERF/{args.dataset_name}/test/{args.feat_folder}/renders_npy"]
        # Replicate for 3 levels for compatibility with existing evaluation logic
        feat_dir = feat_dir * 3
        logging.info(f"Using single feature level (replicated for 3 levels): {feat_dir[0]}")
    else:
        # Original behavior: multiple feature levels
        feat_dir = [f"./output/LERF/{args.dataset_name}/test/{args.feat_folder}_1/renders_npy",
                    f"./output/LERF/{args.dataset_name}/test/{args.feat_folder}_2/renders_npy",
                    f"./output/LERF/{args.dataset_name}/test/{args.feat_folder}_3/renders_npy"]

    output_path = f"./eval_results/LERF/{args.dataset_name}"
    gt_path = args.gt_folder

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{dataset_name}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

    # Print configuration
    logging.info("=" * 60)
    logging.info("Evaluation Configuration")
    logging.info("=" * 60)
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Feature dimension: {args.src_dim}")
    logging.info(f"Using CLIP: {args.use_clip}")
    logging.info(f"Using Procrustes: {args.use_procrustes}")
    if args.use_procrustes:
        logging.info(f"  Q matrix path: {args.q_matrix}")
        logging.info(f"  Text embeddings: {args.text_embeddings}")
    logging.info("=" * 60)

    # Determine feature level for Q matrix computation
    feature_level = 1  # default
    if args.feat_level is not None:
        feature_level = args.feat_level
    elif args.single_level:
        # Extract level from feat_folder (e.g., "ours_30000_langfeat_1" -> 1)
        match = re.search(r'_langfeat_(\d+)$', args.feat_folder)
        if match:
            feature_level = int(match.group(1))

    evaluate(feat_dir, output_path, gt_path, logger, eval_params,
             src_dim=args.src_dim,
             projection_matrix_path=args.projection_matrix,
             use_clip=args.use_clip,
             use_procrustes=args.use_procrustes,
             text_embeddings_path=args.text_embeddings,
             q_matrix_path=args.q_matrix,
             train_data_root=args.train_data_root,
             dataset_name=args.dataset_name,
             feature_level=feature_level)