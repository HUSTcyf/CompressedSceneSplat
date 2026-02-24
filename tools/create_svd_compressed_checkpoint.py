#!/usr/bin/env python3
"""
Create SVD-compressed checkpoint from existing checkpoint and SVD decomposition.

This script supports two modes:

1. SVD Decomposition Mode (default):
   - Loads SVD results from lang_feat_svd.npz (U, S, Vt matrices)
   - Loads existing checkpoint from gaussian_results/lerf_ovs/
   - Compresses language_features from 768-dim to 16-dim using SVD projection
   - Saves checkpoint_with_features.pth to output_features/{scene}/

2. Grid SVD Mode (--use-grid-svd):
   - Loads pre-compressed features from lang_feat_grid_svd_r16.npz
   - Loads existing checkpoint from gaussian_results/lerf_ovs/
   - Uses the compressed features directly (no additional compression needed)
   - Saves checkpoint_with_features.pth to output_features/{scene}/

Usage:
    # SVD decomposition mode
    python tools/create_svd_compressed_checkpoint.py --scene figurines
    python tools/create_svd_compressed_checkpoint.py --all-scenes

    # Grid SVD mode (pre-compressed features)
    python tools/create_svd_compressed_checkpoint.py --scene figurines --use-grid-svd
    python tools/create_svd_compressed_checkpoint.py --all-scenes --use-grid-svd

SVD compression formula:
    compressed_features = language_features @ Vt[:rank, :].T

This projects the 768-dim features onto the top-16 right singular vectors.
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, List, Optional


# Default paths
DATA_ROOT = "/new_data/cyf/projects/SceneSplat"
TRAIN_ROOT = f"{DATA_ROOT}/gaussian_train/lerf_ovs/train"
CHECKPOINT_ROOT = f"{DATA_ROOT}/gaussian_results/lerf_ovs"
OUTPUT_ROOT = f"{DATA_ROOT}/output_features"

# SVD compression rank
SVD_RANK = 16


def get_available_scenes(use_grid_svd: bool = False, rank: int = SVD_RANK) -> List[str]:
    """Get list of scenes that have both SVD files and checkpoints.

    Args:
        use_grid_svd: If True, look for lang_feat_grid_svd_r*.npz files
                     If False, look for lang_feat_svd.npz files
        rank: SVD rank for grid SVD mode (e.g., 16 for r16)

    Returns:
        List of scene names that have both required files
    """
    train_dir = Path(TRAIN_ROOT)
    checkpoint_dir = Path(CHECKPOINT_ROOT)

    # Find all scenes with SVD files
    svd_scenes = set()
    if train_dir.exists():
        if use_grid_svd:
            # Look for lang_feat_grid_svd_r*.npz files
            pattern = f"*/lang_feat_grid_svd_r{rank}.npz"
            for svd_file in train_dir.glob(pattern):
                svd_scenes.add(svd_file.parent.name)
        else:
            # Look for lang_feat_svd.npz files
            for svd_file in train_dir.glob("*/lang_feat_svd.npz"):
                svd_scenes.add(svd_file.parent.name)

    # Find all scenes with ckpts/chkpnt*.pth format
    ckpt_scenes = set()
    if checkpoint_dir.exists():
        for ckpt_dir in checkpoint_dir.glob("*/ckpts"):
            if ckpt_dir.is_dir():
                for ckpt_file in ckpt_dir.glob("chkpnt*.pth"):
                    ckpt_scenes.add(ckpt_dir.parent.name)

    # Return intersection
    available = sorted(list(svd_scenes & ckpt_scenes))
    return available


def load_grid_svd_compressed_features(scene: str, rank: int = SVD_RANK) -> np.ndarray:
    """Load pre-compressed SVD features from lang_feat_grid_svd_r*.npz.

    The grid SVD file contains:
    - compressed: [N_unique, rank] - unique compressed features
    - indices: [N_gaussians] - mapping from each gaussian to its compressed feature index

    The indices array maps each of the N_gaussians to one of the N_unique compressed features.
    Multiple gaussians can share the same compressed feature (grid-based compression).

    Args:
        scene: Scene name
        rank: SVD rank (e.g., 16 for r16)

    Returns:
        compressed_features: [N_gaussians, rank] - full compressed feature array
    """
    grid_svd_path = f"{TRAIN_ROOT}/{scene}/lang_feat_grid_svd_r{rank}.npz"

    if not os.path.exists(grid_svd_path):
        raise FileNotFoundError(f"Grid SVD file not found: {grid_svd_path}")

    print(f"Loading grid SVD from: {grid_svd_path}")
    data = np.load(grid_svd_path)

    compressed = data['compressed']  # [N_unique, rank] - unique compressed features
    indices = data['indices']  # [N_gaussians] - mapping from gaussian to compressed feature index

    print(f"  Compressed features (unique): {compressed.shape}")
    print(f"  Indices (mappings): {indices.shape}")
    print(f"  Total gaussians: {indices.shape[0]}")

    # Use fancy indexing to reconstruct full feature array
    # compressed[indices] selects the appropriate row from compressed for each gaussian
    full_compressed = compressed[indices]  # [N_gaussians, rank]

    print(f"  Full compressed features shape: {full_compressed.shape}")

    return full_compressed


def get_svd_compressed_features_from_original(
    scene: str,
    rank: int = SVD_RANK
) -> np.ndarray:
    """Load original features and compress them using SVD decomposition.

    This function:
    1. Loads original lang_feat.npy and valid_feat_mask.npy
    2. Loads SVD components (Vt) from lang_feat_svd.npz
    3. Compresses valid features using SVD projection
    4. Returns full compressed features array [N_gaussians, rank]

    Args:
        scene: Scene name
        rank: SVD compression rank (default 16)

    Returns:
        svd_compressed: [N_gaussians, rank] - SVD compressed features
    """
    # Load original features
    lang_feat_path = f"{TRAIN_ROOT}/{scene}/lang_feat.npy"
    valid_feat_mask_path = f"{TRAIN_ROOT}/{scene}/valid_feat_mask.npy"

    if not os.path.exists(lang_feat_path):
        raise FileNotFoundError(f"Language features not found: {lang_feat_path}")
    if not os.path.exists(valid_feat_mask_path):
        raise FileNotFoundError(f"Valid feature mask not found: {valid_feat_mask_path}")

    print(f"Loading original features from: {lang_feat_path}")
    original_features = np.load(lang_feat_path).astype(np.float32)  # [N_total, 768]

    print(f"Loading valid feature mask from: {valid_feat_mask_path}")
    original_valid_feat_mask = np.load(valid_feat_mask_path).astype(np.bool_)  # [N_original]
    N_original = original_valid_feat_mask.shape[0]

    print(f"  Original features shape: {original_features.shape}")
    print(f"  Original valid feat mask shape: {original_valid_feat_mask.shape}")

    # Check if lang_feat has been filtered (size mismatch with valid_feat_mask)
    if original_features.shape[0] != N_original:
        print(f"  Warning: lang_feat has been filtered ({original_features.shape[0]} != {N_original})")
        print(f"  All features in filtered lang_feat are valid (will be padded back to original size)")
        # Use all filtered features directly (no masking needed)
        valid_features = original_features
        # Track which original positions are valid (from original mask)
        valid_positions = original_valid_feat_mask  # [N_original]
    else:
        print(f"  Valid count: {original_valid_feat_mask.sum()}, Invalid count: {(~original_valid_feat_mask).sum()}")
        # Extract valid features using mask
        valid_features = original_features[original_valid_feat_mask]  # [N_valid, 768]
        valid_positions = original_valid_feat_mask

    print(f"  Valid features shape: {valid_features.shape}")

    # Load SVD components (only need Vt for compression)
    svd_path = f"{TRAIN_ROOT}/{scene}/lang_feat_svd.npz"
    if not os.path.exists(svd_path):
        raise FileNotFoundError(f"SVD file not found: {svd_path}")

    print(f"Loading SVD components from: {svd_path}")
    svd_data = np.load(svd_path)
    Vt = svd_data['Vt']  # [768, 768]

    # Compress using SVD projection
    Vt_r = Vt[:rank, :].T  # [768, rank]
    svd_compressed_valid = valid_features @ Vt_r  # [N_valid, rank]

    # Create full array with zeros for invalid positions (always use original size)
    svd_compressed = np.zeros((N_original, rank), dtype=np.float32)
    svd_compressed[valid_positions] = svd_compressed_valid

    print(f"  SVD compressed features shape: {svd_compressed.shape} (padded to original size {N_original})")

    return svd_compressed


def compare_compression_methods(
    scene: str,
    rank: int = SVD_RANK
) -> Dict[str, float]:
    """Compare Grid SVD vs SVD compression methods.

    This function:
    1. Loads features compressed using Grid SVD (from lang_feat_grid_svd_r*.npz)
    2. Computes features compressed using SVD (from original features + Vt)
    3. Calculates error metrics between the two methods
    4. Only compares valid features (where grid SVD has non-zero values)

    Args:
        scene: Scene name
        rank: SVD compression rank (default 16)

    Returns:
        Dictionary containing error metrics
    """
    print(f"\n{'='*60}")
    print(f"Comparing compression methods for scene: {scene} (rank={rank})")
    print(f"{'='*60}\n")

    # Load Grid SVD compressed features
    print("Loading Grid SVD compressed features...")
    grid_compressed = load_grid_svd_compressed_features(scene, rank)  # [N_gaussians, rank]

    # Load SVD compressed features from original
    print("\nLoading SVD compressed features from original...")
    svd_compressed = get_svd_compressed_features_from_original(scene, rank)  # [N_gaussians, rank]

    # Verify dimensions match
    if grid_compressed.shape != svd_compressed.shape:
        print(f"Warning: Shape mismatch!")
        print(f"  Grid SVD: {grid_compressed.shape}")
        print(f"  SVD: {svd_compressed.shape}")
        # Truncate to minimum size
        min_n = min(grid_compressed.shape[0], svd_compressed.shape[0])
        grid_compressed = grid_compressed[:min_n]
        svd_compressed = svd_compressed[:min_n]

    # Find valid positions (where grid SVD has non-zero values)
    # Grid SVD only has valid features; invalid ones are padded with zeros
    grid_norm = np.linalg.norm(grid_compressed, axis=1)
    valid_mask = grid_norm > 1e-10  # Positions where Grid SVD has non-zero values

    print(f"\nValid positions for comparison: {valid_mask.sum()} / {valid_mask.shape[0]}")

    # Extract only valid features for comparison
    grid_valid = grid_compressed[valid_mask]  # [N_valid, rank]
    svd_valid = svd_compressed[valid_mask]  # [N_valid, rank]

    print(f"  Grid SVD valid features shape: {grid_valid.shape}")
    print(f"  SVD valid features shape: {svd_valid.shape}")

    # Calculate error metrics
    difference = grid_valid - svd_valid

    # Mean Squared Error (MSE) - per feature and overall
    mse_per_feature = np.mean(difference ** 2, axis=0)
    mse = np.mean(mse_per_feature)

    # Root Mean Squared Error (RMSE)
    rmse_per_feature = np.sqrt(mse_per_feature)
    rmse = np.sqrt(mse)

    # Mean Absolute Error (MAE)
    mae_per_feature = np.mean(np.abs(difference), axis=0)
    mae = np.mean(mae_per_feature)

    # Cosine similarity (per feature vector)
    grid_norms = np.linalg.norm(grid_valid, axis=1, keepdims=True)
    svd_norms = np.linalg.norm(svd_valid, axis=1, keepdims=True)
    dot_products = np.sum(grid_valid * svd_valid, axis=1, keepdims=True)
    cosine_sim = dot_products / (grid_norms * svd_norms + 1e-10)
    mean_cosine_sim = np.mean(cosine_sim)

    # Relative error (L2 norm difference / L2 norm of SVD)
    svd_norm = np.linalg.norm(svd_valid)
    diff_norm = np.linalg.norm(difference)
    relative_error = diff_norm / (svd_norm + 1e-10)

    # Reconstruction quality metrics
    # Signal-to-noise ratio (SNR)
    signal_power = np.mean(svd_valid ** 2)
    noise_power = np.mean(difference ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Print results
    print(f"\n{'='*60}")
    print(f"Error Metrics (Grid SVD vs SVD Compression):")
    print(f"{'='*60}")

    print(f"\nOverall Metrics:")
    print(f"  MSE (Mean Squared Error): {mse:.6f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"  MAE (Mean Absolute Error): {mae:.6f}")
    print(f"  Relative Error: {relative_error:.6f} ({relative_error*100:.2f}%)")
    print(f"  Mean Cosine Similarity: {mean_cosine_sim:.6f}")
    print(f"  SNR (Signal-to-Noise Ratio): {snr_db:.2f} dB")

    print(f"\nPer-Feature Metrics (first {min(16, rank)} dimensions):")
    print(f"  {'Dim':<4} {'MSE':<10} {'RMSE':<10} {'MAE':<10}")
    print(f"  {'-'*42}")
    for i in range(min(16, rank)):
        print(f"  {i:<4} {mse_per_feature[i]:<10.6f} {rmse_per_feature[i]:<10.6f} {mae_per_feature[i]:<10.6f}")

    print(f"\nStatistics Summary:")
    print(f"  Valid features compared: {valid_mask.sum()}")
    print(f"  Compression rank: {rank}")
    print(f"  Grid SVD unique features: {len(np.unique(np.load(f'{TRAIN_ROOT}/{scene}/lang_feat_grid_svd_r{rank}.npz')['indices']))}")

    print(f"\n{'='*60}\n")

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'relative_error': float(relative_error),
        'mean_cosine_similarity': float(mean_cosine_sim),
        'snr_db': float(snr_db),
        'valid_count': int(valid_mask.sum()),
        'rank': rank,
    }


def load_svd_components(scene: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SVD components from lang_feat_svd.npz.

    Returns:
        U: [N, 768] - left singular vectors
        S: [768] - singular values
        Vt: [768, 768] - right singular vectors (transposed)
    """
    svd_path = f"{TRAIN_ROOT}/{scene}/lang_feat_svd.npz"

    if not os.path.exists(svd_path):
        raise FileNotFoundError(f"SVD file not found: {svd_path}")

    print(f"Loading SVD from: {svd_path}")
    data = np.load(svd_path)
    U = data['U']  # [N, 768]
    S = data['S']  # [768]
    Vt = data['Vt']  # [768, 768]

    print(f"  U shape: {U.shape}")
    print(f"  S shape: {S.shape}")
    print(f"  Vt shape: {Vt.shape}")
    print(f"  Singular values (first 10): {S[:10]}")

    return U, S, Vt


def find_checkpoint(scene: str, preferred_iteration: Optional[int] = None) -> str:
    """Find checkpoint file for the given scene.

    Args:
        scene: Scene name
        preferred_iteration: Preferred iteration number (e.g., 30000)

    Returns:
        Path to the checkpoint file
    """
    scene_dir = Path(CHECKPOINT_ROOT) / scene

    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    # Look for checkpoints in ckpts/ subdirectory
    ckpts_dir = scene_dir / "ckpts"
    if not ckpts_dir.exists():
        raise FileNotFoundError(f"ckpts/ directory not found: {ckpts_dir}")

    # Look for specific iteration
    if preferred_iteration is not None:
        preferred_ckpt = ckpts_dir / f"chkpnt{preferred_iteration}.pth"
        if preferred_ckpt.exists():
            return str(preferred_ckpt)

    # Find all checkpoints in ckpts/
    ckpt_files = sorted(ckpts_dir.glob("chkpnt*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in: {ckpts_dir}")

    # Return the highest iteration checkpoint
    return str(ckpt_files[-1])


def load_checkpoint(ckpt_path: str, scene: str) -> Tuple[Tuple, int]:
    """Load checkpoint file (gsplat format).

    Returns:
        model_params: 13-element tuple in capture_language_feature() format
        iteration: iteration number
    """
    print(f"Loading checkpoint from: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # GSPlat format: dict with 'splats' key
    if isinstance(checkpoint, dict) and 'splats' in checkpoint:
        print(f"  Detected gsplat format checkpoint")
        iteration = checkpoint.get('step', 0)
        splats = checkpoint['splats']

        # Extract gsplat components
        xyz = splats['means']  # [N, 3]
        opacity = splats['opacities']
        if opacity.dim() == 1:
            opacity = opacity.unsqueeze(-1)
        features_dc = splats['sh0'].squeeze(1)  # [N, 3]
        features_rest = splats['shN'].reshape(xyz.shape[0], -1)  # [N, 45]
        scaling = splats['scales']
        rotation = splats['quats']

        N = xyz.shape[0]
        print(f"  Extracted {N} Gaussians from gsplat checkpoint")

        # Create placeholders for GaussianModel state
        active_sh_degree = 3
        max_radii2D = torch.zeros(N, dtype=torch.int32)
        xyz_gradient_accum = torch.zeros(N, 1, dtype=torch.float32)
        denom = torch.zeros(N, 1, dtype=torch.float32)
        opt_dict = {}
        spatial_lr_scale = 1.0

        # Load language features from dataset (will be compressed later)
        lang_feat_path = f"{TRAIN_ROOT}/{scene}/lang_feat.npy"
        if not os.path.exists(lang_feat_path):
            raise FileNotFoundError(f"Language features not found: {lang_feat_path}")

        print(f"  Loading language features from: {lang_feat_path}")
        language_features = torch.from_numpy(np.load(lang_feat_path).astype(np.float32))
        print(f"  Language features shape: {language_features.shape}")

        # Verify dimensions match
        if language_features.shape[0] != N:
            print(f"  Warning: Language feature count ({language_features.shape[0]}) != Gaussian count ({N})")

        # Load valid_feat_mask to ensure proper feature alignment
        valid_feat_mask_path = f"{TRAIN_ROOT}/{scene}/valid_feat_mask.npy"
        if os.path.exists(valid_feat_mask_path):
            print(f"  Loading valid_feat_mask from: {valid_feat_mask_path}")
            valid_feat_mask = torch.from_numpy(np.load(valid_feat_mask_path).astype(np.bool_))
            print(f"  Valid feat mask shape: {valid_feat_mask.shape}")
            print(f"  Valid count: {valid_feat_mask.sum().item()}, Invalid count: {(~valid_feat_mask).sum().item()}")
        else:
            print(f"  Warning: valid_feat_mask not found at {valid_feat_mask_path}, assuming all features are valid")
            valid_feat_mask = torch.ones(language_features.shape[0], dtype=torch.bool)

        # Create 13-element tuple in capture_language_feature() format
        model_params = (
            active_sh_degree,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            language_features,  # [N, 768] - will be compressed later
            max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            spatial_lr_scale,
            valid_feat_mask,  # [N] - boolean mask for valid features
        )

        return model_params, iteration

    raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}. Expected gsplat format with 'splats' key.")


def compress_language_features(
    language_features: torch.Tensor,
    Vt: np.ndarray,
    valid_feat_mask: torch.Tensor,
    rank: int = SVD_RANK
) -> torch.Tensor:
    """Compress language features using SVD projection.

    SVD was computed on valid features only. This function:
    1. Extracts valid features from language_features
    2. Compresses them using SVD projection
    3. Creates [N_original, rank] array filled with zeros
    4. Fills valid positions with compressed features
    5. Invalid positions remain zero

    Args:
        language_features: [N, 768] - features (may be pre-filtered)
        Vt: [768, 768] - right singular vectors from SVD (computed on valid features)
        valid_feat_mask: [N_original] - boolean mask for valid features (original size)
        rank: compression rank (default 16)

    Returns:
        compressed_features: [N_original, rank] - compressed features (invalid positions are 0)
    """
    print(f"Compressing language features from {language_features.shape[1]} to {rank} dimensions...")

    N_features = language_features.shape[0]
    N_original = valid_feat_mask.shape[0]
    valid_count = valid_feat_mask.sum().item()
    invalid_count = (~valid_feat_mask).sum().item()

    print(f"  Language features size: {N_features}")
    print(f"  Original size (from mask): {N_original}")
    print(f"  Valid features: {valid_count}, Invalid features: {invalid_count}")

    # Check if features have been pre-filtered
    if N_features != N_original:
        print(f"  Warning: Features are pre-filtered ({N_features} != {N_original})")
        print(f"  Using all features directly and padding to original size...")
        # Use all features (they're already filtered to contain only valid ones)
        valid_features = language_features  # [N_features, 768]
        compressed_valid = valid_features.numpy() @ Vt[:rank, :].T  # [N_features, rank]
    else:
        # Extract only valid features for compression
        valid_features = language_features[valid_feat_mask]  # [M, 768] where M = valid_count
        print(f"  Extracted valid features shape: {valid_features.shape}")
        # Compress valid features using SVD projection
        compressed_valid = valid_features.numpy() @ Vt[:rank, :].T  # [M, rank]

    # Convert to tensor
    compressed_valid_tensor = torch.from_numpy(compressed_valid.astype(np.float32))
    print(f"  Compressed valid features shape: {compressed_valid_tensor.shape}")

    # Create full [N_original, rank] array filled with zeros (always use original size)
    compressed_tensor = torch.zeros(N_original, rank, dtype=torch.float32)

    # Fill valid positions with compressed features
    compressed_tensor[valid_feat_mask] = compressed_valid_tensor

    print(f"  Final compressed features shape: {compressed_tensor.shape} (padded to {N_original})")
    print(f"  Invalid positions ({invalid_count}) filled with 0")

    # Compute compression stats (use original size for accurate ratio)
    original_size = N_original * 768 * 4  # float32
    compressed_size = compressed_tensor.numel() * 4
    compression_ratio = original_size / compressed_size

    print(f"  Original size: {original_size / 1024**2:.2f} MB")
    print(f"  Compressed size: {compressed_size / 1024**2:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")

    return compressed_tensor


def create_compressed_checkpoint(
    model_params: Tuple,
    compressed_features: torch.Tensor
) -> Tuple:
    """Create a new checkpoint with compressed language features.

    Args:
        model_params: Original tuple (may be 13 or 14 elements with valid_feat_mask)
        compressed_features: Compressed language features

    Returns:
        new_model_params: 13-element tuple with compressed features (no valid_feat_mask)
    """
    # Unpack the original tuple, handling both 13 and 14 element cases
    if len(model_params) == 14:
        # Input has valid_feat_mask, but we don't include it in output
        (active_sh_degree, xyz, features_dc, features_rest,
         scaling, rotation, opacity, _language_features,
         max_radii2D, xyz_gradient_accum, denom,
         opt_dict, spatial_lr_scale, _valid_feat_mask) = model_params
        tmp_feats = torch.zeros(_valid_feat_mask.shape[0], compressed_features.shape[1], dtype=torch.float32, device=compressed_features.device)  # Placeholder to maintain tuple structure
        tmp_feats[_valid_feat_mask] = compressed_features
        compressed_features = tmp_feats  # Use the padded version for the checkpoint
        print(f"  Created padded compressed features with shape: {compressed_features.shape} for checkpoint")
    elif len(model_params) == 13:
        # Input doesn't have valid_feat_mask
        (active_sh_degree, xyz, features_dc, features_rest,
         scaling, rotation, opacity, _language_features,
         max_radii2D, xyz_gradient_accum, denom,
         opt_dict, spatial_lr_scale) = model_params
    else:
        raise ValueError(f"Unexpected model_params length: {len(model_params)}, expected 13 or 14")

    # Create new 13-element tuple with compressed features
    new_model_params = (
        active_sh_degree,
        xyz,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity,
        compressed_features,  # Replace with compressed features
        max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        spatial_lr_scale,
    )

    return new_model_params


def save_checkpoint(
    model_params: Tuple,
    iteration: int,
    save_path: str
):
    """Save checkpoint in GaussianModel format.

    Args:
        model_params: 13-element tuple
        iteration: iteration number
        save_path: output file path
    """
    # Create output directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save in format: ((13-element tuple), iteration)
    torch.save((model_params, iteration), save_path)

    print(f"Saved checkpoint to: {save_path}")
    print(f"  Format: ((13-element tuple), iteration={iteration})")

    # Print file size
    file_size = os.path.getsize(save_path)
    print(f"  File size: {file_size / 1024**2:.2f} MB")


def process_scene(
    scene: str,
    iteration: Optional[int] = None,
    rank: int = SVD_RANK,
    output_dir: Optional[str] = None
):
    """Process a single scene.

    Args:
        scene: Scene name
        iteration: Preferred checkpoint iteration (None for latest)
        rank: SVD compression rank
        output_dir: Output directory (default OUTPUT_ROOT)
    """
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene}")
    print(f"{'='*60}\n")

    # Load SVD components
    U, S, Vt = load_svd_components(scene)

    # Load checkpoint
    ckpt_path = find_checkpoint(scene, iteration)
    model_params, iteration_num = load_checkpoint(ckpt_path, scene)

    # Extract language features (element 7 of the tuple)
    original_features = model_params[7]
    print(f"\nOriginal language features shape: {original_features.shape}")

    # Extract valid_feat_mask (element 13 of the tuple)
    if len(model_params) >= 14:
        valid_feat_mask = model_params[13]
        print(f"Valid feat mask: {valid_feat_mask.sum().item()} valid, {(~valid_feat_mask).sum().item()} invalid")
    else:
        print("Warning: valid_feat_mask not found in model_params, assuming all features are valid")
        valid_feat_mask = torch.ones(original_features.shape[0], dtype=torch.bool)

    # Verify dimensions match
    if original_features.shape[1] != 768:
        print(f"Warning: Expected 768-dim features, got {original_features.shape[1]}")
        print("Continuing anyway...")

    # Compress language features (only valid features, then pad with zeros for invalid)
    compressed_features = compress_language_features(original_features, Vt, valid_feat_mask, rank)

    # Create new checkpoint with compressed features
    new_model_params = create_compressed_checkpoint(model_params, compressed_features)

    # Save checkpoint
    if output_dir is None:
        output_dir = OUTPUT_ROOT
    save_path = f"{output_dir}/{scene}/checkpoint_with_features.pth"
    save_checkpoint(new_model_params, iteration_num, save_path)

    # Also save just the language features as .npy for convenience
    lang_feat_path = f"{output_dir}/{scene}/language_features.npy"
    np.save(lang_feat_path, compressed_features.numpy())
    print(f"Also saved language features to: {lang_feat_path}")

    print(f"\n{'='*60}")
    print(f"Scene {scene} processed successfully!")
    print(f"{'='*60}\n")


def process_scene_grid_svd(
    scene: str,
    iteration: Optional[int] = None,
    rank: int = SVD_RANK,
    output_dir: Optional[str] = None
):
    """Process a single scene using pre-compressed grid SVD features.

    This mode:
    1. Loads pre-compressed features from lang_feat_grid_svd_r*.npz
    2. Loads checkpoint from gaussian_results/lerf_ovs/
    3. Replaces language features with pre-compressed features
    4. Saves checkpoint with compressed features

    Args:
        scene: Scene name
        iteration: Preferred checkpoint iteration (None for latest)
        rank: SVD compression rank (default 16)
        output_dir: Output directory (default OUTPUT_ROOT)
    """
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene} (Grid SVD mode, rank={rank})")
    print(f"{'='*60}\n")

    # Load pre-compressed grid SVD features
    compressed_features = load_grid_svd_compressed_features(scene, rank)

    # Convert to torch tensor
    compressed_features_tensor = torch.from_numpy(compressed_features)
    print(f"Converted to tensor: {compressed_features_tensor.shape}")

    # Load checkpoint
    ckpt_path = find_checkpoint(scene, iteration)
    model_params, iteration_num = load_checkpoint(ckpt_path, scene)

    # Verify dimensions match
    original_features = model_params[7]
    print(f"\nCheckpoint language features shape: {original_features.shape}")
    print(f"Grid SVD compressed features shape: {compressed_features_tensor.shape}")

    if compressed_features_tensor.shape[0] != original_features.shape[0]:
        print(f"Warning: Feature count mismatch!")
        print(f"  Checkpoint has {original_features.shape[0]} features")
        print(f"  Grid SVD has {compressed_features_tensor.shape[0]} features")

    # Create new checkpoint with compressed features
    new_model_params = create_compressed_checkpoint(model_params, compressed_features_tensor)

    # Save checkpoint
    if output_dir is None:
        output_dir = OUTPUT_ROOT
    save_path = f"{output_dir}/{scene}/checkpoint_with_features.pth"
    save_checkpoint(new_model_params, iteration_num, save_path)

    # Also save just the language features as .npy for convenience
    lang_feat_path = f"{output_dir}/{scene}/language_features.npy"
    np.save(lang_feat_path, compressed_features_tensor.numpy())
    print(f"Also saved language features to: {lang_feat_path}")

    # Print compression stats
    original_size = original_features.numel() * 4  # float32
    compressed_size = compressed_features_tensor.numel() * 4
    compression_ratio = original_size / compressed_size

    print(f"\nCompression statistics:")
    print(f"  Original size: {original_size / 1024**2:.2f} MB ({original_features.shape})")
    print(f"  Compressed size: {compressed_size / 1024**2:.2f} MB ({compressed_features_tensor.shape})")
    print(f"  Compression ratio: {compression_ratio:.2f}x")

    print(f"\n{'='*60}")
    print(f"Scene {scene} processed successfully! (Grid SVD mode)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create SVD-compressed checkpoint with language features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single scene (SVD decomposition mode)
  python tools/create_svd_compressed_checkpoint.py --scene figurines

  # Process with specific checkpoint iteration
  python tools/create_svd_compressed_checkpoint.py --scene figurines --iteration 30000

  # Process all available scenes
  python tools/create_svd_compressed_checkpoint.py --all-scenes

  # Use different compression rank
  python tools/create_svd_compressed_checkpoint.py --scene figurines --rank 32

  # Use pre-compressed grid SVD features
  python tools/create_svd_compressed_checkpoint.py --scene figurines --use-grid-svd
  python tools/create_svd_compressed_checkpoint.py --scene figurines --use-grid-svd --rank 8

  # Compare Grid SVD vs SVD compression error
  python tools/create_svd_compressed_checkpoint.py --scene figurines --compare
  python tools/create_svd_compressed_checkpoint.py --all-scenes --compare --rank 8
        """
    )

    parser.add_argument(
        "--scene",
        type=str,
        help="Scene name to process"
    )
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Process all available scenes"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Preferred checkpoint iteration (default: use latest)"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=SVD_RANK,
        help=f"SVD compression rank (default: {SVD_RANK})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_ROOT,
        help=f"Output directory (default: {OUTPUT_ROOT})"
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List available scenes and exit"
    )
    parser.add_argument(
        "--use-grid-svd",
        action="store_true",
        help="Use pre-computed grid SVD features (lang_feat_grid_svd_r*.npz) instead of computing from SVD decomposition"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare Grid SVD vs SVD compression error (requires both lang_feat_grid_svd_r*.npz and lang_feat_svd.npz)"
    )

    args = parser.parse_args()

    # List available scenes
    if args.list_scenes:
        scenes = get_available_scenes(use_grid_svd=args.use_grid_svd, rank=args.rank)
        mode = "Grid SVD" if args.use_grid_svd else "SVD Decomposition"
        print(f"Available scenes ({mode} mode, rank={args.rank}):")
        for scene in scenes:
            print(f"  - {scene}")
        return

    # Compare mode
    if args.compare:
        # For compare mode, we need both grid SVD and regular SVD files
        if not args.scene and not args.all_scenes:
            parser.error("Either --scene or --all-scenes must be specified with --compare")

        if args.scene:
            try:
                compare_compression_methods(args.scene, args.rank)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("\nNote: --compare requires both lang_feat_grid_svd_r*.npz and lang_feat_svd.npz files")
        else:
            # Find scenes with both types of files
            train_dir = Path(TRAIN_ROOT)
            scenes_with_both = []
            for scene_dir in train_dir.iterdir():
                if scene_dir.is_dir():
                    scene = scene_dir.name
                    grid_file = scene_dir / f"lang_feat_grid_svd_r{args.rank}.npz"
                    svd_file = scene_dir / "lang_feat_svd.npz"
                    if grid_file.exists() and svd_file.exists():
                        scenes_with_both.append(scene)

            scenes_with_both = sorted(scenes_with_both)

            if not scenes_with_both:
                print(f"No scenes found with both lang_feat_grid_svd_r{args.rank}.npz and lang_feat_svd.npz files!")
                return

            print(f"Found {len(scenes_with_both)} scenes to compare:")
            for scene in scenes_with_both:
                print(f"  - {scene}")
            print()

            all_results = {}
            for scene in scenes_with_both:
                try:
                    result = compare_compression_methods(scene, args.rank)
                    all_results[scene] = result
                except Exception as e:
                    print(f"Error processing scene {scene}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Print summary table
            if all_results:
                print(f"\n{'='*80}")
                print(f"Summary Comparison (rank={args.rank}):")
                print(f"{'='*80}")
                print(f"{'Scene':<20} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'Cosine':<10} {'SNR (dB)':<10}")
                print(f"{'-'*80}")
                for scene, result in all_results.items():
                    print(f"{scene:<20} {result['mse']:<10.6f} {result['rmse']:<10.6f} "
                          f"{result['mae']:<10.6f} {result['mean_cosine_similarity']:<10.6f} "
                          f"{result['snr_db']:<10.2f}")

                # Calculate averages
                avg_mse = np.mean([r['mse'] for r in all_results.values()])
                avg_rmse = np.mean([r['rmse'] for r in all_results.values()])
                avg_mae = np.mean([r['mae'] for r in all_results.values()])
                avg_cosine = np.mean([r['mean_cosine_similarity'] for r in all_results.values()])
                avg_snr = np.mean([r['snr_db'] for r in all_results.values()])

                print(f"{'-'*80}")
                print(f"{'AVERAGE':<20} {avg_mse:<10.6f} {avg_rmse:<10.6f} {avg_mae:<10.6f} {avg_cosine:<10.6f} {avg_snr:<10.2f}")
                print(f"{'='*80}\n")
        return

    # Validate arguments for normal processing
    if not args.scene and not args.all_scenes:
        parser.error("Either --scene or --all-scenes must be specified")

    if args.scene and args.all_scenes:
        parser.error("Cannot specify both --scene and --all-scenes")

    # Choose processing function based on mode
    process_func = process_scene_grid_svd if args.use_grid_svd else process_scene

    # Process scenes
    if args.scene:
        process_func(args.scene, args.iteration, args.rank, args.output_dir)
    else:
        scenes = get_available_scenes(use_grid_svd=args.use_grid_svd, rank=args.rank)
        if not scenes:
            mode = "Grid SVD" if args.use_grid_svd else "SVD Decomposition"
            print(f"No scenes found with both {mode.lower()} files and checkpoints!")
            return

        mode = "Grid SVD" if args.use_grid_svd else "SVD Decomposition"
        print(f"Found {len(scenes)} scenes to process ({mode} mode, rank={args.rank}):")
        for scene in scenes:
            print(f"  - {scene}")
        print()

        for scene in scenes:
            try:
                process_func(scene, args.iteration, args.rank, args.output_dir)
            except Exception as e:
                print(f"Error processing scene {scene}: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()
