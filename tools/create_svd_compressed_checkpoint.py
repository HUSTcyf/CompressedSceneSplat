#!/usr/bin/env python3
"""
Create SVD-compressed checkpoint from existing checkpoint and SVD decomposition.

This script supports three modes:

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

3. Direct Mode (--direct):
   - Loads all Gaussian parameters directly from TRAIN_ROOT (coord, color, opacity, quat, scale)
   - Loads language features from TRAIN_ROOT
   - Creates checkpoint without requiring CHECKPOINT_ROOT
   - Saves checkpoint_with_features.pth to output_features/{scene}/

Usage:
    # SVD decomposition mode
    python tools/create_svd_compressed_checkpoint.py --scene figurines
    python tools/create_svd_compressed_checkpoint.py --all-scenes

    # Grid SVD mode (pre-compressed features)
    python tools/create_svd_compressed_checkpoint.py --scene figurines --use-grid-svd
    python tools/create_svd_compressed_checkpoint.py --all-scenes --use-grid-svd

    # Direct mode (no checkpoint required)
    python tools/create_svd_compressed_checkpoint.py --scene figurines --direct
    python tools/create_svd_compressed_checkpoint.py --all-scenes --direct

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
dataset = "lerf_ovs"
# dataset = "3DOVS"
DATA_ROOT = "/new_data/cyf/projects/SceneSplat"
TRAIN_ROOT = f"{DATA_ROOT}/gaussian_train_clip/{dataset}/train"
CHECKPOINT_ROOT = f"{DATA_ROOT}/gaussian_results/{dataset}"
OUTPUT_ROOT = f"{DATA_ROOT}/output_features"

# OccamLGS paths
OCCAMLGS_OUTPUT_ROOT = "/new_data/cyf/projects/OccamLGS/output/LERF"
OCCAMLGS_BASE_CHECKPOINT_ITER = 30000  # Default iteration for base checkpoint (without language features)

# SVD compression rank
SVD_RANK = 16


def get_available_scenes(
    use_grid_svd: bool = False,
    rank: int = SVD_RANK,
    feature_level: Optional[int] = None,
    direct_mode: bool = False,
    occamlgs_mode: bool = False
) -> List[str]:
    """Get list of scenes that have required files.

    Args:
        use_grid_svd: If True, look for lang_feat_grid_svd_r*.npz files
                     If False, look for lang_feat_svd.npz files
        rank: SVD rank for grid SVD mode (e.g., 16 for r16)
        feature_level: Feature level number (1, 2, 3, etc.) for sequence-based files
        direct_mode: If True, only check for Gaussian parameter files in TRAIN_ROOT
                    If False, require both SVD files and checkpoints

    Returns:
        List of scene names that have both required files
    """
    if occamlgs_mode:
        # For OccamLGS mode, look for checkpoint files in OCCAMLGS_OUTPUT_ROOT
        occamlgs_dir = Path(OCCAMLGS_OUTPUT_ROOT)
        if not occamlgs_dir.exists():
            return []

        scenes = []
        for scene_dir in occamlgs_dir.iterdir():
            if scene_dir.is_dir():
                # Look for chkpnt*_langfeat_*.pth files
                for ckpt_file in scene_dir.glob("chkpnt*_langfeat_*.pth"):
                    scenes.append(scene_dir.name)
                    break

        return sorted(scenes)

    train_dir = Path(TRAIN_ROOT)
    checkpoint_dir = Path(CHECKPOINT_ROOT)

    # Build sequence suffix
    seq_suffix = f"_{feature_level}" if feature_level is not None else ""

    # Find all scenes with SVD files
    svd_scenes = set()
    if train_dir.exists():
        if use_grid_svd:
            # Look for lang_feat_grid_svd_r*_*.npz files (with sequence suffix)
            pattern = f"*/lang_feat_grid_svd_r{rank}{seq_suffix}.npz"
            for svd_file in train_dir.glob(pattern):
                svd_scenes.add(svd_file.parent.name)
        else:
            # Look for lang_feat_svd.npz files (no sequence support for regular SVD)
            for svd_file in train_dir.glob("*/lang_feat_svd.npz"):
                svd_scenes.add(svd_file.parent.name)

    # Direct mode: only require Gaussian parameter files in TRAIN_ROOT
    if direct_mode:
        # Filter scenes that have all required Gaussian parameter files
        available_scenes = []
        required_files = ['coord.npy', 'color.npy', 'opacity.npy', 'quat.npy', 'scale.npy']
        if use_grid_svd:
            # For grid SVD mode, also require the grid SVD file
            required_files.append(f"lang_feat_grid_svd_r{rank}{seq_suffix}.npz")
        else:
            # For regular SVD mode, also require lang_feat and SVD file
            required_files.append(f"lang_feat{seq_suffix}.npy")
            # valid_feat_mask is now unified (no level suffix)
            required_files.append("valid_feat_mask.npy")
            required_files.append("lang_feat_svd.npz")

        for scene in svd_scenes:
            scene_dir = train_dir / scene
            if all((scene_dir / f).exists() for f in required_files):
                available_scenes.append(scene)

        return sorted(available_scenes)

    # Normal mode: Find all scenes with ckpts/chkpnt*.pth format
    ckpt_scenes = set()
    if checkpoint_dir.exists():
        for ckpt_dir in checkpoint_dir.glob("*/ckpts"):
            if ckpt_dir.is_dir():
                for ckpt_file in ckpt_dir.glob("chkpnt*.pth"):
                    ckpt_scenes.add(ckpt_dir.parent.name)

    # Return intersection
    available = sorted(list(svd_scenes & ckpt_scenes))
    return available


def load_grid_svd_compressed_features(scene: str, rank: int = SVD_RANK, feature_level: Optional[int] = None) -> np.ndarray:
    """Load pre-compressed SVD features from lang_feat_grid_svd_r*.npz.

    The grid SVD file contains:
    - compressed: [N_unique, rank] - unique compressed features
    - indices: [N_gaussians] - mapping from each gaussian to its compressed feature index

    The indices array maps each of the N_gaussians to one of the N_unique compressed features.
    Multiple gaussians can share the same compressed feature (grid-based compression).

    Args:
        scene: Scene name
        rank: SVD rank (e.g., 16 for r16)
        feature_level: Feature level number (1, 2, 3, etc.) for sequence-based files

    Returns:
        compressed_features: [N_gaussians, rank] - full compressed feature array
    """
    # Build sequence suffix
    seq_suffix = f"_{feature_level}" if feature_level is not None else ""
    grid_svd_path = f"{TRAIN_ROOT}/{scene}/lang_feat_grid_svd_r{rank}{seq_suffix}.npz"

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
    rank: int = SVD_RANK,
    feature_level: Optional[int] = None
) -> np.ndarray:
    """Load original features and compress them using SVD decomposition.

    This function:
    1. Loads original lang_feat_{seq}.npy and valid_feat_mask_{seq}.npy
    2. Loads SVD components (Vt) from lang_feat_svd.npz
    3. Compresses valid features using SVD projection
    4. Returns full compressed features array [N_gaussians, rank]

    Args:
        scene: Scene name
        rank: SVD compression rank (default 16)
        feature_level: Feature level number (1, 2, 3, etc.) for sequence-based files

    Returns:
        svd_compressed: [N_gaussians, rank] - SVD compressed features
    """
    # Build sequence suffix
    seq_suffix = f"_{feature_level}" if feature_level is not None else ""

    # Load original features
    lang_feat_path = f"{TRAIN_ROOT}/{scene}/lang_feat{seq_suffix}.npy"
    # valid_feat_mask is now unified (no level suffix)
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
    rank: int = SVD_RANK,
    feature_level: Optional[int] = None
) -> Dict[str, float]:
    """Compare Grid SVD vs SVD compression methods.

    This function:
    1. Loads features compressed using Grid SVD (from lang_feat_grid_svd_r*_*.npz)
    2. Computes features compressed using SVD (from original features + Vt)
    3. Calculates error metrics between the two methods
    4. Only compares valid features (where grid SVD has non-zero values)

    Args:
        scene: Scene name
        rank: SVD compression rank (default 16)
        feature_level: Feature level number (1, 2, 3, etc.) for sequence-based files

    Returns:
        Dictionary containing error metrics
    """
    print(f"\n{'='*60}")
    print(f"Comparing compression methods for scene: {scene} (rank={rank}, seq={feature_level})")
    print(f"{'='*60}\n")

    # Load Grid SVD compressed features
    print("Loading Grid SVD compressed features...")
    grid_compressed = load_grid_svd_compressed_features(scene, rank, feature_level)  # [N_gaussians, rank]

    # Load SVD compressed features from original
    print("\nLoading SVD compressed features from original...")
    svd_compressed = get_svd_compressed_features_from_original(scene, rank, feature_level)  # [N_gaussians, rank]

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
    # Build sequence suffix for file path
    seq_suffix = f"_{feature_level}" if feature_level is not None else ""
    print(f"  Grid SVD unique features: {len(np.unique(np.load(f'{TRAIN_ROOT}/{scene}/lang_feat_grid_svd_r{rank}{seq_suffix}.npz')['indices']))}")

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


def find_occamlgs_base_checkpoint(scene: str, iteration: int = 30000) -> str:
    """Find OccamLGS base checkpoint file (without language features).

    OccamLGS base checkpoint naming: chkpnt{iteration}.pth (no language features)

    Args:
        scene: Scene name
        iteration: Iteration number (default 30000)

    Returns:
        Path to the OccamLGS base checkpoint file
    """
    ckpt_path = f"{OCCAMLGS_OUTPUT_ROOT}/{scene}/chkpnt{iteration}.pth"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"OccamLGS base checkpoint not found: {ckpt_path}")

    return ckpt_path


def load_occamlgs_base_checkpoint(ckpt_path: str) -> Tuple[Tuple, int]:
    """Load OccamLGS base checkpoint file (without language features).

    OccamLGS base checkpoint format (12 elements, NO language features):
    - [0]: active_sh_degree (int)
    - [1]: xyz (N, 3)
    - [2]: features_dc (N, 1, 3) - correct SH format
    - [3]: features_rest (N, 15, 3) - correct SH format for sh_degree=3
    - [4]: scaling (N, 3)
    - [5]: rotation (N, 4)
    - [6]: opacity (N, 1)
    - [7]: max_radii2D (N,)
    - [8]: xyz_gradient_accum (N, 1)
    - [9]: denom (N, 1)
    - [10]: opt_dict (dict)
    - [11]: spatial_lr_scale (float)

    Args:
        ckpt_path: Path to OccamLGS base checkpoint file

    Returns:
        model_params: 12-element tuple in OccamLGS base format (no language features)
        iteration: iteration number
    """
    print(f"Loading OccamLGS base checkpoint from: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # OccamLGS format: (model_params, iteration) where model_params is a tuple
    if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        model_params, iteration = checkpoint
        print(f"  Detected OccamLGS base format: (model_params, iteration={iteration})")
        print(f"  Model params length: {len(model_params)}")

        if len(model_params) != 12:
            raise ValueError(f"Expected 12-element OccamLGS base tuple, got {len(model_params)} elements")

        # Print key info
        active_sh_degree = model_params[0]
        xyz = model_params[1]
        features_dc = model_params[2]

        print(f"  active_sh_degree: {active_sh_degree}")
        print(f"  xyz shape: {xyz.shape}")
        print(f"  features_dc shape: {features_dc.shape} (should be [N, 1, 3])")
        print(f"  Note: This base checkpoint has NO language features (12 elements)")

        return model_params, iteration

    raise ValueError(f"Unexpected OccamLGS base checkpoint format: type={type(checkpoint)}, len={len(checkpoint) if isinstance(checkpoint, (list, tuple)) else 'N/A'}")


def add_language_features_to_base_checkpoint(
    base_model_params: Tuple,
    language_features: torch.Tensor
) -> Tuple:
    """Add language features to base checkpoint (12-element -> 13-element tuple).

    Args:
        base_model_params: 12-element tuple (no language features)
        language_features: [N, D] - language features to add

    Returns:
        model_params: 13-element tuple with language features

    Raises:
        ValueError: If dimension mismatch between base checkpoint and language features
    """
    if len(base_model_params) != 12:
        raise ValueError(f"Expected 12-element base tuple, got {len(base_model_params)} elements")

    # Unpack the 12-element tuple
    (active_sh_degree, xyz, features_dc, features_rest,
     scaling, rotation, opacity,
     max_radii2D, xyz_gradient_accum, denom,
     opt_dict, spatial_lr_scale) = base_model_params

    # Check dimension mismatch - raise error if mismatch
    N_base = xyz.shape[0]
    N_features = language_features.shape[0]

    if N_base != N_features:
        raise ValueError(
            f"Dimension mismatch: base checkpoint has {N_base} Gaussians "
            f"but language features have {N_features} features. "
            f"Please ensure the base checkpoint and grid SVD features are from the same training state."
        )

    # Create 13-element tuple by inserting language features
    # Position 7 is after opacity, before max_radii2D
    model_params_with_lang = (
        active_sh_degree,
        xyz,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity,
        language_features,  # Position 7: NEW language features
        max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        spatial_lr_scale,
    )

    print(f"  Created 13-element tuple with language features: {language_features.shape}")

    return model_params_with_lang


def find_occamlgs_checkpoint(scene: str, iteration: int = 30000, feature_level: int = 1) -> str:
    """Find OccamLGS checkpoint file for the given scene.

    OccamLGS checkpoint naming: chkpnt{iteration}_langfeat_{feature_level}.pth

    Args:
        scene: Scene name
        iteration: Iteration number (default 30000)
        feature_level: Feature level (default 1)

    Returns:
        Path to the OccamLGS checkpoint file
    """
    ckpt_path = f"{OCCAMLGS_OUTPUT_ROOT}/{scene}/chkpnt{iteration}_langfeat_{feature_level}.pth"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"OccamLGS checkpoint not found: {ckpt_path}")

    return ckpt_path


def load_occamlgs_checkpoint(ckpt_path: str) -> Tuple[Tuple, int]:
    """Load OccamLGS checkpoint file (13-element tuple format).

    OccamLGS format (13 elements):
    - [0]: active_sh_degree (int)
    - [1]: xyz (N, 3)
    - [2]: features_dc (N, 1, 3) - correct SH format
    - [3]: features_rest (N, 15, 3) - correct SH format for sh_degree=3
    - [4]: scaling (N, 3)
    - [5]: rotation (N, 4)
    - [6]: opacity (N, 1)
    - [7]: language_features (N, 512) - CLIP features to compress
    - [8]: max_radii2D (N,)
    - [9]: xyz_gradient_accum (N, 1)
    - [10]: denom (N, 1)
    - [11]: opt_dict (dict)
    - [12]: spatial_lr_scale (float)

    Args:
        ckpt_path: Path to OccamLGS checkpoint file

    Returns:
        model_params: 13-element tuple in OccamLGS format
        iteration: iteration number
    """
    print(f"Loading OccamLGS checkpoint from: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # OccamLGS format: (model_params, first_iter) where model_params is a tuple
    if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        model_params, iteration = checkpoint
        print(f"  Detected OccamLGS format: (model_params, iteration={iteration})")
        print(f"  Model params length: {len(model_params)}")

        if len(model_params) != 13:
            raise ValueError(f"Expected 13-element OccamLGS tuple, got {len(model_params)} elements")

        # Print key info
        active_sh_degree = model_params[0]
        xyz = model_params[1]
        features_dc = model_params[2]
        language_features = model_params[7]

        print(f"  active_sh_degree: {active_sh_degree}")
        print(f"  xyz shape: {xyz.shape}")
        print(f"  features_dc shape: {features_dc.shape} (should be [N, 1, 3])")
        print(f"  language_features shape: {language_features.shape} (should be [N, 512])")

        return model_params, iteration

    raise ValueError(f"Unexpected OccamLGS checkpoint format: type={type(checkpoint)}, len={len(checkpoint) if isinstance(checkpoint, (list, tuple)) else 'N/A'}")


def compute_svd_on_occamlgs_features(language_features: torch.Tensor, rank: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD on OccamLGS language features (512-dim CLIP).

    Args:
        language_features: [N, 512] - CLIP features
        rank: Target compression rank

    Returns:
        U: [N, rank] - left singular vectors (truncated)
        S: [rank] - singular values (truncated)
        Vt: [rank, 512] - right singular vectors (transposed, truncated)
    """
    print(f"Computing SVD on {language_features.shape} features...")

    # Convert to numpy
    features_np = language_features.numpy().astype(np.float32)

    # Compute full SVD
    U, S, Vt = np.linalg.svd(features_np, full_matrices=False)

    print(f"  Full SVD shapes: U={U.shape}, S={S.shape}, Vt={Vt.shape}")
    print(f"  Singular values (first 10): {S[:10]}")
    print(f"  Cumulative variance (first {rank}): {np.sum(S[:rank]**2) / np.sum(S**2):.4f}")

    # Truncate to target rank
    U_r = U[:, :rank]  # [N, rank]
    S_r = S[:rank]     # [rank]
    Vt_r = Vt[:rank, :]  # [rank, 512]

    print(f"  Truncated SVD shapes: U={U_r.shape}, S={S_r.shape}, Vt={Vt_r.shape}")

    return U_r, S_r, Vt_r


def compress_occamlgs_features(
    language_features: torch.Tensor,
    Vt: np.ndarray,
    rank: int = 16
) -> torch.Tensor:
    """Compress OccamLGS language features using pre-computed SVD.

    Args:
        language_features: [N, 512] - CLIP features
        Vt: [rank, 512] - right singular vectors (transposed, truncated)
        rank: Compression rank

    Returns:
        compressed_features: [N, rank] - compressed features
    """
    print(f"Compressing OccamLGS features from {language_features.shape[1]} to {rank} dimensions...")

    # Compress using SVD projection: features @ Vt.T
    # Vt is [rank, 512], so Vt.T is [512, rank]
    compressed = language_features.numpy() @ Vt.T  # [N, rank]

    print(f"  Compressed features shape: {compressed.shape}")

    # Convert to tensor
    compressed_tensor = torch.from_numpy(compressed.astype(np.float32))

    # Compute compression stats
    original_size = language_features.numel() * 4  # float32
    compressed_size = compressed_tensor.numel() * 4
    compression_ratio = original_size / compressed_size

    print(f"  Original size: {original_size / 1024**2:.2f} MB")
    print(f"  Compressed size: {compressed_size / 1024**2:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")

    return compressed_tensor


def load_checkpoint(ckpt_path: str, scene: str, feature_level: Optional[int] = None) -> Tuple[Tuple, int]:
    """Load checkpoint file (gsplat format).

    Args:
        ckpt_path: Path to checkpoint file
        scene: Scene name (for loading lang_feat and valid_feat_mask)
        feature_level: Feature level number (1, 2, 3, etc.) for sequence-based files

    Returns:
        model_params: 13-element tuple in capture_language_feature() format
        iteration: iteration number
    """
    # Build sequence suffix
    seq_suffix = f"_{feature_level}" if feature_level is not None else ""
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
        lang_feat_path = f"{TRAIN_ROOT}/{scene}/lang_feat{seq_suffix}.npy"
        if not os.path.exists(lang_feat_path):
            raise FileNotFoundError(f"Language features not found: {lang_feat_path}")

        print(f"  Loading language features from: {lang_feat_path}")
        language_features = torch.from_numpy(np.load(lang_feat_path).astype(np.float32))
        print(f"  Language features shape: {language_features.shape}")

        # Verify dimensions match
        if language_features.shape[0] != N:
            print(f"  Warning: Language feature count ({language_features.shape[0]}) != Gaussian count ({N})")

        # Load valid_feat_mask to ensure proper feature alignment
        # valid_feat_mask is now unified (no level suffix)
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
    output_dir: Optional[str] = None,
    feature_level: Optional[int] = None
):
    """Process a single scene.

    Args:
        scene: Scene name
        iteration: Preferred checkpoint iteration (None for latest)
        rank: SVD compression rank
        output_dir: Output directory (default OUTPUT_ROOT)
        feature_level: Feature level number (1, 2, 3, etc.) for sequence-based files
    """
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene} (seq={feature_level})")
    print(f"{'='*60}\n")

    # Load SVD components
    U, S, Vt = load_svd_components(scene)

    # Load checkpoint
    ckpt_path = find_checkpoint(scene, iteration)
    model_params, iteration_num = load_checkpoint(ckpt_path, scene, feature_level)

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
    output_dir: Optional[str] = None,
    feature_level: Optional[int] = None
):
    """Process a single scene using pre-compressed grid SVD features.

    This mode:
    1. Loads pre-compressed features from lang_feat_grid_svd_r*_*.npz
    2. Loads checkpoint from gaussian_results/lerf_ovs/
    3. Replaces language features with pre-compressed features
    4. Saves checkpoint with compressed features

    Args:
        scene: Scene name
        iteration: Preferred checkpoint iteration (None for latest)
        rank: SVD compression rank (default 16)
        output_dir: Output directory (default OUTPUT_ROOT)
        feature_level: Feature level number (1, 2, 3, etc.) for sequence-based files
    """
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene} (Grid SVD mode, rank={rank}, seq={feature_level})")
    print(f"{'='*60}\n")

    # Load pre-compressed grid SVD features
    compressed_features = load_grid_svd_compressed_features(scene, rank, feature_level)

    # Convert to torch tensor
    compressed_features_tensor = torch.from_numpy(compressed_features)
    print(f"Converted to tensor: {compressed_features_tensor.shape}")

    # Load checkpoint
    ckpt_path = find_checkpoint(scene, iteration)
    model_params, iteration_num = load_checkpoint(ckpt_path, scene, feature_level)

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


def load_gaussian_params_from_train_root(
    scene: str,
    feature_level: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Gaussian parameters directly from TRAIN_ROOT.

    Loads coord, color, opacity, quat, scale from .npy files.

    Args:
        scene: Scene name
        feature_level: Feature level number (1, 2, 3, etc.) for sequence-based files

    Returns:
        coord: [N, 3] - 3D coordinates
        color: [N, 3] - RGB colors (0-255 range, will be normalized to 0-1)
        opacity: [N, 1] - opacity values
        quat: [N, 4] - quaternion rotations (w, x, y, z)
        scale: [N, 3] - scale parameters
    """
    scene_dir = Path(TRAIN_ROOT) / scene

    # Load required Gaussian parameter files
    coord_path = scene_dir / "coord.npy"
    color_path = scene_dir / "color.npy"
    opacity_path = scene_dir / "opacity.npy"
    quat_path = scene_dir / "quat.npy"
    scale_path = scene_dir / "scale.npy"

    # Check all files exist
    required_files = {
        'coord.npy': coord_path,
        'color.npy': color_path,
        'opacity.npy': opacity_path,
        'quat.npy': quat_path,
        'scale.npy': scale_path,
    }

    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {scene_dir}: {missing}")

    # Load parameters
    coord = np.load(coord_path).astype(np.float32)  # [N, 3]
    color = np.load(color_path).astype(np.float32)  # [N, 3] - 0-255 range
    opacity = np.load(opacity_path).astype(np.float32)  # [N] or [N, 1]
    quat = np.load(quat_path).astype(np.float32)  # [N, 4] - x, y, z, w or w, x, y, z
    scale = np.load(scale_path).astype(np.float32)  # [N, 3]

    # Normalize color from 0-255 to 0-1
    color = color / 255.0

    # Ensure opacity has shape [N, 1]
    if opacity.ndim == 1:
        opacity = opacity.reshape(-1, 1)

    # Clip opacity to valid range
    opacity = np.clip(opacity, 0.001, 1.0)

    # Clip scale to valid range
    scale = np.clip(scale, 1e-4, 1.0)

    print(f"Loaded Gaussian parameters from {scene_dir}:")
    print(f"  coord: {coord.shape}")
    print(f"  color: {color.shape} (normalized 0-1)")
    print(f"  opacity: {opacity.shape}")
    print(f"  quat: {quat.shape}")
    print(f"  scale: {scale.shape}")

    return coord, color, opacity, quat, scale


def create_model_params_from_gaussian_data(
    coord: np.ndarray,
    color: np.ndarray,
    opacity: np.ndarray,
    quat: np.ndarray,
    scale: np.ndarray,
    language_features: torch.Tensor,
    valid_feat_mask: Optional[torch.Tensor] = None,
    use_occamlgs_format: bool = False,
) -> Tuple:
    """Create model_params tuple from Gaussian data.

    This creates the same format as gsplat/OccamLGS checkpoint but directly from numpy arrays.

    Args:
        coord: [N, 3] - 3D coordinates
        color: [N, 3] - RGB colors (normalized 0-1)
        opacity: [N, 1] - opacity values
        quat: [N, 4] - quaternion rotations
        scale: [N, 3] - scale parameters
        language_features: [N, D] - language features
        valid_feat_mask: [N] - boolean mask for valid features (optional)
        use_occamlgs_format: If True, use OccamLGS format (features_dc=[N,1,3], features_rest=[N,15,3])

    Returns:
        model_params: 13-element tuple in OccamLGS/gsplat format
    """
    N = coord.shape[0]

    # Convert to torch tensors
    xyz = torch.from_numpy(coord).float()  # [N, 3]

    # Create spherical harmonics features
    if use_occamlgs_format:
        # OccamLGS format: features_dc=[N,1,3], features_rest=[N,15,3]
        features_dc = torch.from_numpy(color).float().unsqueeze(1)  # [N, 3] -> [N, 1, 3]
        features_rest = torch.zeros(N, 15, 3, dtype=torch.float32)  # [N, 15, 3] for sh_degree=3
    else:
        # gsplat format: features_dc=[N, 3], features_rest=[N, 45]
        features_dc = torch.from_numpy(color).float()  # [N, 3]
        features_rest = torch.zeros(N, 45, dtype=torch.float32)

    # Scaling
    scaling = torch.from_numpy(scale).float()  # [N, 3]

    # Rotation (quaternion)
    rotation = torch.from_numpy(quat).float()  # [N, 4]

    # Opacity
    opacity_tensor = torch.from_numpy(opacity).float()  # [N, 1]

    # Placeholder values for GaussianModel state
    active_sh_degree = 3
    max_radii2D = torch.zeros(N, dtype=torch.int32)
    xyz_gradient_accum = torch.zeros(N, 1, dtype=torch.float32)
    denom = torch.zeros(N, 1, dtype=torch.float32)
    opt_dict = {}
    spatial_lr_scale = 1.0

    # Create valid_feat_mask if not provided
    if valid_feat_mask is None:
        valid_feat_mask = torch.ones(language_features.shape[0], dtype=torch.bool)

    # Create 13-element tuple in OccamLGS format
    model_params = (
        active_sh_degree,
        xyz,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity_tensor,
        language_features,  # [N, D] - compressed features
        max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        spatial_lr_scale,
        valid_feat_mask,  # [N] - boolean mask for valid features
    )

    return model_params


def process_scene_direct_grid_svd(
    scene: str,
    rank: int = SVD_RANK,
    output_dir: Optional[str] = None,
    feature_level: Optional[int] = None
):
    """Process a scene using direct mode with pre-compressed grid SVD features.

    Direct mode: Loads all Gaussian parameters directly from TRAIN_ROOT
    without requiring checkpoint from CHECKPOINT_ROOT.

    Args:
        scene: Scene name
        rank: SVD compression rank (default 16)
        output_dir: Output directory (default OUTPUT_ROOT)
        feature_level: Feature level number (1, 2, 3, etc.) for sequence-based files
    """
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene} (Direct Grid SVD mode, rank={rank}, seq={feature_level})")
    print(f"{'='*60}\n")

    # Load Gaussian parameters from TRAIN_ROOT
    coord, color, opacity, quat, scale = load_gaussian_params_from_train_root(scene, feature_level)

    # Load pre-compressed grid SVD features
    compressed_features = load_grid_svd_compressed_features(scene, rank, feature_level)

    # Convert to torch tensor
    compressed_features_tensor = torch.from_numpy(compressed_features).float()
    print(f"Compressed features shape: {compressed_features_tensor.shape}")

    # Load valid_feat_mask (now unified, no level suffix)
    valid_mask_path = f"{TRAIN_ROOT}/{scene}/valid_feat_mask.npy"
    if os.path.exists(valid_mask_path):
        valid_feat_mask = torch.from_numpy(np.load(valid_mask_path).astype(np.bool_))
        print(f"Valid feat mask: {valid_feat_mask.sum().item()} valid")
    else:
        valid_feat_mask = None

    # Create model_params directly from Gaussian data with OccamLGS format
    model_params = create_model_params_from_gaussian_data(
        coord, color, opacity, quat, scale, compressed_features_tensor,
        valid_feat_mask=valid_feat_mask,
        use_occamlgs_format=True  # Use OccamLGS format with correct SH structure
    )

    # Extract language features for stats
    original_features = model_params[7]

    # Save checkpoint
    if output_dir is None:
        output_dir = OUTPUT_ROOT
    save_path = f"{output_dir}/{scene}/checkpoint_with_features.pth"
    save_checkpoint(model_params, 0, save_path)  # iteration=0 for direct mode

    # Also save just the language features as .npy for convenience
    lang_feat_path = f"{output_dir}/{scene}/language_features.npy"
    np.save(lang_feat_path, compressed_features_tensor.numpy())
    print(f"Also saved language features to: {lang_feat_path}")

    # Print stats
    print(f"\nStatistics:")
    print(f"  Gaussians: {coord.shape[0]:,}")
    print(f"  Compressed features: {compressed_features_tensor.shape}")
    print(f"  File size: {os.path.getsize(save_path) / 1024**2:.2f} MB")

    print(f"\n{'='*60}")
    print(f"Scene {scene} processed successfully! (Direct Grid SVD mode)")
    print(f"{'='*60}\n")


def process_scene_direct_svd(
    scene: str,
    rank: int = SVD_RANK,
    output_dir: Optional[str] = None,
    feature_level: Optional[int] = None
):
    """Process a scene using direct mode with SVD compression.

    Direct mode: Loads all Gaussian parameters directly from TRAIN_ROOT
    without requiring checkpoint from CHECKPOINT_ROOT.

    Args:
        scene: Scene name
        rank: SVD compression rank (default 16)
        output_dir: Output directory (default OUTPUT_ROOT)
        feature_level: Feature level number (1, 2, 3, etc.) for sequence-based files
    """
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene} (Direct SVD mode, rank={rank}, seq={feature_level})")
    print(f"{'='*60}\n")

    # Build sequence suffix
    seq_suffix = f"_{feature_level}" if feature_level is not None else ""

    # Load Gaussian parameters from TRAIN_ROOT
    coord, color, opacity, quat, scale = load_gaussian_params_from_train_root(scene, feature_level)

    # Load original features and valid mask
    scene_dir = Path(TRAIN_ROOT) / scene
    lang_feat_path = scene_dir / f"lang_feat{seq_suffix}.npy"
    # valid_feat_mask is now unified (no level suffix)
    valid_feat_mask_path = scene_dir / "valid_feat_mask.npy"

    if not lang_feat_path.exists():
        raise FileNotFoundError(f"Language features not found: {lang_feat_path}")
    if not valid_feat_mask_path.exists():
        raise FileNotFoundError(f"Valid feature mask not found: {valid_feat_mask_path}")

    print(f"Loading language features from: {lang_feat_path}")
    original_features = np.load(lang_feat_path).astype(np.float32)  # [N, 768]

    print(f"Loading valid feature mask from: {valid_feat_mask_path}")
    valid_feat_mask = torch.from_numpy(np.load(valid_feat_mask_path).astype(np.bool_))
    print(f"Valid feat mask: {valid_feat_mask.sum().item()} valid, {(~valid_feat_mask).sum().item()} invalid")

    # Load SVD components (only need Vt for compression)
    svd_path = scene_dir / "lang_feat_svd.npz"
    if not svd_path.exists():
        raise FileNotFoundError(f"SVD file not found: {svd_path}")

    print(f"Loading SVD components from: {svd_path}")
    svd_data = np.load(svd_path)
    Vt = svd_data['Vt']  # [768, 768]

    # Compress language features
    N_original = valid_feat_mask.shape[0]
    N_features = original_features.shape[0]

    print(f"  Original features shape: {original_features.shape}")
    print(f"  Original size (from mask): {N_original}")

    # Check if features have been pre-filtered
    if N_features != N_original:
        print(f"  Warning: Features are pre-filtered ({N_features} != {N_original})")
        valid_features = original_features  # [N_features, 768]
        compressed_valid = valid_features @ Vt[:rank, :].T  # [N_features, rank]
    else:
        # Extract only valid features for compression
        valid_features = original_features[valid_feat_mask.numpy()]  # [M, 768]
        print(f"  Valid features shape: {valid_features.shape}")
        # Compress valid features using SVD projection
        compressed_valid = valid_features @ Vt[:rank, :].T  # [M, rank]

    # Create full [N_original, rank] array filled with zeros
    compressed = np.zeros((N_original, rank), dtype=np.float32)

    # Fill valid positions with compressed features
    if N_features != N_original:
        # Features were pre-filtered, use valid_feat_mask to place back
        compressed[valid_feat_mask.numpy()] = compressed_valid
    else:
        compressed[valid_feat_mask.numpy()] = compressed_valid

    print(f"  Compressed features shape: {compressed.shape}")

    # Convert to torch tensor
    compressed_features_tensor = torch.from_numpy(compressed).float()

    # Create model_params directly from Gaussian data
    model_params = create_model_params_from_gaussian_data(
        coord, color, opacity, quat, scale,
        compressed_features_tensor,
        valid_feat_mask
    )

    # Save checkpoint
    if output_dir is None:
        output_dir = OUTPUT_ROOT
    save_path = f"{output_dir}/{scene}/checkpoint_with_features.pth"
    save_checkpoint(model_params, 0, save_path)  # iteration=0 for direct mode

    # Also save just the language features as .npy for convenience
    lang_feat_path_out = f"{output_dir}/{scene}/language_features.npy"
    np.save(lang_feat_path_out, compressed)
    print(f"Also saved language features to: {lang_feat_path_out}")

    # Print stats
    original_size = N_original * 768 * 4  # float32
    compressed_size = compressed_features_tensor.numel() * 4
    compression_ratio = original_size / compressed_size

    print(f"\nStatistics:")
    print(f"  Gaussians: {coord.shape[0]:,}")
    print(f"  Original size: {original_size / 1024**2:.2f} MB")
    print(f"  Compressed size: {compressed_size / 1024**2:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")

    print(f"\n{'='*60}")
    print(f"Scene {scene} processed successfully! (Direct SVD mode)")
    print(f"{'='*60}\n")


def process_scene_occamlgs(
    scene: str,
    iteration: Optional[int] = None,
    feature_level: int = 1,
    rank: int = SVD_RANK,
    output_dir: Optional[str] = None
):
    """Process a single scene from OccamLGS checkpoint.

    This mode:
    1. Loads OccamLGS checkpoint from OCCAMLGS_OUTPUT_ROOT
    2. Extracts 512-dim CLIP language features
    3. Computes SVD on the features and compresses to 16-dim
    4. Saves checkpoint with correct spherical harmonics format

    Args:
        scene: Scene name
        iteration: Checkpoint iteration (default 30000)
        feature_level: Feature level (default 1)
        rank: SVD compression rank (default 16)
        output_dir: Output directory (default OUTPUT_ROOT)
    """
    # Use default iteration of 30000 if not specified
    if iteration is None:
        iteration = 30000

    print(f"\n{'='*60}")
    print(f"Processing scene: {scene} (OccamLGS mode)")
    print(f"  Checkpoint: chkpnt{iteration}_langfeat_{feature_level}.pth")
    print(f"  Compression rank: {rank}")
    print(f"{'='*60}\n")

    # Find and load OccamLGS checkpoint
    ckpt_path = find_occamlgs_checkpoint(scene, iteration, feature_level)
    model_params, iteration_num = load_occamlgs_checkpoint(ckpt_path)

    # Extract language features (element 7 of the tuple)
    language_features = model_params[7]
    print(f"\nOriginal language features shape: {language_features.shape}")

    # Verify it's 512-dim CLIP features
    if language_features.shape[1] != 512:
        raise ValueError(f"Expected 512-dim CLIP features, got {language_features.shape[1]}-dim features")

    # Compute SVD on the OccamLGS features
    print("\nComputing SVD on OccamLGS language features...")
    U, S, Vt = compute_svd_on_occamlgs_features(language_features, rank)

    # Compress features using SVD projection
    compressed_features = compress_occamlgs_features(language_features, Vt, rank)

    # Create new checkpoint with compressed features
    # OccamLGS already has correct SH format, so we can just replace the features
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

    # Also save SVD components for future reference
    svd_path = f"{output_dir}/{scene}/lang_feat_svd_occamlgs.npz"
    np.savez(svd_path, U=U, S=S, Vt=Vt)
    print(f"Also saved SVD components to: {svd_path}")

    print(f"\n{'='*60}")
    print(f"Scene {scene} processed successfully! (OccamLGS mode)")
    print(f"{'='*60}\n")


def process_scene_occamlgs_from_base(
    scene: str,
    iteration: Optional[int] = None,
    feature_level: int = 1,
    rank: int = SVD_RANK,
    output_dir: Optional[str] = None
):
    """Process a single scene from OccamLGS base checkpoint + grid SVD features.

    This mode:
    1. Loads OccamLGS base checkpoint (WITHOUT language features) from OCCAMLGS_OUTPUT_ROOT
    2. Loads pre-compressed grid SVD features from TRAIN_ROOT
    3. Merges them to create checkpoint with compressed features
    4. Saves checkpoint with correct spherical harmonics format

    Args:
        scene: Scene name
        iteration: Base checkpoint iteration (default 30000)
        feature_level: Feature level (default 1)
        rank: SVD compression rank (default 16)
        output_dir: Output directory (default OUTPUT_ROOT)
    """
    # Use default iteration of 30000 if not specified
    if iteration is None:
        iteration = 30000

    print(f"\n{'='*60}")
    print(f"Processing scene: {scene} (OccamLGS base + grid SVD mode)")
    print(f"  Base checkpoint: chkpnt{iteration}.pth (NO language features)")
    print(f"  Grid SVD features: lang_feat_grid_svd_r{rank}_{feature_level}.npz")
    print(f"  Compression rank: {rank}")
    print(f"{'='*60}\n")

    # Step 1: Load OccamLGS base checkpoint (without language features)
    base_ckpt_path = find_occamlgs_base_checkpoint(scene, iteration)
    base_model_params, iteration_num = load_occamlgs_base_checkpoint(base_ckpt_path)

    # Step 2: Load pre-compressed grid SVD features from TRAIN_ROOT
    print(f"\nLoading grid SVD compressed features from TRAIN_ROOT...")
    compressed_features = load_grid_svd_compressed_features(scene, rank, feature_level)
    compressed_features_tensor = torch.from_numpy(compressed_features).float()
    print(f"  Compressed features shape: {compressed_features_tensor.shape}")

    # Step 3: Add language features to base checkpoint (will raise error if dimensions don't match)
    print(f"\nAdding language features to base checkpoint...")
    model_params_with_lang = add_language_features_to_base_checkpoint(
        base_model_params, compressed_features_tensor
    )

    # Step 4: Save checkpoint
    if output_dir is None:
        output_dir = OUTPUT_ROOT
    save_path = f"{output_dir}/{scene}/checkpoint_with_features.pth"
    save_checkpoint(model_params_with_lang, iteration_num, save_path)

    # Also save just the language features as .npy for convenience
    lang_feat_path = f"{output_dir}/{scene}/language_features.npy"
    np.save(lang_feat_path, compressed_features_tensor.numpy())
    print(f"Also saved language features to: {lang_feat_path}")

    # Print stats
    print(f"\nStatistics:")
    print(f"  Gaussians: {xyz.shape[0]:,}")
    print(f"  Compressed features: {compressed_features_tensor.shape}")
    print(f"  File size: {os.path.getsize(save_path) / 1024**2:.2f} MB")

    print(f"\n{'='*60}")
    print(f"Scene {scene} processed successfully! (OccamLGS base + grid SVD mode)")
    print(f"{'='*60}\n")


def create_coordinate_mapping(
    occamlgs_xyz: np.ndarray,
    train_coord: np.ndarray
) -> np.ndarray:
    """Create mapping from TRAIN_ROOT coordinates to OccamLGS coordinates.

    Args:
        occamlgs_xyz: [N_occamlgs, 3] - OccamLGS checkpoint coordinates
        train_coord: [N_train, 3] - TRAIN_ROOT coordinates

    Returns:
        mapping: [N_train, 2] - array of (train_idx, occamlgs_idx) pairs
    """
    print(f"Creating coordinate mapping: {train_coord.shape[0]} TRAIN_ROOT -> {occamlgs_xyz.shape[0]} OccamLGS")

    # Create a dictionary for efficient lookup
    occamlgs_coord_map = {}
    for i, coord in enumerate(occamlgs_xyz):
        coord_tuple = tuple(coord)
        if coord_tuple not in occamlgs_coord_map:
            occamlgs_coord_map[coord_tuple] = i

    # Find mapping for each TRAIN_ROOT coordinate
    mapping = []
    missing_coords = []
    for i, coord in enumerate(train_coord):
        coord_tuple = tuple(coord)
        if coord_tuple in occamlgs_coord_map:
            mapping.append([i, occamlgs_coord_map[coord_tuple]])
        else:
            missing_coords.append(i)

    if missing_coords:
        print(f"  Warning: {len(missing_coords)} TRAIN_ROOT coords not found in OccamLGS checkpoint")

    print(f"  Mapped {len(mapping)}/{len(train_coord)} coordinates")
    return np.array(mapping, dtype=np.int64)


def process_scene_occamlgs_with_grid_svd(
    scene: str,
    iteration: Optional[int] = None,
    feature_level: int = 1,
    rank: int = SVD_RANK,
    output_dir: Optional[str] = None
):
    """Process a scene from OccamLGS checkpoint with grid SVD features from TRAIN_ROOT.

    This mode:
    1. Loads OccamLGS checkpoint (with language features) to get correct Gaussian parameters
    2. Loads pre-compressed grid SVD features from TRAIN_ROOT
    3. Creates coordinate mapping between TRAIN_ROOT and OccamLGS checkpoint
    4. Expands grid SVD features to match OccamLGS checkpoint size using coordinate mapping
    5. Saves checkpoint with OccamLGS format

    Args:
        scene: Scene name
        iteration: Checkpoint iteration (default 30000)
        feature_level: Feature level (default 1)
        rank: SVD compression rank (default 16)
        output_dir: Output directory (default OUTPUT_ROOT)
    """
    # Use default iteration of 30000 if not specified
    if iteration is None:
        iteration = 30000

    print(f"\n{'='*60}")
    print(f"Processing scene: {scene} (OccamLGS + Grid SVD from TRAIN_ROOT mode)")
    print(f"  Checkpoint: chkpnt{iteration}_langfeat_{feature_level}.pth (for Gaussian params only)")
    print(f"  Grid SVD features: lang_feat_grid_svd_r{rank}_{feature_level}.npz (from TRAIN_ROOT)")
    print(f"  Compression rank: {rank}")
    print(f"{'='*60}\n")

    # Step 1: Load OccamLGS checkpoint to get Gaussian parameters with correct format
    ckpt_path = find_occamlgs_checkpoint(scene, iteration, feature_level)
    model_params, iteration_num = load_occamlgs_checkpoint(ckpt_path)

    # Extract Gaussian parameters (we'll keep these)
    active_sh_degree = model_params[0]
    xyz = model_params[1]
    features_dc = model_params[2]
    features_rest = model_params[3]
    scaling = model_params[4]
    rotation = model_params[5]
    opacity = model_params[6]
    max_radii2D = model_params[8]
    xyz_gradient_accum = model_params[9]
    denom = model_params[10]
    opt_dict = model_params[11]
    spatial_lr_scale = model_params[12]
    valid_feat_mask = model_params[13] if len(model_params) >= 14 else None

    N_gaussians = xyz.shape[0]
    print(f"  Loaded {N_gaussians} Gaussians from OccamLGS checkpoint")

    # Step 2: Load pre-compressed grid SVD features from TRAIN_ROOT
    print(f"\nLoading grid SVD compressed features from TRAIN_ROOT...")
    compressed_features = load_grid_svd_compressed_features(scene, rank, feature_level)
    N_features = compressed_features.shape[0]
    print(f"  Compressed features shape: {compressed_features.shape}")

    # Step 3: Load TRAIN_ROOT coordinates for mapping
    print(f"\nLoading TRAIN_ROOT coordinates...")
    seq_suffix = f"_{feature_level}" if feature_level is not None else ""
    train_coord_path = f"{TRAIN_ROOT}/{scene}/coord.npy"
    if not os.path.exists(train_coord_path):
        raise FileNotFoundError(f"TRAIN_ROOT coord file not found: {train_coord_path}")

    train_coord = np.load(train_coord_path)
    print(f"  TRAIN_ROOT coord shape: {train_coord.shape}")

    # Step 4: Create coordinate mapping
    occamlgs_xyz_np = xyz.detach().cpu().numpy()
    mapping = create_coordinate_mapping(occamlgs_xyz_np, train_coord)

    # Step 5: Expand compressed features to match OccamLGS checkpoint size
    print(f"\nExpanding compressed features to match OccamLGS checkpoint size...")
    print(f"  Target size: {N_gaussians} Gaussians")
    print(f"  Source size: {N_features} features")

    # Create full-size compressed features array with zero padding
    full_compressed_features = np.zeros((N_gaussians, rank), dtype=np.float32)

    # Map features from TRAIN_ROOT to OccamLGS using coordinate mapping
    # mapping[:, 0] = TRAIN_ROOT indices, mapping[:, 1] = OccamLGS indices
    for train_idx, occamlgs_idx in mapping:
        full_compressed_features[occamlgs_idx] = compressed_features[train_idx]

    mapped_count = mapping.shape[0]
    unmapped_count = N_gaussians - mapped_count
    print(f"  Mapped features: {mapped_count}")
    print(f"  Zero-padded features: {unmapped_count}")

    # Convert to tensor
    compressed_features_tensor = torch.from_numpy(full_compressed_features).float()

    # Step 6: Create new model_params with compressed features (use full size)
    model_params_with_compressed = (
        active_sh_degree,
        xyz,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity,
        compressed_features_tensor,  # Expanded features with zero padding
        max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        spatial_lr_scale,
        valid_feat_mask,
    )

    # Step 5: Save checkpoint
    if output_dir is None:
        output_dir = OUTPUT_ROOT
    save_path = f"{output_dir}/{scene}/checkpoint_with_features.pth"
    save_checkpoint(model_params_with_compressed, iteration_num, save_path)

    # Also save just the language features as .npy for convenience
    lang_feat_path = f"{output_dir}/{scene}/language_features.npy"
    np.save(lang_feat_path, compressed_features_tensor.numpy())
    print(f"Also saved language features to: {lang_feat_path}")

    # Print stats
    print(f"\nStatistics:")
    print(f"  Total Gaussians: {N_gaussians:,}")
    print(f"  Mapped features: {mapped_count:,}")
    print(f"  Zero-padded features: {unmapped_count:,}")
    print(f"  Compressed features: {compressed_features_tensor.shape}")
    print(f"  File size: {os.path.getsize(save_path) / 1024**2:.2f} MB")

    print(f"\n{'='*60}")
    print(f"Scene {scene} processed successfully! (OccamLGS + Grid SVD from TRAIN_ROOT mode)")
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

  # Direct mode (no checkpoint required, loads from TRAIN_ROOT only)
  python tools/create_svd_compressed_checkpoint.py --scene figurines --direct --use-grid-svd
  python tools/create_svd_compressed_checkpoint.py --all-scenes --direct
  python tools/create_svd_compressed_checkpoint.py --scene figurines --direct --use-grid-svd --feat-seq 1

  # OccamLGS mode: load from OccamLGS checkpoints with 512-dim CLIP features
  python tools/create_svd_compressed_checkpoint.py --scene figurines --occamlgs
  python tools/create_svd_compressed_checkpoint.py --scene figurines --occamlgs --feature-level 1
  python tools/create_svd_compressed_checkpoint.py --all-scenes --occamlgs --rank 32

  # OccamLGS base mode: load base checkpoint (no lang feat) + grid SVD features from TRAIN_ROOT
  python tools/create_svd_compressed_checkpoint.py --scene figurines --occamlgs-base
  python tools/create_svd_compressed_checkpoint.py --scene figurines --occamlgs-base --feature-level 1 --rank 16
  python tools/create_svd_compressed_checkpoint.py --all-scenes --occamlgs-base

  # TRAIN_ROOT + OccamLGS format: load all params from TRAIN_ROOT + grid SVD features
  python tools/create_svd_compressed_checkpoint.py --scene figurines --train-occamlgs
  python tools/create_svd_compressed_checkpoint.py --scene figurines --train-occamlgs --feature-level 1 --rank 16
  python tools/create_svd_compressed_checkpoint.py --all-scenes --train-occamlgs

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
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Direct mode: Load Gaussian parameters from TRAIN_ROOT only, without requiring CHECKPOINT_ROOT"
    )
    parser.add_argument(
        "--occamlgs",
        action="store_true",
        help="OccamLGS mode: Load from OccamLGS checkpoints with 512-dim CLIP features and compute SVD compression"
    )
    parser.add_argument(
        "--occamlgs-base",
        action="store_true",
        help="OccamLGS base mode: Load base checkpoint (without language features) from OccamLGS + grid SVD features from TRAIN_ROOT"
    )
    parser.add_argument(
        "--feature-level",
        type=int,
        default=1,
        help="Feature level for sequence-based files (default: 1). Used for lang_feat_{level}.npy, lang_feat_grid_svd_r{rank}_{level}.npz, etc."
    )

    args = parser.parse_args()

    # List available scenes
    if args.list_scenes:
        scenes = get_available_scenes(
            use_grid_svd=args.use_grid_svd,
            rank=args.rank,
            feature_level=args.feature_level,
            direct_mode=args.direct,
            occamlgs_mode=args.occamlgs
        )
        mode_parts = []
        if args.occamlgs_base:
            mode_parts.append("OccamLGS-Base")
        if args.occamlgs:
            mode_parts.append("OccamLGS")
        if args.direct:
            mode_parts.append("Direct")
        if args.use_grid_svd:
            mode_parts.append("Grid SVD")
        else:
            mode_parts.append("SVD Decomposition")
        mode = " + ".join(mode_parts)
        seq_str = f", seq={args.feature_level}" if args.feature_level is not None else ""
        print(f"Available scenes ({mode} mode, rank={args.rank}{seq_str}):")
        for scene in scenes:
            print(f"  - {scene}")
        return

    # Compare mode (not compatible with direct mode)
    if args.compare:
        if args.direct:
            parser.error("--compare is not compatible with --direct mode")
        # For compare mode, we need both grid SVD and regular SVD files
        if not args.scene and not args.all_scenes:
            parser.error("Either --scene or --all-scenes must be specified with --compare")

        if args.scene:
            try:
                compare_compression_methods(args.scene, args.rank, args.feature_level)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("\nNote: --compare requires both lang_feat_grid_svd_r*.npz and lang_feat_svd.npz files")
        else:
            # Find scenes with both types of files
            train_dir = Path(TRAIN_ROOT)
            scenes_with_both = []
            # Build sequence suffix
            seq_suffix = f"_{args.feature_level}" if args.feature_level is not None else ""
            for scene_dir in train_dir.iterdir():
                if scene_dir.is_dir():
                    scene = scene_dir.name
                    grid_file = scene_dir / f"lang_feat_grid_svd_r{args.rank}{seq_suffix}.npz"
                    svd_file = scene_dir / "lang_feat_svd.npz"
                    if grid_file.exists() and svd_file.exists():
                        scenes_with_both.append(scene)

            scenes_with_both = sorted(scenes_with_both)

            if not scenes_with_both:
                print(f"No scenes found with both lang_feat_grid_svd_r{args.rank}{seq_suffix}.npz and lang_feat_svd.npz files!")
                return

            print(f"Found {len(scenes_with_both)} scenes to compare:")
            for scene in scenes_with_both:
                print(f"  - {scene}")
            print()

            all_results = {}
            for scene in scenes_with_both:
                try:
                    result = compare_compression_methods(scene, args.rank, args.feature_level)
                    all_results[scene] = result
                except Exception as e:
                    print(f"Error processing scene {scene}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Print summary table
            if all_results:
                print(f"\n{'='*80}")
                print(f"Summary Comparison (rank={args.rank}, seq={args.feature_level}):")
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
    if args.occamlgs and args.use_grid_svd:
        # OccamLGS + Grid SVD mode: load OccamLGS checkpoint + pre-computed grid SVD features from TRAIN_ROOT
        process_func = process_scene_occamlgs_with_grid_svd
    elif args.occamlgs_base:
        # OccamLGS base mode: load base checkpoint (no lang feat) + grid SVD features
        process_func = process_scene_occamlgs_from_base
    elif args.occamlgs:
        # OccamLGS mode: load from OccamLGS checkpoints with language features
        process_func = process_scene_occamlgs
    elif args.direct:
        # Direct mode: load from TRAIN_ROOT only
        if args.use_grid_svd:
            process_func = process_scene_direct_grid_svd
        else:
            process_func = process_scene_direct_svd
    else:
        # Normal mode: load from CHECKPOINT_ROOT
        process_func = process_scene_grid_svd if args.use_grid_svd else process_scene

    # Process scenes
    if args.scene:
        # OccamLGS-Base, OccamLGS, and Direct modes use different parameters
        if args.occamlgs_base:
            process_func(args.scene, args.iteration, args.feature_level, args.rank, args.output_dir)
        elif args.occamlgs:
            process_func(args.scene, args.iteration, args.feature_level, args.rank, args.output_dir)
        elif args.direct:
            process_func(args.scene, args.rank, args.output_dir, args.feature_level)
        else:
            process_func(args.scene, args.iteration, args.rank, args.output_dir, args.feature_level)
    else:
        scenes = get_available_scenes(
            use_grid_svd=args.use_grid_svd,
            rank=args.rank,
            feature_level=args.feature_level,
            direct_mode=args.direct,
            occamlgs_mode=args.occamlgs
        )
        if not scenes:
            mode_parts = []
            if args.occamlgs:
                mode_parts.append("OccamLGS")
            if args.direct:
                mode_parts.append("Direct")
            if args.use_grid_svd:
                mode_parts.append("Grid SVD")
            else:
                mode_parts.append("SVD Decomposition")
            mode = " + ".join(mode_parts)
            seq_str = f", seq={args.feature_level}" if args.feature_level is not None else ""
            print(f"No scenes found with required files ({mode}{seq_str})!")
            return

        mode_parts = []
        if args.occamlgs_base:
            mode_parts.append("OccamLGS-Base")
        if args.occamlgs:
            mode_parts.append("OccamLGS")
        if args.direct:
            mode_parts.append("Direct")
        if args.use_grid_svd:
            mode_parts.append("Grid SVD")
        else:
            mode_parts.append("SVD Decomposition")
        mode = " + ".join(mode_parts)
        seq_str = f", seq={args.feature_level}" if args.feature_level is not None else ""
        print(f"Found {len(scenes)} scenes to process ({mode} mode, rank={args.rank}{seq_str}):")
        for scene in scenes:
            print(f"  - {scene}")
        print()

        for scene in scenes:
            try:
                # OccamLGS-Base, OccamLGS+GridSVD, OccamLGS, and Direct modes use different parameters
                if args.occamlgs_base:
                    process_func(scene, args.iteration, args.feature_level, args.rank, args.output_dir)
                elif args.occamlgs:
                    # This covers both --occamlgs and --occamlgs --use-grid-svd (same signature)
                    process_func(scene, args.iteration, args.feature_level, args.rank, args.output_dir)
                elif args.direct:
                    process_func(scene, args.rank, args.output_dir, args.feature_level)
                else:
                    process_func(scene, args.iteration, args.rank, args.output_dir, args.feature_level)
            except Exception as e:
                print(f"Error processing scene {scene}: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()
