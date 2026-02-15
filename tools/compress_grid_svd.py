#!/usr/bin/env python3
"""
Grid-Based SVD Compression Tool for Language Features

This script:
1. Reads lang_feat.npy files and computes grid coordinates
2. Computes average lang feat per grid with corresponding indices
3. Uses RPCA (GPU by default) for SVD decomposition on the grid average lang_feat matrix
4. Saves U matrices for r=8, 16, 32 as lang_feat_grid_svd_r{8,16,32}.npz

Supports both single scene and batch processing modes.
In batch mode, RPCA is applied to each scene separately.

Usage:
    # Single scene mode (GPU RPCA by default)
    python tools/compress_grid_svd.py --data_dir /path/to/scene --grid_size 0.01

    # Batch mode - all scenes in a dataset (processes each scene separately)
    python tools/compress_grid_svd.py --data_root /new_data/cyf/projects/SceneSplat/gaussian_train --dataset 3DOVS --split train

    # Batch mode - specific scenes
    python tools/compress_grid_svd.py --data_root /new_data/cyf/projects/SceneSplat/gaussian_train --dataset 3DOVS --split train --scenes scene1,scene2

    # With custom ranks
    python tools/compress_grid_svd.py --data_root /new_data/cyf/projects/SceneSplat/gaussian_train --dataset 3DOVS --split train --ranks 8,16,32

    # Use CPU RPCA
    python tools/compress_grid_svd.py --data_root /new_data/cyf/projects/SceneSplat/gaussian_train --dataset 3DOVS --split train --gpu -1

    # Disable RPCA
    python tools/compress_grid_svd.py --data_root /new_data/cyf/projects/SceneSplat/gaussian_train --dataset 3DOVS --split train --no_rpca

    # Specify output directory
    python tools/compress_grid_svd.py --data_root /new_data/cyf/projects/SceneSplat/gaussian_train --dataset 3DOVS --split train --output_dir /path/to/output
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary modules
from pointcept.models.utils.structure import Point

# Import RPCA utilities
from tools.rpca_utils import (
    apply_rpca,
    RPCA_CPU_AVAILABLE,
    CUDA_AVAILABLE,
)


def load_scene_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load coord, lang_feat, and valid_feat_mask from a scene directory.

    Args:
        data_dir: Path to scene directory

    Returns:
        coord: [N, 3] - 3D coordinates
        lang_feat: [N, D] - language features
        valid_mask: [N] - valid feature mask (or None)
    """
    coord_path = os.path.join(data_dir, "coord.npy")
    lang_feat_path = os.path.join(data_dir, "lang_feat.npy")
    valid_mask_path = os.path.join(data_dir, "valid_feat_mask.npy")

    # Check required files
    if not os.path.exists(coord_path):
        raise FileNotFoundError(f"coord.npy not found in {data_dir}")
    if not os.path.exists(lang_feat_path):
        raise FileNotFoundError(f"lang_feat.npy not found in {data_dir}")

    # Load data
    coord = np.load(coord_path).astype(np.float32)
    lang_feat = np.load(lang_feat_path).astype(np.float32)

    # Check if lang_feat has fewer points than coord
    if coord.shape[0] != lang_feat.shape[0]:
        # Load valid mask to check if lang_feat only contains valid features
        if os.path.exists(valid_mask_path):
            valid_mask = np.load(valid_mask_path).astype(bool)
            num_valid = np.sum(valid_mask)

            # Check if lang_feat size matches number of valid points
            if lang_feat.shape[0] == num_valid:
                # lang_feat only contains valid features, filter coord and valid_mask to match
                print(f"  [Info] lang_feat.npy contains only valid features ({lang_feat.shape[0]:,} points)")
                print(f"  [Info] Filtering coord and valid_mask to match ({num_valid:,} valid points)")

                # Filter coord to only include valid points
                coord = coord[valid_mask]
                # valid_mask becomes all True (since we only have valid features)
                valid_mask = np.ones(lang_feat.shape[0], dtype=bool)
            else:
                raise ValueError(
                    f"Point count mismatch in {data_dir}:\\n"
                    f"  coord.npy: {coord.shape[0]:,} points\\n"
                    f"  lang_feat.npy: {lang_feat.shape[0]:,} points\\n"
                    f"  valid_feat_mask (True): {num_valid:,} points\\n"
                    f"  lang_feat size doesn't match coord or valid_mask count!"
                )
        else:
            raise ValueError(
                f"Point count mismatch in {data_dir}:\\n"
                f"  coord.npy: {coord.shape[0]:,} points\\n"
                f"  lang_feat.npy: {lang_feat.shape[0]:,} points\\n"
                f"  These files must have the same number of points!"
            )
    else:
        # coord and lang_feat have same number of points, load valid mask if available
        valid_mask = None
        if os.path.exists(valid_mask_path):
            valid_mask = np.load(valid_mask_path).astype(bool)
            # Validate that valid_mask matches coord
            if valid_mask.shape[0] != coord.shape[0]:
                raise ValueError(
                    f"Point count mismatch in {data_dir}:\\n"
                    f"  coord.npy: {coord.shape[0]:,} points\\n"
                    f"  valid_feat_mask.npy: {valid_mask.shape[0]:,} points\\n"
                    f"  These files must have same number of points!"
                )
        else:
            valid_mask = None

    return coord, lang_feat, valid_mask


def compute_grid_average_features(
    coord: np.ndarray,
    lang_feat: np.ndarray,
    valid_mask: Optional[np.ndarray],
    grid_size: float,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute average language features per grid cell.

    Args:
        coord: [N, 3] - 3D coordinates
        lang_feat: [N, D] - language features
        valid_mask: [N] - valid feature mask (optional)
        grid_size: Grid size in meters
        device: Device for torch operations

    Returns:
        grid_avg_feats: [M, D] - average features per grid
        point_to_grid_indices: [N] - final grid index for each point (-1 if not in any grid)
        grid_point_counts: [M] - number of gaussians in each grid
    """
    # Apply valid mask if available and define valid_indices
    if valid_mask is not None:
        # valid_indices = np.where(valid_mask)[0]
        valid_mask = valid_mask.astype(bool)
        coord_t = torch.from_numpy(coord[valid_mask]).to(device)
        lang_feat_t = torch.from_numpy(lang_feat[valid_mask]).to(device)
        valid_indices = np.arange(coord_t.shape[0])
    else:
        valid_indices = np.arange(coord.shape[0])
        coord_t = torch.from_numpy(coord).to(device)
        lang_feat_t = torch.from_numpy(lang_feat).to(device)

    N = coord_t.shape[0]
    batch = torch.zeros(N, dtype=torch.long, device=device)

    # Step 1: Create Point object
    print("\n" + "=" * 60)
    print("Step 1: Creating Point object")
    print("=" * 60)

    point = Point({
        "coord": coord_t,
        # "feat": feat_t,
        "batch": batch,
        "grid_size": grid_size,
    })

    print("Point object created with keys: " + str(list(point.keys())))

    # Step 2: Compute grid coordinates (this calls serialization internally)
    print("\n" + "=" * 60)
    print("Step 2: Computing grid coordinates via serialization")
    print("=" * 60)

    # Get coordinate range for normalization
    coord_min = point.coord.min(dim=0)[0]
    coord_normalized = point.coord - coord_min

    print("Coordinate normalization:")
    print("  - Original range min: [{:.6f}, {:.6f}, {:.6f}]".format(
        coord_min[0], coord_min[1], coord_min[2]
    ))
    coord_norm_min = coord_normalized.min(dim=0)[0]
    coord_norm_max = coord_normalized.max(dim=0)[0]
    print("  - Normalized range: [{:.6f}, {:.6f}, {:.6f}] to [{:.6f}, {:.6f}, {:.6f}]".format(
        coord_norm_min[0], coord_norm_min[1], coord_norm_min[2],
        coord_norm_max[0], coord_norm_max[1], coord_norm_max[2]
    ))

    # Compute grid coordinates (integer division)
    grid_coord_raw = torch.div(
        coord_normalized,
        point.grid_size,
        rounding_mode="trunc"
    ).int()

    grid_min = grid_coord_raw.min(dim=0)[0]
    grid_max = grid_coord_raw.max(dim=0)[0]
    print("Grid coordinate computation (grid_size={}):".format(grid_size))
    print("  - Division by grid_size")
    print("  - Rounding mode: trunc (towards zero)")
    print("  - Grid coord range: [{:.3f}, {:.3f}, {:.3f}] to [{:.3f}, {:.3f}, {:.3f}]".format(
        grid_min[0], grid_min[1], grid_min[2],
        grid_max[0], grid_max[1], grid_max[2]
    ))
    print("  - Grid coord shape: " + str(grid_coord_raw.shape))

    # Store grid_coord back to point
    point["grid_coord"] = grid_coord_raw

    # Step 3: Apply serialization (Z-order Morton encoding)
    print("\n" + "=" * 60)
    print("Step 3: Applying Z-order Morton serialization")
    print("=" * 60)

    # Determine depth based on coordinate range
    coord_max_val = point.grid_coord.max().item()
    # Safety check: LitePT requires depth * 3 + len(offset).bit_length() <= 63
    # Maximum depth allowed is 16 (from structure.py assert)
    max_allowed_depth = 16
    depth = min(int(coord_max_val).bit_length(), max_allowed_depth)
    print("Determining serialization depth:")
    print("  - Max grid coordinate: " + str(coord_max_val))
    print("  - Bit length required: " + str(int(coord_max_val).bit_length()))
    print("  - Max allowed depth: " + str(max_allowed_depth))
    print("  - Using depth={} for Morton encoding".format(depth))

    # Apply serialization
    point.serialization(order="z", depth=depth, shuffle_orders=False)

    # Get Morton codes
    morton_codes = point["serialized_code"]  # [num_orders, N]
    serialized_order = point["serialized_order"]  # [num_orders, N]

    print("Serialization complete:")
    print("  - Number of orders: " + str(morton_codes.shape[0]))
    print("  - Morton codes shape: " + str(morton_codes.shape))
    print("  - Codes range: [{:,}] to [{:,}]".format(
        morton_codes.min(), morton_codes.max()
    ))
    print("  - Serialized order shape: " + str(serialized_order.shape))

    # Step 4: Find unique grids using torch.unique
    print("\n" + "=" * 60)
    print("Step 4: Finding unique grids with torch.unique")
    print("=" * 60)

    grid_coord = point.grid_coord  # [N, 3]
    batch = point.batch  # [N]

    # Combine batch with grid_coord (bit operation for multi-batch support)
    grid_coord_with_batch = grid_coord | (batch.view(-1, 1) << 48)

    print("Input to torch.unique:")
    print("  - grid_coord_with_batch shape: " + str(grid_coord_with_batch.shape))
    print("  - Unique operation on dim=0 (spatial dimension)")

    # Call torch.unique
    unique_grids, cluster, counts = torch.unique(
        grid_coord_with_batch,
        sorted=True,
        return_inverse=True,
        return_counts=True,
        dim=0,
    )

    print("torch.unique results:")
    print("  - unique_grids (M): " + str(unique_grids.shape))
    print("  - cluster (mapping): " + str(cluster.shape))
    print("  - counts (points per grid): " + str(counts.shape))

    M = unique_grids.shape[0]

    # Step 5: Compute for each grid
    print("\n" + "=" * 60)
    print("Step 5: Computing per Grid with Reassignment")
    print("=" * 60)

    skipped_grids = 0  # Count grids skipped due to single point

    # Track single-point grids and multi-point grids
    single_point_grids = []  # List of (grid_idx, point_idx, feat)
    multi_point_grid_indices = []  # List of grid_idx with >1 points

    # First pass: identify single-point and multi-point grids (with progress bar)
    print("\n[First Pass] Identifying single-point and multi-point grids...")
    for grid_idx in tqdm(range(M)):
        point_indices = torch.where(cluster == grid_idx)[0]

        if point_indices.shape[0] > 1:
            multi_point_grid_indices.append(grid_idx)
            # Get language features for points in this grid
            grid_lang_feats = lang_feat_t[point_indices]  # [K, D]
        elif point_indices.shape[0] == 1:
            single_pt_idx = point_indices[0].item()
            single_pt_feat = lang_feat_t[single_pt_idx]  # [D]
            single_point_grids.append((grid_idx, single_pt_idx, single_pt_feat))

    # Reassignment: find K closest grids, merge with most similar among them
    if single_point_grids and multi_point_grid_indices:
        print("\nReassigning single-point grids to most similar among K closest grids:")
        print("  - Single-point grids: {:,}".format(len(single_point_grids)))
        print("  - Multi-point grids: {:,}".format(len(multi_point_grid_indices)))

        # K: number of closest grids to consider
        K = 27  # 26 neighbors + self, can be adjusted

        # Build mapping from grid_idx to grid_coord (center of grid)
        grid_idx_to_coord = {}  # Maps grid_idx to grid_coord [3]

        for grid_idx in range(M):
            # Get the grid coordinate for this grid
            point_indices = torch.where(cluster == grid_idx)[0]
            if point_indices.shape[0] > 0:
                gc = grid_coord[point_indices[0]]  # [3]
                grid_idx_to_coord[grid_idx] = gc

        # Convert to tensors for batch computation
        # grid_coords_tensor: [M, 3] - grid coordinate for each grid_idx
        grid_coords_list = [grid_idx_to_coord[i] for i in range(M)]
        grid_coords_tensor = torch.stack(grid_coords_list)  # [M, 3]

        # Initialize average feature tensor for ALL grids
        D = lang_feat_t.shape[1]
        grid_avg_feats = torch.zeros(M, D, device=device)
        grid_point_counts = torch.zeros(M, dtype=torch.long, device=device)

        # Compute initial average features for ALL grids
        for grid_idx in tqdm(range(M), desc="Computing avg feats"):
            point_indices = torch.where(cluster == grid_idx)[0]
            grid_lang_feats = lang_feat_t[point_indices]  # [K, D]
            grid_avg_feats[grid_idx] = grid_lang_feats.mean(dim=0)
            grid_point_counts[grid_idx] = point_indices.shape[0]

        # Reassign each single point to most similar among K closest grids
        print(f"\n[Reassignment Phase] Assigning single points to most similar among {K} closest grids...")
        reassigned_count = 0
        skipped_grids = 0

        # Track which grids have already been merged (emptied)
        merged_grids = set()

        for grid_idx, single_pt_idx, single_pt_feat in tqdm(single_point_grids):
            # Skip if this grid has already been merged
            if grid_idx in merged_grids:
                skipped_grids += 1
                continue

            # Also skip if the grid was already emptied by a previous merge
            if grid_point_counts[grid_idx].item() == 0:
                skipped_grids += 1
                merged_grids.add(grid_idx)
                continue

            # Get current grid coordinate
            current_gc = grid_idx_to_coord[grid_idx]  # [3]

            # Compute L1 distance to all other grids
            # L1 distance in grid space = Manhattan distance
            distances = torch.sum(torch.abs(grid_coords_tensor - current_gc.unsqueeze(0)), dim=1)  # [M]

            # Find K closest grids (including self, but we'll handle that)
            _, closest_indices = torch.topk(distances, k=min(K, M), largest=False)
            closest_grid_indices = closest_indices.tolist()

            # Remove current grid from consideration (we want to merge elsewhere)
            if grid_idx in closest_grid_indices:
                closest_grid_indices.remove(grid_idx)

            # Skip if no adjacent grids available
            if len(closest_grid_indices) == 0:
                skipped_grids += 1
                continue

            # Normalize single point feature
            single_pt_norm = single_pt_feat / torch.norm(single_pt_feat)

            # Compute similarity only with K closest grids
            closest_grid_feats = grid_avg_feats[closest_grid_indices]  # [K', D]

            # Normalize grid features
            grid_norms = torch.norm(closest_grid_feats, dim=1, keepdim=True)
            grid_normalized = closest_grid_feats / grid_norms.clamp(min=1e-8)

            # Compute cosine similarities
            similarities = torch.mm(grid_normalized, single_pt_norm.unsqueeze(1)).squeeze()

            # Find best matching grid among closest
            best_local_idx = similarities.argmax().item()
            best_grid_idx = closest_grid_indices[best_local_idx]

            # Reassign: update the target grid's average feature
            old_count = grid_point_counts[best_grid_idx].item()
            old_avg = grid_avg_feats[best_grid_idx]
            new_avg = (old_avg * old_count + single_pt_feat) / (old_count + 1)
            grid_avg_feats[best_grid_idx] = new_avg
            grid_point_counts[best_grid_idx] = old_count + 1
            grid_avg_feats[grid_idx] = torch.zeros_like(new_avg)
            grid_point_counts[grid_idx] = 0

            # Mark this grid as merged
            merged_grids.add(grid_idx)
            if old_count == 1:
                merged_grids.add(best_grid_idx)

            # IMPORTANT: Update cluster mapping to reflect reassignment
            cluster[single_pt_idx] = best_grid_idx

            reassigned_count += 1

        print()  # New line after progress bar
        print(f"  - Total reassigned: {reassigned_count}")
        print(f"  - Skipped (already merged): {skipped_grids}")
    elif single_point_grids:
        print("\nNo multi-point grids available for reassignment.")
        print("  - Single-point grids skipped: {:,}".format(len(single_point_grids)))
        skipped_grids = len(single_point_grids)

    # After reassignment, recalculate average features per grid
    grid_to_point_indices = []

    for grid_idx in tqdm(range(M)):
        if grid_point_counts[grid_idx] > 0:
            point_indices = torch.where(cluster == grid_idx)[0]
            # Map back to original indices
            original_indices = valid_indices[point_indices.cpu().numpy()]
            grid_to_point_indices.append(original_indices)
            
            assert grid_point_counts[grid_idx] == point_indices.shape[0]

    non_zero_rows = grid_avg_feats.any(dim=1)
    grid_avg_feats = grid_avg_feats[non_zero_rows]
    grid_point_counts = grid_point_counts[grid_point_counts > 0]
    # Compute point_to_grid_indices from grid_to_point_indices (after reassignment)
    # Initialize with -1 (points not in any grid)
    point_to_grid_indices = np.full(N, -1, dtype=np.int64)
    # Map each point to its final grid index
    for grid_idx, indices in enumerate(grid_to_point_indices):
        point_to_grid_indices[indices] = grid_idx
    return grid_avg_feats.cpu().numpy(), point_to_grid_indices, grid_point_counts.cpu().numpy()


def perform_svd_decomposition(
    valid_feat: np.ndarray,
    grid_avg_feats: np.ndarray,
    point_to_grid_indices: np.ndarray,
    grid_point_counts: np.ndarray,
    scene_name: str,
    ranks: List[int],
    use_rpca: bool = True,
    rpca_max_iter: int = 10,
    rpca_tol: float = 1e-7,
    device: Optional[str] = None,
) -> Dict[int, Dict]:
    """
    Perform SVD decomposition on grid average features for multiple ranks.

    Args:
        valid_feat: [N, D] - original valid features for reconstruction error calculation
        grid_avg_feats: [M, D] - grid average features
        point_to_grid_indices: [N] - mapping from point index to grid index
        grid_point_counts: [M] - number of points in each grid
        scene_name: Name of the scene (for logging)
        ranks: List of target ranks
        use_rpca: Whether to apply RPCA before SVD (default: True)
        rpca_max_iter: Maximum iterations for RPCA
        rpca_tol: Tolerance for RPCA convergence
        device: Device for RPCA (e.g., 'cuda:0', 'cuda:1', 'cpu', None for auto)

    Returns:
        Dictionary mapping rank to decomposition results
    """
    results = {}
    M, D = grid_avg_feats.shape

    print(f"  Scene: {scene_name}")
    print(f"    Grids: {M:,}, Feature dim: {D}")

    # Apply RPCA if requested
    # feat_matrix = torch.from_numpy(grid_avg_feats).to(device)
    if use_rpca:
        print(f"    Applying RPCA preprocessing...")
        rpca_results = apply_rpca(grid_avg_feats, max_iter=rpca_max_iter, tol=rpca_tol, device=device, structured=True, indices=point_to_grid_indices, d=grid_point_counts, return_tensors=True)
        feat_matrix = rpca_results['L']  # L is now a GPU tensor directly
        print("rpca finished")
        # Build weight vectors from grid_point_counts (for weighted SVD) - PyTorch GPU
        # Convert numpy arrays to PyTorch tensors on GPU
        d_gpu = torch.from_numpy(grid_point_counts).to(device)
        d_sqrt = torch.sqrt(d_gpu)  # [M] - sqrt counts
        d_inv_sqrt = 1.0 / d_sqrt  # [M] - inverse sqrt counts
        # Use broadcasting: [M, 1] * [M, D] = [M, D] via GPU (feat_matrix is already a GPU tensor)
        feat_matrix = d_sqrt.unsqueeze(-1) * feat_matrix  # [M, 1] * [M, D] = [M, D] broadcasting
        del rpca_results  # 释放 L 和 S 矩阵
        torch.cuda.empty_cache()  # 清空 CUDA 缓存

    # Perform SVD
    print(f"    Computing SVD...")
    U, S, Vt = torch.linalg.svd(feat_matrix, full_matrices=False)

    # Perform weighted SVD using broadcasting (no large matrix allocation)
    # U [M, D] * D_inv_sqrt [M, 1] -> [M, D] via broadcast
    U = U * d_inv_sqrt.unsqueeze(-1)  # [M, D] * [M, 1] = [M, D] broadcasting

    # Compute energy statistics
    energy = S ** 2
    total_energy = energy.sum()
    energy_cumsum = torch.cumsum(energy, dim=0) / total_energy

    print(f"    SVD Results:")
    print(f"      Total singular values: {len(S)}")
    print(f"      Max singular value: {S[0]:.6f}")
    print(f"      Min singular value: {S[-1]:.6f}")
    print(f"      Condition number: {S[0] / S[-1]:.2e}")

    # Extract U matrices for each rank
    for r in ranks:
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]
        compressed_feat = (U_r * S_r).cpu().numpy()  # Broadcasting: [M, r] * [r] -> [M, r]

        # Compute energy ratio at this rank
        rank_energy_ratio = energy_cumsum[r - 1]

        # Compute reconstruction error
        # Restore U_r from grid-level [M, r] to point-level [N_valid, r] using point_to_grid_indices
        ptg_indices = torch.from_numpy(point_to_grid_indices).to(device)
        valid_mask = ptg_indices >= 0
        U_r_point_level = U_r[ptg_indices[valid_mask]]  # [N_valid, r]

        # Reconstruct at point level and compare with original valid_feat
        # Use broadcasting instead of np.diag: [N, r] * [r, D] instead of [N, r] @ [r, r] @ [r, D]
        feat_reconstructed_point_level = U_r_point_level * S_r[None, :] @ Vt_r  # [N_valid, D]
        valid_feat_tensor = torch.from_numpy(valid_feat[valid_mask.cpu().numpy()]).to(device)
        error = torch.norm(valid_feat_tensor - feat_reconstructed_point_level, p='fro') / torch.norm(valid_feat_tensor, p='fro')

        results[r] = {
            'compressed': compressed_feat.astype(np.float32), 
            'indices': point_to_grid_indices.astype(np.int32), 
            'rank_energy_ratio': rank_energy_ratio,
            'reconstruction_error': error,
        }

        print(f"      Rank {r}: Energy={rank_energy_ratio:.4f} ({rank_energy_ratio*100:.2f}%), Error={error:.6f}, U shape={U_r.shape}")

    return results


def save_svd_results(
    svd_results: Dict[int, Dict],
    scene_name: str,
    output_dir: str,
    base_filename: str = "lang_feat_grid_svd",
) -> Dict[int, str]:
    """
    Save compressed features and point-to-grid indices as .npz files.

    Args:
        svd_results: Dictionary of SVD results per rank (contains 'compressed' and 'indices')
        scene_name: Name of the scene
        output_dir: Output directory path
        base_filename: Base filename for output files

    Returns:
        Dictionary mapping rank to output file path
    """
    output_path = Path(output_dir) / scene_name
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    for r, results in svd_results.items():
        output_file = output_path / f"{base_filename}_r{r}.npz"

        # Save compressed features and point-to-grid indices
        # This matches the loading logic in compute_grid.py
        compressed_feat = results["compressed"]
        indices = results["indices"]

        np.savez(output_file, compressed=compressed_feat, indices=indices)

        saved_paths[r] = str(output_file)

    return saved_paths


def save_grid_meta_data(point_to_grid_indices: np.ndarray, grid_point_counts: np.ndarray, output_dir: str, scene_name: str):
    """
    Save grid metadata to JSON file.

    Args:
        point_to_grid_indices: [N] - final grid index for each point (-1 if not in any grid)
        grid_point_counts: [M] - number of gaussians in each grid
        output_dir: Output directory path
        scene_name: Name of the scene
    """
    output_path = Path(output_dir) / scene_name
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "grid_meta_data.json"

    # Convert to serializable format
    meta_data = {
        # "point_to_grid_indices": point_to_grid_indices.astype(int).tolist(),
        "grid_point_counts": float(grid_point_counts.mean()),
        "num_grids": len(grid_point_counts),
        "num_points_with_grid": int(np.sum(point_to_grid_indices >= 0)),
    }

    with open(output_file, 'w') as f:
        json.dump(meta_data, f)

    print(f"    Saved grid metadata: {output_file}")


def find_scenes(data_root: str, dataset: str, split: str) -> List[str]:
    """
    Find all scene directories for a given dataset and split.

    Args:
        data_root: Root directory containing datasets
        dataset: Dataset name (e.g., "3DOVS", "lerf_ovs")
        split: "train", "val", or "test"

    Returns:
        List of scene directory paths
    """
    dataset_path = os.path.join(data_root, dataset, split)

    if not os.path.exists(dataset_path):
        return []

    scene_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            # Check if this directory contains required files
            coord_path = os.path.join(item_path, "coord.npy")
            lang_feat_path = os.path.join(item_path, "lang_feat.npy")
            if os.path.exists(coord_path) and os.path.exists(lang_feat_path):
                scene_dirs.append(item_path)

    return sorted(scene_dirs)


def process_single_scene(
    data_dir: str,
    grid_size: float,
    ranks: List[int],
    output_dir: str,
    use_rpca: bool,
    rpca_max_iter: int,
    rpca_tol: float,
    device: Optional[str],
    torch_device: str = "cuda",
) -> bool:
    """
    Process a single scene: compute grid average features, apply RPCA+SVD, and save results.

    Args:
        data_dir: Path to scene directory
        grid_size: Grid size in meters
        ranks: List of target ranks for SVD
        output_dir: Output directory path
        use_rpca: Whether to apply RPCA
        rpca_max_iter: Maximum iterations for RPCA
        rpca_tol: Tolerance for RPCA convergence
        device: Device for RPCA
        torch_device: Device for torch operations

    Returns:
        True if successful, False otherwise
    """
    scene_name = os.path.basename(data_dir)

    try:
        # Load data
        coord, lang_feat, valid_mask = load_scene_data(data_dir)

        print(f"\n  Processing scene: {scene_name}")
        print(f"    Points: {coord.shape[0]:,}")

        # Compute grid average features
        grid_avg_feats, point_to_grid_indices, grid_point_counts = compute_grid_average_features(
            coord, lang_feat, valid_mask, grid_size, torch_device
        )

        # Get valid features for reconstruction error calculation
        valid_feat = lang_feat if valid_mask is None else lang_feat[valid_mask]

        # Perform SVD decomposition
        svd_results = perform_svd_decomposition(
            valid_feat,
            grid_avg_feats,
            point_to_grid_indices,
            grid_point_counts,
            scene_name,
            ranks,
            use_rpca=use_rpca,
            rpca_max_iter=rpca_max_iter,
            rpca_tol=rpca_tol,
            device=device,
        )

        # Save SVD results
        saved_paths = save_svd_results(
            svd_results,
            scene_name,
            output_dir,
        )

        # Save grid metadata
        save_grid_meta_data(point_to_grid_indices, grid_point_counts, output_dir, scene_name)

        # Print saved paths
        for r, path in saved_paths.items():
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"    Saved rank {r}: {path} ({file_size_mb:.2f} MB)")

        return True

    except Exception as e:
        print(f"    Error processing scene {scene_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Grid-based SVD compression for language features (RPCA applied per scene)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single scene mode (GPU RPCA by default)
    python tools/compress_grid_svd.py --data_dir /path/to/scene --grid_size 0.01

    # Batch mode - all scenes in a dataset (each scene processed separately)
    python tools/compress_grid_svd.py --data_root /new_data/cyf/projects/SceneSplat/gaussian_train --dataset 3DOVS --split train

    # Batch mode - specific scenes
    python tools/compress_grid_svd.py --data_root /new_data/cyf/projects/SceneSplat/gaussian_train --dataset 3DOVS --split train --scenes scene1,scene2

    # With custom ranks
    python tools/compress_grid_svd.py --data_root /new_data/cyf/projects/SceneSplat/gaussian_train --dataset 3DOVS --split train --ranks 8,16,32

    # Use CPU for RPCA
    python tools/compress_grid_svd.py --data_root /new_data/cyf/projects/SceneSplat/gaussian_train --dataset 3DOVS --split train --device cpu

    # Disable RPCA
    python tools/compress_grid_svd.py --data_root /new_data/cyf/projects/SceneSplat/gaussian_train --dataset 3DOVS --split train --no_rpca
        """
    )

    # Input source (single file or batch)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to a single scene directory (single scene mode)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/new_data/cyf/projects/SceneSplat/gaussian_train",
        help="Root directory containing datasets (batch mode)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="3DOVS",
        choices=["3DOVS", "lerf_ovs"],
        help="Dataset name (for batch mode)",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Comma-separated list of scene names (default: all scenes in split)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.01,
        help="Grid size in meters (default: 0.01)",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="8,16,32",
        help="Comma-separated list of ranks for SVD decomposition (default: 8,16,32)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./grid_svd_output",
        help="Output directory for SVD results (default: ./grid_svd_output)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="lang_feat_grid_svd",
        help="Base filename for output files (default: lang_feat_grid_svd)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for torch and RPCA operations (e.g., cuda, cuda:0, cuda:1, cpu) (default: cuda)",
    )
    parser.add_argument(
        "--no_rpca",
        action="store_true",
        help="Disable RPCA preprocessing (default: enabled)",
    )
    parser.add_argument(
        "--rpca_max_iter",
        type=int,
        default=50,
        help="Maximum iterations for RPCA (default: 50)",
    )
    parser.add_argument(
        "--rpca_tol",
        type=float,
        default=1e-5,
        help="Tolerance for RPCA convergence (default: 1e-5)",
    )

    args = parser.parse_args()

    # Parse ranks
    ranks = [int(r.strip()) for r in args.ranks.split(',')]

    # Determine device string for RPCA
    device = args.device

    # Print device info
    print("=" * 60)
    print("Device Configuration")
    print("=" * 60)
    if CUDA_AVAILABLE:
        if device.startswith('cuda'):
            gpu_id = int(device.split(':')[1]) if ':' in device else 0
            if gpu_id < torch.cuda.device_count():
                print(f"RPCA Device: GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
            else:
                print(f"Warning: GPU {gpu_id} not found, using GPU 0")
                device = 'cuda:0'
                print(f"RPCA Device: GPU 0 ({torch.cuda.get_device_name(0)})")
        else:
            print(f"RPCA Device: CPU (via pyrpca library)")
    else:
        print("GPU (CUDA) not available, using CPU for RPCA")
        device = 'cpu'
    print(f"RPCA (CPU via pyrpca) available: {RPCA_CPU_AVAILABLE}")
    print(f"RPCA enabled by default: {not args.no_rpca}")

    # Check RPCA availability if requested
    use_rpca = not args.no_rpca

    # Process data
    if args.data_dir is not None:
        # Single scene mode
        print("\n" + "=" * 60)
        print("Single Scene Mode")
        print("=" * 60)

        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory not found: {args.data_dir}")
            return

        success = process_single_scene(
            args.data_dir,
            args.grid_size,
            ranks,
            args.output_dir,
            use_rpca,
            args.rpca_max_iter,
            args.rpca_tol,
            device,
            args.device,
        )

        if success:
            print("\n" + "=" * 60)
            print("Done!")
            print("=" * 60)
        else:
            print("\nProcessing failed!")

    else:
        # Batch mode
        print("\n" + "=" * 60)
        print("Batch Processing Mode")
        print("=" * 60)
        print(f"Dataset: {args.dataset}")
        print(f"Split: {args.split}")
        print(f"Grid size: {args.grid_size}m")
        print(f"RPCA per scene: {use_rpca}")

        # Find scenes
        if args.scenes is not None:
            scene_names = [s.strip() for s in args.scenes.split(',')]
            scenes = [os.path.join(args.data_root, args.dataset, args.split, s) for s in scene_names]
        else:
            scenes = find_scenes(args.data_root, args.dataset, args.split)

        if not scenes:
            print(f"Error: No scenes found for dataset {args.dataset} split {args.split}")
            return

        print(f"Found {len(scenes)} scenes to process")
        print("=" * 60)

        # Process each scene
        success_count = 0
        fail_count = 0

        for scene_dir in tqdm(scenes, desc="Processing scenes"):
            success = process_single_scene(
                scene_dir,
                args.grid_size,
                ranks,
                args.output_dir,
                use_rpca,
                args.rpca_max_iter,
                args.rpca_tol,
                device,
                args.device,
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

        # Print final summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Scenes processed: {len(scenes)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Output directory: {args.output_dir}")
        print("\nDone!")


if __name__ == "__main__":
    main()
