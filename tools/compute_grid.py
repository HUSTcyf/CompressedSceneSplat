#!/usr/bin/env python3
"""
Grid Coordinate Computation and Analysis Tool

This script demonstrates how grid coordinates are computed in LitePT:
1. Load 3DGS data (coord, color, opacity, quat, scale)
2. Load lang_feat and valid_feat_mask
3. Filter valid points
4. Concatenate features into 11 channels
5. Compute grid coordinates
6. Apply Z-order Morton encoding
7. Find unique grids using torch.unique
8. Perform grid reassignment (single points -> multi-point grids)
9. Compute cosine similarity: each point vs grid average feature
10. Output detailed statistics

Usage (Single Scene):
    python tools/compute_grid.py --data_dir /path/to/scene --grid_size 0.01

Usage (Batch Mode):
    python tools/compute_grid.py --dataset 3DOVS --split train --scenes bed,sofa --grid_size 0.01

Usage (Save to JSON):
    python tools/compute_grid.py --dataset 3DOVS --split train --output_json /path/to/output_dir
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary modules
from pointcept.models.utils.structure import Point
from pointcept.models.utils.serialization import encode


def load_3dgs_data(data_dir, device="cuda", load_lang_feat=True, use_svd=False, svd_rank=16):
    """Load 3DGS data from .npy files.

    Args:
        data_dir: Path to scene directory
        device: Device to use
        load_lang_feat: Whether to load language features
        use_svd: If True, load lang_feat_svd.npz (compressed, already filtered)
        svd_rank: Number of SVD components to use for compressed features (default: 16)

    Returns:
        coord: [N, 3] - 3D coordinates
        feat: [N, 11] - concatenated features (color+opacity+quat+scale)
        lang_feat: [N, D] - language features (optional)
        valid_mask: [N] - valid feature mask
        grid_size: float - grid size for voxelization
        svd_features_loaded: bool - whether SVD features were loaded (already filtered)
    """
    print("=" * 60)
    print("Loading 3DGS data from: " + data_dir)
    print("=" * 60)

    # Load individual files
    coord_path = os.path.join(data_dir, "coord.npy")
    color_path = os.path.join(data_dir, "color.npy")
    opacity_path = os.path.join(data_dir, "opacity.npy")
    quat_path = os.path.join(data_dir, "quat.npy")
    scale_path = os.path.join(data_dir, "scale.npy")
    valid_mask_path = os.path.join(data_dir, "valid_feat_mask.npy")

    print("Loading files:")
    print("  - coord: " + coord_path)
    print("  - color: " + color_path)
    print("  - opacity: " + opacity_path)
    print("  - quat: " + quat_path)
    print("  - scale: " + scale_path)
    print("  - valid_feat_mask: " + valid_mask_path)

    coord = np.load(coord_path).astype(np.float32)
    color = np.load(color_path).astype(np.float32)
    opacity = np.load(opacity_path).astype(np.float32)
    quat = np.load(quat_path).astype(np.float32)
    scale = np.load(scale_path).astype(np.float32)

    # Concatenate features: color(3) + opacity(1) + quat(4) + scale(3) = 11 channels
    feat = np.concatenate([color, opacity, quat, scale], axis=1)

    # Load language features
    lang_feat = None

    if load_lang_feat:
        if use_svd:
            # Load compressed SVD features (already filtered by valid_feat_mask)
            lang_feat_svd_path = os.path.join(data_dir, "lang_feat_svd.npz")
            print("  - lang_feat_svd: " + lang_feat_svd_path)
            if os.path.exists(lang_feat_svd_path):
                svd_data = np.load(lang_feat_svd_path)
                # Compute compressed features from U, S matrices
                # Take first r components: U[:, :r] * S[:r]
                U = svd_data['U'].astype(np.float32)  # [N_valid, k] where k >= r
                S = svd_data['S'].astype(np.float32)  # [k,] where k >= r

                # Check available dimensions
                k_available = min(U.shape[1], len(S))
                r = min(svd_rank, k_available)

                print("    - Loaded SVD matrices:")
                print("      - U shape: " + str(U.shape))
                print("      - S shape: " + str(S.shape))
                print("      - Available components: " + str(k_available))
                print("      - Using rank r=" + str(r))

                # Take first r components and compute compressed features
                U_r = U[:, :r]  # [N_valid, r]
                S_r = S[:r]     # [r,]
                lang_feat = U_r * S_r  # Broadcasting: [N_valid, r] * [r,] -> [N_valid, r]

                print("    - Compressed feature shape: " + str(lang_feat.shape))
                print("    - Note: SVD features are already filtered by valid_feat_mask")
            else:
                print("    - lang_feat_svd not found, falling back to lang_feat.npy")
                use_svd = False  # Fall back to regular features

        if not use_svd:
            # Load full language features
            lang_feat_path = os.path.join(data_dir, "lang_feat.npy")
            print("  - lang_feat: " + lang_feat_path)
            if os.path.exists(lang_feat_path):
                lang_feat = np.load(lang_feat_path).astype(np.float32)
                print("    - Loaded lang_feat with shape: " + str(lang_feat.shape))
            else:
                print("    - lang_feat not available, skipping cosine similarity computation")

    # Load valid mask
    valid_mask = None
    if os.path.exists(valid_mask_path):
        valid_mask = np.load(valid_mask_path).astype(bool)
        print("    - Loaded valid_feat_mask with shape: " + str(valid_mask.shape))
        print("    - Valid points: " + str(valid_mask.sum()) + "/" + str(valid_mask.shape[0]))
    else:
        print("    - valid_feat_mask not found, using all points")

    num_points = coord.shape[0]
    grid_size = 0.01  # default grid size (1cm)

    print("\nData loaded successfully:")
    print("  - Number of points: {:,}".format(num_points))
    coord_min = coord.min(axis=0)
    coord_max = coord.max(axis=0)
    print("  - Coord range: [{:.3f}, {:.3f}, {:.3f}] to [{:.3f}, {:.3f}, {:.3f}]".format(
        coord_min[0], coord_min[1], coord_min[2],
        coord_max[0], coord_max[1], coord_max[2]
    ))
    print("  - Feature shape: " + str(feat.shape))
    if lang_feat is not None:
        print("  - Lang feat shape: " + str(lang_feat.shape))
    if valid_mask is not None:
        valid_count = valid_mask.sum()
        valid_pct = valid_count * 100.0 / num_points
        print("  - Valid points: {:,}/{:,} ({:.2f}%)".format(valid_count, num_points, valid_pct))
    print("  - Grid size: " + str(grid_size) + "m")

    return coord, feat, lang_feat, valid_mask, grid_size


def compute_cosine_similarity_for_grid(lang_feats_tensor):
    """Compute average pairwise cosine similarity for features in a grid.

    Args:
        lang_feats_tensor: [K, D] - language features for K points in a grid

    Returns:
        avg_cosine_sim: float - average pairwise cosine similarity
    """
    K = lang_feats_tensor.shape[0]

    if K <= 1: return 1.0

    # Normalize features to unit length
    norms = torch.norm(lang_feats_tensor, p=2, dim=1, keepdim=True)  # [K, 1]
    normalized_feats = lang_feats_tensor / norms  # [K, D]

    # Compute pairwise cosine similarity matrix
    # sim_matrix[i, j] = cos(feats[i], feats[j])
    sim_matrix = torch.mm(normalized_feats, normalized_feats.t())  # [K, K]

    # Extract upper triangle (excluding diagonal) to avoid duplicate and self-comparisons
    # triu_indices: returns indices of upper triangle
    K_int = lang_feats_tensor.shape[0]
    triu_indices = torch.triu_indices(K_int, K_int)
    # Only take off-diagonal elements
    mask = triu_indices[0] != triu_indices[1]
    off_diagonal_sim = sim_matrix[triu_indices[0][mask], triu_indices[1][mask]]

    # Return average cosine similarity
    avg_cosine_sim = off_diagonal_sim.mean().item()
    return avg_cosine_sim


def compute_grid_statistics(coord, feat, lang_feat, valid_mask, grid_size, device="cuda"):
    """Compute grid coordinates and analyze statistics.

    This function mimics the exact process used in LitePT:
    1. Create Point object
    2. Call serialization (computes grid_coord)
    3. Analyze grid distribution
    4. Compute cosine similarity for points within each grid
    """
    print("\n" + "=" * 60)
    print("Computing Grid Statistics")
    print("=" * 60)

    # Check if lang_feat is already filtered (SVD features)
    lang_feat_already_filtered = False
    if lang_feat is not None and lang_feat.shape[0] != coord.shape[0]:
        lang_feat_already_filtered = True
        print("Detected SVD features: lang_feat shape ({}) != coord shape ({})".format(
            lang_feat.shape[0], coord.shape[0]))
        print("  - SVD features are already filtered, will not apply valid_mask to lang_feat")

    # Apply valid mask if available
    if valid_mask is not None:
        original_num_points = coord.shape[0]
        valid_indices = np.where(valid_mask)[0]
        coord = coord[valid_indices]
        feat = feat[valid_indices]
        # Only filter lang_feat if it's not already filtered (SVD features)
        if lang_feat is not None and not lang_feat_already_filtered:
            lang_feat = lang_feat[valid_indices]
        print("Filtered to valid points:")
        print("  - Original points: {:,}".format(original_num_points))
        print("  - Valid points: {:,}".format(coord.shape[0]))
        print("  - Filtered out: {:,} points".format(original_num_points - coord.shape[0]))

    # Convert to torch tensors
    coord_t = torch.from_numpy(coord).to(device)
    feat_t = torch.from_numpy(feat).to(device)
    batch = torch.zeros(coord_t.shape[0], dtype=torch.long, device=device)

    # Convert lang_feat if available
    lang_feat_t = None
    if lang_feat is not None:
        lang_feat_t = torch.from_numpy(lang_feat).to(device)
        print("  - Lang feat loaded on device")

    N = coord_t.shape[0]
    print("\nInput tensors:")
    print("  - coord shape: " + str(coord_t.shape))
    print("  - feat shape: " + str(feat_t.shape))
    if lang_feat_t is not None:
        print("  - lang_feat shape: " + str(lang_feat_t.shape))
    print("  - batch shape: " + str(batch.shape))

    # Step 1: Create Point object
    print("\n" + "=" * 60)
    print("Step 1: Creating Point object")
    print("=" * 60)

    point = Point({
        "coord": coord_t,
        "feat": feat_t,
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
    print("\nUnique grids found: {:,}".format(M))

    # Step 5: Compute cosine similarity for each grid
    print("\n" + "=" * 60)
    print("Step 5: Computing Cosine Similarity per Grid with Reassignment")
    print("=" * 60)

    grid_cosine_sims = []
    grid_cosine_counts = []
    skipped_grids = 0  # Count grids skipped due to single point

    if lang_feat_t is not None:
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
            print(f"  - K (closest grids to consider): {K}")

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
                if not closest_grid_indices:
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

        # After reassignment, recalculate cosine similarity using new method
        # New method: compute similarity between each point and grid average feature
        if multi_point_grid_indices:
            print("\n[Re-calculating Cosine Similarity] Computing similarity with updated grids...")
            print("  - Method: each point vs grid average feature (not pairwise)")
            
            final_grid_cosine_sims = []
            final_grid_counts = []
            
            for grid_idx in tqdm(multi_point_grid_indices, desc="final sim"):
                # Get all points in this grid (including reassigned ones)
                point_indices = torch.where(cluster == grid_idx)[0]
                grid_lang_feats = lang_feat_t[point_indices]  # [K, D]
                
                # Compute grid average feature
                grid_avg_feat = grid_lang_feats.mean(dim=0)  # [D]
                
                # Normalize grid average feature
                grid_avg_norm = grid_avg_feat / torch.norm(grid_avg_feat)  # [D]
                
                # Normalize all point features
                point_norms = torch.norm(grid_lang_feats, dim=1, keepdim=True)  # [K, 1]
                point_normalized = grid_lang_feats / point_norms  # [K, D]
                
                # Compute cosine similarity between each point and grid average
                point_sims = torch.mm(point_normalized, grid_avg_norm.unsqueeze(1)).squeeze()  # [K]
                
                # Average similarity across all points in this grid
                avg_sim = point_sims.mean().item()
                final_grid_cosine_sims.append(avg_sim)
                final_grid_counts.append(point_indices.shape[0])
            
            print()  # New line after progress bar
            print("  - Re-calculated for {:,} grids".format(len(final_grid_cosine_sims)))
            
            # Replace old cosine sims with new ones
            grid_cosine_sims = final_grid_cosine_sims
            grid_cosine_counts = final_grid_counts

        # Convert to tensor for statistics
        if len(grid_cosine_sims) > 0:
            grid_cosine_sims = torch.tensor(grid_cosine_sims, device=device)

            print("Cosine similarity computation complete:")
            print("  - Total grids: {:,}".format(M))
            print("  - Grids with >1 point: {:,}".format(len(grid_cosine_sims)))
            print("  - Grids skipped (1 point): {:,}".format(skipped_grids))

        # Step 6: Overall statistics (mean and variance only)
        print("\n" + "=" * 60)
        print("Step 6: Overall Cosine Similarity Statistics")
        print("=" * 60)

        if len(grid_cosine_sims) > 0:
            # Overall mean and variance
            overall_mean = grid_cosine_sims.mean().item()
            overall_var = grid_cosine_sims.var().item()
            overall_std = grid_cosine_sims.std().item()
            overall_min = grid_cosine_sims.min().item()
            overall_max = grid_cosine_sims.max().item()

            print("Overall cosine similarity across all grids:")
            print("  - Mean: {:.4f}".format(overall_mean))
            print("  - Variance: {:.4f}".format(overall_var))
            print("  - Std Dev: {:.4f}".format(overall_std))
            print("  - Min: {:.4f}".format(overall_min))
            print("  - Max: {:.4f}".format(overall_max))

        # Print grid_cosine_counts statistics
        if len(grid_cosine_counts) > 0:
            grid_cosine_counts = torch.tensor(grid_cosine_counts, device=device)
            print("\nGrid point counts (cosine similarity grids):")
            print("  - Number of grids: {:,}".format(grid_cosine_counts.shape[0]))
            print("  - Mean points per grid: {:.2f}".format(grid_cosine_counts.float().mean().item()))
            print("  - Std dev: {:.2f}".format(grid_cosine_counts.float().std().item()))
            print("  - Min: {:.0f}".format(grid_cosine_counts.min().item()))
            print("  - Max: {:.0f}".format(grid_cosine_counts.max().item()))
            print("  - Median: {:.0f}".format(grid_cosine_counts.median().item()))
        else:
            print("No grids with >1 point found, skipping cosine similarity statistics")
    else:
        print("Lang feat not available, skipping cosine similarity computation")

    # Calculate basic grid statistics
    print("\n" + "=" * 60)
    print("Step 7: Basic Grid Statistics")
    print("=" * 60)

    # Calculate statistics
    points_per_grid = counts.float().mean().item()
    points_per_grid_std = counts.float().std().item()
    points_per_grid_min = counts.min().item()
    points_per_grid_max = counts.max().item()

    print("Grid statistics:")
    print("  - Total points (N): {:,}".format(N))
    print("  - Unique grids (M): {:,}".format(M))
    print("  - Downsampling ratio: {:.2f}% (1:{:.2f})".format(
        M / N * 100, 1.0 / (M / N)
    ))
    print("  - Points per grid (mean): {:.2f}".format(points_per_grid))
    print("  - Points per grid (std): {:.2f}".format(points_per_grid_std))
    print("  - Points per grid (min): {:.2f}".format(points_per_grid_min))
    print("  - Points per grid (max): {:.2f}".format(points_per_grid_max))

    # Analyze batch distribution
    print("\nBatch distribution:")
    print("=" * 60)

    if batch.unique().shape[0] > 1:
        print("  - Multiple batches detected: " + str(batch.unique().shape[0]))
        for b in batch.unique()[:10]:  # Show first 10
            batch_mask = (batch == b).sum().item()
            print("  - Batch " + str(b.item()) + ": " + str(batch_mask) + " points")
    else:
        print("  - Single batch: " + str(batch[0].item()))

    result = {
        "num_points": N,
        "num_grids": M,
        "downsampling_ratio": M / N,
        "points_per_grid_mean": points_per_grid,
        "points_per_grid_std": points_per_grid_std,
        "morton_codes": morton_codes.shape,
    }

    if len(grid_cosine_sims) > 0:
        result.update({
            "num_grids": grid_cosine_counts.shape[0],
            "downsampling_ratio": grid_cosine_counts.shape[0] / N,
            "points_per_grid_mean": grid_cosine_counts.float().mean().item(),
            "points_per_grid_std": grid_cosine_counts.float().std().item(),
            "cosine_sim_mean": grid_cosine_sims.mean().item(),
            "cosine_sim_var": grid_cosine_sims.var().item(),
            "cosine_sim_std": grid_cosine_sims.std().item(),
            "cosine_sim_min": grid_cosine_sims.min().item(),
            "cosine_sim_max": grid_cosine_sims.max().item(),
        })

    return result


def find_scenes(data_root, dataset, split="train"):
    """Find all scene directories for a given dataset and split.

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
            if os.path.exists(coord_path):
                scene_dirs.append(item_path)

    return sorted(scene_dirs)


def aggregate_stats(all_stats):
    """Aggregate statistics from multiple scenes.

    Args:
        all_stats: List of statistics dictionaries from each scene

    Returns:
        Aggregated statistics dictionary
    """
    if not all_stats:
        return {
            'num_scenes': 0,
            'valid_scenes': 0,
            'total_points': 0,
            'total_grids': 0,
            'avg_points_per_grid': 0,
            'avg_downsample_ratio': 0,
            'points_per_grid_std': 0,
        }

    # Initialize aggregators
    total_points = 0
    total_grids = 0
    total_downsample_ratio = 0
    total_points_per_grid = []

    if 'cosine_sim_mean' in all_stats[0]:
        cosine_sim_values = []

    num_scenes = len(all_stats)
    valid_scenes = 0

    for stats in all_stats:
        if stats.get('num_points', 0) > 0:
            valid_scenes += 1
            total_points += stats['num_points']
            total_grids += stats['num_grids']
            total_downsample_ratio += stats.get('downsampling_ratio', 0)
            total_points_per_grid.append(stats.get('points_per_grid_mean', 0))

            if 'cosine_sim_mean' in stats:
                cosine_sim_values.append(stats['cosine_sim_mean'])

    # Compute averages
    avg_points_per_grid = sum(total_points_per_grid) / len(total_points_per_grid) if total_points_per_grid else 0
    avg_downsample_ratio = total_downsample_ratio / valid_scenes if valid_scenes > 0 else 0
    avg_cosine_sim = sum(cosine_sim_values) / len(cosine_sim_values) if cosine_sim_values else None

    result = {
        'num_scenes': num_scenes,
        'valid_scenes': valid_scenes,
        'total_points': total_points,
        'total_grids': total_grids,
        'avg_points_per_grid': avg_points_per_grid,
        'avg_downsample_ratio': avg_downsample_ratio,
        'points_per_grid_std': np.std(total_points_per_grid).item() if total_points_per_grid else 0,
    }

    if avg_cosine_sim is not None:
        result['cosine_sim_mean'] = avg_cosine_sim
        result['cosine_sim_std'] = np.std(cosine_sim_values).item()
        result['cosine_sim_min'] = min(cosine_sim_values)
        result['cosine_sim_max'] = max(cosine_sim_values)

    return result


def convert_stats_for_json(stats):
    """Convert stats dict to JSON-serializable format."""
    result = {}
    for key, value in stats.items():
        if isinstance(value, (np.integer, np.floating)):
            result[key] = float(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            result[key] = value.cpu().tolist()
        else:
            result[key] = value
    return result


def process_single_scene(data_dir, grid_size, device, load_lang_feat, use_svd=False, svd_rank=16):
    """Process a single scene and return statistics."""
    coord, feat, lang_feat, valid_mask, loaded_grid_size = load_3dgs_data(
        data_dir, device, load_lang_feat=load_lang_feat, use_svd=use_svd, svd_rank=svd_rank
    )

    if grid_size != loaded_grid_size:
        grid_size = grid_size

    stats = compute_grid_statistics(coord, feat, lang_feat, valid_mask, grid_size, device)
    return stats


def process_batch(data_root, dataset, scenes, split, grid_size, device, load_lang_feat, use_svd=False, svd_rank=16):
    """Process multiple scenes in batch."""
    print("=" * 60)
    print("Batch Processing Mode")
    print("=" * 60)
    print("Dataset: " + dataset)
    print("Split: " + split)
    print("Scenes to process: " + str(len(scenes)))
    print("Grid size: " + str(grid_size))
    if use_svd:
        print("Using SVD compressed features (rank={})".format(svd_rank))
    print("=" * 60)

    all_stats = []
    scene_names = []

    for i, scene_dir in enumerate(scenes):
        scene_name = os.path.basename(scene_dir)
        scene_names.append(scene_name)

        print("\n" + "-" * 60)
        print("Processing scene " + str(i+1) + "/" + str(len(scenes)) + ": " + scene_name)
        print("-" * 60)

        try:
            stats = process_single_scene(scene_dir, grid_size, device, load_lang_feat, use_svd, svd_rank)
            stats['scene_name'] = scene_name
            all_stats.append(stats)
        except Exception as e:
            print("Error processing scene " + scene_name + ": " + str(e))

    # Aggregate statistics
    print("\n" + "=" * 60)
    print("Aggregated Statistics Across All Scenes")
    print("=" * 60)

    agg_stats = aggregate_stats(all_stats)
    print(agg_stats)

    print("Number of scenes: " + str(agg_stats["num_scenes"]))
    print("Valid scenes: " + str(agg_stats['valid_scenes']))
    print("Total points across all scenes: {:,}".format(agg_stats['total_points']))
    print("Total grids across all scenes: {:,}".format(agg_stats['total_grids']))
    print("Average points per grid: {:.2f}".format(agg_stats['avg_points_per_grid']))
    print("Average downsampling ratio: {:.2f}%".format(agg_stats['avg_downsample_ratio'] * 100))

    if 'cosine_sim_mean' in agg_stats:
        print("\nCosine Similarity Summary (aggregated):")
        print("  - Mean: {:.4f}".format(agg_stats['cosine_sim_mean']))
        print("  - Std: {:.4f}".format(agg_stats['cosine_sim_std']))
        print("  - Min: {:.4f}".format(agg_stats['cosine_sim_min']))
        print("  - Max: {:.4f}".format(agg_stats['cosine_sim_max']))

    # Per-scene summary
    print("\nPer-Scene Summary:")
    print("-" * 90)

    for stats in all_stats:
        scene_name = stats.get('scene_name', 'N/A')
        num_points = stats.get('num_points', 0)
        num_grids = stats.get('num_grids', 0)
        downsample = stats.get('downsampling_ratio', 0)
        cosine_sim = stats.get('cosine_sim_mean', -1)

        cos_str = "{:.4f}".format(cosine_sim) if cosine_sim >= 0 else "N/A"
        print(scene_name.ljust(30) + "{:,}".format(num_points).rjust(15) + "{:,}".format(num_grids).rjust(15) + "{:.2f}%".format(downsample * 100).rjust(12) + cos_str.rjust(10))

    return agg_stats, all_stats


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Compute and analyze grid coordinates for 3DGS data"
    )
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
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="z",
        choices=["z", "z-trans", "hilbert", "hilbert-trans"],
        help="Serialization order (default: z)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top grids to show (default: 20)",
    )
    parser.add_argument(
        "--no_lang_feat",
        action="store_true",
        help="Disable lang_feat loading and cosine similarity computation",
    )
    parser.add_argument(
        "--use_svd",
        action="store_true",
        help="Use SVD compressed features (lang_feat_svd.npz) instead of full lang_feat.npy",
    )
    parser.add_argument(
        "--svd_rank",
        type=int,
        default=16,
        help="Number of SVD components to use for compressed features (default: 16)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Directory path to save per-scene results as JSON files (filename = scene_name.json)",
    )

    args = parser.parse_args()

    # Determine mode: single scene or batch
    if args.data_dir is not None:
        # Single scene mode
        print("Single Scene Mode")
        print("=" * 60)

        # Check if data directory exists
        if not os.path.exists(args.data_dir):
            print("Error: Data directory not found: " + args.data_dir)
            return

        load_lang_feat = not args.no_lang_feat
        stats = process_single_scene(args.data_dir, args.grid_size, args.device, load_lang_feat, args.use_svd, args.svd_rank)

        # Print final summary
        print("\n" + "=" * 60)
        print("Final Summary")
        print("=" * 60)
        print("Scene directory: " + args.data_dir)
        print("Grid size: " + str(args.grid_size) + "m")
        print("Serialization order: " + args.order)
        print("Total points: " + str(stats['num_points']))
        print("Unique grids: " + str(stats['num_grids']))
        print("Downsampling ratio: " + str(stats['downsampling_ratio']))
        print("Points per grid (mean): " + str(stats['points_per_grid_mean']))
        print("Points per grid (std): " + str(stats['points_per_grid_std']))
        print("Morton codes shape: " + str(stats['morton_codes']))

        if 'cosine_sim_mean' in stats:
            print("\nCosine Similarity Summary:")
            print("  - Mean: " + "{:.4f}".format(stats['cosine_sim_mean']))
            print("  - Std: " + "{:.4f}".format(stats['cosine_sim_std']))
            print("  - Min: " + "{:.4f}".format(stats['cosine_sim_min']))
            print("  - Max: " + "{:.4f}".format(stats['cosine_sim_max']))
            print("  - Median: " + "{:.4f}".format(stats['cosine_sim_median']))

        # Save to JSON if requested
        if args.output_json is not None:
            scene_name = os.path.basename(args.data_dir)
            stats['scene_name'] = scene_name
            # Convert stats to JSON-serializable format
            stats_json = convert_stats_for_json(stats)
            # Create output directory if needed
            os.makedirs(args.output_json, exist_ok=True)
            # Save with scene_name as filename
            output_path = os.path.join(args.output_json, f"{scene_name}.json")
            with open(output_path, 'w') as f:
                json.dump(stats_json, f, indent=2)
            print("\nResults saved to: " + output_path)
    else:
        # Batch mode
        load_lang_feat = not args.no_lang_feat

        # Find scenes
        if args.scenes is not None:
            scene_names = [s.strip() for s in args.scenes.split(',')]
            scenes = [os.path.join(args.data_root, args.dataset, args.split, s) for s in scene_names]
        else:
            scenes = find_scenes(args.data_root, args.dataset, args.split)

        if not scenes:
            print("Error: No scenes found for dataset " + args.dataset + " split " + args.split)
            return

        # Process batch
        agg_stats, all_stats = process_batch(
            args.data_root, args.dataset, scenes, args.split,
            args.grid_size, args.device, load_lang_feat, args.use_svd, args.svd_rank
        )

        # Save to JSON if requested
        if args.output_json is not None:
            # Create output directory if needed
            os.makedirs(args.output_json, exist_ok=True)
            print("\n" + "=" * 60)
            print("Saving results to JSON files")
            print("=" * 60)

            saved_count = 0
            for stats in all_stats:
                scene_name = stats.get('scene_name', 'unknown')
                stats_json = convert_stats_for_json(stats)
                output_path = os.path.join(args.output_json, f"{scene_name}.json")
                with open(output_path, 'w') as f:
                    json.dump(stats_json, f, indent=2)
                print(f"  - Saved: {output_path}")
                saved_count += 1

            print(f"Total saved: {saved_count} JSON files to {args.output_json}")


if __name__ == "__main__":
    main()
