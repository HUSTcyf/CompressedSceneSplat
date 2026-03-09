"""
Density-Invariant Trainer for Point Cloud Training with Rank-Aware Data Loading

This trainer implements density-invariant training strategy with three scenarios:
1. Dense input (full point cloud)
2. Half density per grid (30%-70% random fluctuation)
3. Single point per grid

Supports multi-GPU training with rank-aware data loading, where each rank only
loads data assigned to it using the pre-computed point-to-grid mappings from
lang_feat_grid_svd_r*.npz files.
"""

import os
import weakref
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tensorboardX import SummaryWriter

from pointcept.engines.train import TrainerBase, TRAINERS
from pointcept.engines.defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.utils.logger import get_root_logger


class GridAwareSampler:
    """
    Grid-based point cloud sampler with rank-aware data loading

    Uses pre-computed point-to-grid mappings for efficient sampling across
    multiple GPUs, where each rank only processes its assigned portion of data.
    """

    def __init__(
        self,
        min_sample_ratio: float = 0.3,
        max_sample_ratio: float = 0.7,
        seed: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        svd_rank: Optional[int] = None,
    ):
        """
        Args:
            min_sample_ratio: Minimum sampling ratio for half-density scenario
            max_sample_ratio: Maximum sampling ratio for half-density scenario
            seed: Random seed for reproducibility
            rank: Current process rank for distributed training
            world_size: Total number of processes
            svd_rank: SVD compression rank to load (e.g., 8, 16, 32). If None, loads highest rank available.
        """
        self.min_sample_ratio = min_sample_ratio
        self.max_sample_ratio = max_sample_ratio
        self.rank = rank
        self.world_size = world_size
        self.svd_rank = svd_rank

        # Create separate RNG for each rank
        self.rng = np.random.RandomState(seed + rank if seed is not None else None)

        # Cache for loaded grid data
        self._grid_cache = {}

    def load_grid_mapping_with_rank(
        self,
        scene_path: str,
        target_rank: Optional[int] = None,
    ) -> Dict:
        """
        Load point-to-grid mapping with rank-specific data partitioning

        Args:
            scene_path: Path to scene directory
            target_rank: Target rank to load data for (None = current rank)

        Returns:
            grid_data: Dictionary with compressed features and point-to-grid mapping
        """
        rank = target_rank if target_rank is not None else self.rank
        scene_path = Path(scene_path)
        scene_name = scene_path.name

        # Check cache
        cache_key = f"{scene_name}_rank{rank}_svd{self.svd_rank or 'auto'}"
        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]

        # Find compressed SVD file
        if self.svd_rank is not None:
            # Load specific SVD rank file
            svd_file = scene_path / f"lang_feat_grid_svd_r{self.svd_rank}.npz"
            if not svd_file.exists():
                raise FileNotFoundError(
                    f"SVD file {svd_file} not found in {scene_path}"
                )
        else:
            # Auto-detect: use the highest rank available
            svd_files = list(scene_path.glob("lang_feat_grid_svd_r*.npz"))
            if not svd_files:
                raise FileNotFoundError(
                    f"No lang_feat_grid_svd_r*.npz found in {scene_path}"
                )
            svd_file = sorted(svd_files, key=lambda x: int(x.stem.split('r')[-1]))[-1]

        # Load data
        data = np.load(svd_file)
        compressed = data['compressed']  # [M, rank]
        indices = data['indices']  # [N] - point to grid mapping

        # Partition data by rank if world_size > 1
        if self.world_size > 1:
            # Partition grids by rank
            num_grids = compressed.shape[0]
            grids_per_rank = (num_grids + self.world_size - 1) // self.world_size

            start_grid = rank * grids_per_rank
            end_grid = min((rank + 1) * grids_per_rank, num_grids)

            # Filter points belonging to this rank's grids
            mask = (indices >= start_grid) & (indices < end_grid)
            filtered_indices = indices[mask]

            # Remap grid indices to be local to this rank
            # global_grid_id -> local_grid_id = global_grid_id - start_grid
            remapped_indices = filtered_indices - start_grid

            # Filter compressed features
            rank_compressed = compressed[start_grid:end_grid]

            grid_data = {
                'compressed_features': torch.from_numpy(rank_compressed).float(),
                'point_to_grid': torch.from_numpy(remapped_indices).long(),
                'num_grids': rank_compressed.shape[0],
                'num_points': remapped_indices.shape[0],
                'svd_rank': rank_compressed.shape[1],
                'start_grid': start_grid,
                'end_grid': end_grid,
                'rank': rank,
            }

            print(f"[Rank {rank}] Loaded {grid_data['num_grids']:,} grids "
                  f"({start_grid}-{end_grid}), {grid_data['num_points']:,} points")

        else:
            # Single GPU: load all data
            grid_data = {
                'compressed_features': torch.from_numpy(compressed).float(),
                'point_to_grid': torch.from_numpy(indices).long(),
                'num_grids': compressed.shape[0],
                'num_points': indices.shape[0],
                'svd_rank': compressed.shape[1],
                'rank': 0,
            }

            print(f"[Rank {rank}] Loaded {grid_data['num_grids']:,} grids, "
                  f"{grid_data['num_points']:,} points")

        # Load meta data if available
        meta_file = scene_path / "grid_meta_data.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
                grid_data['meta'] = meta_data

        # Cache
        self._grid_cache[cache_key] = grid_data

        return grid_data

    def build_inverse_mapping(
        self,
        point_to_grid: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Build inverse mapping: grid_id -> list of point indices

        OPTIMIZED: Vectorized approach using unique_consecutive + split.
        All computation stays on GPU, only final grid_ids moved to CPU once.

        Args:
            point_to_grid: [N] point to grid indices

        Returns:
            grid_to_points: Dictionary mapping grid_id to point indices tensor
        """
        if point_to_grid is None:
            raise ValueError(
                "point_to_grid is None! This indicates SVD file loading failed. "
                "Possible reasons:\n"
                "1. SVD file not found at expected path (scene_path/lang_feat_grid_svd_r{rank}.npz)\n"
                "2. Point count mismatch between SVD file and current data\n"
                "3. Exception during SVD file loading (check previous warning logs)"
            )

        device = point_to_grid.device

        # SUPER OPTIMIZED: Use unique_consecutive + split (all on GPU)
        # unique_consecutive is faster than unique for sorted data
        sorted_idx = torch.argsort(point_to_grid)
        sorted_grids = point_to_grid[sorted_idx]

        # Find unique grids and counts in one pass (GPU only)
        unique_grids, counts = torch.unique_consecutive(sorted_grids, return_counts=True)

        # Convert sorted_idx to list of tensors using split (vectorized)
        # This is much faster than loop-based slicing
        split_indices = torch.split(sorted_idx, counts.tolist())

        # Convert grid_ids to CPU ONCE (not per-item)
        unique_grids_cpu = unique_grids.cpu().numpy()

        # Build dictionary (minimal CPU-GPU transfer)
        grid_to_points = {
            int(grid_id): split_indices[i]
            for i, grid_id in enumerate(unique_grids_cpu)
        }

        return grid_to_points

    def sample_dense(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        point_to_grid: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Scenario 1: Dense input (use all assigned points for this rank)

        Args:
            coord: [N, 3] point coordinates
            feat: [N, C] point features
            point_to_grid: [N] point to grid mapping
            labels: [N] point labels (optional)

        Returns:
            sample_dict: Dictionary with all data
        """
        sample_dict = {
            'coord': coord,
            'feat': feat,
            'point_to_grid': point_to_grid,
            'sampling_ratio': torch.tensor(1.0, device=coord.device),
            'scenario': 'dense',
            'num_points': coord.shape[0],
            'rank': self.rank,
        }

        if labels is not None:
            sample_dict['labels'] = labels

        # Handle batch: if provided in kwargs and not None, use it; otherwise default to zeros
        if kwargs.get('batch') is not None:
            sample_dict['batch'] = kwargs['batch']
        else:
            sample_dict['batch'] = torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)

        for key, value in kwargs.items():
            if key not in ['coord', 'feat', 'labels', 'batch']:
                sample_dict[key] = value

        return sample_dict

    def sample_half_density(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        point_to_grid: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        grid_to_points: Dict[int, torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Scenario 2: Sample ~50% of points per grid (30%-70% random)

        SUPER OPTIMIZED: Fully vectorized with NO Python loops over grids.
        Uses advanced indexing and batch operations for all grids simultaneously.

        Args:
            coord: [N, 3] point coordinates
            feat: [N, C] point features
            point_to_grid: [N] point to grid mapping
            labels: [N] point labels (optional)
            grid_to_points: Pre-computed grid to points mapping (must be provided)

        Returns:
            sample_dict: Dictionary with sampled data
        """
        assert grid_to_points is not None, "grid_to_points must be provided"

        device = coord.device
        num_grids = len(grid_to_points)

        if num_grids == 0:
            # Fallback if no grids
            num_samples = max(1, int(coord.shape[0] * 0.5))
            perm = torch.randperm(coord.shape[0], device=device)
            all_indices = perm[:num_samples]
            actual_ratios = torch.full((num_samples,), 0.5, device=device)
        else:
            # Direct computation of sampling metadata (no cache)
            point_tensors = list(grid_to_points.values())
            grid_counts_list = [len(pts) for pts in point_tensors]
            grid_counts = torch.tensor(grid_counts_list, device=device)

            # Pre-compute grid_offsets
            grid_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                                     grid_counts[:-1].cumsum(dim=0)])

            # Pre-compute positions_in_grid
            positions_in_grid = torch.arange(num_grids, device=device).repeat_interleave(grid_counts)

            # Generate all random ratios at once on GPU
            sample_ratios = torch.empty(num_grids, device=device)
            sample_ratios.uniform_(self.min_sample_ratio, self.max_sample_ratio)

            # Calculate number of samples per grid
            num_samples_per_grid = (grid_counts.float() * sample_ratios).long().clamp(min=1)

            # Flatten all point indices
            flat_point_indices = torch.cat(point_tensors)

            # Generate random permutations for each grid (simple approach, no caching)
            # This reduces memory pressure by not storing permutations across iterations
            unique_counts = torch.unique(grid_counts)
            perm_cache = {}
            for count in unique_counts:
                count_val = count.item()
                # Generate fresh permutation each time (no caching)
                perm_cache[count_val] = torch.randperm(count_val, device=device)

            # Build flat permutation tensor by concatenating perms for each grid
            all_perms = [perm_cache[grid_counts[i].item()] for i in range(num_grids)]
            flat_perms = torch.cat(all_perms)

            # Create position indicators within each grid's permutation (fully vectorized)
            # positions_in_perm: position within that grid's permutation [0, 1, 2, ..., 0, 1, ...]
            # Compute using cumsum instead of .tolist() to stay on GPU
            grid_ends = grid_counts.cumsum(dim=0)
            positions_in_perm_flat = torch.arange(grid_counts.sum().item(), device=device)
            grid_start_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                                           grid_ends[:-1]])
            positions_in_perm = positions_in_perm_flat - grid_start_offsets.repeat_interleave(grid_counts)

            # Create sampling mask: element selected if position < num_samples_for_that_grid
            samples_per_element = num_samples_per_grid[positions_in_grid]
            take_mask = positions_in_perm < samples_per_element

            # Extract selected indices (all vectorized, NO Python loops)
            selected_positions = torch.where(take_mask)[0]
            all_indices = flat_point_indices[selected_positions]

            # Compute actual ratios for each selected point (vectorized)
            selected_grid_ids = positions_in_grid[selected_positions]
            selected_counts = grid_counts[selected_grid_ids].float()
            selected_num_samples = num_samples_per_grid[selected_grid_ids].float()
            actual_ratios = selected_num_samples / selected_counts

        # Build sample dictionary
        sample_dict = {
            'coord': coord[all_indices],
            'feat': feat[all_indices],
            'point_to_grid': point_to_grid[all_indices],
            'sampling_ratio': actual_ratios.mean(),
            'scenario': 'half_density',
            'num_points': all_indices.shape[0],
            'rank': self.rank,
        }

        if labels is not None:
            sample_dict['labels'] = labels[all_indices]

        # Handle batch: if provided in kwargs and not None, index it; otherwise default to zeros
        batch_input = kwargs.get('batch')
        if batch_input is not None:
            sample_dict['batch'] = batch_input[all_indices]
        else:
            sample_dict['batch'] = torch.zeros(all_indices.shape[0], dtype=torch.long, device=device)

        for key, value in kwargs.items():
            if key in ['coord', 'feat', 'labels', 'batch']:
                continue
            if isinstance(value, torch.Tensor) and value.shape[0] == coord.shape[0]:
                sample_dict[key] = value[all_indices]
            else:
                sample_dict[key] = value

        return sample_dict

    def sample_single_per_grid(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        point_to_grid: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        grid_to_points: Dict[int, torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Scenario 3: Sample 1 point per grid

        SUPER OPTIMIZED: Fully vectorized sampling using batch operations.

        Args:
            coord: [N, 3] point coordinates
            feat: [N, C] point features
            point_to_grid: [N] point to grid mapping
            labels: [N] point labels (optional)
            grid_to_points: Pre-computed grid to points mapping (must be provided)

        Returns:
            sample_dict: Dictionary with sampled data (1 point per grid)
        """
        assert grid_to_points is not None, "grid_to_points must be provided"

        device = coord.device
        num_grids = len(grid_to_points)

        if num_grids == 0:
            all_indices = torch.tensor([0], device=device)
        else:
            # Direct computation of sampling metadata (no cache)
            point_tensors = list(grid_to_points.values())
            grid_counts_list = [len(pts) for pts in point_tensors]
            grid_counts = torch.tensor(grid_counts_list, device=device)

            # Pre-compute grid_offsets
            grid_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                                     grid_counts[:-1].cumsum(dim=0)])

            # Generate random offset for each grid (0 to count-1) on GPU
            random_offsets = torch.rand(num_grids, device=device) * grid_counts.float()
            random_offsets = random_offsets.long()

            # Compute global indices: start + random_offset
            all_indices = grid_offsets + random_offsets

            # Map back to original indices
            flat_point_indices = torch.cat(point_tensors)
            all_indices = flat_point_indices[all_indices]

        sample_dict = {
            'coord': coord[all_indices],
            'feat': feat[all_indices],
            'point_to_grid': point_to_grid[all_indices],
            'sampling_ratio': torch.tensor(
                1.0 / num_grids if num_grids > 0 else 0.0,
                device=device
            ),
            'scenario': 'single_per_grid',
            'num_points': all_indices.shape[0],
            'rank': self.rank,
        }

        if labels is not None:
            sample_dict['labels'] = labels[all_indices]

        # Handle batch: if provided in kwargs and not None, index it; otherwise default to zeros
        batch_input = kwargs.get('batch')
        if batch_input is not None:
            sample_dict['batch'] = batch_input[all_indices]
        else:
            sample_dict['batch'] = torch.zeros(all_indices.shape[0], dtype=torch.long, device=device)

        for key, value in kwargs.items():
            if key in ['coord', 'feat', 'labels', 'batch']:
                continue
            if isinstance(value, torch.Tensor) and value.shape[0] == coord.shape[0]:
                sample_dict[key] = value[all_indices]
            else:
                sample_dict[key] = value

        return sample_dict

    def sample_scenarios(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        point_to_grid: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        scenarios: List[str] = ['dense', 'half', 'single'],
        **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Sample multiple scenarios

        OPTIMIZED: Build grid_to_points mapping ONCE and reuse for all scenarios.
        This avoids repeating the expensive O(N log N) sorting operation.

        Args:
            coord: [N, 3] point coordinates
            feat: [N, C] point features
            point_to_grid: [N] point to grid mapping
            labels: [N] point labels (optional)
            scenarios: List of scenarios to generate
            **kwargs: Additional data

        Returns:
            samples: List of sample dictionaries
        """
        samples = []

        # OPTIMIZATION: Build inverse mapping ONCE before sampling
        # All scenarios share the same point_to_grid, so compute mapping once
        # Check if any scenario needs grid_to_points (half and single do)
        needs_grid_mapping = any(s in ['half', 'single'] for s in scenarios)
        grid_to_points = self.build_inverse_mapping(point_to_grid) if needs_grid_mapping else None

        for scenario in scenarios:
            if scenario == 'dense':
                sample = self.sample_dense(coord, feat, point_to_grid, labels, **kwargs)
            elif scenario == 'half':
                # Pass pre-computed grid_to_points
                sample = self.sample_half_density(coord, feat, point_to_grid, labels, grid_to_points, **kwargs)
            elif scenario == 'single':
                # Pass pre-computed grid_to_points
                sample = self.sample_single_per_grid(coord, feat, point_to_grid, labels, grid_to_points, **kwargs)
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            samples.append(sample)

        return samples


class DensityConsistencyLoss(nn.Module):
    """
    Density consistency loss using point-to-grid mapping

    Enforces that model outputs are consistent across different sampling densities
    for points from the same grid.
    """

    def __init__(
        self,
        consistency_type: str = 'mse',
        temperature: float = 0.5,
        use_grid_features: bool = True,
    ):
        super().__init__()
        self.consistency_type = consistency_type
        self.temperature = temperature
        self.use_grid_features = use_grid_features

    def match_features_by_grid(
        self,
        feat1: torch.Tensor,
        grid1: torch.Tensor,
        feat2: torch.Tensor,
        grid2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match features from two scenarios by grid indices

        SUPER OPTIMIZED: Pre-aggregate features by grid using scatter_mean,
        then just lookup common grids. This is O(N) instead of O(G*N).

        Args:
            feat1: [N1, C] features from scenario 1
            grid1: [N1] grid indices from scenario 1
            feat2: [N2, C] features from scenario 2
            grid2: [N2] grid indices from scenario 2

        Returns:
            aligned_feat1: [M, C] aligned features from scenario 1
            aligned_feat2: [M, C] aligned features from scenario 2
        """
        device = feat1.device

        # OPTIMIZED: Pre-aggregate features by grid (one-time O(N) operation)
        # Instead of scanning for each grid, compute all means at once

        def aggregate_features_by_grid(feat, grid):
            """Aggregate features to grid level using scatter_add"""
            # Get unique grids and assign indices
            unique_grids, inverse_indices = torch.unique(grid, return_inverse=True)

            num_grids = unique_grids.shape[0]
            num_channels = feat.shape[1]

            # Initialize sum and count tensors
            feat_sum = torch.zeros(num_grids, num_channels, device=device, dtype=feat.dtype)
            grid_count = torch.zeros(num_grids, device=device, dtype=feat.dtype)

            # Scatter add features and counts
            feat_sum.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, num_channels), feat)
            grid_count.scatter_add_(0, inverse_indices, torch.ones_like(grid, dtype=feat.dtype))

            # Compute mean (avoid division by zero)
            grid_count = grid_count.clamp(min=1.0)
            feat_mean = feat_sum / grid_count.unsqueeze(1)

            return unique_grids, feat_mean

        # Aggregate both feature sets
        unique_grids1, agg_feat1 = aggregate_features_by_grid(feat1, grid1)
        unique_grids2, agg_feat2 = aggregate_features_by_grid(feat2, grid2)

        # Find common grids using isin
        common_mask = torch.isin(unique_grids1, unique_grids2)
        common_grids = unique_grids1[common_mask]

        if len(common_grids) == 0:
            return (
                torch.zeros(0, feat1.shape[1], device=device),
                torch.zeros(0, feat2.shape[1], device=device)
            )

        # Extract aggregated features for common grids
        # Use isin again to find indices
        common_idx1 = torch.where(common_mask)[0]
        common_idx2 = torch.where(torch.isin(unique_grids2, common_grids))[0]

        aligned_feat1 = agg_feat1[common_idx1]
        aligned_feat2 = agg_feat2[common_idx2]

        return aligned_feat1, aligned_feat2

    def forward(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        inputs: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute density consistency loss between different density scenarios

        Args:
            outputs: List of model outputs for each scenario
            inputs: List of inputs for each scenario

        Returns:
            consistency_loss: Scalar loss value
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        total_weight = 0.0

        # Compare all pairs
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                feat_i = outputs[i].get('feat')
                feat_j = outputs[j].get('feat')
                grid_i = inputs[i].get('point_to_grid')
                grid_j = inputs[j].get('point_to_grid')

                # Original assertions (commented out, now using continue with warning instead)
                assert feat_i is not None, f"outputs[{i}]['feat'] is None"
                assert feat_j is not None, f"outputs[{j}]['feat'] is None"
                assert grid_i is not None, f"inputs[{i}]['point_to_grid'] is None"
                assert grid_j is not None, f"inputs[{j}]['point_to_grid'] is None"

                # Align features by grid
                # Note: using .detach() here since consistency loss is auxiliary
                # The main task loss will provide gradients for the model
                with torch.no_grad():
                    aligned_feat_i, aligned_feat_j = self.match_features_by_grid(
                        feat_i, grid_i, feat_j, grid_j
                    )

                num_common_grids = aligned_feat_i.shape[0]
                if num_common_grids == 0:
                    raise RuntimeError(
                        f"[DensityConsistencyLoss] No common grids found between scenario {i} ({scenario_i}) "
                        f"and scenario {j} ({scenario_j}). "
                        f"This indicates a data loading error where point_to_grid mappings "
                        f"do not share any grid indices. "
                        f"feat_i.shape={feat_i.shape}, grid_i range=[{grid_i.min()}, {grid_i.max()}], "
                        f"feat_j.shape={feat_j.shape}, grid_j range=[{grid_j.min()}, {grid_j.max()}]"
                    )

                # Compute consistency loss (detached, no gradients to model)
                if self.consistency_type == 'mse':
                    loss = F.mse_loss(aligned_feat_i, aligned_feat_j)
                elif self.consistency_type == 'cosine':
                    loss = 1 - F.cosine_similarity(
                        aligned_feat_i, aligned_feat_j, dim=-1
                    ).mean()
                elif self.consistency_type == 'kl':
                    log_prob_i = F.log_softmax(
                        aligned_feat_i / self.temperature, dim=-1
                    )
                    prob_j = F.softmax(
                        aligned_feat_j / self.temperature, dim=-1
                    )
                    loss = F.kl_div(
                        log_prob_i, prob_j, reduction='batchmean'
                    ) * (self.temperature ** 2)
                else:
                    raise ValueError(f"Unknown consistency type: {self.consistency_type}")

                total_loss += loss * num_common_grids
                total_weight += num_common_grids

                scenario_i = inputs[i].get('scenario', f'scenario_{i}')
                scenario_j = inputs[j].get('scenario', f'scenario_{j}')
                loss_dict[f'{scenario_i}_vs_{scenario_j}'] = loss.item()

        if total_weight > 0:
            total_loss = total_loss / total_weight
        else:
            # Fallback: get device from first input
            device = inputs[0].get('coord').device if inputs else torch.device('cuda')
            total_loss = torch.tensor(0.0, device=device)

        loss_dict['total_consistency'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        loss_dict['num_common_grids'] = total_weight if isinstance(total_weight, float) else total_weight.item()

        return total_loss, loss_dict


@TRAINERS.register_module("DensityInvariantTrainer")
class DensityInvariantTrainer(TrainerBase):
    """
    Density-Invariant Trainer with rank-aware data loading

    Implements multi-scenario training using pre-computed point-to-grid mappings
    from lang_feat_grid_svd_r*.npz files. Each rank loads only its assigned portion
    of the data for memory efficiency.
    """

    def __init__(self, cfg):
        super().__init__()

        # Get rank info
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()

        # Base configuration
        self.epoch = 0
        self.start_epoch = 0
        # CRITICAL FIX: Use cfg.epoch for max_epoch, NOT cfg.eval_epoch
        # eval_epoch is for evaluation frequency (e.g., every 20 epochs)
        # epoch is the total number of training epochs (e.g., 200)
        self.max_epoch = cfg.epoch
        self.best_metric_value = -torch.inf

        # Logger
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")

        # Density-invariant training configuration
        self.density_config = cfg.get('density_invariant', {})
        self.min_sample_ratio = self.density_config.get('min_sample_ratio', 0.3)
        self.max_sample_ratio = self.density_config.get('max_sample_ratio', 0.7)
        self.consistency_weight = self.density_config.get('consistency_weight', 0.1)
        self.consistency_type = self.density_config.get('consistency_type', 'mse')
        self.enabled_scenarios = self.density_config.get(
            'scenarios', ['dense', 'half', 'single']
        )
        self.scenario_weights = self.density_config.get(
            'scenario_weights', {'dense': 1.0, 'half': 1.0, 'single': 1.0}
        )
        self.use_compressed_features = self.density_config.get('use_compressed_features', True)
        self.svd_rank = self.density_config.get('svd_rank', None)  # SVD compression rank to load
        self.batched_forward = self.density_config.get('batched_forward', True)  # NEW: batched or separate forward

        # Per-dimension correlation monitoring
        self.enable_per_dim_monitor = cfg.get('enable_per_dim_monitor', True)
        self.per_dim_log_freq = cfg.get('per_dim_log_freq', 100)
        self.per_dim_monitor = None

        if self.enable_per_dim_monitor and self.rank == 0:  # Only on main process
            from pointcept.utils.per_dim_monitor import PerDimMonitor
            self.per_dim_monitor = PerDimMonitor(
                num_dims=self.svd_rank if self.svd_rank else 16,
                log_freq=self.per_dim_log_freq,
                target_dim0_corr=cfg.get('target_dim0_corr', 0.90),
                target_minor_corr=cfg.get('target_minor_corr', 0.30),
                warmup_iters=cfg.get('per_dim_warmup_iters', 200),
            )
            self.logger.info("=> Per-Dimension Monitoring enabled:")
            self.logger.info(f"   Log frequency: every {self.per_dim_log_freq} iterations")
            self.logger.info(f"   Target Dim 0 correlation: {cfg.get('target_dim0_corr', 0.90)}")
            self.logger.info(f"   Target Minor correlation: {cfg.get('target_minor_corr', 0.30)}")

        self.logger.info("=> Density-Invariant Training Configuration:")
        self.logger.info(f"   Rank: {self.rank}/{self.world_size}")
        self.logger.info(f"   Sample ratio range: [{self.min_sample_ratio}, {self.max_sample_ratio}]")
        self.logger.info(f"   Consistency weight: {self.consistency_weight}")
        self.logger.info(f"   Consistency type: {self.consistency_type}")
        self.logger.info(f"   Enabled scenarios: {self.enabled_scenarios}")
        self.logger.info(f"   Scenario weights: {self.scenario_weights}")
        self.logger.info(f"   Use compressed features: {self.use_compressed_features}")
        self.logger.info(f"   SVD rank: {self.svd_rank if self.svd_rank else 'auto (highest available)'}")
        self.logger.info(f"   Batched forward: {self.batched_forward}")

        # Build components
        self.logger.info("=> Building model ...")
        self.model = self.build_model()

        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()

        if not cfg.test_only:
            self.logger.info("=> Building train dataset & dataloader ...")
            self.train_loader = self.build_train_loader()
            self.logger.info("=> Building val dataset & dataloader ...")
            self.val_loader = self.build_val_loader()
            self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
            self.optimizer = self.build_optimizer()
            self.scheduler = self.build_scheduler()
            self.scaler = self.build_scaler()
        else:
            self.train_loader = None
            self.val_loader = None
            self.optimizer = None
            self.scheduler = None
            self.scaler = None

        # Build density sampling components
        self.sampler = GridAwareSampler(
            min_sample_ratio=self.min_sample_ratio,
            max_sample_ratio=self.max_sample_ratio,
            seed=cfg.get('seed', None),
            rank=self.rank,
            world_size=self.world_size,
            svd_rank=self.svd_rank,
        )

        self.consistency_loss = DensityConsistencyLoss(
            consistency_type=self.consistency_type,
        )

        # Per-scene loss tracking for debugging and visualization
        # Dictionary structure: {scene_name: {'total_loss': [], 'l1_loss': [], 'cosine_loss': [], 'iterations': []}}
        # Tracks REAL 3D scene names (from dataset), not training scenarios
        self.per_scene_losses = {}
        # Track all unique scene names discovered during training
        self._all_real_scenes = set()

        self.logger.info("=> Building hooks ...")
        self.register_hooks(cfg.hooks)

    def build_model(self):
        """Build model"""
        from pointcept.models import build_model
        import torch.nn as nn

        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available!")
        else:
            self.logger.info("CUDA is available!")

        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )

        return model

    def build_writer(self):
        """Build tensorboard writer"""
        # TEMPORARILY DISABLED: Prevent writing to events file to save disk space
        writer = None  # SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        if comm.is_main_process():
            self.logger.info(f"Tensorboard writer DISABLED (not logging to: {self.cfg.save_path})")
        return writer

    def build_train_loader(self):
        """Build train dataloader"""
        from pointcept.datasets import build_dataset
        from pointcept.datasets import point_collate_fn
        from functools import partial
        import torch.utils.data

        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True if self.cfg.num_worker_per_gpu > 0 else False,
        )

        return train_loader

    def build_val_loader(self):
        """Build val dataloader"""
        from pointcept.datasets import build_dataset, collate_fn
        import torch.utils.data

        if not self.cfg.evaluate:
            return None

        val_cfg = self.cfg.data.val
        val_cfg = val_cfg if isinstance(val_cfg, (list, tuple)) else [val_cfg]

        loaders = []
        for cfg_i in val_cfg:
            val_data = build_dataset(cfg_i)
            sampler = (
                torch.utils.data.distributed.DistributedSampler(val_data)
                if comm.get_world_size() > 1
                else None
            )
            loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=sampler,
                collate_fn=collate_fn,
            )
            loaders.append(loader)

        return loaders[0] if len(loaders) == 1 else loaders

    def build_optimizer(self):
        """Build optimizer"""
        from pointcept.utils.optimizer import build_optimizer
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        """Build scheduler"""
        from pointcept.utils.scheduler import build_scheduler

        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        """Build gradient scaler with very conservative init_scale to prevent NaN.

        OBSERVED ISSUE: decoder0.conv.1.bias has persistent large gradients (~0.77)
        - This is NOT a random spike, but a structural issue from dim[0] dominance
        - Large gradients accumulate in weights → activation anomalies → NaN

        VERY CONSERVATIVE: init_scale=16.0 to minimize overflow risk.
        - This is 32x smaller than the initial 512, and 4096x smaller than PyTorch default 65536
        - Priority: Stability over learning speed
        - GradScaler will automatically increase scale when training is stable (growth_factor=2.0)

        With init_scale=16:
        - loss (1.6) × scale (16) = 25.6
        - 25.6 × activation (2) × channels (504) = 25.8K >> 65K fp16 max
        - BUT: This is MUCH smaller than before, reducing overflow risk significantly

        Trade-off:
        - Learning will be slower initially (smaller gradients)
        - But training will be stable (no NaN)
        - GradScaler will auto-increase scale over time if no overflow occurs

        Key protection layers:
        1. Very small init_scale (16) - minimizes initial gradient magnitude
        2. Loss spike detection (2.5x MA)
        3. Gradient clipping (1.0 norm)
        4. Consecutive large gradient detection (5 in a row)
        5. GradScaler auto-adjustment (will decrease on overflow, increase when stable)
        """
        if self.cfg.enable_amp:
            scaler = torch.amp.GradScaler(
                "cuda",
                init_scale=16.0,        # Very conservative: 32x smaller than 512
                growth_factor=2.0,       # Double scale every 2000 successful iters
                backoff_factor=0.5,      # Halve scale when inf/NaN detected
                growth_interval=2000,    # How often to increase scale
            )
        else:
            scaler = None
        return scaler

    def train(self):
        """Main training loop"""
        from pointcept.utils.events import EventStorage

        with EventStorage() as self.storage:
            if self.cfg.test_only:
                self.before_eval()
                self.logger.info(">>>>>>>>>>>>>>>> Test Only, Skip Training >>>>>>>>>>>>>>>>")
            else:
                self.before_train()
                self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    if comm.get_world_size() > 1:
                        self.train_loader.sampler.set_epoch(self.epoch)
                    self.model.train()
                    self.data_iterator = enumerate(self.train_loader)
                    self.before_epoch()

                    for (
                        self.comm_info["iter"],
                        self.comm_info["input_dict"],
                    ) in self.data_iterator:
                        self.before_step()
                        self.run_step()
                        self.after_step()

                    self.after_epoch()
            self.after_train()

            import datetime
            self.logger.info(f"Training finished at {datetime.datetime.now()}")

    def run_step(self):
        """Run a single training step with density-invariant strategy"""
        import time

        # TIMING: Start of iteration
        iter_start_time = time.time()

        input_dict = self.comm_info["input_dict"]

        # TIMING: Data loading
        data_load_start = time.time()
        # Move to GPU
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        data_load_time = time.time() - data_load_start

        # Extract input data
        coord = input_dict.get('coord')
        feat = input_dict.get('feat')
        lang_feat = input_dict.get('lang_feat')  # Ground truth language features
        labels = input_dict.get('segment')
        valid_feat_mask = input_dict.get('valid_feat_mask')

        if coord is None or feat is None:
            raise ValueError("Input must contain 'coord' and 'feat'")

        # Log initial data statistics
        scene_name = input_dict.get('name', 'unknown')
        self.logger.debug(f"[Rank {self.rank}] Processing scene '{scene_name}': "
                         f"{coord.shape[0]} points initially")

        # CRITICAL: SVD files contain point_to_grid mapping for ONLY valid points
        # We need to filter input data to valid points before using the mapping
        # Note: FilterValidPoints transform has already been applied in the dataset,
        # but we apply it again here for safety in case the data loader modified things

        # Get point_to_grid mapping FIRST (before any filtering)
        # This mapping was loaded in GenericGSDataset and should align with coord/feat
        point_to_grid = input_dict.get('point_to_grid')

        # Handle batched data: DataLoader wraps items in lists
        if isinstance(point_to_grid, list):
            if len(point_to_grid) == 0:
                point_to_grid = None
            elif len(point_to_grid) == 1:
                point_to_grid = point_to_grid[0]
            else:
                # Multiple scenes in batch - concatenate all elements
                # This matches how coord/feat are handled by the default collate_fn
                # All tensors should be on the same device (CPU at this point)
                point_to_grid = torch.cat(point_to_grid, dim=0)

        # Store original sizes for validation
        num_total = coord.shape[0]
        point_to_grid_size = point_to_grid.shape[0] if point_to_grid is not None else 0

        # Apply valid_feat_mask filtering if needed
        if valid_feat_mask is not None:
            valid_mask = valid_feat_mask > 0
            num_valid = valid_mask.sum().item()

            # Warn if very few valid points
            if num_valid < 100:
                self.logger.warning(f"[Rank {self.rank}] Scene '{scene_name}' has very few valid points ({num_valid}). "
                                   f"This may indicate a data problem.")

            # IMPORTANT: Filter point_to_grid along with other data to maintain alignment
            # The GenericGSDataset and transforms should have already done this, but we ensure it here
            if point_to_grid is not None and point_to_grid.shape[0] == num_total:
                point_to_grid = point_to_grid[valid_mask]
                self.logger.debug(f"[Rank {self.rank}] Filtered point_to_grid to match valid points")

            # Filter to valid points
            coord = coord[valid_mask]
            feat = feat[valid_mask]
            if lang_feat is not None:
                lang_feat = lang_feat[valid_mask]
            if labels is not None:
                labels = labels[valid_mask]

            self.logger.debug(f"[Rank {self.rank}] Filtered to {num_valid:,} valid points "
                            f"out of {num_total:,} total points")

        # Validate point_to_grid alignment
        if point_to_grid is not None:
            current_num_points = coord.shape[0]
            if point_to_grid.shape[0] != current_num_points:
                raise ValueError(
                    f"[Rank {self.rank}] point_to_grid size mismatch!\n"
                    f"  point_to_grid.shape[0] = {point_to_grid.shape[0]}\n"
                    f"  coord.shape[0] = {current_num_points}\n"
                    f"  Original sizes: coord={num_total}, point_to_grid={point_to_grid_size}\n"
                    f"This indicates a bug in the data pipeline - transforms didn't maintain alignment."
                )

            # Validate indices are non-negative (basic sanity check)
            if (point_to_grid < 0).any():
                raise ValueError(
                    f"[Rank {self.rank}] point_to_grid contains negative indices!\n"
                    f"  min(point_to_grid) = {point_to_grid.min().item()}\n"
                    f"This indicates corrupted SVD file or incorrect filtering."
                )

        # Assert point_to_grid is available
        assert point_to_grid is not None, (
            f"[Rank {self.rank}] point_to_grid is None! Cannot proceed with training.\n"
            f"This indicates SVD file loading failed for scene '{scene_name}'.\n"
        )

        # Generate multi-scenario samples
        scenarios_to_use = [
            s for s in self.enabled_scenarios
            if s in ['dense', 'half', 'single']
        ]

        # TIMING: Sampling
        sampling_start = time.time()
        scenario_samples = self.sampler.sample_scenarios(
            coord=coord,
            feat=feat,
            point_to_grid=point_to_grid,
            labels=labels,
            scenarios=scenarios_to_use,
            batch=input_dict.get('batch'),
            grid_size=input_dict.get('grid_size', 0.01),
            origin_coord=input_dict.get('origin_coord'),
            name=input_dict.get('name'),
            lang_feat=lang_feat,  # Pass lang_feat for sampling
            # Pass Gaussian parameters for Rendered2DLoss
            opacity=input_dict.get('opacity'),
            quat=input_dict.get('quat'),
            scale=input_dict.get('scale'),
            scene_path=input_dict.get('scene_path'),
        )
        sampling_time = time.time() - sampling_start

        # Forward pass for each scenario
        scenario_outputs = []
        scenario_losses = []

        # OPTIMIZED: Pre-compute common values outside the loop to avoid repeated access
        grid_size = input_dict.get('grid_size', 0.01)
        epoch_progress = self.epoch / self.max_epoch
        device = coord.device
        # Get scene name (may be list due to collate_fn, so handle that)
        scene_name_raw = input_dict.get('name', 'unknown')
        if isinstance(scene_name_raw, list):
            # For batched data, use the first scene name or join them
            scene_name = scene_name_raw[0] if len(scene_name_raw) == 1 else "_".join(str(s) for s in scene_name_raw)
        else:
            scene_name = scene_name_raw

        # TIMING: Forward pass
        forward_start = time.time()

        # Get backbone model (used in both modes)
        if hasattr(self.model, 'module'):
            backbone = self.model.module.backbone
        else:
            backbone = self.model.backbone

        # NAN CHECK: Check backbone weights before forward pass (EVERY iteration now)
        # This was previously done every 10 iters, but NaN can happen at any time
        has_nan_in_backbone = False
        nan_params = []
        for name, param in backbone.named_parameters():
            if torch.isnan(param).any():
                has_nan_in_backbone = True
                nan_count = torch.isnan(param).sum().item()
                nan_params.append(f"{name} ({nan_count} NaNs)")

        if has_nan_in_backbone:
            print(f"\n🚨 NaN DETECTED IN BACKBONE WEIGHTS BEFORE FORWARD PASS!")
            print(f"  Iteration: {self.comm_info['iter']}, Epoch: {self.epoch}")
            print(f"  Corrupted parameters:")
            for p in nan_params[:5]:  # Show first 5
                print(f"    - {p}")
            print(f"  Model is corrupted from previous iterations!")
            print(f"  Need to restart from a clean checkpoint.")
            raise RuntimeError("Backbone weights contain NaN! Cannot continue training.")

        # Check minimum point count for all scenarios
        for sample_dict in scenario_samples:
            num_points = sample_dict['coord'].shape[0]
            if num_points < 4:
                raise ValueError(
                    f"[Rank {self.rank}] Scenario '{sample_dict.get('scenario', 'unknown')}' "
                    f"has only {num_points} point(s), but sparse convolution requires >= 4 points. "
                    f"This may indicate the scene has too few grids or points. "
                    f"Scene: {scene_name}"
                )

        if self.batched_forward:
            # =====================================================================
            # MODE 1: BATCHED FORWARD (default, faster but may cross-contaminate)
            # =====================================================================
            # Concatenate all scenarios into a single batch
            # OPTIMIZATION: Single forward pass leverages GPU parallelism

            batched_coord = []
            batched_feat = []
            batch_indices = []  # To track which scenario each point belongs to
            scenario_point_counts = []  # To split outputs back

            for i, sample_dict in enumerate(scenario_samples):
                batched_coord.append(sample_dict['coord'])
                batched_feat.append(sample_dict['feat'])
                num_points = sample_dict['coord'].shape[0]
                scenario_point_counts.append(num_points)
                batch_indices.append(torch.full((num_points,), i, device=device))

            # Concatenate all data
            batched_coord = torch.cat(batched_coord, dim=0)  # [Total_points, 3]
            batched_feat = torch.cat(batched_feat, dim=0)    # [Total_points, C]
            batch_indices = torch.cat(batch_indices, dim=0)  # [Total_points]

            # Store total points before deleting tensors
            total_points = batched_coord.shape[0]

            # Create batch input dict
            batched_input = {
                'coord': batched_coord,
                'feat': batched_feat,
                'batch': batch_indices,
                'grid_size': grid_size,
                'epoch_progress': epoch_progress,
            }

            # Clear concatenated tensors to free memory (after creating batched_input)
            del batched_coord, batched_feat, batch_indices

            # Create valid_feat_mask for all points (all valid after filtering)
            batched_input['valid_feat_mask'] = torch.ones(
                total_points, dtype=torch.bool, device=device
            )

            # Forward pass through backbone (batched for all scenarios)
            # NAN CHECK: Check input data for anomalies
            coord = batched_input['coord']
            feat = batched_input['feat']
            if torch.isnan(coord).any() or torch.isinf(coord).any():
                print(f"\n🚨 NaN/Inf DETECTED IN INPUT COORD!")
                print(f"  coord NaN: {torch.isnan(coord).any().item()}")
                print(f"  coord Inf: {torch.isinf(coord).any().item()}")
                print(f"  coord stats: min={coord.min():.2f}, max={coord.max():.2f}")
                assert False, "Input coord contains NaN/Inf! Check data loading."
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                print(f"\n🚨 NaN/Inf DETECTED IN INPUT FEAT!")
                print(f"  feat NaN: {torch.isnan(feat).any().item()}")
                print(f"  feat Inf: {torch.isinf(feat).any().item()}")
                print(f"  feat stats: min={feat.min():.6f}, max={feat.max():.6f}")
                assert False, "Input feat contains NaN/Inf! Check data loading."

            # ========================================
            # Decoder Layer Monitoring Setup (register hooks BEFORE forward pass)
            # ========================================
            decoder_layer_outputs = {}
            decoder_hooks = []

            def create_debug_hook(layer_name):
                def hook(module, input, output):
                    # Store output statistics
                    if isinstance(output, torch.Tensor):
                        output_data = output
                    elif isinstance(output, tuple) and len(output) > 0:
                        output_data = output[0]
                    else:
                        return

                    with torch.no_grad():
                        output_np = output_data.detach().cpu()
                        decoder_layer_outputs[layer_name] = {
                            'shape': list(output_data.shape),
                            'has_nan': torch.isnan(output_data).any().item(),
                            'has_inf': torch.isinf(output_data).any().item(),
                            'min': output_np.min().item(),
                            'max': output_np.max().item(),
                            'mean': output_np.mean().item(),
                            'std': output_np.std().item(),
                            'abs_mean': output_np.abs().mean().item(),
                            'abs_max': output_np.abs().max().item(),
                            'num_zeros': (output_np == 0).sum().item(),
                            'num_elements': output_np.numel(),
                        }
                return hook
            return create_debug_hook

            # Register hooks for all decoder layers BEFORE forward pass
            if hasattr(backbone, 'dec'):
                decoder = backbone.dec
                for dec_name in ['dec0', 'dec1', 'dec2', 'dec3']:
                    if hasattr(decoder, dec_name):
                        dec_layer = getattr(decoder, dec_name)
                        # Hook for upsampling layers
                        if hasattr(dec_layer, 'up'):
                            hook = dec_layer.up.register_forward_hook(
                                create_debug_hook(f'dec.{dec_name}.up')
                            )
                            decoder_hooks.append(hook)
                        # Hook for skip connection layers
                        if hasattr(dec_layer, 'up_skip'):
                            hook = dec_layer.up_skip.register_forward_hook(
                                create_debug_hook(f'dec.{dec_name}.up_skip')
                            )
                            decoder_hooks.append(hook)
                        # Hook for block layers
                        if hasattr(dec_layer, 'blocks'):
                            for block_idx, block in enumerate(decoder.blocks):
                                hook = block.register_forward_hook(
                                    create_debug_hook(f'dec.{dec_name}.block{block_idx}')
                                )
                                decoder_hooks.append(hook)

            with torch.amp.autocast("cuda", enabled=self.cfg.enable_amp):
                from pointcept.models.utils.structure import Point
                point = Point(batched_input)
                point_feat = backbone(point)

                # NAN CHECK: Detect NaN in backbone output immediately
                feat_before_scale = point_feat["feat"]
                if torch.isnan(feat_before_scale).any():
                    print(f"\n🚨 NaN DETECTED IN BACKBONE OUTPUT!")
                    print(f"  feat_before_scale contains NaN!")
                    print(f"  NaN count: {torch.isnan(feat_before_scale).sum().item()}")
                    print(f"  feat stats: min={feat_before_scale.min().item():.6f}, max={feat_before_scale.max().item():.6f}")
                    print(f"  Iteration: {self.comm_info['iter']}, Epoch: {self.epoch}")

                    # Print scenario names
                    scenario_names = [s.get('scenario', 'unknown') for s in scenario_samples]
                    print(f"  Scenarios in this batch: {scenario_names}")
                    print(f"  Point counts per scenario: {scenario_point_counts}")

                    # Check which dimensions have NaN
                    for dim in range(feat_before_scale.shape[1]):
                        if torch.isnan(feat_before_scale[:, dim]).any():
                            print(f"    dim[{dim}] has NaN!")

                    # Check if AMP is enabled
                    print(f"  AMP enabled: {self.cfg.enable_amp}")
                    print(f"  If AMP is True, try setting enable_amp=False in config to test.")

                    assert False, "Backbone output contains NaN! Training stopped to prevent corruption."

                # CRITICAL FIX: Do NOT normalize features
                # Normalization causes L2 Loss = 2*(1-Cosine), leading to mode collapse
                # point_feat["feat"] = F.normalize(point_feat["feat"], p=2, dim=1)  # REMOVED

                # ====================================================================
                # DEBUG: Comprehensive model collapse analysis (first iteration only)
                # ====================================================================
                if self.comm_info['iter'] == 0:
                    print("\n" + "="*80, flush=True)
                    print("==== [MODEL COLLAPSE ANALYSIS - ITER 0] ====", flush=True)
                    print("="*80, flush=True)

                    # Print collected decoder layer statistics
                    print(f"\n[0] Decoder Layer Statistics ({len(decoder_layer_outputs)} layers monitored):", flush=True)

                    # Sort layers by name for consistent output
                    sorted_layers = sorted(decoder_layer_outputs.keys())

                    for layer_name in sorted_layers:
                        stats = decoder_layer_outputs[layer_name]
                        status_icon = "✓"
                        status_msg = "OK"

                        # Check for anomalies
                        if stats['has_nan']:
                            status_icon = "🚨"
                            status_msg = "NaN DETECTED!"
                        elif stats['has_inf']:
                            status_icon = "🚨"
                            status_msg = "Inf DETECTED!"
                        elif stats['abs_max'] > 1000:
                            status_icon = "🚨"
                            status_msg = f"EXTREME (max={stats['abs_max']:.2f})"
                        elif stats['abs_max'] > 100:
                            status_icon = "⚠️"
                            status_msg = f"Large (max={stats['abs_max']:.2f})"
                        elif stats['abs_mean'] < 1e-6:
                            status_icon = "⚠️"
                            status_msg = f"Near-zero (mean={stats['abs_mean']:.2e})"
                        elif stats['num_zeros'] / stats['num_elements'] > 0.5:
                            status_icon = "⚠️"
                            status_msg = f"50% zeros ({stats['num_zeros']}/{stats['num_elements']})"

                        print(f"    {status_icon} {layer_name:40s} {status_msg}", flush=True)
                        if status_icon in ["🚨", "⚠️"]:
                            print(f"       shape={stats['shape']}, min={stats['min']:.4f}, max={stats['max']:.4f}, "
                                  f"mean={stats['mean']:.6f}, std={stats['std']:.6f}", flush=True)

                    # Remove hooks after collecting data
                    for hook in decoder_hooks:
                        hook.remove()

                    # Get decoder last layer bias
                    if hasattr(backbone, 'dec'):
                        decoder = backbone.dec
                        # Look for the final output layer (usually named 'out', 'cls', or is the last layer)
                        if hasattr(decoder, 'out'):
                            dec_bias = decoder.out.bias
                        elif hasattr(decoder, 'cls'):
                            dec_bias = decoder.cls.bias
                        elif hasattr(decoder, 'fc'):
                            dec_bias = decoder.fc.bias
                        else:
                            # Try to find the last Linear layer
                            for name, module in reversed(list(decoder.named_modules())):
                                if isinstance(module, torch.nn.Linear):
                                    dec_bias = module.bias
                                    break

                    if dec_bias is not None:
                        dec_bias_np = dec_bias.detach().cpu().numpy()
                        bias_max = abs(dec_bias_np).max()
                        bias_mean = abs(dec_bias_np).mean()
                        icon = "✓" if bias_max < 10 else "⚠️"
                        print(f"\n[0.1] Decoder last layer bias: {icon} max={bias_max:.4f}, mean={bias_mean:.4f}", flush=True)

                # Simplified backbone output info
                if self.comm_info['iter'] == 0:
                    print(f"\n[1] Backbone output shape: {feat_before_scale.shape}", flush=True)

                # CRITICAL: Apply tanh activation to match LangPretrainer.forward()
                # The LangPretrainer wrapper applies tanh, but trainer bypasses it
                # We MUST apply tanh here for consistency!
                batched_features = torch.tanh(feat_before_scale)

                # Simplified after tanh info
                if self.comm_info['iter'] == 0:
                    print(f"\n[2] After tanh shape: {batched_features.shape}", flush=True)
                    print("="*80 + "\n", flush=True)

                # NAN CHECK: Detect NaN after tanh
                if torch.isnan(batched_features).any():
                    raise AssertionError("🚨 NaN AFTER TANH! Check feat_before_scale.")

                # STABILITY CHECK: Detect extreme values that might cause numerical instability
                feat_max = batched_features.abs().max().item()
                feat_mean = batched_features.abs().mean().item()
                if feat_max > 1000.0 or feat_mean > 100.0:
                    print(f"⚠️ WARNING: Extreme features (max={feat_max:.2f}, mean={feat_mean:.2f}) - may cause instability")

            # Clean up large intermediate tensors to free memory before loss computation
            del point, point_feat, batched_input

            # Now compute loss for each scenario separately
            # (Each scenario has its own lang_feat target)

            # Get REAL 3D scene name from input_dict (e.g., "figurines", "kitchen", etc.)
            # This is the actual scene name from the dataset, NOT the training scenario (dense/single)
            real_scene_name = scene_name  # Already extracted above at line 1150-1155

            # Track all unique real scene names discovered during training
            self._all_real_scenes.add(real_scene_name)

            # Aggregate losses from all scenarios for this real scene
            # We'll average the losses across scenarios for the final loss value
            scenario_losses_for_real_scene = {}  # {scenario: {'total': float, 'l1': float, 'cos': float}}

            start_idx = 0
            for i, sample_dict in enumerate(scenario_samples):
                end_idx = start_idx + scenario_point_counts[i]
                # Extract features for this scenario (creates new tensor, not a view)
                scenario_feat = batched_features[start_idx:end_idx].clone()

                # Check if sample_dict has valid_feat_mask, otherwise use all ones
                if 'valid_feat_mask' in sample_dict:
                    valid_mask = sample_dict['valid_feat_mask']
                    valid_ratio = valid_mask.float().mean().item()
                    # Warn about low valid ratios
                    if valid_ratio < 0.5:
                        print(f"⚠️ WARNING: Scenario {i} has low valid_feat_mask ratio: {valid_ratio:.2%} (may cause instability)")
                else:
                    valid_mask = torch.ones(scenario_point_counts[i], dtype=torch.bool, device=device)

                # Build scenario-specific input for loss computation
                scenario_input = {
                    'coord': sample_dict['coord'],
                    'feat': sample_dict['feat'],
                    'grid_size': grid_size,
                    'epoch_progress': epoch_progress,
                    'valid_feat_mask': valid_mask,  # Use actual valid mask from dataset
                }

                # Add lang_feat if present (required for loss computation)
                if 'lang_feat' in sample_dict:
                    scenario_input['lang_feat'] = sample_dict['lang_feat']
                    # Check if lang_feat contains NaN
                    if torch.isnan(scenario_input['lang_feat']).any():
                        raise AssertionError(f"🚨 NaN in lang_feat for scenario {i}! Check data pipeline.")

                # Add segment labels if present
                if 'labels' in sample_dict:
                    scenario_input['segment'] = sample_dict['labels']

                # Add Gaussian parameters for Rendered2DLoss
                # Extract from feat tensor (11 channels: color(3) + opacity(1) + quat(4) + scale(3))
                feat = sample_dict['feat']

                # Always extract from feat if it has 11 channels
                if feat.shape[-1] == 11:
                    # Extract opacity from feat channels [3:4]
                    scenario_input['opacity'] = feat[:, 3:4]
                    # Extract quat from feat channels [4:8]
                    scenario_input['quat'] = feat[:, 4:8]
                    # Extract scale from feat channels [8:11]
                    scenario_input['scale'] = feat[:, 8:11]
                else:
                    # Fallback: try to get from sample_dict
                    if 'opacity' in sample_dict:
                        scenario_input['opacity'] = sample_dict['opacity']
                    if 'quat' in sample_dict:
                        scenario_input['quat'] = sample_dict['quat']
                    if 'scale' in sample_dict:
                        scenario_input['scale'] = sample_dict['scale']

                if 'scene_path' in sample_dict:
                    scenario_input['scene_path'] = sample_dict['scene_path']

                # Get scenario name for debug/info
                scenario_name = sample_dict.get('scenario', f'scenario_{i}')

                # Compute loss using criteria
                with torch.amp.autocast("cuda", enabled=self.cfg.enable_amp):
                    # Get criteria
                    if hasattr(self.model, 'module'):
                        criteria = self.model.module.criteria
                    else:
                        criteria = self.model.criteria

                    segment = scenario_input.get("segment")
                    lang_feat = scenario_input.get('lang_feat')

                    # DEBUG: Compare pred vs target at loss computation (first iteration, first scenario only)
                    if self.comm_info['iter'] == 0 and i == 0:
                        print(f"\n[6] Pred vs Target Analysis (scenario {i}):", flush=True)

                        if lang_feat is not None:
                            import torch.nn.functional as F
                            import numpy as np

                            # Compute overall difference statistics
                            pred_np = scenario_feat.detach().cpu().numpy()
                            gt_np = lang_feat.detach().cpu().numpy()
                            diff = np.abs(pred_np - gt_np)
                            overall_max_diff = diff.max()
                            overall_mean_diff = diff.mean()

                            # Compute cosine similarity
                            pred_norm = F.normalize(scenario_feat, p=2, dim=1)  # [N, 16]
                            gt_norm = F.normalize(lang_feat, p=2, dim=1)  # [N, 16]
                            cos_sim_per_row = (pred_norm * gt_norm).sum(dim=1)  # [N]

                            cos_mean = cos_sim_per_row.mean().item()
                            cos_min = cos_sim_per_row.min().item()

                            # Compact output: one line for differences, one for similarity
                            status = "✓" if cos_mean > 0.5 else ("⚠️" if cos_mean > 0.1 else "🚨")
                            print(f"    {status} Diff: max={overall_max_diff:.4f}, mean={overall_mean_diff:.4f} | "
                                  f"CosSim: mean={cos_mean:.4f}, min={cos_min:.4f}", flush=True)

                            if cos_mean < 0.1:
                                print(f"    🚨 WARNING: Very low cosine similarity! Model may be learning trivial solutions.", flush=True)

                            print("="*80 + "\n", flush=True)

                    # Prepare kwargs for criteria call (include Gaussian params for Rendered2DLoss)
                    criteria_kwargs = {
                        'valid_feat_mask': scenario_input['valid_feat_mask'],
                        'segment': segment,
                        'epoch_progress': epoch_progress,
                        'scenario': scenario_name,  # Pass scenario for Rendered2DLoss (only dense computes loss)
                    }
                    # Add optional parameters for Rendered2DLoss
                    if 'coord' in scenario_input:
                        criteria_kwargs['coord'] = scenario_input['coord']
                    if 'opacity' in scenario_input:
                        criteria_kwargs['opacity'] = scenario_input['opacity']
                    if 'quat' in scenario_input:
                        criteria_kwargs['quat'] = scenario_input['quat']
                    if 'scale' in scenario_input:
                        criteria_kwargs['scale'] = scenario_input['scale']
                    if 'scene_path' in scenario_input:
                        criteria_kwargs['scene_path'] = scenario_input['scene_path']

                    # Compute loss (returns tuple when verbose_losses=True)
                    loss_result = criteria(
                        scenario_feat,
                        lang_feat,  # Use raw features, not normalized
                        **criteria_kwargs,
                    )

                    # Handle return formats:
                    # - (loss, loss_dict) or (loss, loss_dict, per_dim_losses, per_dim_weights) when verbose_losses=True
                    # - just loss when verbose_losses=False
                    if isinstance(loss_result, tuple):
                        loss = loss_result[0]
                        if len(loss_result) == 2:
                            loss_dict = loss_result[1]
                            per_dim_losses = None
                            per_dim_weights = None
                        elif len(loss_result) == 4:
                            loss_dict = loss_result[1]
                            per_dim_losses = loss_result[2]
                            per_dim_weights = loss_result[3]
                        else:
                            loss_dict = None
                            per_dim_losses = None
                            per_dim_weights = None
                    else:
                        loss = loss_result
                        loss_dict = None
                        per_dim_losses = None
                        per_dim_weights = None

                    # Store this scenario's loss for current iteration (will be averaged later)
                    # NOTE: This is only this scenario's individual loss, NOT the total training loss
                    # The actual total_loss used for backprop is computed later (sum of all scenarios + consistency)
                    scenario_losses_for_real_scene[scenario_name] = {
                        'total': loss.item(),  # Single scenario loss
                        'l1': loss_dict.get('l1_loss', 0.0) if loss_dict else 0.0,
                        'cos': loss_dict.get('cos_loss', 0.0) if loss_dict else 0.0,
                        'contrast': loss_dict.get('contrast_loss', 0.0) if loss_dict else 0.0,
                        'per_dim_l1': per_dim_losses.get('per_dim_l1') if per_dim_losses else None,
                        'per_dim_l1_weights': per_dim_weights.get('per_dim_l1_weights') if per_dim_weights else None,
                    }

                output = dict(loss=loss, feat=scenario_feat)
                scenario_outputs.append(output)
                scenario_losses.append(loss)

                # Clean up scenario-specific tensors to free memory
                del scenario_feat, scenario_input

                start_idx = end_idx

            # UNIFIED LOSS TRACKING: Record losses for ALL real scenes at each iteration
            # For the scene trained this iteration: use the newly computed loss (averaged across scenarios)
            # For other scenes: repeat the last recorded loss value
            # This ensures all scenes have aligned loss curves for comparison and averaging
            #
            # CRITICAL FIX: Calculate TRUE global iteration that continues across epochs
            # self.comm_info["iter"] resets to 0 at the start of each epoch
            # We need: global_iter = epoch * iters_per_epoch + iter_within_epoch
            local_iter = self.comm_info["iter"]
            iters_per_epoch = len(self.train_loader)
            true_global_iter = self.epoch * iters_per_epoch + local_iter

            # Aggregate losses from all scenarios for the current real scene
            # Average the losses across scenarios (dense, single) for a single value per real scene
            if scenario_losses_for_real_scene:
                avg_total = sum(v['total'] for v in scenario_losses_for_real_scene.values()) / len(scenario_losses_for_real_scene)
                avg_l1 = sum(v['l1'] for v in scenario_losses_for_real_scene.values()) / len(scenario_losses_for_real_scene)
                avg_cos = sum(v['cos'] for v in scenario_losses_for_real_scene.values()) / len(scenario_losses_for_real_scene)
                avg_contrast = sum(v['contrast'] for v in scenario_losses_for_real_scene.values()) / len(scenario_losses_for_real_scene)

                # Aggregate per-dimension L1 losses (average across scenarios)
                per_dim_l1_list = [v['per_dim_l1'] for v in scenario_losses_for_real_scene.values() if v['per_dim_l1'] is not None]
                if per_dim_l1_list:
                    # Stack and average per-dimension losses
                    avg_per_dim_l1 = torch.stack(per_dim_l1_list).mean(dim=0)  # [D]
                else:
                    avg_per_dim_l1 = None

                # Aggregate per-dimension weights (take first scenario's weights, as they should be the same)
                per_dim_weights_list = [v['per_dim_l1_weights'] for v in scenario_losses_for_real_scene.values() if v['per_dim_l1_weights'] is not None]
                avg_per_dim_l1_weights = per_dim_weights_list[0] if per_dim_weights_list else None
            else:
                avg_total = avg_l1 = avg_cos = avg_rendered2d = 0.0
                avg_per_dim_l1 = None
                avg_per_dim_l1_weights = None

            # Update loss tracking for ALL known real scenes
            # For the current scene being trained: append the averaged loss
            # For other scenes: append their last loss value (repeat)
            for scene_name in self._all_real_scenes:
                # Initialize loss tracking dict for this scene if needed
                if scene_name not in self.per_scene_losses:
                    self.per_scene_losses[scene_name] = {
                        'total_loss': [],
                        'l1_loss': [],
                        'cos_loss': [],
                        'contrast_loss': [],
                        'per_dim_l1': [],  # List of tensors, each [D]
                        'per_dim_l1_weights': [],  # List of tensors, each [D]
                        'iterations': [],
                        'epochs': [],
                    }

                # Migrate old format (if per_dim_l1 keys are missing, add them)
                if 'per_dim_l1' not in self.per_scene_losses[scene_name]:
                    self.per_scene_losses[scene_name]['per_dim_l1'] = []
                if 'per_dim_l1_weights' not in self.per_scene_losses[scene_name]:
                    self.per_scene_losses[scene_name]['per_dim_l1_weights'] = []
                if 'contrast_loss' not in self.per_scene_losses[scene_name]:
                    self.per_scene_losses[scene_name]['contrast_loss'] = []

                # Get the loss values for this iteration
                if scene_name == real_scene_name:
                    # This is the scene being trained this iteration - use new averaged values
                    total_loss = avg_total
                    l1_loss = avg_l1
                    cos_loss = avg_cos
                    contrast_loss = avg_contrast
                    per_dim_l1 = avg_per_dim_l1
                    per_dim_l1_weights = avg_per_dim_l1_weights
                else:
                    # This scene was NOT trained this iteration - repeat last values
                    if self.per_scene_losses[scene_name]['total_loss']:
                        total_loss = self.per_scene_losses[scene_name]['total_loss'][-1]
                        l1_loss = self.per_scene_losses[scene_name]['l1_loss'][-1]
                        cos_loss = self.per_scene_losses[scene_name]['cos_loss'][-1]
                        contrast_loss = self.per_scene_losses[scene_name]['contrast_loss'][-1] if self.per_scene_losses[scene_name]['contrast_loss'] else 0.0
                        per_dim_l1 = self.per_scene_losses[scene_name]['per_dim_l1'][-1] if self.per_scene_losses[scene_name]['per_dim_l1'] else None
                        per_dim_l1_weights = self.per_scene_losses[scene_name]['per_dim_l1_weights'][-1] if self.per_scene_losses[scene_name]['per_dim_l1_weights'] else None
                    else:
                        # No previous loss - this shouldn't happen, but handle gracefully
                        total_loss = 0.0
                        l1_loss = 0.0
                        cos_loss = 0.0
                        contrast_loss = 0.0
                        per_dim_l1 = None
                        per_dim_l1_weights = None

                # Record the loss for this scene at this iteration
                self.per_scene_losses[scene_name]['total_loss'].append(total_loss)
                self.per_scene_losses[scene_name]['l1_loss'].append(l1_loss)
                self.per_scene_losses[scene_name]['cos_loss'].append(cos_loss)
                self.per_scene_losses[scene_name]['contrast_loss'].append(contrast_loss)
                self.per_scene_losses[scene_name]['per_dim_l1'].append(per_dim_l1)
                self.per_scene_losses[scene_name]['per_dim_l1_weights'].append(per_dim_l1_weights)
                self.per_scene_losses[scene_name]['iterations'].append(true_global_iter)
                self.per_scene_losses[scene_name]['epochs'].append(self.epoch)

            # Clean up batched_features after all scenarios processed
            del batched_features

        # else:
        #     # =====================================================================
        #     # MODE 2: SEPARATE FORWARD (slower but no cross-contamination)
        #     # =====================================================================
        #     # Forward each scenario independently to avoid interference
        #     # This prevents sparse convolution from operating across scenario boundaries
        #
        #     for i, sample_dict in enumerate(scenario_samples):
        #         num_points = sample_dict['coord'].shape[0]
        #
        #         # Build scenario-specific input dict
        #         scenario_input = {
        #             'coord': sample_dict['coord'],
        #             'feat': sample_dict['feat'],
        #             'batch': torch.zeros(num_points, dtype=torch.long, device=device),  # Always batch 0
        #             'grid_size': grid_size,
        #             'epoch_progress': epoch_progress,
        #             'valid_feat_mask': torch.ones(num_points, dtype=torch.bool, device=device),
        #         }
        #
        #         # Add lang_feat if present (required for loss computation)
        #         if 'lang_feat' in sample_dict:
        #             scenario_input['lang_feat'] = sample_dict['lang_feat']
        #
        #         # Add segment labels if present
        #         if 'labels' in sample_dict:
        #             scenario_input['segment'] = sample_dict['labels']
        #
        #         # Forward pass through backbone (separate for each scenario)
        #         with torch.amp.autocast("cuda", enabled=self.cfg.enable_amp):
        #             from pointcept.models.utils.structure import Point
        #             point = Point(scenario_input)
        #             point_feat = backbone(point)
        #
        #             # CRITICAL FIX: Do NOT normalize features
        #             # Normalization causes L2 Loss = 2*(1-Cosine), leading to mode collapse
        #             # point_feat["feat"] = F.normalize(point_feat["feat"], p=2, dim=1)  # REMOVED
        #
        #             scenario_feat = point_feat["feat"]  # [N_scenario, D]
        #
        #             # DEBUG: Verify model preserves point ordering (only print once at first iteration)
        #             if self.comm_info["iter"] == 0 and i == 0:
        #                 print(f"\n[DEBUG] Model Forward Verification:")
        #                 print(f"  Input coord shape: {scenario_input['coord'].shape}")
        #                 print(f"  Input feat shape: {scenario_input['feat'].shape}")
        #                 print(f"  Output feat shape: {scenario_feat.shape}")
        #                 # Check if point object has coord preserved
        #                 if hasattr(point_feat, 'coord') or 'coord' in point_feat.keys():
        #                     output_coord = point_feat.coord if hasattr(point_feat, 'coord') else point_feat['coord']
        #                     print(f"  Output coord shape: {output_coord.shape}")
        #                     # Verify coordinates match (should be identical if ordering preserved)
        #                     coords_match = torch.allclose(scenario_input['coord'], output_coord, atol=1e-5)
        #                     print(f"  Input/Output coords match: {coords_match}")
        #                     if not coords_match:
        #                         print(f"  WARNING: Coordinates don't match! This may indicate reordering.")
        #                         # Check if it's just a permutation
        #                         coord_diff = (scenario_input['coord'] - output_coord).abs().max()
        #                         print(f"  Max coord difference: {coord_diff.item()}")
        #
        #         # Clean up intermediate tensors
        #         del point, point_feat
        #
        #         # Compute loss using criteria
        #         with torch.amp.autocast("cuda", enabled=self.cfg.enable_amp):
        #             # Get criteria
        #             if hasattr(self.model, 'module'):
        #                 criteria = self.model.module.criteria
        #             else:
        #                 criteria = self.model.criteria
        #
        #             segment = scenario_input.get("segment")
        #             lang_feat = scenario_input.get('lang_feat')
        #
        #             # VALIDATION: Verify target (lang_feat) correctness before loss computation
        #             if self.comm_info["iter"] == 0 and i == 0:
        #                 print(f"\n[VALIDATION] Target Correctness Check - Scenario {i}:")
        #                 print(f"  pred (scenario_feat) shape: {scenario_feat.shape}")
        #                 print(f"  target (lang_feat) shape: {lang_feat.shape if lang_feat is not None else 'None'}")
        #                 print(f"  valid_feat_mask shape: {scenario_input['valid_feat_mask'].shape}")
        #
        #                 if lang_feat is not None:
        #                     # Check 1: NaN/Inf in target
        #                     has_nan = torch.isnan(lang_feat).any().item()
        #                     has_inf = torch.isinf(lang_feat).any().item()
        #                     print(f"  Target has NaN: {has_nan}, has Inf: {has_inf}")
        #
        #                     # Check 2: Target scale statistics
        #                     print(f"  Target statistics:")
        #                     print(f"    mean: {lang_feat.mean().item():.6f}, std: {lang_feat.std().item():.6f}")
        #                     print(f"    min: {lang_feat.min().item():.6f}, max: {lang_feat.max().item():.6f}")
        #
        #                     # Check 3: Pred statistics for comparison
        #                     print(f"  Pred statistics:")
        #                     print(f"    mean: {scenario_feat.mean().item():.6f}, std: {scenario_feat.std().item():.6f}")
        #                     print(f"    min: {scenario_feat.min().item():.6f}, max: {scenario_feat.max().item():.6f}")
        #
        #                     # Check 4: Scale difference (important for convergence)
        #                     target_scale = lang_feat.abs().mean().item()
        #                     pred_scale = scenario_feat.abs().mean().item()
        #                     scale_ratio = pred_scale / (target_scale + 1e-8)
        #                     print(f"  Scale ratio (pred/target): {scale_ratio:.4f}")
        #                     if scale_ratio > 100 or scale_ratio < 0.01:
        #                         print(f"    WARNING: Large scale mismatch! This may prevent convergence.")
        #
        #                     # Check 5: point_to_grid validity
        #                     point_to_grid = sample_dict.get('point_to_grid')
        #                     if point_to_grid is not None:
        #                         print(f"  point_to_grid range: [{point_to_grid.min()}, {point_to_grid.max()}]")
        #                         # Check if indices match lang_feat length
        #                         if point_to_grid.shape[0] != lang_feat.shape[0]:
        #                             print(f"    ERROR: point_to_grid length {point_to_grid.shape[0]} != lang_feat length {lang_feat.shape[0]}")
        #                         else:
        #                             print(f"    ✓ point_to_grid length matches lang_feat")
        #
        #                     # Check 6: Valid mask coverage
        #                     valid_mask = scenario_input['valid_feat_mask']
        #                     valid_ratio = valid_mask.float().mean().item()
        #                     print(f"  Valid feat mask coverage: {valid_ratio:.2%}")
        #                     if valid_ratio < 0.5:
        #                         print(f"    WARNING: Less than 50% of features are valid!")
        #
        #                     # Check 7: Per-dimension statistics (for 16-dim SVD)
        #                     if lang_feat.shape[1] == 16:
        #                         print(f"  Per-dimension target stats:")
        #                         for dim in range(min(4, 16)):  # Show first 4 dims
        #                             dim_mean = lang_feat[:, dim].mean().item()
        #                             dim_std = lang_feat[:, dim].std().item()
        #                             print(f"    dim[{dim}]: mean={dim_mean:.4f}, std={dim_std:.4f}")
        #
        #         loss = criteria(
        #             scenario_feat,
        #             lang_feat,  # Use raw features, not normalized
        #             valid_feat_mask=scenario_input['valid_feat_mask'],
        #             segment=segment,
        #             epoch_progress=epoch_progress,
        #         )
        #
        #         output = dict(loss=loss, feat=scenario_feat)
        #         scenario_outputs.append(output)
        #         scenario_losses.append(loss)
        #
        #         # Clean up scenario-specific tensors to free memory
        #         del scenario_feat, scenario_input

        forward_time = time.time() - forward_start

        # TIMING: Consistency loss computation
        consistency_start = time.time()
        # Compute density consistency loss between different sampling scenarios
        try:
            consistency_loss, consistency_loss_dict = self.consistency_loss(
                outputs=scenario_outputs,
                inputs=scenario_samples,
            )
        except Exception as e:
            self.logger.warning(f"Consistency loss computation failed: {e}")
            consistency_loss = torch.tensor(0.0, device=self.device)
            consistency_loss_dict = {'total_consistency': 0.0, 'num_common_grids': 0}
        consistency_time = time.time() - consistency_start

        # Optionally: add grid feature alignment loss
        # Disabled to save memory - this auxiliary loss doesn't contribute gradients
        # if self.use_compressed_features:
        #     grid_alignment_loss = self.compute_grid_alignment_loss(
        #         scenario_outputs, scenario_samples, lang_feat
        #     )
        #     consistency_loss = consistency_loss + 0.1 * grid_alignment_loss
        #     consistency_loss_dict['grid_alignment'] = grid_alignment_loss.item()

        # Compute weighted total loss
        total_loss = 0.0
        for loss, scenario in zip(scenario_losses, scenarios_to_use):
            weight = self.scenario_weights.get(scenario, 1.0)
            total_loss = total_loss + weight * loss

        # Add density consistency loss
        total_loss = total_loss + self.consistency_weight * consistency_loss

        # LOSS SPIKE DETECTION: Skip iteration to prevent gradient explosion in AMP
        # Chain reaction: loss spike → large gradients → fp16 overflow → NaN
        # Example: loss 0.91 → 1.93 (2x) → gradients overflow fp16 → NaN
        #
        # CRITICAL FIX: Instead of just clipping the loss, we SKIP the iteration entirely.
        # This prevents large gradients from being computed and corrupting the weights.
        # The previous approach (clipping loss) still allowed backward pass with large gradients.
        if self.cfg.enable_amp:
            loss_value = total_loss.item() if torch.is_tensor(total_loss) else total_loss

            # Initialize loss history tracking
            if not hasattr(self, '_loss_history'):
                self._loss_history = []
                self._loss_ma = None  # Moving average
                self._loss_ma_window = 10  # Moving average window

            # Track loss for spike detection
            self._loss_history.append(loss_value)
            if len(self._loss_history) > 50:  # Keep recent history
                self._loss_history.pop(0)

            # Update moving average
            if len(self._loss_history) >= self._loss_ma_window:
                recent_losses = self._loss_history[-self._loss_ma_window:]
                self._loss_ma = sum(recent_losses) / len(recent_losses)

            # Detection thresholds
            # NOTE: Normal loss range depends on checkpoint and data:
            # - After loading pretrained checkpoint: ~17-18 is NORMAL (each scene ~8.6 + 0.5)
            # - After fine-tuning: may decrease to lower values
            #
            # Therefore:
            # - fixed_threshold = 30.0 (safety net for EXTREME anomalies only)
            # - spike_multiplier = 2.5x MA (main detection mechanism)
            fixed_threshold = 30.0  # Extreme outliers only (1.7x normal initial loss)
            spike_multiplier = 2.5  # Allow 2.5x increase before skipping

            should_skip = False
            skip_reason = ""

            # Check 1: Spike detection (relative to moving average) - PRIMARY mechanism
            # This catches sudden loss increases that indicate gradient explosion
            if self._loss_ma is not None and len(self._loss_history) >= self._loss_ma_window:
                if loss_value > self._loss_ma * spike_multiplier:
                    should_skip = True
                    skip_reason = f"spike detected: {loss_value:.4f} > {spike_multiplier}x MA ({self._loss_ma:.4f})"

            # Check 2: Fixed threshold (catch EXTREME outliers only)
            # This is a safety net for completely broken training
            elif loss_value > fixed_threshold:
                should_skip = True
                skip_reason = f"exceeds extreme threshold {fixed_threshold}"

            if should_skip:
                ma_info = f" MA={self._loss_ma:.4f} ({loss_value/(self._loss_ma+1e-8):.1f}x)" if self._loss_ma is not None else ""
                print(f"⚠️ Loss spike ({skip_reason}): curr={loss_value:.4f}{ma_info} - skipping backward pass")
                # IMPORTANT: Return early to skip backward pass and optimizer step
                # This is the key fix - we don't compute gradients or update weights
                return

        # NAN CHECK: Detect and handle NaN loss before backward pass
        # This prevents cascading NaN failures that can corrupt the entire model
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print(f"🚨 NaN/Inf loss (iter={self.iter}, epoch={self.epoch}): total={total_loss.item()}, scenarios={[f'{l.item():.2f}' for l in scenario_losses]}")
            # Try to identify which scenario caused the NaN
            for i, (loss, scenario) in enumerate(zip(scenario_losses, scenarios_to_use)):
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"  Scenario '{scenario}' has NaN/Inf loss!")
            # Skip this iteration to prevent corruption
            return

        # TIMING: Backward pass
        backward_start = time.time()
        # Backward pass
        self.optimizer.zero_grad()

        if self.cfg.enable_amp:
            self.scaler.scale(total_loss).backward()

            # Check gradients before unscale (they're still scaled)
            has_nan_grad_before_unscale = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"🚨 NaN/Inf in scaled grad: {name}")
                        has_nan_grad_before_unscale = True

            if has_nan_grad_before_unscale:
                print(f"  Skipping unscale_() and optimizer step, reducing scale...")
                self.scaler.update()
                self.optimizer.zero_grad()
                return

            self.scaler.unscale_(self.optimizer)

            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )

            # LARGE GRADIENT DETECTION: Skip iteration if gradients are dangerously large
            # Large gradients can corrupt weights even if loss looks normal
            #
            # CRITICAL ISSUE: decoder0.conv.1.bias has persistent large gradients (~0.77)
            # - This is NOT a random spike, but a structural issue from dim[0] dominance
            # - Persistent large gradients → weight accumulation → activation anomalies → NaN
            #
            # Solution: Global threshold + consecutive detection
            # 1. Global threshold (1.0) - unified for all decoder layers
            # 2. Consecutive large gradient detection (5 in a row) - prevent accumulation
            # 3. Sudden spike detection (2.5x previous)
            #
            # Gradient norm analysis (with init_scale=512):
            # - decoder0.conv.1.bias: 0.7-0.8 (persistent, but within tolerance)
            # - Other decoder params: 0.1-0.3 (normal)
            # - Warning range: 0.5-1.0
            # - Danger range: > 1.0

            # Track max gradient
            max_grad = 0.0
            max_grad_param = ""

            for name, param in self.model.named_parameters():
                if param.grad is not None and 'dec' in name:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > max_grad:
                        max_grad = grad_norm
                        max_grad_param = name

            # Check for dangerous gradients that will corrupt weights
            should_skip_grad = False
            skip_reason = ""

            # Check 1: Global absolute threshold
            if max_grad > self.cfg.max_grad_threshold:
                should_skip_grad = True
                skip_reason = f"gradient {max_grad:.4f} > {self.cfg.max_grad_threshold}"

            # Check 2: Consecutive large gradient detection
            # DISABLED during warmup: LR is low, gradients may be consistently large
            # Warmup detection: current LR < 0.0005 (warmup goes from 0.0001 to 0.001)
            current_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            in_warmup = current_lr < 0.0005

            # Store previous gradient value BEFORE any checks (for comparison)
            prev_max_grad = getattr(self, '_prev_max_grad', 0)

            if in_warmup:
                # Reset consecutive count during warmup, don't accumulate
                self._consecutive_large_grad_count = 0
            else:
                # CRITICAL FIX: Only count consecutive GROWING gradients, not just large gradients
                # A stable gradient of 0.58 is fine, but a gradient growing from 0.5 → 0.7 → 0.9 is dangerous
                if hasattr(self, '_consecutive_large_grad_count'):
                    if prev_max_grad > 0:
                        # Only increment if gradient is GROWING (significantly larger than previous)
                        if max_grad > 1.2 * prev_max_grad:
                            self._consecutive_large_grad_count += 1
                        elif max_grad < 0.8 * prev_max_grad:
                            # Gradient decreased - reset counter
                            self._consecutive_large_grad_count = 0
                        # else: gradient is stable (within 20% of previous) - don't change counter
                    else:
                        # First iteration after warmup - initialize
                        self._consecutive_large_grad_count = 0

                    # Also increment if gradient is very large (>1.0) regardless of growth
                    if max_grad > 1.0:
                        self._consecutive_large_grad_count += 1
                else:
                    self._consecutive_large_grad_count = 0

                # Skip if we've had 10+ consecutive iterations with growing/large gradients
                # Increased from 5 to 10 to be more lenient
                if self._consecutive_large_grad_count >= 10:
                    should_skip_grad = True
                    skip_reason = f"{self._consecutive_large_grad_count} consecutive growing/large grads (max={max_grad:.4f})"

            # Check 3: Sudden spike detection (relative to previous)
            # Apply to all iterations regardless of warmup status
            if prev_max_grad > 0:
                if max_grad > 2.5 * prev_max_grad:
                    should_skip_grad = True
                    skip_reason = f"sudden spike: {max_grad:.4f} > 2.5x previous ({prev_max_grad:.4f})"

            # Store current gradient for next iteration's comparison
            self._prev_max_grad = max_grad

            # Handle dangerous gradients with reduced step size instead of skipping
            if should_skip_grad:
                # Strategy: Reduce all gradients to 1/10 of original and proceed with update
                # This is better than completely skipping because:
                # 1. Maintains training momentum
                # 2. Allows gradual weight adjustment even with large gradients
                # 3. Prevents complete stagnation when gradients are consistently large
                reduced_grad_scale = 0.1  # Scale factor for gradient reduction

                print(f"⚠️ Large grad ({skip_reason}): max={max_grad:.4f} in {max_grad_param}, scaling grads by {reduced_grad_scale}")

                # Apply reduced scale to all gradients
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(reduced_grad_scale)

                # Reset consecutive counter after applying gradient reduction
                # The reduced update should help stabilize training
                self._consecutive_large_grad_count = 0

            # WARNING: Monitor decoder layer gradients (no clipping, just warn)
            # Decoder layers receive large gradients from dim[0] and need monitoring
            for name, param in self.model.named_parameters():
                if param.grad is not None and 'dec' in name:
                    decoder_grad_norm = param.grad.norm().item()
                    if decoder_grad_norm > self.cfg.decoder_grad_warn_threshold:
                        print(f"⚠️ WARNING: Large decoder grad: {name} norm={decoder_grad_norm:.4f} (threshold: {self.cfg.decoder_grad_warn_threshold})")

            # Store old scale to check if scaler reduced it
            old_scale = self.scaler.get_scale()

            # Let GradScaler handle inf/NaN gradients automatically
            # If gradients contain inf/NaN, scaler.step() will skip the update
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # CHECK: Verify backbone weights are still valid after optimizer step
            # This catches cases where the optimizer step creates NaN/Inf weights
            if hasattr(self.model, 'module'):
                backbone = self.model.module.backbone
            else:
                backbone = self.model.backbone

            has_nan_in_backbone_after_step = False
            nan_params_after_step = []
            for name, param in backbone.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    has_nan_in_backbone_after_step = True
                    nan_count = torch.isnan(param).sum().item() + torch.isinf(param).sum().item()
                    nan_params_after_step.append(f"  {name} ({nan_count} NaN/Inf)")

            if has_nan_in_backbone_after_step:
                params_str = ", ".join(nan_params_after_step[:3])
                if len(nan_params_after_step) > 3:
                    params_str += f" +{len(nan_params_after_step)-3} more"
                raise RuntimeError(f"🚨 NaN/Inf in backbone AFTER step! Scale={new_scale}. Params: {params_str}")

            # WARNING: Monitor extreme weight values (no clipping, just warn)
            # This warns about weights that might cause NaN in subsequent iterations
            max_weight_value = 10.0  # Warning threshold for absolute weight values
            for name, param in self.model.named_parameters():
                max_abs_weight = param.data.abs().max().item()
                if max_abs_weight > max_weight_value:
                    print(f"⚠️ WARNING: Extreme weight: {name} max={max_abs_weight:.4f} (threshold: {max_weight_value})")

            # Check if scaler was reduced (indicates inf/NaN gradients were detected)
            new_scale = self.scaler.get_scale()
            if new_scale < old_scale:
                print(f"⚠️ WARNING: GradScaler reduced {old_scale} → {new_scale} (inf/NaN grads auto-skipped)")

            if old_scale <= new_scale:
                self.scheduler.step()
        else:
            total_loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()

        backward_time = time.time() - backward_start

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()

        # TIMING: Total iteration time
        total_iter_time = time.time() - iter_start_time

        # Store outputs - use total_loss (including consistency_loss) for logging
        output_dict = scenario_outputs[0].copy()
        output_dict["loss"] = total_loss
        self.comm_info["model_output_dict"] = output_dict
        self.comm_info["scenario_outputs"] = {
            scenario: output for scenario, output in zip(scenarios_to_use, scenario_outputs)
        }
        self.comm_info["consistency_loss"] = consistency_loss_dict

        # TIMING: Store timing information for logging
        self.comm_info["timing"] = {
            "data_load": data_load_time,
            "sampling": sampling_time,
            "forward": forward_time,
            "consistency": consistency_time,
            "backward": backward_time,
            "total": total_iter_time,
        }

        # Log losses
        if comm.is_main_process() and self.writer is not None:
            step = self.comm_info.get("iter", 0)
            global_step = self.epoch * len(self.train_loader) + step

            for scenario, loss in zip(scenarios_to_use, scenario_losses):
                self.writer.add_scalar(
                    f"train/loss_{scenario}",
                    loss.item(),
                    global_step
                )

            for key, value in consistency_loss_dict.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(
                        f"train/consistency_{key}",
                        value,
                        global_step
                    )

            self.writer.add_scalar(
                "train/loss_total",
                total_loss.item(),
                global_step
            )

            # # Print consistency loss to console
            # if consistency_loss_dict:
            #     # Print float losses
            #     consistency_str = ", ".join([f"{k}={v:.6f}" for k, v in consistency_loss_dict.items() if isinstance(v, float)])
            #     if consistency_str:
            #         print(f"[Epoch {self.epoch}, Iter {step}] Consistency Loss: {consistency_str}")
            #     # Print num_common_grids separately
            #     num_grids = consistency_loss_dict.get('num_common_grids')
            #     if num_grids is not None:
            #         num_grids_val = int(num_grids) if isinstance(num_grids, (int, float)) else num_grids
            #         print(f"[Epoch {self.epoch}, Iter {step}] Common Grids: {num_grids_val}")

            # for scenario, loss in zip(scenarios_to_use, scenario_losses):
            #     print(f"[Epoch {self.epoch}, Iter {step}] Loss {scenario}: {loss.item():.6f}")

        # =====================================================================
        # Per-Dimension Correlation Monitoring
        # =====================================================================
        if (
            self.enable_per_dim_monitor
            and self.per_dim_monitor is not None
            and self.comm_info["iter"] % self.per_dim_log_freq == 0
        ):
            # Get predictions and ground truth from the first scenario
            if len(scenario_outputs) > 0:
                first_output = scenario_outputs[0]
                pred_feat = first_output.get('feat')

                # Get ground truth from input_dict
                gt_feat = input_dict.get('lang_feat')

                if pred_feat is not None and gt_feat is not None:
                    # Apply valid mask if available
                    valid_mask = input_dict.get('valid_feat_mask', None)
                    if valid_mask is None:
                        # Create all-ones mask if not available
                        valid_mask = torch.ones(pred_feat.shape[0], dtype=torch.bool, device=pred_feat.device)

                    # Update per-dim monitor
                    self.per_dim_monitor.update(
                        pred=pred_feat,
                        gt=gt_feat,
                        valid_mask=valid_mask,
                        iteration=self.comm_info["iter"],
                        epoch=self.epoch,
                    )

                    # Check for trivial solution
                    if self.per_dim_monitor.is_trivial_solution():
                        self.logger.warning(
                            f"[Iteration {self.comm_info['iter']}] ⚠️ TRIVIAL SOLUTION DETECTED! "
                            f"Model may be predicting constant values. "
                            f"Dim 0 corr: {self.per_dim_monitor.history['dim0_corr'][-1]:.4f}, "
                            f"Minor corr: {self.per_dim_monitor.history['minor_corr_mean'][-1]:.4f}"
                        )

                    # Check for convergence
                    if self.per_dim_monitor.is_learning_principal_component():
                        self.logger.info(
                            f"[Iteration {self.comm_info['iter']}] ✓ Principal component converged! "
                            f"Dim 0 correlation: {self.per_dim_monitor.best_dim0_corr:.4f}"
                        )

                    if self.per_dim_monitor.is_learning_minor_components():
                        self.logger.info(
                            f"[Iteration {self.comm_info['iter']}] ✓ Minor components converged! "
                            f"Minor correlation: {self.per_dim_monitor.best_minor_corr:.4f}"
                        )
            # print(f"[Epoch {self.epoch}, Iter {step}] Total Loss: {total_loss.item():.6f}")

            # # TIMING: Print timing information to console
            # timing = self.comm_info.get("timing", {})
            # if timing:
            #     total = timing.get("total", 0)
            #     print(f"[Epoch {self.epoch}, Iter {step}] Time: "
            #           f"total={total:.4f}s, "
            #           f"data={timing.get('data_load', 0):.4f}s, "
            #           f"sampling={timing.get('sampling', 0):.4f}s, "
            #           f"forward={timing.get('forward', 0):.4f}s, "
            #           f"consistency={timing.get('consistency', 0):.4f}s, "
            #           f"backward={timing.get('backward', 0):.4f}s")
            #     # Print percentages for better understanding
            #     if total > 0:
            #         print(f"[Epoch {self.epoch}, Iter {step}] Time %: "
            #               f"sampling={timing.get('sampling', 0)/total*100:.1f}%, "
            #               f"forward={timing.get('forward', 0)/total*100:.1f}%, "
            #               f"consistency={timing.get('consistency', 0)/total*100:.1f}%, "
            #               f"backward={timing.get('backward', 0)/total*100:.1f}%")

            for sample_dict in scenario_samples:
                scenario = sample_dict['scenario']
                ratio = sample_dict['sampling_ratio']
                self.writer.add_scalar(
                    f"train/sampling_ratio_{scenario}",
                    ratio.item(),
                    global_step
                )

                num_points = sample_dict['num_points']
                self.writer.add_scalar(
                    f"train/num_points_{scenario}_rank{self.rank}",
                    num_points,
                    global_step
                )

    def compute_grid_alignment_loss(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        inputs: List[Dict[str, torch.Tensor]],
        compressed_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute grid feature alignment loss"""
        alignment_loss = 0.0
        count = 0

        for output, sample_input in zip(outputs, inputs):
            point_to_grid = sample_input.get('point_to_grid')
            model_feat = output.get('feat')

            if point_to_grid is None or model_feat is None:
                raise RuntimeError(
                    f"[compute_grid_alignment_loss] Missing required data! "
                    f"point_to_grid is None: {point_to_grid is None}, "
                    f"model_feat is None: {model_feat is None}. "
                    f"This indicates a data pipeline error where scenario outputs "
                    f"or inputs are not properly populated."
                )

            unique_grids = torch.unique(point_to_grid)

            for grid_id in unique_grids:
                grid_id = grid_id.item()
                if grid_id < compressed_features.shape[0]:
                    mask = (point_to_grid == grid_id)
                    if mask.sum() > 0:
                        model_feat_in_grid = model_feat[mask]
                        compressed_feat = compressed_features[grid_id]

                        if model_feat_in_grid.shape[-1] != compressed_feat.shape[-1]:
                            target_dim = min(model_feat_in_grid.shape[-1], compressed_feat.shape[-1])
                            model_proj = model_feat_in_grid[:, :target_dim]
                            compressed_proj = compressed_feat[:target_dim]
                        else:
                            model_proj = model_feat_in_grid
                            compressed_proj = compressed_feat

                        loss = F.mse_loss(
                            model_proj.mean(dim=0),
                            compressed_proj
                        )
                        alignment_loss += loss
                        count += 1

        if count > 0:
            alignment_loss = alignment_loss / count

        return alignment_loss

    # Hook methods
    def before_eval(self):
        for h in self.hooks:
            h.before_eval()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        # Reset scenario iteration counters at the start of each epoch
        # This ensures each epoch starts counting from iteration 0 for each scenario
        if hasattr(self, '_scenario_iter_in_epoch'):
            # Reset all scenario counters to 0
            for scenario_name in self._scenario_iter_in_epoch:
                self._scenario_iter_in_epoch[scenario_name] = 0
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

        for h in self.hooks:
            h.after_epoch()
            if self.cfg.empty_cache_per_epoch:
                torch.cuda.empty_cache()

        self.storage.reset_histories()

    def after_train(self):
        comm.synchronize()
        torch.cuda.empty_cache()

        # Save per-dim monitoring history
        if self.per_dim_monitor is not None and comm.is_main_process():
            history_path = os.path.join(self.cfg.save_path, "per_dim_correlation_history.json")
            self.per_dim_monitor.save_history(history_path)

            # Print final summary
            summary = self.per_dim_monitor.get_summary()
            print("\n" + "="*80)
            print("PER-DIMENSION CORRELATION MONITORING SUMMARY")
            print("="*80)
            print(f"\nTotal iterations: {summary['total_iterations']}")
            print(f"Final Dim 0 correlation: {summary['final_dim0_corr']:.4f}")
            print(f"Final Minor correlation: {summary['final_minor_corr']:.4f}")
            print(f"Best Dim 0 correlation: {summary['best_dim0_corr']:.4f}")
            print(f"Best Minor correlation: {summary['best_minor_corr']:.4f}")
            print(f"\nDim 0 converged (target {self.target_dim0_corr:.2f}): {summary['converged_dim0']}")
            print(f"Minor converged (target {self.target_minor_corr:.2f}): {summary['converged_minor']}")
            print(f"Trivial solution detected: {summary['is_trivial']}")
            print("="*80 + "\n")

        for h in self.hooks:
            h.after_train()

        if comm.is_main_process() and self.writer is not None:
            self.writer.close()


# Import required functions
from pointcept.engines.defaults import worker_init_fn
