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
                    continue

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
        self.max_epoch = cfg.eval_epoch
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

        self.logger.info("=> Density-Invariant Training Configuration:")
        self.logger.info(f"   Rank: {self.rank}/{self.world_size}")
        self.logger.info(f"   Sample ratio range: [{self.min_sample_ratio}, {self.max_sample_ratio}]")
        self.logger.info(f"   Consistency weight: {self.consistency_weight}")
        self.logger.info(f"   Consistency type: {self.consistency_type}")
        self.logger.info(f"   Enabled scenarios: {self.enabled_scenarios}")
        self.logger.info(f"   Scenario weights: {self.scenario_weights}")
        self.logger.info(f"   Use compressed features: {self.use_compressed_features}")
        self.logger.info(f"   SVD rank: {self.svd_rank if self.svd_rank else 'auto (highest available)'}")

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
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        if comm.is_main_process():
            self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
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
        """Build gradient scaler"""
        scaler = torch.amp.GradScaler("cuda") if self.cfg.enable_amp else None
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
            else:
                point_to_grid = point_to_grid[0]  # Take first element

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
        )
        sampling_time = time.time() - sampling_start

        # Forward pass for each scenario
        scenario_outputs = []
        scenario_losses = []

        # OPTIMIZED: Pre-compute common values outside the loop to avoid repeated access
        grid_size = input_dict.get('grid_size', 0.01)
        epoch_progress = self.epoch / self.max_epoch
        device = coord.device
        scene_name = input_dict.get('name', 'unknown')

        # TIMING: Forward pass
        forward_start = time.time()

        # OPTIMIZATION: Batched forward pass for all scenarios
        # Instead of calling model 3 times, concatenate all scenarios and call once
        # This leverages GPU parallelism and reduces kernel launch overhead

        # Check minimum point count for all scenarios first
        for sample_dict in scenario_samples:
            num_points = sample_dict['coord'].shape[0]
            if num_points < 4:
                raise ValueError(
                    f"[Rank {self.rank}] Scenario '{sample_dict.get('scenario', 'unknown')}' "
                    f"has only {num_points} point(s), but sparse convolution requires >= 4 points. "
                    f"This may indicate the scene has too few grids or points. "
                    f"Scene: {scene_name}"
                )

        # Concatenate all scenarios into a single batch
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
        with torch.amp.autocast("cuda", enabled=self.cfg.enable_amp):
            # Get backbone model
            if hasattr(self.model, 'module'):
                # DDP wrapped model
                backbone = self.model.module.backbone
            else:
                backbone = self.model.backbone

            # Create Point object and pass through backbone
            from pointcept.models.utils.structure import Point
            point = Point(batched_input)
            point_feat = backbone(point)

            # Normalize features (consistent with LangPretrainer)
            import torch.nn.functional as F
            point_feat["feat"] = F.normalize(point_feat["feat"], p=2, dim=1)

            # Split features back to individual scenarios
            batched_features = point_feat["feat"]  # [Total_points, D]

        # Clean up large intermediate tensors to free memory before loss computation
        del point, point_feat, batched_input

        # Now compute loss for each scenario separately
        # (Each scenario has its own lang_feat target)
        start_idx = 0
        for i, sample_dict in enumerate(scenario_samples):
            end_idx = start_idx + scenario_point_counts[i]
            # Extract features for this scenario (creates new tensor, not a view)
            scenario_feat = batched_features[start_idx:end_idx].clone()

            # Build scenario-specific input for loss computation
            scenario_input = {
                'coord': sample_dict['coord'],
                'feat': sample_dict['feat'],
                'grid_size': grid_size,
                'epoch_progress': epoch_progress,
                'valid_feat_mask': torch.ones(
                    scenario_point_counts[i], dtype=torch.bool, device=device
                ),
            }

            # Add lang_feat if present (required for loss computation)
            if 'lang_feat' in sample_dict:
                scenario_input['lang_feat'] = sample_dict['lang_feat']

            # Add segment labels if present
            if 'labels' in sample_dict:
                scenario_input['segment'] = sample_dict['labels']

            # Compute loss using criteria
            with torch.amp.autocast("cuda", enabled=self.cfg.enable_amp):
                # Get criteria
                if hasattr(self.model, 'module'):
                    criteria = self.model.module.criteria
                else:
                    criteria = self.model.criteria

                segment = scenario_input.get("segment")

                # Normalize GT features to match model output (consistent with LangPretrainer)
                lang_feat = scenario_input.get('lang_feat')
                if lang_feat is not None:
                    import torch.nn.functional as F
                    lang_feat_normalized = F.normalize(lang_feat, p=2, dim=1)
                else:
                    lang_feat_normalized = None

                loss = criteria(
                    scenario_feat,
                    lang_feat_normalized,
                    valid_feat_mask=scenario_input['valid_feat_mask'],
                    segment=segment,
                    epoch_progress=epoch_progress,
                )

            output = dict(loss=loss, feat=scenario_feat)
            scenario_outputs.append(output)
            scenario_losses.append(loss)

            # Clean up scenario-specific tensors to free memory
            del scenario_feat, scenario_input

            start_idx = end_idx

        # Clean up batched_features after all scenarios processed
        del batched_features

        forward_time = time.time() - forward_start

        # TIMING: Consistency loss computation
        consistency_start = time.time()
        # Compute density consistency loss
        with torch.amp.autocast("cuda", enabled=self.cfg.enable_amp):
            consistency_loss, consistency_loss_dict = self.consistency_loss(
                outputs=scenario_outputs,
                inputs=scenario_samples,
            )
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

        total_loss = total_loss + self.consistency_weight * consistency_loss

        # TIMING: Backward pass
        backward_start = time.time()
        # Backward pass
        self.optimizer.zero_grad()

        if self.cfg.enable_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
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

            # Print consistency loss to console
            if consistency_loss_dict:
                # Print float losses
                consistency_str = ", ".join([f"{k}={v:.6f}" for k, v in consistency_loss_dict.items() if isinstance(v, float)])
                if consistency_str:
                    print(f"[Epoch {self.epoch}, Iter {step}] Consistency Loss: {consistency_str}")
                # Print num_common_grids separately
                num_grids = consistency_loss_dict.get('num_common_grids')
                if num_grids is not None:
                    num_grids_val = int(num_grids) if isinstance(num_grids, (int, float)) else num_grids
                    print(f"[Epoch {self.epoch}, Iter {step}] Common Grids: {num_grids_val}")

            for scenario, loss in zip(scenarios_to_use, scenario_losses):
                print(f"[Epoch {self.epoch}, Iter {step}] Loss {scenario}: {loss.item():.6f}")
            print(f"[Epoch {self.epoch}, Iter {step}] Total Loss: {total_loss.item():.6f}")

            # TIMING: Print timing information to console
            timing = self.comm_info.get("timing", {})
            if timing:
                total = timing.get("total", 0)
                print(f"[Epoch {self.epoch}, Iter {step}] Time: "
                      f"total={total:.4f}s, "
                      f"data={timing.get('data_load', 0):.4f}s, "
                      f"sampling={timing.get('sampling', 0):.4f}s, "
                      f"forward={timing.get('forward', 0):.4f}s, "
                      f"consistency={timing.get('consistency', 0):.4f}s, "
                      f"backward={timing.get('backward', 0):.4f}s")
                # Print percentages for better understanding
                if total > 0:
                    print(f"[Epoch {self.epoch}, Iter {step}] Time %: "
                          f"sampling={timing.get('sampling', 0)/total*100:.1f}%, "
                          f"forward={timing.get('forward', 0)/total*100:.1f}%, "
                          f"consistency={timing.get('consistency', 0)/total*100:.1f}%, "
                          f"backward={timing.get('backward', 0)/total*100:.1f}%")

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
                continue

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

        for h in self.hooks:
            h.after_train()

        if comm.is_main_process():
            self.writer.close()


# Import required functions
from pointcept.engines.defaults import worker_init_fn
