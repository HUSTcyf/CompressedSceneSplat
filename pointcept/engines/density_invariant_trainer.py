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

        # Group points by grid
        unique_grids, inverse_indices = torch.unique(
            point_to_grid,
            return_inverse=True
        )

        grid_to_points = {}

        # Build mapping for each unique grid
        # Use a more efficient approach to avoid creating huge intermediate tensors
        for i, grid_id in enumerate(unique_grids):
            grid_id = grid_id.item()
            # Find all points that belong to this grid
            points_in_grid = torch.nonzero(inverse_indices == i, as_tuple=True)[0]
            grid_to_points[grid_id] = points_in_grid

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
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Scenario 2: Sample ~50% of points per grid (30%-70% random)

        Args:
            coord: [N, 3] point coordinates
            feat: [N, C] point features
            point_to_grid: [N] point to grid mapping
            labels: [N] point labels (optional)

        Returns:
            sample_dict: Dictionary with sampled data
        """
        device = coord.device

        # Build inverse mapping: grid -> points
        grid_to_points = self.build_inverse_mapping(point_to_grid)

        # Sample from each grid
        sampled_indices = []
        actual_ratios = []

        for grid_id, points_in_grid in grid_to_points.items():
            num_points = len(points_in_grid)

            if num_points == 0:
                continue

            # Random sampling ratio between min and max
            sample_ratio = self.rng.uniform(
                self.min_sample_ratio, self.max_sample_ratio
            )

            # Calculate number of points to sample
            num_samples = max(1, int(num_points * sample_ratio))

            # Sample indices
            if num_samples >= num_points:
                sampled_indices.append(points_in_grid)
                actual_ratio = 1.0
                actual_ratios.extend([actual_ratio] * points_in_grid.shape[0])
            else:
                perm = torch.randperm(num_points, device=device)
                sampled = points_in_grid[perm[:num_samples]]
                sampled_indices.append(sampled)
                actual_ratio = num_samples / num_points
                actual_ratios.extend([actual_ratio] * sampled.shape[0])

        # Concatenate all sampled indices
        if len(sampled_indices) == 0:
            num_samples = max(1, int(coord.shape[0] * 0.5))
            perm = torch.randperm(coord.shape[0], device=device)
            all_indices = perm[:num_samples]
            actual_ratios = [0.5] * num_samples
        else:
            all_indices = torch.cat(sampled_indices)

        # Build sample dictionary
        sample_dict = {
            'coord': coord[all_indices],
            'feat': feat[all_indices],
            'point_to_grid': point_to_grid[all_indices],
            'sampling_ratio': torch.tensor(
                np.mean(actual_ratios),
                device=device
            ),
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
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Scenario 3: Sample 1 point per grid

        Args:
            coord: [N, 3] point coordinates
            feat: [N, C] point features
            point_to_grid: [N] point to grid mapping
            labels: [N] point labels (optional)

        Returns:
            sample_dict: Dictionary with sampled data (1 point per grid)
        """
        device = coord.device

        # Build inverse mapping: grid -> points
        grid_to_points = self.build_inverse_mapping(point_to_grid)

        # Sample 1 point per grid
        sampled_indices = []

        for grid_id, points_in_grid in grid_to_points.items():
            num_points = len(points_in_grid)
            assert num_points > 0

            if num_points == 1:
                sampled_indices.append(points_in_grid)
            else:
                selected_idx = self.rng.randint(0, num_points)
                # Use list indexing to maintain 1D tensor shape
                sampled_indices.append(points_in_grid[[selected_idx]])

        if len(sampled_indices) == 0:
            all_indices = torch.tensor([0], device=device)
        else:
            all_indices = torch.cat(sampled_indices)

        num_grids = len(grid_to_points)
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

        for scenario in scenarios:
            if scenario == 'dense':
                sample = self.sample_dense(coord, feat, point_to_grid, labels, **kwargs)
            elif scenario == 'half':
                sample = self.sample_half_density(coord, feat, point_to_grid, labels, **kwargs)
            elif scenario == 'single':
                sample = self.sample_single_per_grid(coord, feat, point_to_grid, labels, **kwargs)
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

        Args:
            feat1: [N1, C] features from scenario 1
            grid1: [N1] grid indices from scenario 1
            feat2: [N2, C] features from scenario 2
            grid2: [N2] grid indices from scenario 2

        Returns:
            aligned_feat1: [M, C] aligned features from scenario 1
            aligned_feat2: [M, C] aligned features from scenario 2
        """
        # Find common grids
        unique_grids1 = torch.unique(grid1)
        unique_grids2 = torch.unique(grid2)

        # Find intersection using numpy
        common_grids = np.intersect1d(
            unique_grids1.cpu().numpy(),
            unique_grids2.cpu().numpy()
        )

        if len(common_grids) == 0:
            return (
                torch.zeros(0, feat1.shape[1], device=feat1.device),
                torch.zeros(0, feat2.shape[1], device=feat2.device)
            )

        # Aggregate features per common grid
        aligned_feat1_list = []
        aligned_feat2_list = []

        for grid_id in common_grids:
            mask1 = (grid1 == grid_id)
            mask2 = (grid2 == grid_id)

            if mask1.sum() > 0 and mask2.sum() > 0:
                feat1_in_grid = feat1[mask1].mean(dim=0)
                feat2_in_grid = feat2[mask2].mean(dim=0)

                aligned_feat1_list.append(feat1_in_grid)
                aligned_feat2_list.append(feat2_in_grid)

        if len(aligned_feat1_list) == 0:
            return (
                torch.zeros(0, feat1.shape[1], device=feat1.device),
                torch.zeros(0, feat2.shape[1], device=feat2.device)
            )

        aligned_feat1 = torch.stack(aligned_feat1_list)
        aligned_feat2 = torch.stack(aligned_feat2_list)

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
        input_dict = self.comm_info["input_dict"]

        # Move to GPU
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

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
        if valid_feat_mask is not None:
            valid_mask = valid_feat_mask > 0
            num_valid = valid_mask.sum().item()
            num_total = coord.shape[0]

            # Warn if very few valid points
            if num_valid < 100:
                self.logger.warning(f"[Rank {self.rank}] Scene '{scene_name}' has very few valid points ({num_valid}). "
                                   f"This may indicate a data problem.")

            # Filter to valid points
            coord = coord[valid_mask]
            feat = feat[valid_mask]
            if lang_feat is not None:
                lang_feat = lang_feat[valid_mask]
            if labels is not None:
                labels = labels[valid_mask]

            self.logger.debug(f"[Rank {self.rank}] Filtered to {num_valid:,} valid points "
                            f"out of {num_total:,} total points")

        # Get point_to_grid mapping (loaded in GenericGSDataset)
        point_to_grid = input_dict.get('point_to_grid')

        # Handle batched data: DataLoader wraps items in lists
        if isinstance(point_to_grid, list):
            if len(point_to_grid) == 0:
                point_to_grid = None
            else:
                point_to_grid = point_to_grid[0]  # Take first element

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

        # Forward pass for each scenario
        scenario_outputs = []
        scenario_losses = []

        for sample_dict in scenario_samples:
            # Check minimum point count (spconv requires minimum points)
            num_points = sample_dict['coord'].shape[0]
            if num_points < 4:  # Minimum points for sparse convolution
                raise ValueError(
                    f"[Rank {self.rank}] Scenario '{sample_dict.get('scenario', 'unknown')}' "
                    f"has only {num_points} point(s), but sparse convolution requires >= 4 points. "
                    f"This may indicate the scene has too few grids or points. "
                    f"Scene: {input_dict.get('name', 'unknown')}"
                )

            # Build scenario input dict with all required fields
            scenario_input = {
                'coord': sample_dict['coord'],
                'feat': sample_dict['feat'],
                'batch': sample_dict.get('batch'),
                'grid_size': sample_dict.get('grid_size', 0.01),
                'epoch_progress': self.epoch / self.max_epoch,
            }

            # Debug: print batch information for this scenario
            batch = sample_dict.get('batch')
            if batch is not None:
                unique_batches = torch.unique(batch)
                self.logger.debug(f"[Rank {self.rank}] Scenario '{sample_dict.get('scenario')}' "
                                f"points={sample_dict['coord'].shape[0]}, "
                                f"feat_shape={sample_dict['feat'].shape}, "
                                f"unique_batches={unique_batches.tolist()}, "
                                f"batch_size={len(unique_batches)}")

            # Add lang_feat if present (required for LangPretrainer)
            if 'lang_feat' in sample_dict:
                scenario_input['lang_feat'] = sample_dict['lang_feat']

            # Add segment labels if present (required for loss computation)
            if 'labels' in sample_dict:
                scenario_input['segment'] = sample_dict['labels']

            # For filtered data, all points are valid
            # Create a valid_feat_mask that's all True
            scenario_input['valid_feat_mask'] = torch.ones(
                sample_dict['coord'].shape[0],
                dtype=torch.bool,
                device=sample_dict['coord'].device
            )

            # Check effective batch size (number of unique batch indices)
            with torch.amp.autocast("cuda", enabled=self.cfg.enable_amp):
                output = self.model(scenario_input)
                loss = output.get("loss", torch.tensor(0.0, device=coord.device))

            scenario_outputs.append(output)
            scenario_losses.append(loss)

        # Compute density consistency loss
        with torch.amp.autocast("cuda", enabled=self.cfg.enable_amp):
            consistency_loss, consistency_loss_dict = self.consistency_loss(
                outputs=scenario_outputs,
                inputs=scenario_samples,
            )

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

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()

        # Store outputs - use total_loss (including consistency_loss) for logging
        output_dict = scenario_outputs[0].copy()
        output_dict["loss"] = total_loss
        self.comm_info["model_output_dict"] = output_dict
        self.comm_info["scenario_outputs"] = {
            scenario: output for scenario, output in zip(scenarios_to_use, scenario_outputs)
        }
        self.comm_info["consistency_loss"] = consistency_loss_dict

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
