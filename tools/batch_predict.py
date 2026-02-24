#!/usr/bin/env python3
"""
Batch Prediction Script for SceneSplat

This script loads 3D Gaussian Splatting models from arbitrary paths,
predicts language features using SceneSplat, and saves the results.

Features:
- Support for both .ply files and preprocessed .npy folders
- Batch processing of multiple scenes
- Automatic preprocessing of .ply files
- Memory-efficient chunking for large scenes
- Feature extraction and saving
- Save features as .npy files for benchmark evaluation (compatible with gaussian_world_3d_semseg_benchmarks)
- Support for LitePT model with LangPretrainer wrapper
- Auto-detection of original checkpoint paths for preprocessed data
- Pruning of Gaussians with invalid language features using valid_feat_mask

Usage:
    # Single file (standard SceneSplat model)
    python batch_predict.py --input scene.ply --output ./output --weight model.pth

    # Directory of .ply files
    python batch_predict.py --input ./ply_files --output ./output --weight model.pth

    # Preprocessed .npy folders
    python batch_predict.py --input ./npy_scenes --output ./output --weight model.pth --preprocessed

    # With custom config
    python batch_predict.py --input ./scenes --output ./output --weight model.pth \
        --config configs/scannet/lang-pretrain-scannet-mcmc-wo-normal-contrastive.py

    # Save .npy files for benchmark evaluation
    python batch_predict.py --input ./scenes --output ./output --weight model.pth \
        --save_npy --npy_output_root ./language_features_siglip2

    # With specific scene ID for .npy saving
    python batch_predict.py --input scene.ply --output ./output --weight model.pth \
        --save_npy --scene_id scene0010

    # Using LitePT model
    python batch_predict.py --input ./scenes --output ./output --weight litept_model.pth \
        --model_type litept --config configs/custom/lang-pretrain-litept-ovs.py

    # Preprocessed data with checkpoint base directory (recursive mode)
    python batch_predict.py --input /path/to/preprocessed_data --output ./output \
        --weight model.pth --preprocessed --recursive \
        --original_checkpoint /path/to/checkpoints --iterations 30000
    # This will look for checkpoints in patterns:
    #   /path/to/checkpoints/{scene_name}/ckpts/chkpnt30000.pth
    #   /path/to/checkpoints/{scene_name}/chkpnt30000.pth
    #   /path/to/checkpoints/chkpnt30000.pth

    # With pruning enabled (remove invalid Gaussians from checkpoint)
    python batch_predict.py --input ./scenes --output ./output --weight model.pth \
        --preprocessed --prune_invalid
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.logger import get_root_logger


def build_test_transform_pipeline(
    grid_size: float = 0.02,
    normalize_color: bool = True,
    center_shift_z: bool = True,
    model_type: str = "scenesplat",
    use_multi_crop: bool = False,
    crop_radius: float = 5.0,
    crop_overlap: float = 0.2,
    max_crops: int = 50,
    skip_gridsample: bool = False,  # NEW: Skip GridSample, use full point cloud
):
    """
    Build test transform pipeline following SceneSplat pattern.

    Returns:
        tuple: (initial_transform, voxelize_transform, post_transform)
        - initial_transform: Compose applied before voxelization
        - voxelize_transform: GridSample that creates fragments
        - post_transform: Compose applied to each fragment

    Args:
        use_multi_crop: If True, use GridCrop for full scene coverage
        crop_radius: Radius of each spherical crop (in world units)
        crop_overlap: Overlap ratio between adjacent crops (0-1)
        max_crops: Maximum number of crops to generate
    """
    # GridSample keys - follow SceneSplat pattern exactly
    if model_type == "litept":
        grid_sample_keys = ("coord", "color", "opacity", "quat", "scale")
        feat_keys = ("color", "opacity", "quat", "scale")  # 11 channels (no coord, matches config)
    else:
        grid_sample_keys = ("coord", "color", "opacity", "quat", "scale")
        feat_keys = ("color", "opacity", "quat", "scale")  # 11 channels

    # Initial transform (applied before voxelization)
    initial_transforms = []
    if center_shift_z:
        initial_transforms.append(dict(type="CenterShift", apply_z=True))
    if normalize_color:
        initial_transforms.append(dict(type="NormalizeColor"))
    # Normalize coordinates to prevent overflow for large scenes
    # NOTE: We don't use NormalizeCoord to preserve more points
    # Instead, we use SphereCrop/GridCrop + RecomputeGridCoord to prevent depth overflow
    initial_transforms.append(
        dict(
            type="Copy",
            keys_dict={
                "coord": "origin_coord",
            },
        )
    )
    initial_transforms.append(
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            keys=grid_sample_keys,
            return_inverse=True,
            return_grid_coord=True,
        )
    )

    if use_multi_crop:
        if not skip_gridsample:
            # ORIGINAL: Use GridSample first
            initial_transforms.append(
                dict(
                    type="GridSample",
                    grid_size=grid_size,
                    hash_type="fnv",
                    mode="train",
                    keys=grid_sample_keys,
                    return_inverse=True,
                    return_grid_coord=True,
                )
            )

        # Step 1: Filter coordinate outliers BEFORE GridCrop to prevent uneven distribution
        # This removes extreme values that cause large scene_size and poor crop distribution
        initial_transforms.append(dict(
            type="FilterCoordOutliers",
            percentile_low=0.5,   # Remove bottom 0.5%
            percentile_high=99.5,  # Remove top 0.5%
            min_points=10000,      # Keep at least this many points
            verbose=True,
        ))
        # Step 2: Use GridCrop for full scene coverage with multiple overlapping crops
        # crop_radius=None means auto-calculate based on scene size (default: 10% of max dimension)
        # min_points=2000 is the new default (adaptive based on scene size)
        # max_crops limits the number of crops for large scenes
        initial_transforms.append(dict(
            type="GridCrop",
            crop_radius=None,  # Auto-calculate based on scene size
            overlap=crop_overlap,
            min_points=2000,   # Lower threshold for better coverage
            max_crops=max_crops,
        ))
        # Note: Don't use RecomputeGridCoord here, as it will be applied per-crop
        initial_transform = Compose(initial_transforms)
        # Return None for voxelize since GridCrop handles the splitting
        return initial_transform, None, None
    else:
        # Single center crop mode
        # CRITICAL: Use SphereCrop to reduce spatial extent and prevent depth overflow
        # mode="center" crops from center, ensuring consistent results
        # This reduces coord span from ~100k to ~10 units, keeping depth <= 16
        initial_transforms.append(dict(type="SphereCrop", point_max=204800, mode="center"))
        # CRITICAL: Recompute grid_coord from local coord to prevent depth overflow
        # This ensures grid_coord values start from 0, keeping depth <= 16
        initial_transforms.append(dict(type="RecomputeGridCoord", grid_size=grid_size))
        initial_transform = Compose(initial_transforms)

    # Voxelization (creates test fragments)
    voxelize_transform = TRANSFORMS.build(
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="test",
            keys=grid_sample_keys,
            return_grid_coord=True,
        )
    )

    # Post transform (applied to each fragment)
    post_transform = Compose([
        dict(type="CenterShift", apply_z=False),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "index"),
            feat_keys=feat_keys,
            offset_keys_dict=dict(batch="coord"),  # Create "batch" key for model
        ),
    ])

    return initial_transform, voxelize_transform, post_transform


def create_occamlgs_checkpoint(
    features: np.ndarray,
    index: np.ndarray,
    original_checkpoint_path: str,
    valid_feat_mask: Optional[np.ndarray] = None,
    prune_invalid: bool = True,
) -> Tuple:
    """
    Create GaussianModel checkpoint compatible with feature_map_renderer.py.

    This function returns the capture_language_feature() format which is:
    (active_sh_degree, xyz, features_dc, features_rest,
     scaling, rotation, opacity, language_features,
     max_radii2D, xyz_gradient_accum, denom,
     opt_dict, spatial_lr_scale)

    IMPORTANT: language_features are stored separately, NOT in features_dc!
    features_dc and features_rest store SH coefficients for color rendering.

    Args:
        features: Predicted language features [M, feat_dim]
        index: Inverse mapping from original points to voxels [N]
        original_checkpoint_path: Path to original checkpoint
        valid_feat_mask: Boolean mask for valid features [N]
        prune_invalid: Whether to prune Gaussians with invalid features

    Returns:
        GaussianModel format tuple (13 elements)
    """
    print(f"Loading original checkpoint from: {original_checkpoint_path}")
    orig_ckpt = torch.load(original_checkpoint_path, map_location="cpu", weights_only=False)

    # Handle gsplat format
    if isinstance(orig_ckpt, dict) and "splats" in orig_ckpt:
        # gsplat format: {"splats": {...}, "step": ...}
        print("Detected gsplat format checkpoint")
        splats = orig_ckpt["splats"]

        # Extract gsplat components
        xyz = splats["means"]  # [N, 3]
        opacity = splats["opacities"]
        if opacity.dim() == 1:
            opacity = opacity.unsqueeze(-1)
        features_dc = splats["sh0"].squeeze(1)  # [N, 3] - SH DC coefficients for COLOR
        features_rest = splats["shN"].reshape(xyz.shape[0], -1)  # [N, 45] - SH rest for COLOR
        scaling = splats["scales"]
        rotation = splats["quats"]

        N = xyz.shape[0]

        # Create placeholders for GaussianModel state
        active_sh_degree = 3
        max_radii2D = torch.zeros(N, dtype=torch.int32, device=xyz.device)
        xyz_gradient_accum = torch.zeros(N, 1, dtype=torch.float32, device=xyz.device)
        denom = torch.zeros(N, 1, dtype=torch.float32, device=xyz.device)
        opt_dict = {}  # Empty optimizer state dict
        spatial_lr_scale = 1.0

        # Apply valid_feat_mask pruning
        if valid_feat_mask is not None and prune_invalid:
            valid_feat_mask = np.asarray(valid_feat_mask, dtype=bool)
            if valid_feat_mask.shape[0] != N:
                print(f"Warning: valid_feat_mask length mismatch, skipping pruning")
            else:
                invalid_count = np.sum(~valid_feat_mask)
                valid_count = np.sum(valid_feat_mask)
                print(f"Pruning {invalid_count} invalid Gaussians (keeping {valid_count}/{N})")

                xyz = xyz[valid_feat_mask]
                features_dc = features_dc[valid_feat_mask]
                features_rest = features_rest[valid_feat_mask]
                scaling = scaling[valid_feat_mask]
                rotation = rotation[valid_feat_mask]
                opacity = opacity[valid_feat_mask]
                max_radii2D = max_radii2D[valid_feat_mask]
                xyz_gradient_accum = xyz_gradient_accum[valid_feat_mask]
                denom = denom[valid_feat_mask]

                N = valid_count
        elif valid_feat_mask is not None and not prune_invalid:
            print(f"Note: valid_feat_mask provided but prune_invalid=False")

        # Map language features back to original points
        if valid_feat_mask is not None and prune_invalid:
            features_orig = features[index]  # [N_orig, feat_dim]
            language_features = features_orig[valid_feat_mask].astype(np.float32)  # [N_pruned, feat_dim]
        else:
            features_orig = features[index]  # [N, feat_dim]
            if valid_feat_mask is not None:
                features_orig = features_orig.copy()
                features_orig[~valid_feat_mask] = 0.0
            language_features = features_orig.astype(np.float32)

        # Convert language features to tensor
        language_features_tensor = torch.from_numpy(language_features)

        # Return in capture_language_feature() format (13 elements)
        # NOTE: features_dc and features_rest are NOT modified - they store SH coefficients for color
        # language_features is stored SEPARATELY as the 8th element
        return (
            active_sh_degree,
            xyz,              # [N, 3] - Gaussian positions
            features_dc,      # [N, 3] - SH DC coefficients for COLOR (NOT language features!)
            features_rest,    # [N, 45] - SH rest coefficients for COLOR
            scaling,          # [N, 3] - Scaling
            rotation,         # [N, 4] - Rotation quaternions
            opacity,          # [N, 1] - Opacity
            language_features_tensor,  # [N, feat_dim] - LANGUAGE FEATURES (separate field!)
            max_radii2D,      # [N] - Max 2D radii
            xyz_gradient_accum,  # [N, 1] - XYZ gradient accumulator
            denom,            # [N, 1] - Denominator
            opt_dict,         # dict - Optimizer state dict
            spatial_lr_scale, # float - Spatial learning rate scale
        )
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(orig_ckpt)}")


class SceneSplatFeatureExtractor:
    """Feature extractor for SceneSplat models."""

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        chunk_size: int = 200000,
        visualize_layers: bool = False,
        layer_pattern: Optional[List[str]] = None,
        vis_output_dir: Optional[str] = None,
        model_type: str = "scenesplat",
    ):
        """
        Initialize feature extractor.

        Args:
            model_path: Path to model checkpoint
            config_path: Path to config file (auto-detected if None)
            device: Device to run inference on
            chunk_size: Chunk size for processing large scenes
            visualize_layers: Whether to visualize intermediate layer features
            layer_pattern: Pattern for layer names to visualize
            vis_output_dir: Output directory for visualizations
            model_type: Model type ("scenesplat" or "litept")
        """
        self.device = device
        self.chunk_size = chunk_size
        self.model_type = model_type
        self.visualize_layers = visualize_layers
        self.layer_pattern = layer_pattern
        self.vis_output_dir = vis_output_dir

        # Load config
        if config_path is None:
            if model_type == "litept":
                config_path = str(PROJECT_ROOT / "configs/custom/lang-pretrain-litept-ovs.py")
            else:
                config_path = str(PROJECT_ROOT / "configs/scannet/lang-pretrain-scannet-mcmc-wo-normal-contrastive.py")

        print(f"Loading config from: {config_path}")
        self.cfg = Config.fromfile(config_path)

        # Determine feature dimension based on model type
        if model_type == "litept":
            self.feat_dim = getattr(self.cfg, 'lang_feat_dim', 768)
        else:
            self.feat_dim = self.cfg.model.backbone.enc_channels[-1] * 3

        print(f"Model type: {model_type}")
        print(f"Feature dimension: {self.feat_dim}")

        # Build model
        self.model = build_model(self.cfg.model)
        self.model.to(device)
        self.model.eval()

        # Load checkpoint
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remove 'module.' prefix if present (from DistributedDataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict, strict=True)
        elif "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=True)
        else:
            self.model.load_state_dict(checkpoint, strict=True)

        print("Model loaded successfully")

        # Setup visualization if requested
        self.layer_outputs = {}
        if self.visualize_layers:
            self._setup_layer_hooks()

    def _setup_layer_hooks(self):
        """Setup forward hooks to capture intermediate layer outputs."""
        if self.layer_pattern is None:
            self.layer_pattern = ["backbone", "encoder", "decoder"]

        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, dict):
                    self.layer_outputs[name] = output.get("feat", output)
                else:
                    self.layer_outputs[name] = output
            return hook

        for name, module in self.model.named_modules():
            for pattern in self.layer_pattern:
                if pattern in name:
                    module.register_forward_hook(get_activation(name))
                    print(f"Registered hook for layer: {name}")

    @torch.no_grad()
    def extract(self, data_dict: Dict) -> np.ndarray:
        """
        Extract features from input data.

        Args:
            data_dict: Input data dictionary with keys:
                - coord: Point coordinates [N, 3]
                - feat: Input features [N, C]
                - batch: Batch indices [1] (required for sparsify)
                - grid_coord: Grid coordinates (optional)

        Returns:
            Extracted features [N, feat_dim]
        """
        coord = data_dict["coord"].to(self.device)
        feat = data_dict["feat"].to(self.device)

        N = coord.shape[0]
        # Create proper batch indices: [0, 0, 0, ..., 0] for N points (single scene)
        batch = torch.zeros(N, dtype=torch.long, device=self.device)

        features_list = []

        # Process in chunks
        for i in range(0, N, self.chunk_size):
            end_idx = min(i + self.chunk_size, N)
            coord_chunk = coord[i:end_idx]
            feat_chunk = feat[i:end_idx]
            batch_chunk = batch[i:end_idx]

            input_dict = {"coord": coord_chunk, "feat": feat_chunk, "batch": batch_chunk}

            if "grid_coord" in data_dict:
                grid_coord = data_dict["grid_coord"].to(self.device)
                input_dict["grid_coord"] = grid_coord[i:end_idx]

            # Forward pass
            try:
                output = self.model(input_dict)
            except AssertionError as e:
                if "serialization" in str(e).lower() or "depth" in str(e).lower() or "bit_length" in str(e).lower():
                    raise RuntimeError(
                        f"Scene is too large for PointOctree serialization (depth > 16). "
                        f"Solution: Increase --grid_size parameter. "
                        f"Try --grid_size 0.04 or --grid_size 0.05 for very large scenes."
                    ) from e
                raise

            # Extract features
            if isinstance(output, dict):
                chunk_features = output["point_feat"]["feat"]
            else:
                chunk_features = output

            features_list.append(chunk_features.cpu().numpy())

        # Concatenate all chunks
        features = np.concatenate(features_list, axis=0)
        return features.astype(np.float32)


class BatchPredictor:
    """Batch prediction for SceneSplat."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        chunk_size: int = 200000,
        grid_size: float = 0.02,
        save_npy: bool = False,
        npy_output_root: Optional[str] = None,
        scene_id: Optional[str] = None,
        model_type: str = "scenesplat",
        preprocessed: bool = False,
        recursive: bool = False,
        original_checkpoint: Optional[str] = None,
        iterations: int = 30000,
        prune_invalid: bool = True,
        # Multi-crop parameters
        use_multi_crop: bool = False,
        crop_radius: float = 5.0,
        crop_overlap: float = 0.2,
        max_crops: int = 50,
        skip_gridsample: bool = False,  # NEW: Skip GridSample, use full point cloud
    ):
        """
        Initialize batch predictor.

        Args:
            input_path: Path to input .ply file or directory
            output_path: Path to output directory
            model_path: Path to model checkpoint
            config_path: Path to config file (auto-detected if None)
            device: Device to run inference on
            chunk_size: Chunk size for processing
            grid_size: Grid size for voxelization
            save_npy: Whether to save .npy files
            npy_output_root: Root directory for .npy files
            scene_id: Scene ID for .npy saving
            model_type: Model type ("scenesplat" or "litept")
            preprocessed: Whether input is preprocessed .npy folders
            recursive: Whether to recursively process directories
            original_checkpoint: Base directory for original checkpoints
            iterations: Checkpoint iteration number
            prune_invalid: Whether to prune invalid Gaussians
            use_multi_crop: Whether to use multi-crop processing for full scene coverage
            crop_radius: Radius of each spherical crop (in world units)
            crop_overlap: Overlap ratio between adjacent crops (0-1)
            max_crops: Maximum number of crops to generate
            skip_gridsample: Whether to skip GridSample and use full point cloud (NEW)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.chunk_size = chunk_size
        self.grid_size = grid_size
        self.save_npy = save_npy
        self.npy_output_root = Path(npy_output_root) if npy_output_root else None
        self.scene_id = scene_id
        self.model_type = model_type
        self.preprocessed = preprocessed
        self.recursive = recursive
        self.original_checkpoint = original_checkpoint
        self.iterations = iterations
        self.prune_invalid = prune_invalid
        self.use_multi_crop = use_multi_crop
        self.crop_radius = crop_radius
        self.crop_overlap = crop_overlap
        self.max_crops = max_crops
        self.skip_gridsample = skip_gridsample

        # Initialize feature extractor
        self.extractor = SceneSplatFeatureExtractor(
            model_path=model_path,
            config_path=config_path,
            device=device,
            chunk_size=chunk_size,
            model_type=model_type,
        )

        # Build test transform pipeline (three-stage: initial, voxelize, post)
        self.initial_transform, self.voxelize_transform, self.post_transform = build_test_transform_pipeline(
            grid_size=grid_size,
            model_type=model_type,
            use_multi_crop=use_multi_crop,
            crop_radius=crop_radius,
            crop_overlap=crop_overlap,
            max_crops=max_crops,
            skip_gridsample=skip_gridsample,  # NEW: Pass skip_gridsample parameter
        )

        # Collect data paths
        self.data_paths = self._collect_data_paths()

        print(f"\nBatch prediction initialized:")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Model: {model_path}")
        print(f"  Model type: {model_type}")
        print(f"  Scenes to process: {len(self.data_paths)}")
        print(f"  Preprocessed mode: {preprocessed}")
        print(f"  Recursive mode: {recursive}")
        print(f"  Prune invalid Gaussians: {prune_invalid}")

    def _collect_data_paths(self) -> List[Path]:
        """Collect all data paths to process."""
        data_paths = []

        if self.preprocessed:
            # Preprocessed .npy folders
            if self.recursive:
                # Recursive mode: find all subdirectories containing .npy files
                for root, dirs, files in os.walk(self.input_path):
                    root_path = Path(root)
                    npy_files = list(root_path.glob("*.npy"))
                    if npy_files:
                        data_paths.append(root_path)
            else:
                # Non-recursive: expect direct subdirectories to be scene folders
                for item in self.input_path.iterdir():
                    if item.is_dir():
                        # Check if it contains .npy files
                        if list(item.glob("*.npy")):
                            data_paths.append(item)
        else:
            # .ply files
            if self.input_path.is_file():
                data_paths = [self.input_path]
            else:
                data_paths = list(self.input_path.glob("*.ply"))

        return sorted(data_paths)

    def _find_original_checkpoint(self, scene_name: str) -> Optional[str]:
        """Find original checkpoint for preprocessed scene."""
        if self.original_checkpoint is None:
            return None

        base_dir = Path(self.original_checkpoint)

        # Try different patterns
        patterns = [
            base_dir / scene_name / "ckpts" / f"chkpnt{self.iterations}.pth",
            base_dir / scene_name / f"chkpnt{self.iterations}.pth",
            base_dir / f"chkpnt{self.iterations}.pth",
        ]

        for pattern in patterns:
            if pattern.exists():
                print(f"Found original checkpoint: {pattern}")
                return str(pattern)

        print(f"Warning: Could not find original checkpoint for scene {scene_name}")
        return None

    def _process_multi_crop(
        self,
        transformed_data: dict,
        scene_start_time: float,
        transform_time: float,
        valid_feat_mask: Optional[np.ndarray],
        scene_name: str,
        data_path: Path,
        gridsample_inverse: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[str], Optional[np.ndarray]]:
        """
        Process multiple crops for full scene coverage.

        Args:
            transformed_data: Data from initial_transform (contains crops and crop_indices)
            scene_start_time: Start time for timing
            transform_time: Time taken for initial transform
            valid_feat_mask: Valid feature mask
            scene_name: Scene name
            data_path: Path to scene data
            gridsample_inverse: GridSample inverse mapping (original_loaded -> sampled)

        Returns:
            (features, inverse, original_checkpoint_path, valid_feat_mask)
        """
        import time

        crops = transformed_data["crops"]
        crop_indices_list = transformed_data["crop_indices"]
        n_original_points = transformed_data["n_original_points"]

        # Get filter metadata (if FilterCoordOutliers was used)
        n_original_before_filter = transformed_data.get("_n_original_points_before_filter", n_original_points)
        filtered_out_indices = transformed_data.get("_filtered_out_indices", None)
        inverse_filter_map = transformed_data.get("_inverse_filter_map", None)

        if filtered_out_indices is not None and len(filtered_out_indices) > 0:
            print(f"  Filtered outliers: {len(filtered_out_indices):,} points (excluded from output)")
            print(f"  Points before filter: {n_original_before_filter:,}, after filter: {n_original_points:,}")

        print(f"  Processing {len(crops)} crops for full scene coverage")
        print(f"  Working with {n_original_points:,} points (after filtering)")

        # Get feature dimension from first crop
        feat_dim = self.extractor.feat_dim

        # Accumulate features and counts for each original point
        accumulated_features = np.zeros((n_original_points, feat_dim), dtype=np.float32)
        point_counts = np.zeros(n_original_points, dtype=np.int32)

        # Build per-crop transform
        # Note: RecomputeGridCoord is already applied by GridCrop, no need to repeat
        from pointcept.datasets.transform import Compose
        crop_transform = Compose([
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord"),
                feat_keys=("color", "opacity", "quat", "scale"),
                offset_keys_dict=dict(batch="coord"),
            ),
        ])

        # Process each crop
        total_feature_time = 0
        for crop_idx, (crop, crop_indices) in enumerate(zip(crops, crop_indices_list)):
            crop_start = time.time()

            # Apply per-crop transform
            crop = crop_transform(crop)

            # Extract features from this crop
            crop_features = self.extractor.extract(crop)
            crop_time = time.time() - crop_start
            total_feature_time += crop_time

            # Accumulate features for points in this crop
            accumulated_features[crop_indices] += crop_features
            point_counts[crop_indices] += 1

            if (crop_idx + 1) % 10 == 0:
                print(f"    Processed {crop_idx + 1}/{len(crops)} crops")

        # Average features for points covered by multiple crops
        valid_mask = point_counts > 0
        accumulated_features[valid_mask] /= point_counts[valid_mask][:, np.newaxis]

        # Create output for covered points only
        covered_indices = np.where(valid_mask)[0]
        features = accumulated_features[covered_indices]

        # Build inverse mapping chain: original_loaded -> gridsample -> filtered -> feature
        # Step 1: filtered_point -> covered_feature_index
        inverse_filtered = np.zeros(n_original_points, dtype=np.int64)
        inverse_filtered[covered_indices] = np.arange(len(covered_indices))
        # For uncovered filtered points, map to 0 (will have zero features)
        inverse_filtered[~valid_mask] = 0

        # Step 2: If GridSample inverse exists, build complete mapping: original_loaded -> feature
        if gridsample_inverse is not None:
            # gridsample_inverse maps: original_loaded (1,000,000) -> gridsample_sampled (385,422)
            # We need to map: original_loaded -> feature
            # But we have: original -> gridsample (gridsample_inverse)
            #              gridsample -> filtered (inverse_filter_map)
            #              filtered -> feature (inverse_filtered)

            n_original_loaded = len(gridsample_inverse)

            if inverse_filter_map is not None:
                # Full chain: original -> gridsample -> filtered -> feature
                # Build gridsample -> filtered mapping
                # n_original_before_filter is the GridSample output size (before filtering)
                n_gridsample = n_original_before_filter
                gridsample_to_filtered = np.zeros(n_gridsample, dtype=np.int64)
                # inverse_filter_map contains indices of kept points in the pre-filter data
                # We set these indices to map to the new filtered indices
                gridsample_to_filtered[inverse_filter_map] = np.arange(n_original_points)

                # Vectorized chain: original -> gridsample -> filtered -> feature
                inverse = np.zeros(n_original_loaded, dtype=np.int64)

                # Step 1: original -> gridsample (already have gridsample_inverse)
                # Step 2: gridsample -> filtered
                gs_to_filt_valid = (gridsample_inverse >= 0) & (gridsample_inverse < n_gridsample)
                filt_indices = np.full(n_original_loaded, -1, dtype=np.int64)
                filt_indices[gs_to_filt_valid] = gridsample_to_filtered[gridsample_inverse[gs_to_filt_valid]]

                # Step 3: filtered -> feature
                filt_to_feat_valid = (filt_indices >= 0) & (filt_indices < n_original_points)
                inverse[filt_to_feat_valid] = inverse_filtered[filt_indices[filt_to_feat_valid]]

                print(f"  Inverse mapping chain: original_loaded({n_original_loaded}) -> gridsample({n_gridsample}) -> filtered({n_original_points}) -> feature({len(features)})")
            else:
                # Simpler case: original -> gridsample -> feature
                n_gridsample = n_original_points
                n_original_loaded = len(gridsample_inverse)
                inverse = np.zeros(n_original_loaded, dtype=np.int64)

                # Vectorized: original -> gridsample -> feature
                gs_valid = (gridsample_inverse >= 0) & (gridsample_inverse < n_gridsample)
                inverse[gs_valid] = inverse_filtered[gridsample_inverse[gs_valid]]

                print(f"  Inverse mapping: original_loaded({n_original_loaded}) -> gridsample({n_gridsample}) -> feature({len(features)})")
        else:
            # No GridSample inverse, use filtered inverse directly
            inverse = inverse_filtered
            print(f"  Inverse mapping: filtered({n_original_points}) -> feature({len(features)})")

        scene_total_time = time.time() - scene_start_time
        coverage = 100 * len(covered_indices) / n_original_points
        print(f"  Extracted features: {features.shape} from {len(crops)} crops")
        print(f"  Coverage: {coverage:.1f}% ({len(covered_indices)}/{n_original_points} points)")
        print(f"  Timing: transform={transform_time:.3f}s, feature={total_feature_time:.3f}s, total={scene_total_time:.3f}s")

        # Find original checkpoint
        original_checkpoint = self._find_original_checkpoint(scene_name)

        # Note: Don't update valid_feat_mask here since inverse now correctly maps
        # from original_loaded points to features
        # The save function will handle valid_feat_mask correctly

        return features, inverse, original_checkpoint, valid_feat_mask

    def _process_ply(self, ply_path: Path) -> Dict:
        """Process a single .ply file."""
        print(f"\nProcessing: {ply_path}")

        # Load .ply file using plyfile or similar
        # This is a placeholder - actual implementation depends on ply format
        raise NotImplementedError("Processing .ply files directly is not yet implemented")

    def _process_preprocessed(self, data_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[str], Optional[np.ndarray]]:
        """
        Process preprocessed .npy folder.

        Returns:
            (features, inverse, original_checkpoint_path, valid_feat_mask)
        """
        scene_name = data_path.name
        print(f"\nProcessing: {scene_name}")

        # Load all required .npy files
        coord = np.load(data_path / "coord.npy")
        color = np.load(data_path / "color.npy")
        opacity = np.load(data_path / "opacity.npy")
        quat = np.load(data_path / "quat.npy")
        scale = np.load(data_path / "scale.npy")

        # Load valid_feat_mask.npy if available
        valid_feat_mask_path = data_path / "valid_feat_mask.npy"
        if valid_feat_mask_path.exists():
            valid_feat_mask = np.load(valid_feat_mask_path)
            valid_count = np.sum(valid_feat_mask)
            total_count = len(valid_feat_mask)
            print(f"Found valid_feat_mask: {valid_count}/{total_count} points have valid features")
            print(f"Using all {total_count} Gaussians for inference (real mask applied during post-processing)")
        else:
            valid_feat_mask = None

        print(f"  Loaded {coord.shape[0]} points")

        # Build input data dictionary - follow SceneSplat pattern with separate keys
        data_dict = {
            "coord": coord,
            "color": color,
            "opacity": opacity,
            "quat": quat,
            "scale": scale,
        }

        # Record inference timing
        import time
        scene_start_time = time.time()

        # Apply SceneSplat test transform pipeline:
        # 1. Initial transform (preprocessing + GridSample with mode="train")
        transform_start = time.time()
        transformed_data = self.initial_transform(data_dict)
        transform_time = time.time() - transform_start

        # Check if multi-crop mode (GridCrop was used)
        if self.voxelize_transform is None:
            # Multi-crop mode
            # Get GridSample inverse mapping (from original loaded points to sampled points)
            gridsample_inverse = transformed_data.get("inverse", None)
            return self._process_multi_crop(transformed_data, scene_start_time, transform_time, valid_feat_mask, scene_name, data_path, gridsample_inverse)

        # Single crop mode (original logic)
        # Save inverse mapping before voxelization overwrites it
        # GridSample(mode="train") creates "inverse" key, not "index"
        inverse = transformed_data["inverse"]

        # 2. Voxelization (GridSample with mode="test" - creates fragments)
        voxelize_start = time.time()
        fragment_list = self.voxelize_transform(transformed_data)
        voxelize_time = time.time() - voxelize_start
        print(f"  Created {len(fragment_list)} test fragments (transform: {transform_time:.3f}s, voxelize: {voxelize_time:.3f}s)")

        # 3. Process each fragment with post_transform and extract features
        all_features = []
        all_indices = []  # Track fragment indices for aggregation
        total_feature_time = 0
        for i, fragment in enumerate(fragment_list):
            # Apply post_transform to each fragment
            fragment = self.post_transform(fragment)

            # Extract features from this fragment
            feature_start = time.time()
            fragment_features = self.extractor.extract(fragment)
            feature_time = time.time() - feature_start
            total_feature_time += feature_time
            all_features.append(fragment_features)

            # Get the index mapping from fragment to original sampled points
            if "index" in fragment:
                all_indices.append(fragment["index"])
            else:
                # Fallback: use sequential indices
                all_indices.append(np.arange(len(fragment_features)))

        # Aggregate fragment features using inverse mapping
        # Strategy: Average features from all fragments for each sampled point
        # Each fragment has indices mapping back to the original sampled points
        if len(all_indices) > 0 and all_indices[0] is not None:
            # Get the number of original sampled points
            max_idx = max(max(idx) if len(idx) > 0 else 0 for idx in all_indices) + 1
            feat_dim = all_features[0].shape[1]

            # Accumulate features and counts
            accumulated_features = np.zeros((max_idx, feat_dim), dtype=np.float32)
            feature_counts = np.zeros(max_idx, dtype=np.int32)

            # Aggregate features from all fragments
            for fragment_feat, fragment_idx in zip(all_features, all_indices):
                for i, idx in enumerate(fragment_idx):
                    if idx < max_idx:
                        accumulated_features[idx] += fragment_feat[i]
                        feature_counts[idx] += 1

            # Average the features (avoid division by zero)
            valid_mask = feature_counts > 0
            accumulated_features[valid_mask] /= feature_counts[valid_mask][:, np.newaxis]

            # Remove any points that weren't covered by any fragment
            features = accumulated_features[valid_mask]
            # Create a new inverse mapping for valid points
            valid_indices = np.where(valid_mask)[0]
            # Map from aggregated features back to original points
            # The inverse maps original points -> sampled points (before fragment aggregation)
            # We need to filter inverse to only include valid sampled points
            new_inverse = np.zeros_like(inverse)
            inverse_mapping = np.full(max_idx, -1)  # Maps old sampled idx -> new aggregated idx
            inverse_mapping[valid_indices] = np.arange(len(valid_indices))
            # Update inverse: for each original point, find its new aggregated index
            for i, inv_val in enumerate(inverse):
                if inv_val < max_idx and inverse_mapping[inv_val] >= 0:
                    new_inverse[i] = inverse_mapping[inv_val]
                else:
                    new_inverse[i] = 0  # Fallback
            inverse = new_inverse
        else:
            # Fallback: use first fragment only
            features = all_features[0]

        scene_total_time = time.time() - scene_start_time
        print(f"  Extracted features: {features.shape} from {len(fragment_list)} fragments (feature extraction: {total_feature_time:.3f}s, total: {scene_total_time:.3f}s)")

        # Find original checkpoint
        original_checkpoint = self._find_original_checkpoint(scene_name)

        return features, inverse, original_checkpoint, valid_feat_mask

    def _save_output(
        self,
        scene_name: str,
        features: np.ndarray,
        inverse: np.ndarray,
        original_checkpoint: Optional[str],
        valid_feat_mask: Optional[np.ndarray],
    ):
        """Save output features and checkpoint."""
        scene_output_dir = self.output_path / scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        # Map fragment features back to all original points using inverse mapping
        # features: [M, feat_dim] (M = sampled points)
        # inverse: [N_orig] maps each original point to a sampled point index
        # features_orig: [N_orig, feat_dim] - Full array with zeros for invalid points
        features_orig = features[inverse]  # [N_orig, feat_dim]

        # Save features as .npy (expanded to all original Gaussians)
        feat_path = scene_output_dir / "language_features.npy"
        np.save(feat_path, features_orig)
        print(f"  Saved features to: {feat_path}")

        # Store original features for later use
        original_features = features_orig

        # Save checkpoint if original checkpoint is available
        if original_checkpoint is not None:
            # Save in GaussianModel format compatible with feature_map_renderer.py
            # This creates a checkpoint with capture_language_feature() format (13 elements)
            gaussian_checkpoint_path = scene_output_dir / f"checkpoint_with_features.pth"
            try:
                checkpoint_state = create_occamlgs_checkpoint(
                    features=features,
                    index=inverse,
                    original_checkpoint_path=original_checkpoint,
                    valid_feat_mask=valid_feat_mask,
                    prune_invalid=self.prune_invalid,
                )

                # Save in format: ((13-element tuple), iteration)
                # This matches gaussian_feature_extractor.py line 435 format
                torch.save((checkpoint_state, 0), gaussian_checkpoint_path)
                print(f"  Saved GaussianModel checkpoint to: {gaussian_checkpoint_path}")
                print(f"    Format: capture_language_feature() (13-element tuple)")

                if valid_feat_mask is not None and self.prune_invalid:
                    print(f"    Checkpoint: Invalid Gaussians pruned")
                elif valid_feat_mask is not None and not self.prune_invalid:
                    print(f"    Checkpoint: Invalid Gaussians kept with zero features")
            except Exception as e:
                print(f"  Warning: Failed to create GaussianModel checkpoint: {e}")
                import traceback
                traceback.print_exc()

        # Save .npy file for benchmark evaluation if requested
        if self.save_npy and self.npy_output_root is not None:
            npy_output_dir = self.npy_output_root / scene_name
            npy_output_dir.mkdir(parents=True, exist_ok=True)

            # Determine scene ID
            output_scene_id = self.scene_id if self.scene_id else scene_name

            # Save .npy file
            npy_path = npy_output_dir / f"{output_scene_id}.npy"
            np.save(npy_path, original_features)
            print(f"  Saved .npy for benchmark to: {npy_path}")

    def run(self):
        """Run batch prediction on all collected data paths."""
        print("\n" + "=" * 60)
        print("Starting batch prediction")
        print("=" * 60)

        for data_path in tqdm(self.data_paths, desc="Processing scenes"):
            try:
                if self.preprocessed:
                    features, inverse, original_checkpoint, valid_feat_mask = self._process_preprocessed(data_path)
                    scene_name = data_path.name
                else:
                    raise NotImplementedError("Processing .ply files directly is not yet implemented")

                self._save_output(
                    scene_name=scene_name,
                    features=features,
                    inverse=inverse,
                    original_checkpoint=original_checkpoint,
                    valid_feat_mask=valid_feat_mask,
                )

            except Exception as e:
                print(f"\nError processing {data_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "=" * 60)
        print("Batch prediction completed")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Batch prediction script for SceneSplat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/output arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input .ply file or directory containing .ply/.npy files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory",
    )

    # Model arguments
    parser.add_argument(
        "--weight",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (auto-detected if not specified)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["scenesplat", "litept"],
        default="scenesplat",
        help="Model type (default: scenesplat)",
    )

    # Processing arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200000,
        help="Chunk size for processing large scenes (default: 200000)",
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.02,
        help="Grid size for voxelization (default: 0.02)",
    )

    # Preprocessed data arguments
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Input is preprocessed .npy folders",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process directories (for preprocessed mode)",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--original_checkpoint",
        type=str,
        default=None,
        help="Base directory for original checkpoints (for preprocessed mode)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Checkpoint iteration number (default: 30000)",
    )

    # .npy saving arguments
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Save features as .npy files for benchmark evaluation",
    )
    parser.add_argument(
        "--npy_output_root",
        type=str,
        default=None,
        help="Root directory for .npy files",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Scene ID for .npy saving",
    )

    # Pruning arguments
    parser.add_argument(
        "--prune_invalid",
        action="store_true",
        default=True,
        dest="prune_invalid",
        help="Prune Gaussians with invalid language features from checkpoint (default: True). "
             "When enabled, Gaussians where valid_feat_mask=False will be completely removed.",
    )

    # Multi-crop arguments
    parser.add_argument(
        "--use_multi_crop",
        action="store_true",
        help="Use multi-crop processing for full scene coverage (slower but more complete)",
    )
    parser.add_argument(
        "--crop_radius",
        type=float,
        default=5.0,
        help="Radius of each spherical crop in world units (default: 5.0)",
    )
    parser.add_argument(
        "--crop_overlap",
        type=float,
        default=0.2,
        help="Overlap ratio between adjacent crops, 0-1 (default: 0.2)",
    )
    parser.add_argument(
        "--max_crops",
        type=int,
        default=50,
        help="Maximum number of crops to generate (default: 50)",
    )
    parser.add_argument(
        "--skip_gridsample",
        action="store_true",
        help="Skip GridSample and use full point cloud directly (higher memory, simpler inverse mapping)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SceneSplat Batch Prediction")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.weight}")
    print(f"Config: {args.config or 'auto-detected'}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {args.device}")
    print(f"Preprocessed: {args.preprocessed}")
    print(f"Recursive: {args.recursive}")
    print(f"Original checkpoint: {args.original_checkpoint or 'N/A'}")
    print(f"Iterations: {args.iterations}")
    print(f"Save .npy: {args.save_npy}")
    print(f"Prune invalid Gaussians: {args.prune_invalid}")
    if args.prune_invalid:
        print(f"  Note: Invalid Gaussians will be removed from checkpoint")
    else:
        print(f"  Note: Invalid Gaussians will be kept with zero features")
    print(f"Multi-crop mode: {args.use_multi_crop}")
    if args.use_multi_crop:
        print(f"  Crop radius: {args.crop_radius}")
        print(f"  Crop overlap: {args.crop_overlap}")
        print(f"  Max crops: {args.max_crops}")
    print("=" * 60)

    # Initialize predictor
    predictor = BatchPredictor(
        input_path=args.input,
        output_path=args.output,
        model_path=args.weight,
        config_path=args.config,
        device=args.device,
        chunk_size=args.chunk_size,
        grid_size=args.grid_size,
        save_npy=args.save_npy,
        npy_output_root=args.npy_output_root,
        scene_id=args.scene_id,
        model_type=args.model_type,
        preprocessed=args.preprocessed,
        recursive=args.recursive,
        original_checkpoint=args.original_checkpoint,
        iterations=args.iterations,
        prune_invalid=args.prune_invalid,
        use_multi_crop=args.use_multi_crop,
        crop_radius=args.crop_radius,
        crop_overlap=args.crop_overlap,
        max_crops=args.max_crops,
        skip_gridsample=args.skip_gridsample,  # NEW: Pass skip_gridsample
    )

    # Run prediction
    predictor.run()

# python tools/batch_predict.py --input /new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/val --output ./output_features --weight exp/lite-16-gridsvd/model/model_last.pth --config configs/custom/lang-pretrain-litept-ovs-gridsvd.py --model_type litept --chunk_size 1000000 --grid_size 0.01 --device cuda:1 --iterations 30000 --preprocessed --recursive --original_checkpoint /new_data/cyf/projects/SceneSplat/gaussian_results/lerf_ovs
#
# Single crop mode (center crop only, faster):
# python tools/batch_predict.py --input /new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/val \
#     --output ./output_features_single \
#     --weight exp/lite-16-gridsvd/model/model_last.pth \
#     --config configs/custom/lang-pretrain-litept-ovs-gridsvd.py \
#     --model_type litept --chunk_size 1000000 --grid_size 0.01 --device cuda:1 \
#     --iterations 30000 --preprocessed --recursive \
#     --original_checkpoint /new_data/cyf/projects/SceneSplat/gaussian_results/lerf_ovs
#
# Multi-crop mode (full scene coverage, slower):
# python tools/batch_predict.py --input /new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/val \
#     --output ./output_features_full \
#     --weight exp/lite-16-gridsvd/model/model_last.pth \
#     --config configs/custom/lang-pretrain-litept-ovs-gridsvd.py \
#     --model_type litept --chunk_size 1000000 --grid_size 0.01 --device cuda:1 \
#     --iterations 30000 --preprocessed --recursive \
#     --original_checkpoint /new_data/cyf/projects/SceneSplat/gaussian_results/lerf_ovs \
#     --use_multi_crop --crop_radius 5.0 --crop_overlap 0.2 --max_crops 50
if __name__ == "__main__":
    main()
