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
):
    """
    Build test transform pipeline following SceneSplat pattern.

    Returns:
        tuple: (initial_transform, voxelize_transform, post_transform)
        - initial_transform: Compose applied before voxelization
        - voxelize_transform: GridSample that creates fragments
        - post_transform: Compose applied to each fragment
    """
    # GridSample keys - follow SceneSplat pattern exactly
    if model_type == "litept":
        grid_sample_keys = ("coord", "color", "opacity", "quat", "scale")
        feat_keys = ("color", "opacity", "quat", "scale", "coord")  # 14 channels
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
    initial_transforms.append(dict(type="NormalizeCoord"))
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

    def _process_ply(self, ply_path: Path) -> Dict:
        """Process a single .ply file."""
        print(f"\nProcessing: {ply_path}")

        # Load .ply file using plyfile or similar
        # This is a placeholder - actual implementation depends on ply format
        raise NotImplementedError("Processing .ply files directly is not yet implemented")

    def _process_preprocessed(self, data_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """
        Process preprocessed .npy folder.

        Returns:
            (features, inverse, original_checkpoint_path)
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
        # This will be used for post-processing; inference uses ALL Gaussians
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
        data_dict = self.initial_transform(data_dict)
        transform_time = time.time() - transform_start

        # Save inverse mapping before voxelization overwrites it
        # GridSample(mode="train") creates "inverse" key, not "index"
        inverse = data_dict["inverse"]

        # 2. Voxelization (GridSample with mode="test" - creates fragments)
        voxelize_start = time.time()
        fragment_list = self.voxelize_transform(data_dict)
        voxelize_time = time.time() - voxelize_start
        print(f"  Created {len(fragment_list)} test fragments (transform: {transform_time:.3f}s, voxelize: {voxelize_time:.3f}s)")

        # 3. Process each fragment with post_transform and extract features
        all_features = []
        total_feature_time = 0
        for fragment in fragment_list:
            # Apply post_transform to each fragment
            fragment = self.post_transform(fragment)

            # Extract features from this fragment
            feature_start = time.time()
            fragment_features = self.extractor.extract(fragment)
            feature_time = time.time() - feature_start
            total_feature_time += feature_time
            all_features.append(fragment_features)

        # Aggregate fragment features using inverse mapping
        # For now, use the first fragment's features (can be improved with averaging)
        features = all_features[0]

        scene_total_time = time.time() - scene_start_time
        print(f"  Extracted features: {features.shape} (feature extraction: {total_feature_time:.3f}s, total: {scene_total_time:.3f}s)")

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

        # Save features as .npy
        feat_path = scene_output_dir / "language_features.npy"
        np.save(feat_path, features)
        print(f"  Saved features to: {feat_path}")

        # Map fragment features back to all original points using inverse mapping
        # features: [M, feat_dim] (M = sampled points)
        # inverse: [N_orig] maps each original point to a sampled point index
        original_features = features[inverse]  # [N_orig, feat_dim]

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
    )

    # Run prediction
    predictor.run()

# python tools/batch_predict.py --input /new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/val --output ./output_features --weight exp/default/model-lite-768/model_last.pth --config configs/custom/lang-pretrain-litept-ovs.py --model_type litept --chunk_size 1000000 --grid_size 0.01 --device cuda:1 --iterations 30000 --preprocessed --recursive --original_checkpoint /new_data/cyf/projects/SceneSplat/gaussian_results/lerf_ovs
if __name__ == "__main__":
    main()
