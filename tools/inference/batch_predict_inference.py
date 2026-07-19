#!/usr/bin/env python3
"""
Batch Prediction Script for SceneSplat using LangPretrainerInference

This script uses the LangPretrainerInference class for batch processing,
with support for Gaussian checkpoint loading/saving.

Features:
- Load original SceneSplat PT-v3m1 model (768-dim output)
- Load LitePT model (smaller, faster)
- Output 768-dimensional language features
- Save Gaussian checkpoint with embedded language features
- Optional coordinate outlier filtering with customizable percentiles

Usage:
    # Basic inference with LitePT model (no filtering)
    python tools/batch_predict_inference.py \\
        --config configs/inference/lang-pretrain-litept-3dgs.py \\
        --checkpoint /path/to/model.pth \\
        --input-root /path/to/npy/folder \\
        --output-dir /path/to/output

    # Use original SceneSplat PT-v3m1 model (768-dim features)
    python tools/batch_predict_inference.py \\
        --use-original-model \\
        --checkpoint /path/to/model.pth \\
        --input-root /path/to/npy/folder \\
        --output-dir /path/to/output

    # With Gaussian checkpoint saving (768-dim language features)
    python tools/batch_predict_inference.py \\
        --use-original-model \\
        --checkpoint /path/to/model.pth \\
        --input-root /path/to/npy/folder \\
        --output-dir /path/to/output \\
        --original-checkpoint /path/to/gaussian/checkpoints \\
        --iterations 30000 \\
        --save-checkpoint

    # Process a single scene
    python tools/batch_predict_inference.py \\
        --use-original-model \\
        --checkpoint /path/to/model.pth \\
        --input-root /path/to/npy/folder \\
        --output-dir /path/to/output \\
        --scene scene_name

    # Specify custom config with feature dimension
    python tools/batch_predict_inference.py \\
        --config configs/custom/my_config.py \\
        --checkpoint /path/to/model.pth \\
        --input-root /path/to/npy/folder \\
        --output-dir /path/to/output \\
        --feature-dim 512

    # With coordinate outlier filtering (0.1%-99.9%)
    python tools/batch_predict_inference.py \\
        --config configs/inference/lang-pretrain-litept-3dgs.py \\
        --checkpoint /path/to/model.pth \\
        --input-root /path/to/npy/folder \\
        --output-dir /path/to/output \\
        --filter-percentile 0.1

    # With stricter outlier filtering (0.5%-99.5%)
    python tools/batch_predict_inference.py \\
        --config configs/inference/lang-pretrain-litept-3dgs.py \\
        --checkpoint /path/to/model.pth \\
        --input-root /path/to/npy/folder \\
        --output-dir /path/to/output \\
        --filter-percentile 0.5
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Import PROJECT_ROOT - handle both script and module execution
try:
    from .. import PROJECT_ROOT  # Relative import when run as module
except ImportError:
    # Fallback when run as script: add parent dir to sys.path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from pointcept.inference.lang_pretrainer import LangPretrainerInference
from pointcept.utils.config import Config


class GaussianCheckpointHandler:
    """Handler for loading and saving GaussianModel checkpoints."""

    @staticmethod
    def load_original_checkpoint(checkpoint_path: str) -> Dict:
        """Load original GaussianModel checkpoint."""
        print(f"Loading original checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Detect format
        if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
            state_dict, iteration = checkpoint
            print(f"Detected tuple format: (state_dict, iteration={iteration})")
            return {"state_dict": state_dict, "iteration": iteration}
        elif isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                print("Detected dict format with state_dict")
                return checkpoint
            elif "iteration" in checkpoint or "model" in checkpoint:
                print("Detected training checkpoint format")
                return checkpoint
            else:
                # GaussianModel format (13-element tuple)
                keys = list(checkpoint.keys())
                print(f"Detected GaussianModel format with {len(keys)} elements")
                return checkpoint
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    @staticmethod
    def create_checkpoint_with_features(
        features: np.ndarray,
        original_checkpoint: Dict,
        valid_feat_mask: Optional[np.ndarray] = None,
        prune_invalid: bool = False,
    ) -> Tuple:
        """
        Create GaussianModel checkpoint with language features.

        Args:
            features: Language features (should already be mapped to the target space)
            original_checkpoint: Original Gaussian checkpoint
            valid_feat_mask: Boolean mask indicating valid features (same size as features)
            prune_invalid: If True, prune invalid Gaussians; if False, fill invalid with zeros

        Returns a 13-element tuple compatible with capture_language_feature() format:
        (active_sh_degree, xyz, features_dc, features_rest,
         scaling, rotation, opacity, language_features,
         max_radii2D, xyz_gradient_accum, denom,
         opt_dict, spatial_lr_scale)
        """
        print("Creating GaussianModel checkpoint with features...")

        # Extract original state
        if isinstance(original_checkpoint, dict) and "state_dict" in original_checkpoint:
            state_dict = original_checkpoint["state_dict"]
        else:
            state_dict = original_checkpoint

        # Handle SceneSplat/OccamLGS tuple format checkpoint
        if isinstance(state_dict, tuple):
            print("Detected SceneSplat/OccamLGS tuple format checkpoint")
            # state_dict is a flat tuple: (N, means, sh0, shN, scales, quats, opacities, lang_feat, ...)
            print(f"  Checkpoint tuple length: {len(state_dict)}")

            # state_dict contains:
            # 0: N (number of Gaussians or flags)
            # 1: means [N, 3]
            # 2: sh0 [N, 1, 3] or [N, 3] - DC (spherical harmonics degree 0, i.e., color)
            # 3: shN [N, 15, 3] - higher degree spherical harmonics
            # 4: scales [N, 3]
            # 5: quats [N, 4] - rotation quaternions
            # 6: opacities [N]
            # 7: lang_feat [N, 768] - optional language features (to be replaced)
            xyz = state_dict[1].cpu()  # [N, 3]
            sh0 = state_dict[2].cpu()  # [N, 1, 3] or [N, 3]
            shN = state_dict[3].cpu()  # [N, 15, 3]
            scaling = state_dict[4].cpu()  # [N, 3]
            rotation = state_dict[5].cpu()  # [N, 4]
            opacity = state_dict[6].cpu()  # [N]

            # Extract colors from sh0
            if sh0.dim() == 3:
                features_dc = sh0.squeeze(1)  # [N, 3]
            else:
                features_dc = sh0  # [N, 3]

            # Reshape shN to [N, 45] for features_rest
            N = xyz.shape[0]
            features_rest = shN.reshape(N, -1)  # [N, 45]

            # Ensure opacity has correct shape
            if opacity.dim() == 1:
                opacity = opacity.unsqueeze(-1)  # [N, 1]

            # Create placeholders for GaussianModel state
            active_sh_degree = 3
            max_radii2D = torch.zeros(N, dtype=torch.int32, device=xyz.device)
            xyz_gradient_accum = torch.zeros(N, 1, dtype=torch.float32, device=xyz.device)
            denom = torch.zeros(N, 1, dtype=torch.float32, device=xyz.device)
            opt_dict = {}  # Empty optimizer state dict
            spatial_lr_scale = 1.0

            # Apply valid_feat_mask pruning or fill with zeros
            valid_feat_mask_torch = None
            if valid_feat_mask is not None and prune_invalid:
                valid_feat_mask_array = np.asarray(valid_feat_mask, dtype=bool)
                if valid_feat_mask_array.shape[0] != N:
                    print(f"Warning: valid_feat_mask length mismatch ({valid_feat_mask_array.shape[0]} vs {N}), skipping pruning")
                    print(f"  This may indicate the checkpoint was preprocessed differently than the language features.")
                    print(f"  Creating a full-valid mask to match xyz size...")
                    valid_feat_mask_array = np.ones(N, dtype=bool)
                else:
                    invalid_count = np.sum(~valid_feat_mask_array)
                    valid_count = np.sum(valid_feat_mask_array)
                    print(f"Pruning {invalid_count} invalid Gaussians (keeping {valid_count}/{N})")

                    valid_feat_mask_torch = torch.from_numpy(valid_feat_mask_array)

                    xyz = xyz[valid_feat_mask_torch]
                    features_dc = features_dc[valid_feat_mask_torch]
                    features_rest = features_rest[valid_feat_mask_torch]
                    scaling = scaling[valid_feat_mask_torch]
                    rotation = rotation[valid_feat_mask_torch]
                    opacity = opacity[valid_feat_mask_torch]
                    max_radii2D = max_radii2D[valid_feat_mask_torch]
                    xyz_gradient_accum = xyz_gradient_accum[valid_feat_mask_torch]
                    denom = denom[valid_feat_mask_torch]

                    # Also filter language features to match pruned xyz
                    features = features[valid_feat_mask_array]
                    N = valid_count
            elif valid_feat_mask is not None and not prune_invalid:
                print(f"Note: valid_feat_mask provided but prune_invalid=False (filling invalid with zeros)")
                if valid_feat_mask.shape[0] != N:
                    print(f"Warning: valid_feat_mask length mismatch ({valid_feat_mask.shape[0]} vs {N})")
                    print(f"  Assuming all {N} Gaussians are valid (mask mismatch indicates checkpoint was preprocessed differently)")
                    # Don't use the mismatched mask, keep all features as-is
                else:
                    # Fill invalid features with zeros to keep all Gaussians
                    bool_mask = valid_feat_mask.astype(bool)
                    features = features.copy()
                    features[~bool_mask] = 0.0
                    n_invalid = (~bool_mask).sum()
                    print(f"  Filled {n_invalid} invalid features with zeros")

            # Convert language features to tensor
            language_features_tensor = torch.from_numpy(features.astype(np.float32))

            # Return in capture_language_feature() format (13 elements)
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
        # Handle gsplat format checkpoint
        elif isinstance(state_dict, dict) and "splats" in state_dict:
            print("Detected gsplat format checkpoint")
            splats = state_dict["splats"]

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

            # Apply valid_feat_mask pruning or fill with zeros
            # Features should already be in the same space as xyz (after expansion in process_scene)
            valid_feat_mask_torch = None
            if valid_feat_mask is not None and prune_invalid:
                valid_feat_mask_array = np.asarray(valid_feat_mask, dtype=bool)
                if valid_feat_mask_array.shape[0] != N:
                    print(f"Warning: valid_feat_mask length mismatch ({valid_feat_mask_array.shape[0]} vs {N}), skipping pruning")
                    print(f"  This may indicate the checkpoint was preprocessed differently than the language features.")
                    print(f"  Creating a full-valid mask to match xyz size...")
                    valid_feat_mask_array = np.ones(N, dtype=bool)
                else:
                    invalid_count = np.sum(~valid_feat_mask_array)
                    valid_count = np.sum(valid_feat_mask_array)
                    print(f"Pruning {invalid_count} invalid Gaussians (keeping {valid_count}/{N})")

                    valid_feat_mask_torch = torch.from_numpy(valid_feat_mask_array)

                    xyz = xyz[valid_feat_mask_torch]
                    features_dc = features_dc[valid_feat_mask_torch]
                    features_rest = features_rest[valid_feat_mask_torch]
                    scaling = scaling[valid_feat_mask_torch]
                    rotation = rotation[valid_feat_mask_torch]
                    opacity = opacity[valid_feat_mask_torch]
                    max_radii2D = max_radii2D[valid_feat_mask_torch]
                    xyz_gradient_accum = xyz_gradient_accum[valid_feat_mask_torch]
                    denom = denom[valid_feat_mask_torch]

                    # Also filter language features to match pruned xyz
                    features = features[valid_feat_mask_array]
                    N = valid_count
            elif valid_feat_mask is not None and not prune_invalid:
                print(f"Note: valid_feat_mask provided but prune_invalid=False (filling invalid with zeros)")
                if valid_feat_mask.shape[0] != N:
                    print(f"Warning: valid_feat_mask length mismatch ({valid_feat_mask.shape[0]} vs {N})")
                    print(f"  Assuming all {N} Gaussians are valid (mask mismatch indicates checkpoint was preprocessed differently)")
                    # Don't use the mismatched mask, keep all features as-is
                else:
                    # Fill invalid features with zeros to keep all Gaussians
                    bool_mask = valid_feat_mask.astype(bool)
                    features = features.copy()
                    features[~bool_mask] = 0.0
                    n_invalid = (~bool_mask).sum()
                    print(f"  Filled {n_invalid} invalid features with zeros")

            # Convert language features to tensor
            language_features_tensor = torch.from_numpy(features.astype(np.float32))

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
            # Try to extract from dict format with standard keys
            ckpt_keys = ["means", "features_dc", "features_rest", "opacities", "scales", "rotations"]

            xyz = state_dict.get("means")
            features_dc = state_dict.get("features_dc")
            features_rest = state_dict.get("features_rest")
            opacity = state_dict.get("opacities")
            scaling = state_dict.get("scales")
            rotation = state_dict.get("rotations")

            if xyz is None:
                raise ValueError(f"Unsupported checkpoint format - missing 'means' key. Available keys: {list(state_dict.keys())}")

            N = xyz.shape[0]

            # Create placeholders for GaussianModel state
            active_sh_degree = 3
            max_radii2D = torch.zeros(N, dtype=torch.int32, device=xyz.device)
            xyz_gradient_accum = torch.zeros(N, 1, dtype=torch.float32, device=xyz.device)
            denom = torch.zeros(N, 1, dtype=torch.float32, device=xyz.device)
            opt_dict = {}
            spatial_lr_scale = 1.0

            # Handle opacity shape
            if opacity is not None and opacity.dim() == 1:
                opacity = opacity.unsqueeze(-1)

            # Handle pruning or fill with zeros
            # Features should already be in the same space as xyz (after expansion in process_scene)
            valid_feat_mask_torch = None
            if valid_feat_mask is not None and prune_invalid:
                valid_feat_mask_array = np.asarray(valid_feat_mask, dtype=bool)
                if valid_feat_mask_array.shape[0] != N:
                    print(f"Warning: valid_feat_mask length mismatch ({valid_feat_mask_array.shape[0]} vs {N}), skipping pruning")
                    print(f"  This may indicate the checkpoint was preprocessed differently than the language features.")
                    print(f"  Creating a full-valid mask to match xyz size...")
                    valid_feat_mask_array = np.ones(N, dtype=bool)
                else:
                    invalid_count = np.sum(~valid_feat_mask_array)
                    valid_count = np.sum(valid_feat_mask_array)
                    print(f"Pruning {invalid_count} invalid Gaussians (keeping {valid_count}/{N})")

                    valid_feat_mask_torch = torch.from_numpy(valid_feat_mask_array)

                    xyz = xyz[valid_feat_mask_torch]
                    if features_dc is not None:
                        features_dc = features_dc[valid_feat_mask_torch]
                    if features_rest is not None:
                        features_rest = features_rest[valid_feat_mask_torch]
                    if opacity is not None:
                        opacity = opacity[valid_feat_mask_torch]
                    if scaling is not None:
                        scaling = scaling[valid_feat_mask_torch]
                    if rotation is not None:
                        rotation = rotation[valid_feat_mask_torch]
                    max_radii2D = max_radii2D[valid_feat_mask_torch]
                    xyz_gradient_accum = xyz_gradient_accum[valid_feat_mask_torch]
                    denom = denom[valid_feat_mask_torch]

                    # Also filter language features to match pruned xyz
                    features = features[valid_feat_mask_array]
                    N = valid_count
            elif valid_feat_mask is not None and not prune_invalid:
                print(f"Note: valid_feat_mask provided but prune_invalid=False (filling invalid with zeros)")
                if valid_feat_mask.shape[0] != N:
                    print(f"Warning: valid_feat_mask length mismatch ({valid_feat_mask.shape[0]} vs {N})")
                    print(f"  Assuming all {N} Gaussians are valid (mask mismatch indicates checkpoint was preprocessed differently)")
                    # Don't use the mismatched mask, keep all features as-is
                else:
                    # Fill invalid features with zeros to keep all Gaussians
                    bool_mask = valid_feat_mask.astype(bool)
                    features = features.copy()
                    features[~bool_mask] = 0.0
                    n_invalid = (~bool_mask).sum()
                    print(f"  Filled {n_invalid} invalid features with zeros")

            # Convert language features to tensor
            language_features_tensor = torch.from_numpy(features.astype(np.float32))

            # Provide defaults for missing components
            if features_dc is None:
                print("Warning: features_dc missing, using zeros")
                features_dc = torch.zeros(N, 3, dtype=torch.float32, device=xyz.device)
            if features_rest is None:
                print("Warning: features_rest missing, using zeros")
                features_rest = torch.zeros(N, 45, dtype=torch.float32, device=xyz.device)
            if opacity is None:
                print("Warning: opacity missing, using ones")
                opacity = torch.ones(N, 1, dtype=torch.float32, device=xyz.device)
            if scaling is None:
                print("Warning: scaling missing, using ones")
                scaling = torch.ones(N, 3, dtype=torch.float32, device=xyz.device)
            if rotation is None:
                print("Warning: rotation missing, using identity quaternions")
                rotation = torch.zeros(N, 4, dtype=torch.float32, device=xyz.device)
                rotation[:, 0] = 1.0  # w=1 for identity quaternion

            return (
                active_sh_degree,
                xyz,
                features_dc,
                features_rest,
                scaling,
                rotation,
                opacity,
                language_features_tensor,
                max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                spatial_lr_scale,
            )

    @staticmethod
    def save_checkpoint(checkpoint: Tuple, save_path: str):
        """Save checkpoint in GaussianModel format.

        The checkpoint is a 13-element tuple from create_checkpoint_with_features.
        We wrap it with iteration number as ((13-element tuple), iteration).
        """
        # Save in format: ((13-element tuple), iteration)
        # This matches gaussian_feature_extractor.py and feature_map_renderer.py format
        torch.save((checkpoint, 0), save_path)
        print(f"Saved GaussianModel checkpoint to: {save_path}")
        print(f"  Format: ((13-element tuple), iteration=0)")


class BatchPredictorWithInference:
    """Batch predictor using LangPretrainerInference with custom model wrapper."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        output_dir: str,
        device: str = "cuda",
        original_checkpoint: Optional[str] = None,
        iterations: int = 30000,
        prune_invalid: bool = False,
        save_checkpoint: bool = False,
        filter_percentile: Optional[float] = None,
        use_original_model: bool = False,
    ):
        """
        Initialize batch predictor.

        Args:
            config_path: Path to inference config
            checkpoint_path: Path to model checkpoint
            output_dir: Output directory
            device: Device to use
            original_checkpoint: Base directory for original Gaussian checkpoints
            iterations: Checkpoint iteration number
            prune_invalid: Whether to prune invalid Gaussians (default: False, keeps all Gaussians)
            save_checkpoint: Whether to save Gaussian checkpoint (when original checkpoint is found)
            filter_percentile: Filter percentile for coord outliers (e.g., 0.1 means 0.1%-99.9%).
                           If None, no filtering is applied (default: None).
            use_original_model: Whether using original PT-v3m1 model (affects checkpoint naming)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.original_checkpoint_base = original_checkpoint
        self.iterations = iterations
        self.prune_invalid = prune_invalid
        self.save_checkpoint = save_checkpoint
        self.filter_percentile = filter_percentile
        self.use_original_model = use_original_model

        # Load config
        self.cfg = Config.fromfile(config_path)

        # Initialize LangPretrainerInference (for transform pipeline)
        # But we'll use custom model wrapper for inference
        self.inference = LangPretrainerInference(
            self.cfg,
            checkpoint_path,
            device=device,
        )
        self.inference.output_dir = str(self.output_dir)

        # Wrap model to return backbone_feat in expected format
        self._wrap_model_for_inference()

        self.checkpoint_handler = GaussianCheckpointHandler()

        # Setup logging
        self.logger = logging.getLogger("BatchPredictor")

    def _wrap_model_for_inference(self):
        """Wrap the model to return backbone_feat in the expected format."""
        original_forward = self.inference.model.forward

        def wrapped_forward(self, input_dict, **kwargs):
            """Wrapped forward that extracts features correctly."""
            # Extract chunk_size if present
            chunk_size = kwargs.pop("chunk_size", None)

            # Call original forward with chunk_size if provided
            if chunk_size is not None:
                result = original_forward(input_dict, chunk_size=chunk_size)
            else:
                result = original_forward(input_dict)

            # Extract features from result
            if "point_feat" in result:
                point_feat = result["point_feat"]
                if isinstance(point_feat, dict):
                    feat = point_feat.get("feat")
                elif hasattr(point_feat, "feat"):
                    feat = point_feat.feat
                else:
                    feat = point_feat
                result["backbone_feat"] = feat

            return result

        import types
        # Bind the method properly to the model instance
        self.inference.model.forward = types.MethodType(wrapped_forward, self.inference.model)

    def find_original_checkpoint(self, scene_name: str) -> Optional[str]:
        """Find original checkpoint for a scene."""
        if self.original_checkpoint_base is None:
            return None

        base_dir = Path(self.original_checkpoint_base)

        # Try different patterns - SceneSplat7K uses _langfeat_0 suffix
        patterns = [
            # Pattern with _langfeat_N suffix (SceneSplat7K naming)
            base_dir / scene_name / f"chkpnt{self.iterations}_langfeat_0.pth",
            base_dir / scene_name / "ckpts" / f"chkpnt{self.iterations}_langfeat_0.pth",
            # Pattern without suffix (original naming)
            base_dir / scene_name / "ckpts" / f"chkpnt{self.iterations}.pth",
            base_dir / scene_name / f"chkpnt{self.iterations}.pth",
            base_dir / f"chkpnt{self.iterations}.pth",
        ]

        for pattern in patterns:
            if pattern.exists():
                return str(pattern)
        return None

    def load_scene_data(self, scene_path: Path) -> Dict[str, np.ndarray]:
        """Load .npy files from a scene directory.

        Handles SVD-compressed language features if configured.
        Filters valid points (with valid language features) to match training distribution.
        Filters coordinate outliers to prevent PointOctree depth overflow.

        Filter order (CRITICAL - must match training):
        1. FilterValidPoints: Removes points with invalid language features (for SVD data)
        2. FilterCoordOutliers: Removes coordinate outliers (if filter_percentile specified)

        Index tracking for expansion:
        - _n_original_points_before_filter: Total original point count (e.g., 1M)
        - _valid_kept_indices: Indices in original space that are valid (for Stage 2 expansion)
        - _outlier_removed_indices_in_valid_space: Indices in valid space removed by outliers (for Stage 1 expansion)

        Two-stage expansion (in process_scene):
        - Stage 1: Expand from outlier-filtered to valid-space (fill outliers with zeros)
        - Stage 2: Expand from valid-space to original-space (fill invalid points with zeros)
        """
        data_dict = {}
        for file_path in scene_path.glob("*.npy"):
            key = file_path.stem
            data_dict[key] = np.load(file_path)

        # Handle SVD-compressed language features
        # Only load SVD features when using LitePT model, NOT for original PT-v3m1
        cfg = self.inference.cfg
        svd_rank = getattr(cfg, 'svd_rank', None)
        model_type = cfg.model.get('backbone', {}).get('type', '')

        # Only load SVD-compressed features for LitePT models (not PT-v3m1)
        is_litept = 'litept' in model_type.lower()

        if svd_rank is not None and 'lang_feat' in data_dict and is_litept:
            svd_file = scene_path / f"lang_feat_grid_svd_r{svd_rank}.npz"
            if svd_file.exists():
                print(f"Loading SVD-{svd_rank} compressed lang_feat from {svd_file}")
                try:
                    svd_data = np.load(svd_file)
                    compressed = svd_data['compressed']  # [M, rank] - grid cells only
                    indices = svd_data['indices']  # [N_valid] - valid points only (after FilterValidPoints)

                    # CRITICAL: SVD file contains features for valid points only
                    # The indices length (N_valid) is the number of valid points after FilterValidPoints
                    # We need to filter coord and other arrays to match this
                    n_valid_points = len(indices)
                    n_grid_cells = len(compressed)
                    print(f"  SVD contains {n_valid_points} valid points, {n_grid_cells} grid cells")

                    # Check if valid_feat_mask exists and needs to be used for filtering
                    if "valid_feat_mask" in data_dict:
                        # Convert valid_feat_mask to boolean (it's stored as int64)
                        valid_mask = data_dict["valid_feat_mask"].astype(bool)
                        data_dict["valid_feat_mask"] = valid_mask  # Update to boolean
                        n_total = len(valid_mask)
                        n_valid_in_mask = valid_mask.sum()

                        print(f"  valid_feat_mask: {n_valid_in_mask}/{n_total} valid points")

                        # Verify consistency between mask and SVD size
                        if n_valid_in_mask != n_valid_points:
                            print(f"  Warning: Mask count ({n_valid_in_mask}) != SVD count ({n_valid_points})")
                            print(f"  Using SVD count as ground truth (assuming mask was created before SVD)")

                        # Now filter all arrays to match training distribution
                        # Training: FilterValidPoints filters coord from 1M → n_valid_points
                        # Then SVD lang_feat is loaded with n_valid_points
                        # So we need to do the same: filter coord first, then use SVD features
                        print(f"  Filtering coord from {n_total} to {n_valid_points} points to match training")

                        # Store original indices for later expansion
                        # _valid_kept_indices: which points in original 1M space are kept (valid)
                        # _n_original_points_before_filter: original point count (1M)
                        valid_indices = np.where(valid_mask)[0]
                        data_dict["_valid_kept_indices"] = valid_indices  # Indices in original space
                        data_dict["_n_original_points_before_filter"] = n_total

                        # Filter coord and other arrays to match SVD size
                        keys_to_filter = ["coord", "color", "opacity", "quat", "scale", "segment"]
                        for key in keys_to_filter:
                            if key in data_dict and isinstance(data_dict[key], np.ndarray):
                                if data_dict[key].shape[0] == n_total:
                                    data_dict[key] = data_dict[key][valid_mask]

                        # Update valid_feat_mask to the filtered size (all points are now valid)
                        data_dict["valid_feat_mask"] = np.ones(n_valid_points, dtype=bool)
                        print(f"  Filtered: coord now has {data_dict['coord'].shape[0]} points")

                    # Add point_to_grid mapping (already has correct size n_valid_points)
                    data_dict["point_to_grid"] = indices.astype(np.int64)

                    # Expand grid-level features to point-level: [N_valid, rank]
                    point_lang_feat = compressed[indices]
                    data_dict["lang_feat"] = point_lang_feat.astype(np.float32)
                    print(f"  Loaded SVD lang_feat: {point_lang_feat.shape}")
                except Exception as e:
                    print(f"  Warning: Failed to load SVD file: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: SVD rank {svd_rank} configured but file not found: {svd_file}")
                print(f"  Using original lang_feat.npy instead")
        elif svd_rank is not None and 'lang_feat' in data_dict and not is_litept:
            print(f"Note: Using original PT-v3m1 model, ignoring SVD config (using full 768-dim features)")

        # Filter valid points to match training distribution (CRITICAL: must be BEFORE FilterCoordOutliers)
        # For SVD: Filtering already done above (coord filtered to match SVD lang_feat size)
        # For non-SVD: Use valid_feat_mask to filter
        if "valid_feat_mask" in data_dict and svd_rank is None:
            valid_mask = data_dict["valid_feat_mask"]
            n_before = len(valid_mask)

            # CRITICAL: Convert to bool first to handle int64/uint8 masks correctly
            # For int64/uint8 arrays, ~ does bitwise NOT, not logical NOT
            bool_mask = valid_mask.astype(bool)
            n_invalid = (~bool_mask).sum()

            if n_invalid > 0:
                print(f"[FilterValidPoints] Removing {n_invalid}/{n_before} invalid points "
                      f"({n_invalid/n_before*100:.1f}%) - matching training distribution")

                # Store original point count before filtering
                if "_n_original_points_before_filter" not in data_dict:
                    data_dict["_n_original_points_before_filter"] = n_before

                # Store indices of invalid points for later expansion
                invalid_indices = np.where(~bool_mask)[0]
                if "_filtered_out_indices" in data_dict:
                    # Merge with existing filtered indices
                    data_dict["_filtered_out_indices"] = np.concatenate([
                        data_dict["_filtered_out_indices"], invalid_indices
                    ])
                else:
                    data_dict["_filtered_out_indices"] = invalid_indices

                # Store indices of valid points for Stage 2 expansion (expand back to original space)
                valid_kept_indices = np.where(bool_mask)[0]
                data_dict["_valid_kept_indices"] = valid_kept_indices
                print(f"  Stored {len(valid_kept_indices)} valid point indices for expansion")

                # Filter all arrays to match training distribution
                keys_to_filter = ["coord", "color", "opacity", "quat", "scale", "lang_feat",
                                  "point_to_grid", "segment", "valid_feat_mask"]
                for key in keys_to_filter:
                    if key in data_dict and isinstance(data_dict[key], np.ndarray):
                        if data_dict[key].shape[0] == n_before:
                            data_dict[key] = data_dict[key][bool_mask]
            else:
                print(f"[FilterValidPoints] All {n_before} points have valid features - no filtering needed")

        # Filter coordinate outliers to prevent PointOctree depth overflow
        # This filtering happens BEFORE the transform pipeline
        # The transform pipeline does NOT include FilterCoordOutliers to avoid duplicate filtering
        # Only apply filtering if filter_percentile is specified
        if "coord" in data_dict and self.filter_percentile is not None:
            coord = data_dict["coord"]
            n_points = coord.shape[0]

            # Calculate percentiles to identify outliers
            # filter_percentile is the margin from both ends (e.g., 0.1 means 0.1%-99.9%)
            percentile_low = self.filter_percentile
            percentile_high = 100 - self.filter_percentile
            pct_low = np.percentile(coord, percentile_low, axis=0)
            pct_high = np.percentile(coord, percentile_high, axis=0)

            # Create mask for points within percentile range
            mask = np.ones(n_points, dtype=bool)
            for dim in range(3):
                mask &= (coord[:, dim] >= pct_low[dim]) & (coord[:, dim] <= pct_high[dim])

            n_filtered_out = (~mask).sum()
            if n_filtered_out > 0:
                print(f"[FilterCoordOutliers] Filtered {n_filtered_out}/{n_points} points ({n_filtered_out/n_points*100:.1f}%)")
                print(f"  Filter range: {percentile_low}%-{percentile_high}%")
                print(f"  Original range: x=[{coord[:, 0].min():.2f}, {coord[:, 0].max():.2f}], "
                      f"y=[{coord[:, 1].min():.2f}, {coord[:, 1].max():.2f}], "
                      f"z=[{coord[:, 2].min():.2f}, {coord[:, 2].max():.2f}]")

                # Store filtered indices in the CURRENT (post-valid-filter) space
                # These are indices 0 to n_points-1 (in the already-filtered space)
                outlier_removed_indices_in_valid_space = np.where(~mask)[0]
                data_dict["_outlier_removed_indices_in_valid_space"] = outlier_removed_indices_in_valid_space
                data_dict["_outlier_removed_count"] = n_filtered_out

                # Filter all arrays with matching first dimension
                keys_to_filter = ["coord", "color", "opacity", "quat", "scale", "lang_feat", "valid_feat_mask",
                                  "point_to_grid", "segment"]
                for key in keys_to_filter:
                    if key in data_dict and isinstance(data_dict[key], np.ndarray) and data_dict[key].shape[0] == n_points:
                        data_dict[key] = data_dict[key][mask]

                # Recompute grid_coord if it exists
                if "grid_coord" in data_dict:
                    coord_filtered = data_dict["coord"]
                    grid_size = 0.01  # Default grid size
                    grid_coord = np.floor((coord_filtered - coord_filtered.min(axis=0)) / grid_size).astype(int)
                    data_dict["grid_coord"] = grid_coord
                    print(f"  Filtered range: x=[{coord_filtered[:, 0].min():.2f}, {coord_filtered[:, 0].max():.2f}], "
                          f"y=[{coord_filtered[:, 1].min():.2f}, {coord_filtered[:, 1].max():.2f}], "
                          f"z=[{coord_filtered[:, 2].min():.2f}, {coord_filtered[:, 2].max():.2f}]")
                    print(f"  grid_coord range: [{grid_coord.min(axis=0)}, {grid_coord.max(axis=0)}]")
        elif "coord" in data_dict and self.filter_percentile is None:
            n_points = data_dict["coord"].shape[0]
            print(f"[FilterCoordOutliers] Disabled (filter_percentile=None), keeping all {n_points} points")

        return data_dict

    def process_scene(
        self,
        scene_path: Path,
        scene_name: Optional[str] = None,
    ) -> Dict:
        """Process a single scene."""
        scene_name = scene_name or scene_path.name
        scene_start_time = time.time()

        print(f"\nProcessing: {scene_name}")

        # Load data (with outlier filtering)
        load_start = time.time()
        data_dict = self.load_scene_data(scene_path)
        load_time = time.time() - load_start

        # Store valid_feat_mask before it gets removed by transforms
        valid_feat_mask = data_dict.get("valid_feat_mask")
        if valid_feat_mask is not None:
            print(f"Found valid_feat_mask: {valid_feat_mask.sum()}/{len(valid_feat_mask)} points have valid features")

        # Store index tracking info for expansion
        n_original_points = data_dict.get("_n_original_points_before_filter")  # Total original points (1M)
        valid_kept_indices = data_dict.get("_valid_kept_indices")  # Indices in original space that are valid
        outlier_removed_indices = data_dict.get("_outlier_removed_indices_in_valid_space")  # Indices in valid space removed by outliers
        outlier_removed_count = data_dict.get("_outlier_removed_count", 0)

        # Find original checkpoint
        original_ckpt_path = None
        if self.original_checkpoint_base is not None:
            original_ckpt_path = self.find_original_checkpoint(scene_name)

        # Run inference
        inference_start = time.time()
        outputs = self.inference(
            data_dict,
            scene_name=scene_name,
            save=False,  # We'll handle saving ourselves
        )
        inference_time = time.time() - inference_start

        # Extract results
        features = outputs["backbone_features"]  # [N_final, feat_dim]
        metadata = outputs["metadata"]

        # Post-processing: handle inverse mapping and expansion
        postprocess_start = time.time()

        # Handle inverse mapping from GridSample
        inverse = metadata.get("inverse")
        if inverse is not None:
            # Map features back to pre-filter points
            features = features[inverse]

        # Two-stage expansion back to original size
        # Stage 1: Expand from outlier-filtered to valid-space (fill outliers with zeros)
        # Stage 2: Expand from valid-space to original-space (fill invalid points with zeros)

        if n_original_points is not None and valid_kept_indices is not None:
            feat_dim = features.shape[1]

            # Stage 1: Expand outlier-removed points back to valid space
            if outlier_removed_indices is not None and len(outlier_removed_indices) > 0:
                n_valid_space = len(valid_kept_indices)
                features_in_valid_space = np.zeros((n_valid_space, feat_dim), dtype=features.dtype)

                # Indices in valid space that were kept (not removed by outlier filtering)
                kept_in_valid_space = np.setdiff1d(np.arange(n_valid_space), outlier_removed_indices)

                # Assign features to kept positions in valid space
                features_in_valid_space[kept_in_valid_space] = features

                print(f"[Expansion Stage 1] Expanded from {features.shape[0]} to {features_in_valid_space.shape[0]} "
                      f"(filled {len(outlier_removed_indices)} outlier-removed points with zeros)")
            else:
                # No outlier filtering, features already in valid space
                features_in_valid_space = features
                print(f"[Expansion Stage 1] No outlier filtering, features already in valid space: {features_in_valid_space.shape[0]}")

            # Stage 2: Expand from valid space to original space (fill invalid points with zeros)
            full_features = np.zeros((n_original_points, feat_dim), dtype=features_in_valid_space.dtype)
            full_features[valid_kept_indices] = features_in_valid_space

            n_invalid_filled = n_original_points - len(valid_kept_indices)
            print(f"[Expansion Stage 2] Expanded from {features_in_valid_space.shape[0]} to {full_features.shape[0]} "
                  f"(filled {n_invalid_filled} invalid points with zeros)")

            # Expand valid_feat_mask back to original size
            if valid_feat_mask is not None:
                # Create full-size mask with all points marked invalid
                full_valid_mask = np.zeros(n_original_points, dtype=bool)
                # Mark valid points as True
                full_valid_mask[valid_kept_indices] = True
                valid_feat_mask = full_valid_mask
                print(f"[Expansion] valid_feat_mask: {valid_feat_mask.sum()}/{len(valid_feat_mask)} valid points")

            features = full_features
        else:
            # No expansion needed
            print(f"[Expansion] No expansion needed, features shape: {features.shape}")

        postprocess_time = time.time() - postprocess_start

        # Save features as .npy
        save_start = time.time()
        feat_path = self.output_dir / scene_name / "language_features.npy"
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feat_path, features)
        save_time = time.time() - save_start

        print(f"Saved features to: {feat_path}")
        print(f"  Final feature shape: {features.shape}")

        # Handle Gaussian checkpoint if available and save_checkpoint is True
        if original_ckpt_path is not None and self.save_checkpoint:
            try:
                original_ckpt = self.checkpoint_handler.load_original_checkpoint(original_ckpt_path)

                # Create checkpoint with features
                # Note: features should already be in the correct space after expansion
                new_ckpt = self.checkpoint_handler.create_checkpoint_with_features(
                    features=features,
                    original_checkpoint=original_ckpt,
                    valid_feat_mask=valid_feat_mask,
                    prune_invalid=self.prune_invalid,
                )

                # Save checkpoint with appropriate filename based on model type
                # "checkpoint_with_features_s.pth" for original PT-v3m1 model (768-dim features)
                # "checkpoint_with_features_p.pth" for LitePT or other models
                if self.use_original_model:
                    ckpt_path = self.output_dir / scene_name / "checkpoint_with_features_s.pth"
                    model_desc = "PT-v3m1 (original)"
                else:
                    ckpt_path = self.output_dir / scene_name / "checkpoint_with_features_p.pth"
                    model_desc = "LitePT/other"
                print(f"Saving checkpoint for {model_desc} model to: {ckpt_path}")
                self.checkpoint_handler.save_checkpoint(new_ckpt, str(ckpt_path))

            except Exception as e:
                print(f"Warning: Failed to process Gaussian checkpoint: {e}")
                import traceback
                traceback.print_exc()
        elif original_ckpt_path is not None and not self.save_checkpoint:
            print(f"Found original checkpoint but --save-checkpoint not specified, skipping Gaussian checkpoint creation")
        elif self.save_checkpoint and original_ckpt_path is None:
            print(f"--save-checkpoint specified but no original checkpoint found for scene {scene_name}")

        # Print timing summary
        scene_total_time = time.time() - scene_start_time
        print(f"Timing: load={load_time:.3f}s, inference={inference_time:.3f}s, "
              f"postprocess={postprocess_time:.3f}s, save={save_time:.3f}s, total={scene_total_time:.3f}s")

        return {
            "scene_name": scene_name,
            "features_shape": features.shape,
            "has_original_checkpoint": original_ckpt_path is not None,
            "checkpoint_saved": (original_ckpt_path is not None and self.save_checkpoint),
            "timing": {
                "load": load_time,
                "inference": inference_time,
                "postprocess": postprocess_time,
                "save": save_time,
                "total": scene_total_time,
            }
        }

    def run(
        self,
        input_root: str,
        recursive: bool = False,
        max_scenes: Optional[int] = None,
        scene: Optional[str] = None,
    ):
        """Run batch prediction on all scenes or a single scene.

        Args:
            input_root: Input directory containing scene folders
            recursive: Recursively find scene directories
            max_scenes: Maximum number of scenes to process
            scene: Single scene name to process (relative to input_root)
        """
        input_path = Path(input_root)
        scenes = []

        # Handle single scene mode
        if scene is not None:
            scene_path = input_path / scene
            if not scene_path.exists():
                raise ValueError(f"Scene path does not exist: {scene_path}")
            if not scene_path.is_dir():
                raise ValueError(f"Scene path is not a directory: {scene_path}")
            if not any(scene_path.glob("*.npy")):
                raise ValueError(f"No .npy files found in scene directory: {scene_path}")
            scenes = [scene_path]
            print(f"\nProcessing single scene: {scene}")
        elif recursive:
            # Find all directories with .npy files
            for root, dirs, files in os.walk(input_path):
                root_path = Path(root)
                npy_files = list(root_path.glob("*.npy"))
                if npy_files:
                    scenes.append(root_path)
        else:
            # Non-recursive: direct subdirectories
            for item in input_path.iterdir():
                if item.is_dir() and any(item.glob("*.npy")):
                    scenes.append(item)

        scenes = sorted(scenes)

        if max_scenes:
            scenes = scenes[:max_scenes]

        if scene is None:
            print(f"\nFound {len(scenes)} scenes to process")

        results = []
        for scene_item in scenes:
            try:
                result = self.process_scene(scene_item)
                results.append(result)
            except Exception as e:
                print(f"Error processing {scene_item}: {e}")
                import traceback
                traceback.print_exc()

        # Summary
        print("\n" + "=" * 60)
        if scene is not None:
            print("Single Scene Processing Summary")
        else:
            print("Batch Prediction Summary")
        print("=" * 60)
        for r in results:
            timing = r.get('timing', {})
            total_time = timing.get('total', 0)
            inference_time = timing.get('inference', 0)
            print(f"{r['scene_name']}: shape={r['features_shape']}, "
                  f"ckpt={'Yes' if r['has_original_checkpoint'] else 'No'}, "
                  f"time={total_time:.3f}s (inference={inference_time:.3f}s)")

        # Calculate aggregate timing stats
        if results and 'timing' in results[0]:
            total_load = sum(r.get('timing', {}).get('load', 0) for r in results)
            total_inference = sum(r.get('timing', {}).get('inference', 0) for r in results)
            total_postprocess = sum(r.get('timing', {}).get('postprocess', 0) for r in results)
            total_save = sum(r.get('timing', {}).get('save', 0) for r in results)
            total_time_all = sum(r.get('timing', {}).get('total', 0) for r in results)
            if len(results) > 1:
                print(f"\nAggregate timing over {len(results)} scenes:")
                print(f"  Total load: {total_load:.3f}s")
                print(f"  Total inference: {total_inference:.3f}s")
                print(f"  Total postprocess: {total_postprocess:.3f}s")
                print(f"  Total save: {total_save:.3f}s")
                print(f"  Total time: {total_time_all:.3f}s")
                print(f"  Average per scene: {total_time_all / len(results):.3f}s")

        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved summary to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch prediction using LangPretrainerInference"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to inference config (auto-selected by --use-original-model if not specified)",
    )
    parser.add_argument(
        "--use-original-model",
        action="store_true",
        help="Use original SceneSplat PT-v3m1 model (configs/inference/lang-pretrain-pt-v3m1-3dgs.py)",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=None,
        help="Feature dimension to output (default: auto-detected from config, 768 for PT-v3m1)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Input directory with .npy scene folders",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--original-checkpoint",
        default=None,
        help="Base directory for original Gaussian checkpoints",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Checkpoint iteration number (default: 30000)",
    )
    parser.add_argument(
        "--prune-invalid",
        action="store_true",
        default=False,
        help="Prune Gaussians with invalid features (default: False)",
    )
    parser.add_argument(
        "--no-prune-invalid",
        dest="prune_invalid",
        action="store_false",
        help="Keep all Gaussians including invalid ones",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        default=False,
        help="Save Gaussian checkpoint with language features when original checkpoint is found",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find scene directories",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Process a single scene by name (relative to input_root)",
    )
    parser.add_argument(
        "--filter-percentile",
        type=float,
        default=None,
        help="Filter percentile for coord outliers (e.g., 0.1 means 0.1%%-99.9%%). "
             "If not specified, no filtering is applied (default: None)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate and auto-select config
    config_path = args.config
    feature_dim = args.feature_dim
    model_type = "unknown"

    if args.use_original_model:
        if config_path is not None:
            print("Warning: --use-original-model specified with --config, using --config")
        else:
            # Auto-select original SceneSplat PT-v3m1 config
            config_path = "configs/inference/lang-pretrain-pt-v3m1-3dgs.py"
            model_type = "PT-v3m1 (original SceneSplat)"
            # Set default feature dim for PT-v3m1 if not specified
            if feature_dim is None:
                feature_dim = 768
    elif config_path is None:
        raise ValueError("Either --config or --use-original-model must be specified")
    else:
        # Detect model type from config path
        if "litept" in config_path.lower():
            model_type = "LitePT"
        elif "pt-v3" in config_path.lower() or "ptv3" in config_path.lower():
            model_type = "PT-v3m1 (original SceneSplat)"
        else:
            model_type = "Custom"

    # Resolve config path relative to project root
    if config_path is not None and not os.path.isabs(config_path):
        config_path = str(PROJECT_ROOT / config_path)

    print("=" * 60)
    print("SceneSplat Batch Prediction (LangPretrainerInference)")
    print("=" * 60)
    print(f"Model type: {model_type}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input_root}")
    if args.scene is not None:
        print(f"Scene: {args.scene} (single scene mode)")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    if feature_dim is not None:
        print(f"Feature dim: {feature_dim}")
    print(f"Original checkpoint: {args.original_checkpoint or 'N/A'}")
    print(f"Iterations: {args.iterations}")
    print(f"Prune invalid: {args.prune_invalid}")
    print(f"Save checkpoint: {args.save_checkpoint}")
    print(f"Filter percentile: {args.filter_percentile if args.filter_percentile is not None else 'Disabled (no filtering)'}")
    if args.scene is None:
        print(f"Recursive: {args.recursive}")
    print("=" * 60)

    predictor = BatchPredictorWithInference(
        config_path=config_path,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        original_checkpoint=args.original_checkpoint,
        iterations=args.iterations,
        prune_invalid=args.prune_invalid,
        save_checkpoint=args.save_checkpoint,
        filter_percentile=args.filter_percentile,
        use_original_model=args.use_original_model,
    )

    predictor.run(
        input_root=args.input_root,
        recursive=args.recursive,
        max_scenes=args.max_scenes,
        scene=args.scene,
    )


if __name__ == "__main__":
    main()
