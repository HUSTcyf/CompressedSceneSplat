#!/usr/bin/env python3
"""
Comprehensive Model Collapse Analysis Script
Integrates all analysis functionality for checkpoint comparison and mode collapse detection.

Checkpoint Structure:
    checkpoint = (model_params, first_iter)
    model_params = (active_sh_degree, xyz, features_dc, features_rest,
                    scaling, rotation, opacity, language_features,
                    max_radii2D, xyz_gradient_accum, denom,
                    opt_dict, spatial_lr_scale, valid_feat_mask)

    language_features is at index 7 of model_params tuple.

Usage:
    # Compare two checkpoints (all samples)
    python analyze_model_collapse_integrated.py --path1 checkpoint1.pth --path2 checkpoint2.pth

    # Compare two checkpoints (common non-zero rows only)
    python analyze_model_collapse_integrated.py --path1 checkpoint1.pth --path2 checkpoint2.pth --nonzero-only

    # Batch compare all scenes in output_features directory
    python analyze_model_collapse_integrated.py --batch-compare --output-dir output_features

    # Analyze single checkpoint
    python analyze_model_collapse_integrated.py --checkpoint model.pth

    # Analyze detailed value ranges
    python analyze_model_collapse_integrated.py --checkpoint model.pth --analyze-ranges
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class CollapseAnalyzer:
    """Comprehensive analyzer for model mode collapse."""

    def __init__(self, device='cpu'):
        self.device = torch.device(device)

    def load_checkpoint(self, path: str) -> Optional[Dict]:
        """
        Load checkpoint and extract model parameters.

        Returns:
            Dict with keys: 'model_params', 'first_iter', 'language_features', etc.
        """
        print(f"\n{'='*70}")
        print(f"Loading: {path}")
        print(f"{'='*70}")

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            if not isinstance(checkpoint, tuple):
                print(f"ERROR: Expected tuple, got {type(checkpoint)}")
                return None

            print(f"Checkpoint is a tuple with {len(checkpoint)} elements")

            # Extract model_params and first_iter
            # SceneSplat format: checkpoint = (model_params, first_iter)
            # where model_params is itself a tuple

            if len(checkpoint) >= 2:
                model_params = checkpoint[0]
                first_iter = checkpoint[1]
            else:
                print("ERROR: Checkpoint tuple has < 2 elements")
                return None

            print(f"First iteration: {first_iter}")
            print(f"Model params length: {len(model_params)}")

            # Parse model_params tuple
            # SceneSplat format has 14 elements
            result = {
                'first_iter': first_iter,
                'model_params': model_params,
            }

            if len(model_params) == 13:
                # Standard OccamLGS format (13 elements)
                (result['active_sh_degree'],
                 result['xyz'],
                 result['features_dc'],
                 result['features_rest'],
                 result['scaling'],
                 result['rotation'],
                 result['opacity'],
                 result['language_features'],  # Index 7
                 result['max_radii2D'],
                 result['xyz_gradient_accum'],
                 result['denom'],
                 result['opt_dict'],
                 result['spatial_lr_scale']) = model_params

                result['valid_feat_mask'] = None

                print(f"\nLoaded OccamLGS format checkpoint (13 elements)")
                print(f"  xyz shape: {result['xyz'].shape}")
                print(f"  features_dc shape: {result['features_dc'].shape}")
                print(f"  features_rest shape: {result['features_rest'].shape}")
                print(f"  scaling shape: {result['scaling'].shape}")
                print(f"  rotation shape: {result['rotation'].shape}")
                print(f"  opacity shape: {result['opacity'].shape}")
                print(f"  language_features shape: {result['language_features'].shape}")

                return result
            elif len(model_params) == 14:
                # SceneSplat checkpoint_with_features format (includes valid_feat_mask)
                (result['active_sh_degree'],
                 result['xyz'],
                 result['features_dc'],
                 result['features_rest'],
                 result['scaling'],
                 result['rotation'],
                 result['opacity'],
                 result['language_features'],  # Index 7
                 result['max_radii2D'],
                 result['xyz_gradient_accum'],
                 result['denom'],
                 result['opt_dict'],
                 result['spatial_lr_scale'],
                 result['valid_feat_mask']) = model_params

                print(f"\nLoaded SceneSplat format checkpoint (14 elements)")
                print(f"  xyz shape: {result['xyz'].shape}")
                print(f"  features_dc shape: {result['features_dc'].shape}")
                print(f"  features_rest shape: {result['features_rest'].shape}")
                print(f"  scaling shape: {result['scaling'].shape}")
                print(f"  rotation shape: {result['rotation'].shape}")
                print(f"  opacity shape: {result['opacity'].shape}")
                print(f"  language_features shape: {result['language_features'].shape}")
                if result['valid_feat_mask'] is not None:
                    print(f"  valid_feat_mask shape: {result['valid_feat_mask'].shape}")

                return result
            else:
                print(f"ERROR: Unknown model_params format (expected 13 or 14 elements, got {len(model_params)})")
                return None

        except Exception as e:
            print(f"ERROR loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_tensor(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
        """Comprehensive tensor analysis."""
        if not isinstance(tensor, torch.Tensor):
            return None

        # Convert to float if needed
        if tensor.dtype in [torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8]:
            tensor = tensor.float()

        result = {
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'var': tensor.var().item(),
            'zeros': (tensor == 0).sum().item(),
            'nan': tensor.isnan().sum().item(),
            'inf': tensor.isinf().sum().item(),
            'neg': (tensor < 0).sum().item(),
            'pos': (tensor > 0).sum().item(),
        }

        # Coefficient of variation
        result['cv'] = result['std'] / (abs(result['mean']) + 1e-8)
        result['zero_ratio'] = result['zeros'] / tensor.numel()

        # Per-dimension analysis
        if len(tensor.shape) >= 2:
            means = tensor.mean(dim=tuple(range(len(tensor.shape)-1)))
            stds = tensor.std(dim=tuple(range(len(tensor.shape)-1)))
            vars = tensor.var(dim=tuple(range(len(tensor.shape)-1)))

            result['dim_means'] = means
            result['dim_stds'] = stds
            result['dim_vars'] = vars
            result['collapsed_dims'] = (vars < 1e-6).nonzero().flatten().tolist()
        else:
            result['collapsed_dims'] = []

        # Mode collapse detection
        result['is_collapsed'] = (
            result['var'] < 1e-6 or
            result['cv'] < 0.01 or
            result['zero_ratio'] > 0.9 or
            result['std'] < 0.001
        )

        return result

    def compare_tensors(self, t1: torch.Tensor, t2: torch.Tensor, name: str, num_samples: int) -> Dict[str, Any]:
        """Compare two tensors with comprehensive metrics."""
        if t1.shape != t2.shape:
            print(f"WARNING: Shape mismatch for {name}: {t1.shape} vs {t2.shape}")
            return None

        # Convert to float if needed
        if t1.dtype in [torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8]:
            t1 = t1.float()
        if t2.dtype in [torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8]:
            t2 = t2.float()

        # Per-dimension analysis
        results = []
        num_dims = t1.shape[-1] if len(t1.shape) >= 2 else 1

        for dim_idx in range(num_dims):
            if len(t1.shape) >= 2:
                d1 = t1[:, dim_idx]
                d2 = t2[:, dim_idx]
            else:
                d1 = t1.flatten()
                d2 = t2.flatten()

            # Statistics
            stats = {
                'dim': dim_idx,
                'mean1': d1.mean().item(),
                'std1': d1.std().item(),
                'var1': d1.var().item(),
                'min1': d1.min().item(),
                'max1': d1.max().item(),
                'zeros1': (d1 == 0).sum().item() / len(d1),
                'mean2': d2.mean().item(),
                'std2': d2.std().item(),
                'var2': d2.var().item(),
                'min2': d2.min().item(),
                'max2': d2.max().item(),
                'zeros2': (d2 == 0).sum().item() / len(d2),
            }

            # Distance metrics
            diff = d1 - d2
            stats['total_l1'] = torch.abs(diff).sum().item()
            stats['total_l2'] = torch.norm(diff, p=2).item()
            stats['per_sample_l1'] = stats['total_l1'] / num_samples
            stats['mad'] = torch.abs(diff).mean().item()

            # Cosine similarity
            stats['cosine_sim'] = torch.nn.functional.cosine_similarity(
                d1.unsqueeze(0), d2.unsqueeze(0)
            ).item()
            stats['cosine_dist'] = 1 - stats['cosine_sim']

            # Correlation
            try:
                stacked = torch.stack([d1, d2])
                stats['corr'] = torch.corrcoef(stacked)[0, 1].item()
            except:
                stats['corr'] = 0.0

            results.append(stats)

        return {
            'name': name,
            'num_samples': num_samples,
            'num_dims': num_dims,
            'dimensions': results,
        }

    def print_tensor_summary(self, stats: Dict, name: str):
        """Print summary statistics for a tensor."""
        print(f"\n{name}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  Variance: {stats['var']:.8f}")
        print(f"  CV: {stats['cv']:.6f}, Zero ratio: {stats['zero_ratio']:.2%}")

        if 'collapsed_dims' in stats and stats['collapsed_dims']:
            print(f"  ⚠️ Collapsed dimensions: {stats['collapsed_dims']}")

        if stats['is_collapsed']:
            print(f"  ⚠️ COLLAPSED!")

    def print_per_dimension_comparison(self, comp: Dict):
        """Print detailed per-dimension comparison."""
        print(f"\n{'='*70}")
        print(f"Per-Dimension Comparison: {comp['name']}")
        print(f"{'='*70}")

        num_dims = comp['num_dims']
        num_samples = comp['num_samples']

        print(f"\nNote: Total L1 is summed over {num_samples} samples")
        print(f"      Per-sample L1 = Total L1 / {num_samples}")

        print(f"\n{'Dim':<4} {'Range1':<20} {'Range2':<20} {'PerSampL1':<12} {'CosSim':<10} {'Status1':<10} {'Status2':<10}")
        print(f"{'-'*80}")

        for d in comp['dimensions']:
            dim_idx = d['dim']
            range1 = f"[{d['min1']:.4f}, {d['max1']:.4f}]"
            range2 = f"[{d['min2']:.4f}, {d['max2']:.4f}]"
            per_samp_l1 = d['per_sample_l1']
            cos_sim = d['cosine_sim']

            status1 = "✓" if d['var1'] >= 1e-6 else "✗ COLLAPSED"
            status2 = "✓" if d['var2'] >= 1e-6 else "✗ COLLAPSED"

            print(f"{dim_idx:<4} {range1:<20} {range2:<20} {per_samp_l1:<12.6f} {cos_sim:<10.6f} {status1:<10} {status2:<10}")

    def analyze_checkpoint(self, path: str, label: str = ""):
        """Analyze a single checkpoint."""
        data = self.load_checkpoint(path)
        if data is None or 'language_features' not in data:
            return {}

        print(f"\n{'='*70}")
        print(f"ANALYSIS: {label or path}")
        print(f"{'='*70}")

        results = {}

        # Analyze language_features (the main tensor of interest)
        lang_feat = data['language_features']
        stats = self.analyze_tensor(lang_feat, 'language_features')
        results['language_features'] = stats
        self.print_tensor_summary(stats, 'language_features')

        # Also analyze other key tensors
        for key in ['xyz', 'features_dc', 'features_rest', 'scaling', 'rotation', 'opacity']:
            if key in data and data[key] is not None:
                stats = self.analyze_tensor(data[key], key)
                results[key] = stats
                if stats:
                    self.print_tensor_summary(stats, key)

        return results

    def compare_checkpoints(self, path1: str, path2: str):
        """Compare two checkpoints comprehensively."""
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE CHECKPOINT COMPARISON")
        print(f"{'='*70}")
        print(f"File 1: {path1}")
        print(f"File 2: {path2}")

        # Load both checkpoints
        data1 = self.load_checkpoint(path1)
        data2 = self.load_checkpoint(path2)

        if data1 is None or data2 is None:
            print("ERROR: Could not load checkpoints")
            return

        # Get language_features
        lang_feat1 = data1.get('language_features')
        lang_feat2 = data2.get('language_features')

        if lang_feat1 is None or lang_feat2 is None:
            print("ERROR: language_features not found in checkpoints")
            return

        num_samples = lang_feat1.shape[0]

        # Analyze each individually
        print(f"\n{'='*70}")
        print(f"INDIVIDUAL ANALYSIS")
        print(f"{'='*70}")

        results1 = self.analyze_tensor(lang_feat1, 'language_features')
        results2 = self.analyze_tensor(lang_feat2, 'language_features')

        self.print_tensor_summary(results1, 'File 1: language_features')
        self.print_tensor_summary(results2, 'File 2: language_features')

        # Print summary of collapsed layers
        print(f"\n{'='*70}")
        print(f"COLLAPSED LAYERS SUMMARY")
        print(f"{'='*70}")

        if results1['is_collapsed']:
            print(f"File 1: ⚠️ COLLAPSED (collapsed dims: {results1.get('collapsed_dims', [])})")
        else:
            print(f"File 1: ✓ OK")

        if results2['is_collapsed']:
            print(f"File 2: ⚠️ COLLAPSED (collapsed dims: {results2.get('collapsed_dims', [])})")
        else:
            print(f"File 2: ✓ OK")

        # Compare language_features
        print(f"\n{'='*70}")
        print(f"LANGUAGE FEATURES COMPARISON")
        print(f"{'='*70}")

        comp = self.compare_tensors(lang_feat1, lang_feat2, 'language_features', num_samples)
        if comp:
            self.print_per_dimension_comparison(comp)

        return {
            'results1': results1,
            'results2': results2,
            'comparison': comp,
        }

    def find_common_nonzero_rows(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        threshold: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Find rows that are non-zero in both feature matrices.

        A row is non-zero if its L2 norm is above threshold.

        Args:
            feat_a: [N_a, D] feature matrix
            feat_b: [N_b, D] feature matrix
            threshold: Threshold for considering a value as non-zero

        Returns:
            (nonzero_a, nonzero_b, info): Filtered feature matrices with only common non-zero rows
        """
        # Check if both have same number of rows
        original_rows_a = feat_a.shape[0]
        original_rows_b = feat_b.shape[0]

        if feat_a.shape[0] != feat_b.shape[0]:
            print(f"Warning: Different row counts - feat_a: {feat_a.shape[0]}, feat_b: {feat_b.shape[0]}")
            min_rows = min(feat_a.shape[0], feat_b.shape[0])
            feat_a = feat_a[:min_rows]
            feat_b = feat_b[:min_rows]

        # Find non-zero rows in each matrix using L2 norm
        norm_a = torch.norm(feat_a, p=2, dim=1)  # [N]
        norm_b = torch.norm(feat_b, p=2, dim=1)  # [N]

        nonzero_mask_a = norm_a > threshold
        nonzero_mask_b = norm_b > threshold

        # Common non-zero rows
        common_mask = nonzero_mask_a & nonzero_mask_b

        info = {
            'original_rows_a': original_rows_a,
            'original_rows_b': original_rows_b,
            'nonzero_rows_a': nonzero_mask_a.sum().item(),
            'nonzero_rows_b': nonzero_mask_b.sum().item(),
            'common_rows': common_mask.sum().item(),
            'total_rows': len(common_mask),
        }

        print(f"  Non-zero rows in feat_a: {info['nonzero_rows_a']}/{info['total_rows']}")
        print(f"  Non-zero rows in feat_b: {info['nonzero_rows_b']}/{info['total_rows']}")
        print(f"  Common non-zero rows: {info['common_rows']}/{info['total_rows']}")

        return feat_a[common_mask], feat_b[common_mask], info

    def compute_l1_loss(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
        """Compute L1 loss (mean absolute error) between two feature matrices."""
        return torch.abs(feat_a - feat_b).mean().item()

    def compute_cosine_similarity_loss(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
        """
        Compute cosine similarity and cosine similarity loss (1 - cosine similarity).

        Returns:
            (mean_cosine_sim, cosine_loss, per_row_cosine_sim): Mean cosine similarity, loss, and per-row values
        """
        # Compute cosine similarity for each row
        # cos_sim = (a · b) / (||a|| * ||b||)
        dot_product = (feat_a * feat_b).sum(dim=1)  # [N]
        norm_a = torch.norm(feat_a, p=2, dim=1)  # [N]
        norm_b = torch.norm(feat_b, p=2, dim=1)  # [N]

        # Avoid division by zero
        cosine_sim = dot_product / (norm_a * norm_b + 1e-8)

        # Mean cosine similarity
        mean_cosine_sim = cosine_sim.mean().item()
        cosine_loss = 1.0 - mean_cosine_sim

        return mean_cosine_sim, cosine_loss, cosine_sim

    def compare_nonzero_checkpoints(self, path1: str, path2: str, threshold: float = 1e-6):
        """
        Compare two checkpoints using only common non-zero rows.

        This method filters out zero rows before computing metrics.
        """
        print(f"\n{'='*70}")
        print(f"NON-ZERO ROWS COMPARISON")
        print(f"{'='*70}")
        print(f"File 1: {path1}")
        print(f"File 2: {path2}")
        print(f"Zero threshold: {threshold}")

        # Load checkpoints
        data1 = self.load_checkpoint(path1)
        data2 = self.load_checkpoint(path2)

        if data1 is None or data2 is None:
            print("ERROR: Could not load checkpoints")
            return None

        lang_feat1 = data1.get('language_features')
        lang_feat2 = data2.get('language_features')

        if lang_feat1 is None or lang_feat2 is None:
            print("ERROR: language_features not found")
            return None

        # Find common non-zero rows
        print(f"\nFinding common non-zero rows...")
        feat1_common, feat2_common, info = self.find_common_nonzero_rows(
            lang_feat1, lang_feat2, threshold
        )

        if feat1_common.shape[0] == 0:
            print("\nERROR: No common non-zero rows found!")
            return None

        # Compute losses
        l1_loss = self.compute_l1_loss(feat1_common, feat2_common)
        cosine_sim, cosine_loss, per_row_cosine = self.compute_cosine_similarity_loss(
            feat1_common, feat2_common
        )

        # Print results
        print(f"\n{'='*70}")
        print(f"RESULTS (Common Non-Zero Rows Only)")
        print(f"{'='*70}")
        print(f"Common non-zero rows: {feat1_common.shape[0]}")
        print(f"Feature dimension: {feat1_common.shape[1]}")
        print(f"L1 Loss: {l1_loss:.6f}")
        print(f"Cosine Similarity: {cosine_sim:.6f}")
        print(f"Cosine Loss (1 - cos): {cosine_loss:.6f}")

        # Additional statistics
        print(f"\nPer-row cosine similarity statistics:")
        print(f"  Mean: {per_row_cosine.mean().item():.6f}")
        print(f"  Std: {per_row_cosine.std().item():.6f}")
        print(f"  Min: {per_row_cosine.min().item():.6f}")
        print(f"  Max: {per_row_cosine.max().item():.6f}")
        print(f"  Median: {per_row_cosine.median().item():.6f}")

        return {
            'common_rows': feat1_common.shape[0],
            'feature_dim': feat1_common.shape[1],
            'l1_loss': l1_loss,
            'cosine_similarity': cosine_sim,
            'cosine_loss': cosine_loss,
            'per_row_cosine': per_row_cosine,
            'info': info,
        }

    def batch_compare_scenes(
        self,
        output_dir: str,
        pattern_p: str = "checkpoint_with_features_p.pth",
        pattern: str = "checkpoint_with_features.pth",
        threshold: float = 1e-6
    ):
        """
        Batch compare all scenes in the output_features directory.

        Finds all scenes with both checkpoint patterns and compares them.
        """
        output_path = Path(output_dir)

        if not output_path.exists():
            print(f"ERROR: Output directory not found: {output_dir}")
            return None

        print(f"\n{'='*70}")
        print(f"BATCH SCENE COMPARISON")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}")
        print(f"Pattern 1: {pattern_p}")
        print(f"Pattern 2: {pattern}")

        # Find all scenes with both checkpoint types
        scenes = []
        for scene_dir in output_path.iterdir():
            if not scene_dir.is_dir():
                continue

            checkpoint_p = scene_dir / pattern_p
            checkpoint = scene_dir / pattern

            if checkpoint_p.exists() and checkpoint.exists():
                scenes.append((scene_dir.name, str(checkpoint_p), str(checkpoint)))

        print(f"\nFound {len(scenes)} scenes with both checkpoint types")

        if not scenes:
            print("No matching scenes found!")
            return None

        results = []
        for scene_name, checkpoint_p, checkpoint in scenes:
            try:
                print(f"\n{'='*70}")
                print(f"Processing: {scene_name}")
                print(f"{'='*70}")

                result = self.compare_nonzero_checkpoints(
                    checkpoint_p, checkpoint, threshold
                )

                if result is not None:
                    result['scene'] = scene_name
                    results.append(result)
            except Exception as e:
                print(f"Error processing {scene_name}: {e}")
                import traceback
                traceback.print_exc()

        # Print summary
        if results:
            self._print_batch_summary(results)

        return results

    def _print_batch_summary(self, results: List[Dict]):
        """Print summary table for batch comparison results."""
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Scene':<20} {'Rows':<10} {'L1 Loss':<12} {'Cosine Sim':<12} {'Cosine Loss':<12}")
        print("-" * 66)

        for r in results:
            if r["common_rows"] > 0:
                print(f"{r['scene']:<20} {r['common_rows']:<10} "
                      f"{r['l1_loss']:<12.6f} {r['cosine_similarity']:<12.6f} {r['cosine_loss']:<12.6f}")
            else:
                print(f"{r['scene']:<20} {r['common_rows']:<10} "
                      f"{'N/A':<12} {'N/A':<12} {'N/A':<12}")

        # Calculate averages
        valid_results = [r for r in results if r["common_rows"] > 0]
        if valid_results:
            avg_l1 = np.mean([r["l1_loss"] for r in valid_results])
            avg_cosine_sim = np.mean([r["cosine_similarity"] for r in valid_results])
            avg_cosine_loss = np.mean([r["cosine_loss"] for r in valid_results])

            print("-" * 66)
            print(f"{'Average':<20} {'':<10} {avg_l1:<12.6f} {avg_cosine_sim:<12.6f} {avg_cosine_loss:<12.6f}")

    def analyze_value_ranges(self, path: str):
        """Analyze detailed value ranges for language features."""
        data = self.load_checkpoint(path)
        if data is None or 'language_features' not in data:
            print(f"ERROR: language_features not found")
            return

        tensor = data['language_features']

        print(f"\n{'='*80}")
        print(f"DETAILED VALUE RANGE ANALYSIS: language_features")
        print(f"{'='*80}")
        print(f"Shape: {tensor.shape}")

        num_dims = tensor.shape[-1] if len(tensor.shape) >= 2 else 1
        num_samples = tensor.shape[0]

        print(f"\nAnalyzing {num_dims} dimensions across {num_samples} samples")

        # Per-dimension statistics
        print(f"\n{'='*80}")
        print(f"Value Range Per Dimension")
        print(f"{'='*80}")
        print(f"{'Dim':<4} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12} {'Median':<12} {'Variance':<12} {'Status'}")
        print(f"{'-'*80}")

        for dim_idx in range(num_dims):
            if len(tensor.shape) >= 2:
                dim_data = tensor[:, dim_idx].float()
            else:
                dim_data = tensor.flatten().float()

            min_val = dim_data.min().item()
            max_val = dim_data.max().item()
            mean_val = dim_data.mean().item()
            std_val = dim_data.std().item()
            median_val = dim_data.median().item()
            var_val = dim_data.var().item()

            if var_val < 1e-6:
                status = "✗ Collapsed"
            elif var_val < 1e-4:
                status = "⚠️ Near-collapse"
            elif std_val < 0.01:
                status = "⚠️ Low variance"
            else:
                status = "✓ Healthy"

            print(f"{dim_idx:<4} {min_val:<12.6f} {max_val:<12.6f} {mean_val:<12.6f} {std_val:<12.6f} {median_val:<12.6f} {var_val:<12.8f} {status}")

        # Sample diversity
        print(f"\n{'='*80}")
        print(f"Sample Diversity Analysis")
        print(f"{'='*80}")

        unique_samples = torch.unique(tensor, dim=0).shape[0]
        print(f"Unique samples: {unique_samples}/{num_samples} ({100*unique_samples/num_samples:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Model Collapse Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two checkpoints (all samples)
  %(prog)s --path1 checkpoint1.pth --path2 checkpoint2.pth

  # Compare two checkpoints (common non-zero rows only)
  %(prog)s --path1 checkpoint_with_features_p.pth --path2 checkpoint_with_features.pth --nonzero-only

  # Batch compare all scenes in output_features directory
  %(prog)s --batch-compare --output-dir output_features

  # Analyze single checkpoint
  %(prog)s --checkpoint model.pth

  # Analyze detailed value ranges
  %(prog)s --checkpoint model.pth --analyze-ranges
        """
    )

    # Path arguments
    parser.add_argument('--path1', type=str, help='First checkpoint path')
    parser.add_argument('--path2', type=str, help='Second checkpoint path (for comparison)')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path (alias for path1)')

    # Batch comparison arguments
    parser.add_argument('--batch-compare', action='store_true',
                        help='Batch compare all scenes in output directory')
    parser.add_argument('--output-dir', type=str, default='output_features',
                        help='Output directory containing scene checkpoints (default: output_features)')
    parser.add_argument('--pattern-p', type=str, default='checkpoint_with_features_p.pth',
                        help='Pattern for first checkpoint type (default: checkpoint_with_features_p.pth)')
    parser.add_argument('--pattern', type=str, default='checkpoint_with_features.pth',
                        help='Pattern for second checkpoint type (default: checkpoint_with_features.pth)')

    # Comparison mode arguments
    parser.add_argument('--nonzero-only', action='store_true',
                        help='Compare only common non-zero rows (filters out zero rows)')
    parser.add_argument('--zero-threshold', type=float, default=1e-6,
                        help='L2 norm threshold for considering a row as non-zero (default: 1e-6)')

    # Analysis arguments
    parser.add_argument('--analyze-ranges', action='store_true',
                        help='Analyze detailed value ranges for language features')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (default: cpu)')

    args = parser.parse_args()

    # Create analyzer
    analyzer = CollapseAnalyzer(device=args.device)

    # Handle batch comparison mode
    if args.batch_compare:
        analyzer.batch_compare_scenes(
            args.output_dir,
            args.pattern_p,
            args.pattern,
            args.zero_threshold
        )
        return

    # Determine paths for single/batch comparison
    path1 = args.path1 or args.checkpoint
    path2 = args.path2

    if not path1:
        # Use default paths if no paths specified
        output_dir = Path('/new_data/cyf/projects/SceneSplat/output_features/bed')
        path1 = str(output_dir / 'checkpoint_with_features_s.pth')
        path2 = str(output_dir / 'checkpoint_with_features.pth')
        print(f"Using default paths:")
        print(f"  File 1: {path1}")
        print(f"  File 2: {path2}")

    # Handle different analysis modes
    if args.analyze_ranges:
        # Analyze value ranges for single checkpoint
        analyzer.analyze_value_ranges(path1)
    elif path2:
        # Compare two checkpoints
        if args.nonzero_only:
            # Compare using only common non-zero rows
            analyzer.compare_nonzero_checkpoints(path1, path2, args.zero_threshold)
        else:
            # Compare all samples
            analyzer.compare_checkpoints(path1, path2)
    else:
        # Analyze single checkpoint
        analyzer.analyze_checkpoint(path1)


if __name__ == "__main__":
    main()
