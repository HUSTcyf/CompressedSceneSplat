#!/usr/bin/env python3
"""
SVD Analysis Tool for Language Features

Analyzes Singular Value Decomposition of language features from OccamLGS format checkpoints.

Usage:
    python tools/analyze_svd.py --checkpoint output_features/teatime_iteration_18000_point_cloud_langfeat.pth
    python tools/analyze_svd.py --npy gaussian_train/scene0011_00/lang_feat.npy
    python tools/analyze_svd.py --batch_dir gaussian_train --pattern "*/lang_feat.npy"
    python tools/analyze_svd.py --batch_dir gaussian_train --use_rpca --rank 32
    python tools/analyze_svd.py --npy lang_feat.npy --gpu 1 --use_rpca
    python tools/analyze_svd.py --npy lang_feat.npy --save_decomp  # Save SVD decomposition
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import os
from typing import List, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import RPCA utilities
from tools.rpca_utils import (
    RPCA_GPU,
    RPCA_CPU,
    apply_rpca,
    RPCA_CPU_AVAILABLE,
    CUDA_AVAILABLE,
)

def print_device_info(gpu_id: Optional[int] = None):
    """Print GPU availability and device information."""
    # Handle CPU mode (gpu_id == -1)
    if gpu_id is not None and gpu_id == -1:
        print("Device: CPU (RPCA via pyrpca library)")
        print(f"RPCA (CPU via pyrpca) available: {RPCA_CPU_AVAILABLE}")
        print(f"Total CPU threads available: {os.cpu_count()}")
        return

    if CUDA_AVAILABLE:
        if gpu_id is not None:
            if gpu_id < 0:
                print(f"Warning: Invalid GPU ID {gpu_id}, will use GPU 0")
                gpu_id = 0
            if gpu_id >= torch.cuda.device_count():
                print(f"Warning: GPU {gpu_id} not found. Available GPUs: {torch.cuda.device_count()}")
                print(f"Will use GPU 0 instead")
                gpu_id = 0
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print(f"GPU (CUDA) available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU (CUDA) not available, will use CPU for RPCA if pyrpca is installed")
    print(f"RPCA (CPU via pyrpca) available: {RPCA_CPU_AVAILABLE}")


def extract_lang_feat(checkpoint_path: str) -> torch.Tensor:
    """
    Extract language features from an OccamLGS format checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Language features tensor [N, feat_dim]
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Parse checkpoint format
    if isinstance(ckpt, tuple) and len(ckpt) == 2:
        model_state, iteration = ckpt
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        model_state = ckpt["model_state"]
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    # Check model state length
    num_items = len(model_state)
    print(f"Model state: {num_items} items")

    if num_items == 13:
        # 13-item format with language features
        lang_feat = model_state[7]
        print(f"Found lang_feat at index 7: {lang_feat.shape}")
    elif num_items == 12:
        raise ValueError("Checkpoint has 12 items (no language features)")
    else:
        raise ValueError(f"Expected 13 items in model_state, got {num_items}")

    return lang_feat


def load_feat_from_npy(npy_path: str) -> torch.Tensor:
    """
    Load features from a .npy file, optionally filtering with valid_feat_mask.npy.

    Args:
        npy_path: Path to the .npy file

    Returns:
        Feature tensor [N, feat_dim]
    """
    print(f"Loading features from: {npy_path}")

    # Handle .npz files (SceneSplat format with 'arr_0' key)
    if str(npy_path).endswith('.npz'):
        feat_np = np.load(npy_path)['arr_0']
    else:
        feat_np = np.load(npy_path)

    print(f"Loaded features shape: {feat_np.shape}, dtype: {feat_np.dtype}")

    # Try to load valid_feat_mask.npy from the same directory
    feat_path = Path(npy_path)
    mask_path = feat_path.parent / "valid_feat_mask.npy"

    if mask_path.exists():
        print(f"Loading valid mask from: {mask_path}")
        mask = np.load(mask_path)
        print(f"Valid mask shape: {mask.shape}, dtype: {mask.dtype}")

        # Check if mask matches feature dimension
        if len(mask) != feat_np.shape[0]:
            print(f"Warning: Mask length ({len(mask)}) doesn't match feature rows ({feat_np.shape[0]})")
            print("  Skipping mask application")
        else:
            # Count valid features
            valid_count = mask.sum()
            total_count = len(mask)
            print(f"Valid features: {valid_count}/{total_count} ({100.0 * valid_count / total_count:.2f}%)")

            # Filter features using valid mask
            feat_np = feat_np[mask > 0]
            print(f"Filtered features shape: {feat_np.shape}")
    else:
        print(f"No valid_feat_mask.npy found at {mask_path}, using all features")

    # Convert to torch tensor
    feat_tensor = torch.from_numpy(feat_np).float()

    return feat_tensor


def find_npy_files(
    batch_dir: str,
    pattern: str = "**/lang_feat.npy",
    max_files: Optional[int] = None,
) -> List[Path]:
    """
    Find all .npy files matching the pattern in the batch directory.

    Args:
        batch_dir: Root directory to search
        pattern: Glob pattern for matching files (supports wildcards)
        max_files: Maximum number of files to process (None for all)

    Returns:
        List of Path objects for matching files
    """
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        raise ValueError(f"Batch directory does not exist: {batch_dir}")

    print(f"Searching for '{pattern}' in {batch_dir}...")

    # Use glob to find matching files
    matching_files = sorted(batch_path.glob(pattern))

    # Filter to only .npy and .npz files
    matching_files = [f for f in matching_files if f.suffix in ['.npy', '.npz']]

    if max_files:
        matching_files = matching_files[:max_files]

    print(f"Found {len(matching_files)} files")
    return matching_files


def print_batch_summary(all_results: List[Dict], target_rank: int):
    """Print batch processing summary."""
    print("\n" + "=" * 100)
    print(f"{'Batch Processing Summary':^100}")
    print("=" * 100)

    if not all_results:
        print("\nNo results to display (all files failed to process)")
        print("=" * 100)
        return

    print(f"\n{'File':<60} {'N':<8} {'D':<6} {'Rank Energy':<12} {'Recon Error':<12}")
    print("-" * 100)

    for file_path, results in all_results:
        N, D = results['shape']
        rank_ratio = results['target_rank_energy_ratio']
        error = results['reconstruction_error']
        use_rpca = results.get('use_rpca', False)

        # Truncate file name for display
        display_name = str(file_path)
        if len(display_name) > 57:
            display_name = "..." + display_name[-54:]

        rpca_marker = " [RPCA]" if use_rpca else ""
        print(f"{display_name:<60} {N:<8} {D:<6} {rank_ratio:>11.1%} {error:>12.6f}{rpca_marker}")

    print("-" * 100)

    # Compute statistics
    rank_ratios = [r['target_rank_energy_ratio'] for _, r in all_results]
    errors = [r['reconstruction_error'] for _, r in all_results]
    shapes = [r['shape'] for _, r in all_results]

    print(f"\nStatistics across {len(all_results)} files:")
    print(f"  Rank {target_rank} Energy Ratio: {np.mean(rank_ratios):.4f} ± {np.std(rank_ratios):.4f}")
    print(f"  Reconstruction Error: {np.mean(errors):.6f} ± {np.std(errors):.6f}")
    print(f"  Matrix sizes: N=[{min(s[0] for s in shapes)}, {max(s[0] for s in shapes)}], "
          f"D={[min(s[1] for s in shapes), max(s[1] for s in shapes)]}")

    print("=" * 100)


def save_batch_results_csv(all_results: List[Dict], output_path: str, target_rank: int):
    """Save batch results to CSV file."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'file_path', 'N', 'D', 'target_rank', 'rank_energy_ratio',
            'reconstruction_error', 'condition_number', 'use_rpca'
        ])

        for file_path, results in all_results:
            N, D = results['shape']
            S = results['S']
            use_rpca = results.get('use_rpca', False)

            writer.writerow([
                str(file_path),
                N,
                D,
                target_rank,
                f"{results['target_rank_energy_ratio']:.6f}",
                f"{results['reconstruction_error']:.8f}",
                f"{S[0] / S[-1]:.4e}",
                use_rpca,
            ])

    print(f"Saved batch results to: {output_path}")


def analyze_svd(features: torch.Tensor, target_rank: int = 16, use_rpca: bool = False,
                rpca_max_iter: int = 10, rpca_tol: float = 1e-7, device: Optional[str] = None,
                n_threads: Optional[int] = None):
    """
    Perform SVD analysis on the feature matrix.

    Args:
        features: Feature tensor [N, D]
        target_rank: Target rank for low-rank approximation
        use_rpca: Whether to apply RPCA before SVD
        rpca_max_iter: Maximum iterations for RPCA
        rpca_tol: Tolerance for RPCA convergence
        device: Device string for RPCA (e.g., 'cuda:0', 'cuda:1', 'cpu', None for auto)
        n_threads: Number of threads for CPU RPCA (None for auto-detect)

    Returns:
        Dictionary with SVD analysis results
    """
    N, D = features.shape
    print(f"\nFeature matrix shape: [{N}, {D}]")

    # Convert to numpy for SVD (detach if requires grad)
    if features.requires_grad:
        features = features.detach()
    feat_np = features.cpu().numpy().astype(np.float64)

    # Apply RPCA if requested
    rpca_results = None
    if use_rpca:
        print(f"\nUsing RPCA with device: {device if device else 'auto-select'}")
        if n_threads:
            print(f"Using {n_threads} threads for CPU RPCA")
        rpca_results = apply_rpca(feat_np, max_iter=rpca_max_iter, tol=rpca_tol,
                                  device=device, n_threads=n_threads)
        feat_np = rpca_results['L']  # Use low-rank matrix for SVD
        print(f"Using RPCA low-rank matrix for SVD analysis")

    print("Computing SVD...")
    U, S, Vt = np.linalg.svd(feat_np, full_matrices=False)

    # Compute energy (explained variance)
    energy = S ** 2
    total_energy = energy.sum()
    energy_cumsum = np.cumsum(energy) / total_energy

    # Find contribution at target rank
    rank_energy_ratio = energy_cumsum[target_rank - 1] if target_rank <= len(S) else 1.0

    # Find rank for common thresholds
    thresholds = [0.90, 0.95, 0.99]
    ranks_for_thresholds = {}
    for thresh in thresholds:
        rank = (energy_cumsum >= thresh).nonzero()[0][0] + 1 if thresh <= energy_cumsum[-1] else len(S)
        ranks_for_thresholds[thresh] = rank

    # Compute low-rank approximation error
    if target_rank < len(S):
        # Reconstruct with target rank
        U_r = U[:, :target_rank]
        S_r = S[:target_rank]
        Vt_r = Vt[:target_rank, :]
        feat_reconstructed = U_r @ np.diag(S_r) @ Vt_r

        # Compute relative error
        error = np.linalg.norm(feat_np - feat_reconstructed, 'fro') / np.linalg.norm(feat_np, 'fro')
    else:
        error = 0.0

    results = {
        'U': U,
        'S': S,
        'Vt': Vt,
        'energy_cumsum': energy_cumsum,
        'total_energy': total_energy,
        'target_rank_energy_ratio': rank_energy_ratio,
        'ranks_for_thresholds': ranks_for_thresholds,
        'reconstruction_error': error,
        'shape': (N, D),
        'use_rpca': use_rpca,
    }

    # Add RPCA results if available
    if rpca_results is not None:
        results['rpca'] = rpca_results

    return results


def print_svd_summary(results: dict, target_rank: int):
    """Print SVD analysis summary."""
    N, D = results['shape']
    S = results['S']
    energy_cumsum = results['energy_cumsum']
    use_rpca = results.get('use_rpca', False)

    print("\n" + "=" * 80)
    title = "SVD Analysis Summary"
    if use_rpca:
        title += " (with RPCA preprocessing)"
    print(title)
    print("=" * 80)

    # Print RPCA info if available
    if use_rpca and 'rpca' in results:
        rpca = results['rpca']
        print(f"\nRPCA Results:")
        print(f"  Estimated rank of low-rank matrix: {rpca['rank']}")
        print(f"  Sparse component nonzeros: {np.count_nonzero(rpca['S'])} / {rpca['S'].size}")

    print(f"\nFeature Matrix: [{N}, {D}]")
    print(f"Original size: {N * D * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"Total singular values: {len(S)}")
    print(f"Max singular value: {S[0]:.6f}")
    print(f"Min singular value: {S[-1]:.6f}")
    print(f"Condition number: {S[0] / S[-1]:.2e}")

    print(f"\n{'Rank':<10} {'Energy Ratio':<15} {'Compressed Size':<20} {'Compression':<15}")
    print("-" * 80)

    # Original
    original_size = N * D * 4
    print(f"{'Original':<10} {'100.0%':<15} {original_size/1024/1024:>10.2f} MB")

    # Target rank
    rank_size = (N * target_rank + target_rank + target_rank * D) * 4
    rank_ratio = results['target_rank_energy_ratio']
    print(f"{target_rank:<10} {rank_ratio:>14.1%} {rank_size/1024/1024:>10.2f} MB         "
          f"{rank_size/original_size:>14.3f}")

    # Common thresholds
    for thresh, rank in results['ranks_for_thresholds'].items():
        thresh_size = (N * rank + rank + rank * D) * 4
        print(f"{rank:<10} {thresh:>14.0%} {thresh_size/1024/1024:>10.2f} MB         "
              f"{thresh_size/original_size:>14.3f}")

    print("-" * 80)

    # Key result: contribution at rank 16
    print(f"\n{'='*80}")
    print(f"Rank {target_rank} Contribution: {rank_ratio:.4f} ({rank_ratio*100:.2f}%)")
    print(f"Reconstruction Error: {results['reconstruction_error']:.6f}")
    print(f"{'='*80}")

    # Singular value distribution
    print("\nTop 20 Singular Values:")
    print(f"{'Rank':<8} {'Value':<15} {'Energy':<15} {'Cumulative':<15}")
    print("-" * 60)
    for i in range(min(20, len(S))):
        energy_i = S[i] ** 2 / results['total_energy']
        print(f"{i+1:<8} {S[i]:<15.6f} {energy_i:>14.2e} {energy_cumsum[i]:>14.2%}")


def save_singular_values_plot(results: dict, output_path: str):
    """Save singular values plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        S = results['S']
        energy_cumsum = results['energy_cumsum']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Singular values (log scale)
        axes[0].semilogy(S, 'b-', linewidth=2)
        axes[0].axvline(x=16, color='r', linestyle='--', label='Rank 16')
        axes[0].set_xlabel('Rank', fontsize=12)
        axes[0].set_ylabel('Singular Value (log scale)', fontsize=12)
        axes[0].set_title('Singular Values', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # 2. Energy contribution
        axes[1].plot(energy_cumsum * 100, 'g-', linewidth=2)
        axes[1].axhline(y=90, color='r', linestyle='--', label='90%')
        axes[1].axhline(y=95, color='orange', linestyle='--', label='95%')
        axes[1].axvline(x=16, color='b', linestyle='--', label='Rank 16')
        axes[1].set_xlabel('Rank', fontsize=12)
        axes[1].set_ylabel('Cumulative Energy (%)', fontsize=12)
        axes[1].set_title('Cumulative Energy Contribution', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # 3. Scree plot (individual energy contribution)
        energy_individual = (S ** 2) / results['total_energy'] * 100
        axes[2].bar(range(1, min(50, len(S)) + 1), energy_individual[:50], alpha=0.7)
        axes[2].axvline(x=16, color='r', linestyle='--', label='Rank 16')
        axes[2].set_xlabel('Rank', fontsize=12)
        axes[2].set_ylabel('Energy Contribution (%)', fontsize=12)
        axes[2].set_title('Individual Energy Contribution (Top 50)', fontsize=14)
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved singular values plot to: {output_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available, skipping plot generation")


def save_svd_decomposition(results: dict, output_path: str):
    """
    Save SVD decomposition (U, S, Vt) to npz file.

    Args:
        results: SVD analysis results dictionary containing U, S, Vt
        output_path: Path to save the npz file (will be saved as .npz)
    """
    output_path = Path(output_path)

    # Change extension to .npz for multiple arrays
    if output_path.suffix == '.npy':
        output_path = output_path.with_suffix('.npz')

    # Save only U, S, Vt matrices
    np.savez_compressed(
        output_path,
        U=results['U'],
        S=results['S'],
        Vt=results['Vt'],
    )

    print(f"\nSaved SVD decomposition to: {output_path}")

    # Print shapes info
    print(f"  U shape: {results['U'].shape}")
    print(f"  S shape: {results['S'].shape}")
    print(f"  Vt shape: {results['Vt'].shape}")

    # Print file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="SVD Analysis for Language Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single checkpoint file
    python tools/analyze_svd.py --checkpoint output_features/teatime_iteration_18000_point_cloud_langfeat.pth

    # Analyze single .npy file
    python tools/analyze_svd.py --npy gaussian_train/scene0011_00/lang_feat.npy

    # Batch process all npy files in a directory
    python tools/analyze_svd.py --batch_dir gaussian_train

    # Batch process with custom pattern
    python tools/analyze_svd.py --batch_dir gaussian_train --pattern "*/lang_feat.npy"

    # Use RPCA preprocessing for batch
    python tools/analyze_svd.py --batch_dir gaussian_train --use_rpca --rank 32

    # Generate plot for single file
    python tools/analyze_svd.py --npy lang_feat.npy --plot --output_dir ./svd_analysis

    # Batch process with plots
    python tools/analyze_svd.py --batch_dir gaussian_train --plot --output_dir ./svd_analysis

    # Batch process with plots, RPCA, and CSV export
    python tools/analyze_svd.py --batch_dir gaussian_train --use_rpca --plot --save_csv --rank 32

    # Save SVD decomposition results
    python tools/analyze_svd.py --npy lang_feat.npy --save_decomp
    python tools/analyze_svd.py --batch_dir gaussian_train --save_decomp
        """
    )

    # Input source (single file modes are mutually exclusive, batch is separate)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file (.pth)",
    )
    input_group.add_argument(
        "--npy",
        type=str,
        help="Path to feature file (.npy or .npz)",
    )

    # Batch processing options
    parser.add_argument(
        "--batch_dir",
        type=str,
        help="Directory for batch processing (searches for lang_feat.npy files)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/lang_feat.npy",
        help="Glob pattern for finding files in batch mode (default: **/lang_feat.npy)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process in batch mode (default: all)",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save batch results to CSV file",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="Target rank for low-rank approximation (default: 16)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./svd_analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate singular values plot (works for both single and batch modes)",
    )
    parser.add_argument(
        "--save_decomp",
        action="store_true",
        help="Save SVD decomposition (U, S, Vt) to npz file in same directory as input",
    )

    # RPCA options
    parser.add_argument(
        "--use_rpca",
        action="store_true",
        help="Apply Robust PCA (RPCA) before SVD to get low-rank matrix",
    )
    parser.add_argument(
        "--rpca_max_iter",
        type=int,
        default=1000,
        help="Maximum iterations for RPCA (default: 1000)",
    )
    parser.add_argument(
        "--rpca_tol",
        type=float,
        default=1e-5,
        help="Tolerance for RPCA convergence (default: 1e-7)",
    )

    # GPU device option
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID for RPCA (default: 0, use -1 for CPU)",
    )

    # Threading option for CPU RPCA
    parser.add_argument(
        "--n_threads",
        type=int,
        default=None,
        help="Number of threads for CPU RPCA (default: auto-detect based on CPU cores)",
    )

    args = parser.parse_args()

    # Determine device string
    device = None
    if args.gpu is not None:
        if args.gpu >= 0:
            device = f'cuda:{args.gpu}'
        else:
            device = 'cpu'

    # Print device info
    print_device_info(args.gpu)

    # Print thread info for CPU mode
    if args.n_threads or device == 'cpu':
        import os
        n_threads = args.n_threads or os.cpu_count()
        print(f"Thread configuration: {n_threads} threads for CPU operations")

    # Validate input arguments
    if not any([args.checkpoint, args.npy, args.batch_dir]):
        parser.error("Must specify one of: --checkpoint, --npy, or --batch_dir")

    # Check RPCA availability if requested
    if args.use_rpca and not (RPCA_CPU_AVAILABLE or CUDA_AVAILABLE):
        print("Error: --use_rpca specified but neither pyrpca (CPU) nor CUDA (GPU) is available.")
        print("For CPU RPCA, install with: pip install pyrpca")
        print("For GPU RPCA, ensure PyTorch is installed with CUDA support")
        return

    # Batch processing mode
    if args.batch_dir:
        # Find all matching files
        npy_files = find_npy_files(args.batch_dir, args.pattern, args.max_files)

        if not npy_files:
            print(f"No files found matching pattern '{args.pattern}' in {args.batch_dir}")
            return

        print(f"\nProcessing {len(npy_files)} files...")
        print("=" * 100)

        # Prepare output directory for plots
        if args.plot:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        for i, file_path in enumerate(npy_files, 1):
            print(f"\n[{i}/{len(npy_files)}] Processing: {file_path}")
            print("-" * 100)

            try:
                # Load features
                lang_feat = load_feat_from_npy(str(file_path))

                # Perform SVD analysis
                results = analyze_svd(
                    lang_feat,
                    args.rank,
                    use_rpca=args.use_rpca,
                    rpca_max_iter=args.rpca_max_iter,
                    rpca_tol=args.rpca_tol,
                    device=device,
                    n_threads=args.n_threads,
                )

                all_results.append((file_path, results))

                # Save SVD decomposition if requested
                if args.save_decomp:
                    # Save in same directory as input file with _svd.npz suffix
                    decomp_path = file_path.parent / f"{file_path.stem}_svd.npz"
                    save_svd_decomposition(results, str(decomp_path))

                # Save plot if requested
                if args.plot:
                    # Create a unique filename based on relative path
                    rel_path = file_path.relative_to(Path(args.batch_dir))
                    # Replace path separators with underscores
                    plot_name = str(rel_path).replace('/', '_').replace('\\', '_')
                    # Remove file extension and add plot suffix
                    plot_name = plot_name.replace('.npy', '').replace('.npz', '')

                    suffix = "_rpca" if args.use_rpca else ""
                    plot_path = output_dir / f"{plot_name}_rank{args.rank}{suffix}_svd.png"
                    save_singular_values_plot(results, str(plot_path))

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        # Print batch summary
        print_batch_summary(all_results, args.rank)

        # Save CSV if requested
        if args.save_csv:
            suffix = "_rpca" if args.use_rpca else ""
            csv_path = output_dir / f"batch_results_rank{args.rank}{suffix}.csv"
            save_batch_results_csv(all_results, str(csv_path), args.rank)

        return

    # Single file mode
    # Load features
    if args.checkpoint:
        lang_feat = extract_lang_feat(args.checkpoint)
        input_name = Path(args.checkpoint).stem
    else:  # args.npy
        lang_feat = load_feat_from_npy(args.npy)
        input_name = Path(args.npy).stem

    # Perform SVD analysis
    results = analyze_svd(
        lang_feat,
        args.rank,
        use_rpca=args.use_rpca,
        rpca_max_iter=args.rpca_max_iter,
        rpca_tol=args.rpca_tol,
        device=device,
        n_threads=args.n_threads,
    )

    # Print summary
    print_svd_summary(results, args.rank)

    # Save SVD decomposition if requested
    if args.save_decomp:
        # Determine input file path for saving alongside
        if args.checkpoint:
            input_path = Path(args.checkpoint)
        else:  # args.npy
            input_path = Path(args.npy)

        # Save in same directory as input file with _svd.npz suffix
        decomp_path = input_path.parent / f"{input_path.stem}_svd.npz"
        save_svd_decomposition(results, str(decomp_path))

    # Save plot if requested
    if args.plot:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = "_rpca" if args.use_rpca else ""
        plot_path = output_dir / f"{input_name}_rank{args.rank}{suffix}_svd.png"
        save_singular_values_plot(results, str(plot_path))


if __name__ == "__main__":
    main()
