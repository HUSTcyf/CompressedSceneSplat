#!/usr/bin/env python3
"""
SVD Decomposition Tool for Linear Layer Weights

Applies Singular Value Decomposition (SVD) to all linear layer weights in a checkpoint
to analyze potential compression through low-rank approximation.

Usage:
    python tools/apply_svd.py --checkpoint /path/to/checkpoint.pth --energy_threshold 0.9
    python tools/apply_svd.py --checkpoint /path/to/checkpoint.pth --energy_threshold 0.95 --output ./checkpoints
"""

import argparse
import torch
import pickle
from pathlib import Path
import sys
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_linear_layers(state_dict):
    """
    Extract all linear layer weights (2D tensors) from state_dict.

    Args:
        state_dict: Model state_dict

    Returns:
        Dictionary mapping key -> tensor for all linear layers
    """
    linear_layers = {}

    for key, tensor in state_dict.items():
        if not hasattr(tensor, 'numel'):
            continue

        # Linear layers have 2D weight tensors
        clean_key = key.replace('module.', '')

        if len(tensor.shape) == 2 and 'weight' in clean_key:
            # Exclude convolution weights (handled by shape check above)
            # Also exclude embeddings which might be 2D but are not "linear" layers in the traditional sense
            if 'embed' not in clean_key.lower():
                linear_layers[key] = tensor

    return linear_layers


def compute_svd_with_energy_threshold(weight, energy_threshold=0.9):
    """
    Compute SVD and determine rank needed to reach energy threshold.

    Args:
        weight: Weight tensor [out_features, in_features]
        energy_threshold: Fraction of singular value energy to preserve (0-1)

    Returns:
        dict containing:
            - U: Left singular vectors
            - S: Singular values
            - Vh: Right singular vectors (transposed)
            - rank_90: Rank needed for energy_threshold
            - energy_cumsum: Cumulative energy distribution
            - original_shape: Original weight shape
    """
    # Compute SVD
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    # Compute energy (squared singular values)
    energy = S ** 2
    total_energy = energy.sum()
    energy_cumsum = torch.cumsum(energy, dim=0) / total_energy

    # Find rank for energy threshold
    rank_90 = (energy_cumsum >= energy_threshold).nonzero(as_tuple=True)[0][0].item() + 1

    return {
        'U': U,
        'S': S,
        'Vh': Vh,
        'rank_90': rank_90,
        'energy_cumsum': energy_cumsum,
        'original_shape': weight.shape,
    }


def compute_compressed_size(shape, rank, dtype_size=4):
    """
    Compute size of compressed weight using low-rank approximation.

    For a weight matrix W of shape (m, n), rank-r approximation:
    W ≈ U_r @ diag(S_r) @ Vh_r
    where U_r is (m, r), S_r is (r,), Vh_r is (r, n)

    Storage: m*r + r + r*n = r * (m + n + 1)

    Args:
        shape: Original weight shape (m, n)
        rank: Target rank
        dtype_size: Size per element in bytes (default: 4 for float32)

    Returns:
        Size in bytes
    """
    m, n = shape
    # U: m x rank, S: rank, Vh: rank x n
    compressed_elements = (m * rank) + rank + (rank * n)
    return compressed_elements * dtype_size


def analyze_svd_for_layers(state_dict, energy_threshold=0.9):
    """
    Analyze SVD compression for all linear layers.

    Args:
        state_dict: Model state_dict
        energy_threshold: Energy threshold for rank selection

    Returns:
        Dictionary with analysis results for each layer
    """
    linear_layers = get_linear_layers(state_dict)
    results = {}

    print(f"\nFound {len(linear_layers)} linear layer weights")
    print("=" * 120)

    # Header
    print(f"{'Layer Name':<50} {'Shape':>20} {'Original':>12} {'Rank@{:.0f}%':>12} {'Compressed':>12} {'Ratio':>10}".format(
        energy_threshold * 100))
    print("-" * 120)

    total_original = 0
    total_compressed = 0

    for key, weight in sorted(linear_layers.items(), key=lambda x: x[1].numel(), reverse=True):
        # Compute SVD
        svd_result = compute_svd_with_energy_threshold(weight, energy_threshold)

        m, n = svd_result['original_shape']
        rank = svd_result['rank_90']

        # Sizes
        original_size = weight.numel() * 4  # float32
        compressed_size = compute_compressed_size((m, n), rank, 4)

        total_original += original_size
        total_compressed += compressed_size

        # Compression ratio
        ratio = compressed_size / original_size

        # Format layer name (remove module. prefix for readability)
        display_name = key.replace('module.', '')
        if len(display_name) > 48:
            display_name = '...' + display_name[-45:]

        print(f"{display_name:<50} {str(svd_result['original_shape']):>20} "
              f"{original_size/1024/1024:>10.2f}MB "
              f"{rank:>12} "
              f"{compressed_size/1024/1024:>10.2f}MB "
              f"{ratio:>9.3f}")

        results[key] = {
            'weight': weight,
            'svd_result': svd_result,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': ratio,
        }

    print("-" * 120)
    print(f"{'TOTAL':<50} {'':>20} {total_original/1024/1024:>10.2f}MB "
          f"{'':>12} {total_compressed/1024/1028:>10.2f}MB "
          f"{total_compressed/total_original:>9.3f}")

    overall_ratio = total_compressed / total_original
    print(f"\nOverall compression ratio: {overall_ratio:.4f} (saving {(1-overall_ratio)*100:.2f}%)")
    print(f"Space saved: {(total_original - total_compressed) / 1024 / 1024:.2f} MB")

    return results


def create_svd_compressed_state_dict(state_dict, svd_results, energy_threshold=0.9):
    """
    Create a new state_dict with SVD-compressed linear layers.

    For each linear layer weight W, store U_r, S_r, Vh_r instead of W.
    The keys become:
        original_key -> U_r
        original_key + '_svd_S' -> S_r
        original_key + '_svd_Vh' -> Vh_r

    Args:
        state_dict: Original state_dict
        svd_results: Results from analyze_svd_for_layers
        energy_threshold: Energy threshold used for rank selection

    Returns:
        New state_dict with compressed layers
    """
    new_state_dict = {}

    # Copy non-linear layer parameters as-is
    for key, tensor in state_dict.items():
        if key not in svd_results:
            new_state_dict[key] = tensor

    # Add SVD-compressed linear layers
    for key, result in svd_results.items():
        svd = result['svd_result']
        rank = svd['rank_90']

        # Truncate to target rank
        U_r = svd['U'][:, :rank]  # [m, rank]
        S_r = svd['S'][:rank]     # [rank]
        Vh_r = svd['Vh'][:rank, :]  # [rank, n]

        # Store as separate tensors
        base_key = key
        new_state_dict[base_key] = U_r
        new_state_dict[base_key + '_svd_S'] = S_r
        new_state_dict[base_key + '_svd_Vh'] = Vh_r

    return new_state_dict


def reconstruct_weight(U, S, Vh):
    """
    Reconstruct weight matrix from SVD components.

    Args:
        U: Left singular vectors [m, rank]
        S: Singular values [rank]
        Vh: Right singular vectors [rank, n]

    Returns:
        Reconstructed weight matrix [m, n]
    """
    return U @ torch.diag(S) @ Vh


def main():
    parser = argparse.ArgumentParser(description="Apply SVD decomposition to linear layer weights")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to input checkpoint file (.pth)",
    )
    parser.add_argument(
        "--energy_threshold",
        type=float,
        default=0.9,
        help="Energy threshold for SVD rank selection (default: 0.9 = 90%%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for compressed checkpoint (default: same as input)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the compressed checkpoint to disk",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-layer statistics",
    )
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Get state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        meta_keys = [k for k in checkpoint.keys() if k != 'state_dict']
    else:
        state_dict = checkpoint
        meta_keys = []

    print(f"State dict loaded with {len(state_dict)} parameters")

    # Analyze SVD for all linear layers
    svd_results = analyze_svd_for_layers(state_dict, args.energy_threshold)

    # Show detailed statistics if requested
    if args.verbose:
        print("\n" + "=" * 120)
        print("Detailed Per-Layer Statistics")
        print("=" * 120)

        for key, result in sorted(svd_results.items(), key=lambda x: x[1]['original_size'], reverse=True):
            svd = result['svd_result']
            m, n = svd['original_shape']
            rank = svd['rank_90']
            max_rank = min(m, n)

            display_name = key.replace('module.', '')
            print(f"\n[{display_name}]")
            print(f"  Original shape: {svd['original_shape']}")
            print(f"  Max possible rank: {max_rank}")
            print(f"  Rank for {args.energy_threshold*100:.0f}% energy: {rank} ({rank/max_rank*100:.1f}% of max)")
            print(f"  Original size: {result['original_size'] / 1024 / 1024:.2f} MB")
            print(f"  Compressed size: {result['compressed_size'] / 1024 / 1024:.2f} MB")
            print(f"  Compression ratio: {result['compression_ratio']:.4f}")
            print(f"  Space saved: {(1 - result['compression_ratio']) * 100:.2f}%")

    # Save compressed checkpoint if requested
    if args.save:
        # Determine output path
        input_path = Path(args.checkpoint)
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = input_path.parent

        output_name = input_path.stem + '-svd.pth'
        output_path = output_dir / output_name

        print(f"\nCreating SVD-compressed checkpoint...")

        # Create new state_dict with SVD components
        compressed_state_dict = create_svd_compressed_state_dict(state_dict, svd_results, args.energy_threshold)

        # Create new checkpoint
        new_checkpoint = {}
        if meta_keys:
            # Copy metadata
            for key in meta_keys:
                new_checkpoint[key] = checkpoint[key]
        new_checkpoint['state_dict'] = compressed_state_dict
        new_checkpoint['svd_compression'] = {
            'energy_threshold': args.energy_threshold,
            'original_size_mb': sum(r['original_size'] for r in svd_results.values()) / 1024 / 1024,
            'compressed_size_mb': sum(r['compressed_size'] for r in svd_results.values()) / 1024 / 1024,
        }

        # Save
        torch.save(new_checkpoint, output_path)
        print(f"Compressed checkpoint saved to: {output_path}")

        # Verify reconstruction
        print("\nVerifying reconstruction accuracy...")
        max_error = 0
        for key, result in svd_results.items():
            svd = result['svd_result']
            rank = svd['rank_90']

            # Reconstruct
            U_r = svd['U'][:, :rank]
            S_r = svd['S'][:rank]
            Vh_r = svd['Vh'][:rank, :]
            reconstructed = reconstruct_weight(U_r, S_r, Vh_r)

            # Compute error
            original = result['weight']
            error = torch.norm(original - reconstructed, p='fro').item() / torch.norm(original, p='fro').item()
            max_error = max(max_error, error)

        print(f"Maximum relative reconstruction error: {max_error:.6f}")


if __name__ == "__main__":
    main()
