#!/usr/bin/env python3
"""
Checkpoint Memory Analysis Tool

Analyzes the memory usage of different layer types in a PyTorch checkpoint.

Usage:
    python tools/analyse_mem.py --checkpoint /path/to/checkpoint.pth
    python tools/analyse_mem.py --checkpoint /path/to/checkpoint.pth --detail
"""

import argparse
import torch
import pickle
from collections import defaultdict
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def analyze_layer_types(state_dict):
    """
    Analyze memory usage by layer types.

    Args:
        state_dict: Model state_dict

    Returns:
        Dictionary with statistics for each layer type
    """
    stats = {
        'linear': {'count': 0, 'bytes': 0, 'keys': []},
        'conv3d': {'count': 0, 'bytes': 0, 'keys': []},
        'conv3d_sparse': {'count': 0, 'bytes': 0, 'keys': []},
        'conv2d': {'count': 0, 'bytes': 0, 'keys': []},
        'conv1d': {'count': 0, 'bytes': 0, 'keys': []},
        'layer_norm': {'count': 0, 'bytes': 0, 'keys': []},
        'batch_norm': {'count': 0, 'bytes': 0, 'keys': []},
        'embedding': {'count': 0, 'bytes': 0, 'keys': []},
        'other': {'count': 0, 'bytes': 0, 'keys': []},
        'total': {'count': 0, 'bytes': 0},
    }

    total_bytes = 0

    for key, tensor in state_dict.items():
        if not hasattr(tensor, 'numel'):
            continue

        # Calculate size in bytes (assuming float32)
        num_params = tensor.numel()
        size_bytes = num_params * 4
        total_bytes += size_bytes

        # Classify by tensor shape and key name
        clean_key = key.replace('module.', '')

        # Detect layer type primarily by tensor shape
        tensor_dim = len(tensor.shape)

        # 3D Convolution: [out_ch, in_ch, d, h, w] - 5D tensor
        if tensor_dim == 5:
            if 'cpe' in clean_key and 'SubM' in str(type(state_dict.get(key, {}))):
                stats['conv3d_sparse']['count'] += 1
                stats['conv3d_sparse']['bytes'] += size_bytes
                stats['conv3d_sparse']['keys'].append((key, tensor.shape))
            else:
                stats['conv3d']['count'] += 1
                stats['conv3d']['bytes'] += size_bytes
                stats['conv3d']['keys'].append((key, tensor.shape))
            stats['total']['count'] += 1
            stats['total']['bytes'] += size_bytes
            continue

        # 2D Convolution: [out_ch, in_ch, h, w] - 4D tensor
        if tensor_dim == 4:
            stats['conv2d']['count'] += 1
            stats['conv2d']['bytes'] += size_bytes
            stats['conv2d']['keys'].append((key, tensor.shape))
            stats['total']['count'] += 1
            stats['total']['bytes'] += size_bytes
            continue

        # 1D Convolution: [out_ch, in_ch, w] - 3D tensor (excluding bias)
        if tensor_dim == 3 and ('weight' in clean_key or 'bias' in clean_key):
            # Check if this is actually a linear layer (fc) by looking at dimensions
            # Linear layers typically have [out_features, in_features] for weight
            if 'fc.' in clean_key or 'mlp.' in clean_key or 'proj.' in clean_key:
                stats['linear']['count'] += 1
                stats['linear']['bytes'] += size_bytes
                stats['linear']['keys'].append((key, tensor.shape))
            else:
                stats['conv1d']['count'] += 1
                stats['conv1d']['bytes'] += size_bytes
                stats['conv1d']['keys'].append((key, tensor.shape))
            stats['total']['count'] += 1
            stats['total']['bytes'] += size_bytes
            continue

        # 2D tensors for weights/biases - Linear layers
        if tensor_dim == 2 and ('weight' in clean_key or 'bias' in clean_key):
            stats['linear']['count'] += 1
            stats['linear']['bytes'] += size_bytes
            stats['linear']['keys'].append((key, tensor.shape))
            stats['total']['count'] += 1
            stats['total']['bytes'] += size_bytes
            continue

        # 1D tensors for norm layers
        if tensor_dim == 1:
            # Layer norm or batch norm
            if any(x in clean_key for x in ['weight', 'bias', 'running_mean', 'running_var']):
                if any(x in clean_key for x in ['ln', 'layer', 'norm.weight', 'norm.bias']):
                    stats['layer_norm']['count'] += 1
                    stats['layer_norm']['bytes'] += size_bytes
                    stats['layer_norm']['keys'].append((key, tensor.shape))
                elif any(x in clean_key for x in ['bn', 'batch']):
                    stats['batch_norm']['count'] += 1
                    stats['batch_norm']['bytes'] += size_bytes
                    stats['batch_norm']['keys'].append((key, tensor.shape))
                else:
                    stats['other']['count'] += 1
                    stats['other']['bytes'] += size_bytes
                    stats['other']['keys'].append((key, tensor.shape))
                stats['total']['count'] += 1
                stats['total']['bytes'] += size_bytes
                continue

        # Embedding layers
        if 'embed' in clean_key:
            stats['embedding']['count'] += 1
            stats['embedding']['bytes'] += size_bytes
            stats['embedding']['keys'].append((key, tensor.shape))
            stats['total']['count'] += 1
            stats['total']['bytes'] += size_bytes
            continue

        # Everything else
        stats['other']['count'] += 1
        stats['other']['bytes'] += size_bytes
        stats['other']['keys'].append((key, tensor.shape))
        stats['total']['count'] += 1
        stats['total']['bytes'] += size_bytes

    return stats, total_bytes


def print_summary(stats, total_bytes, checkpoint_size=None):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("Layer Type Memory Usage Summary")
    print("=" * 80)
    print(f"{'Layer Type':<15} {'Count':>8} {'Size (MB)':>12} {'% of State':>12} {'% of Total':>12}")
    print("-" * 80)

    # Sort by size descending
    layer_types = ['linear', 'conv3d', 'conv2d', 'conv1d', 'layer_norm', 'batch_norm', 'embedding', 'other']

    for layer_type in layer_types:
        data = stats[layer_type]
        if data['count'] == 0:
            continue
        size_mb = data['bytes'] / (1024 ** 2)
        pct_state = (data['bytes'] / total_bytes) * 100
        pct_total = (data['bytes'] / stats['total']['bytes']) * 100
        print(f"{layer_type:<15} {data['count']:>8} {size_mb:>12.2f} {pct_state:>11.1f}% {pct_total:>11.1f}%")

    print("-" * 80)
    total_mb = stats['total']['bytes'] / (1024 ** 2)
    print(f"{'Total':<15} {stats['total']['count']:>8} {total_mb:>12.2f} {100.0:>11.1f}% 100.0%")

    if checkpoint_size:
        checkpoint_mb = checkpoint_size / (1024 ** 2)
        print("\n" + "=" * 80)
        print(f"State Dict:      {total_mb:.2f} MB ({(total_bytes/checkpoint_size)*100:.1f}% of checkpoint)")
        print(f"Full Checkpoint: {checkpoint_mb:.2f} MB")
        print("=" * 80)


def print_detailed_stats(stats, state_dict, max_keys=10):
    """Print detailed statistics for each layer type."""
    print("\n" + "=" * 80)
    print("Detailed Layer Statistics")
    print("=" * 80)

    layer_types = ['linear', 'conv3d', 'conv2d', 'conv1d', 'layer_norm', 'batch_norm', 'embedding', 'other']

    for layer_type in layer_types:
        data = stats[layer_type]
        if data['count'] == 0:
            continue

        print(f"\n[{layer_type.upper()}] - {data['count']} layers, {data['bytes'] / (1024**2):.2f} MB")

        # Show largest layers
        if data['keys']:
            # Sort by size
            sorted_keys = sorted(data['keys'], key=lambda x: x[1].numel(), reverse=True)
            print(f"  Top {min(len(sorted_keys), max_keys)} largest layers:")
            for i, (key, shape) in enumerate(sorted_keys[:max_keys]):
                size_mb = shape.numel() * 4 / (1024 ** 2)
                num_params = shape.numel()
                clean_key = key.replace("module.", "")
                print(f"    {i+1}. {clean_key}")
                print(f"       Shape: {shape}, Params: {num_params:,}, Size: {size_mb:.2f} MB")
            if len(sorted_keys) > max_keys:
                print(f"    ... and {len(sorted_keys) - max_keys} more")


def main():
    parser = argparse.ArgumentParser(description="Analyze checkpoint memory usage")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pth)",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show detailed per-layer statistics",
    )
    parser.add_argument(
        "--max_keys",
        type=int,
        default=10,
        help="Maximum number of keys to show per layer type (default: 10)",
    )
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Get checkpoint size
    checkpoint_size = len(pickle.dumps(checkpoint))

    # Get state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    print(f"State dict loaded with {len(state_dict)} parameters")

    # Analyze
    stats, total_bytes = analyze_layer_types(state_dict)

    # Print summary
    print_summary(stats, total_bytes, checkpoint_size)

    # Print detailed stats if requested
    if args.detail:
        print_detailed_stats(stats, state_dict, args.max_keys)


if __name__ == "__main__":
    main()
