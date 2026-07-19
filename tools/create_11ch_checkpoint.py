#!/usr/bin/env python3
"""
Convert checkpoint from 5-channel (xyz+rgb) to 11-channel (color+opacity+quat+scale) input
by loading model with new config and original weights.

Usage:
    python tools/create_11ch_checkpoint.py \
        --config configs/inference/lang-pretrain-pt-v3m1-3dgs-16.py \
        --weight checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth \
        --output checkpoints/lang-pretrain-pt-v3m1-3dgs-11ch.pth
"""

import os
import sys
import argparse
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pointcept.models import build_model
from pointcept.utils.config import Config


def create_checkpoint(config_path: str, weight_path: str, output_path: str):
    """
    Create a new checkpoint by loading model with new config and original weights.

    Args:
        config_path: Path to the new config file (11-channel input)
        weight_path: Path to the original checkpoint (5-channel input)
        output_path: Path to save the new checkpoint
    """
    print(f"Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)

    print(f"Building model with in_channels=11...")
    model = build_model(cfg.model)
    model.eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    print(f"\nLoading original checkpoint from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)

    # Extract state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        metadata = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        metadata = {k: v for k, v in checkpoint.items() if k != 'model'}
    else:
        state_dict = checkpoint
        metadata = {}

    print(f"  Original state_dict keys: {len(state_dict)}")

    # Remove 'module.' prefix if present (DDP checkpoints)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Load weights with custom handling for mismatched layers
    print("\nLoading weights with custom mismatch handling...")
    loaded_keys = []
    skipped_keys = []
    shape_mismatch_keys = []
    force_skip_keys = []

    model_state_dict = model.state_dict()

    for key, value in new_state_dict.items():
        # Force skip stem layer to use new model's random initialization
        # This ensures the new in_channels=11 setting is respected
        if 'stem' in key and 'embedding' in key:
            force_skip_keys.append((key, model_state_dict[key].shape, value.shape))
            skipped_keys.append(key)
            continue

        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                model_state_dict[key] = value
                loaded_keys.append(key)
            else:
                shape_mismatch_keys.append((key, model_state_dict[key].shape, value.shape))
                skipped_keys.append(key)
        else:
            skipped_keys.append(key)

    # Load the matched weights
    load_result = model.load_state_dict(model_state_dict, strict=False)

    # Print loading results
    print(f"\nLoading results:")
    print(f"  Successfully loaded: {len(loaded_keys)} keys")
    print(f"  Force skipped (stem layer): {len(force_skip_keys)} keys")
    print(f"  Skipped (shape mismatch): {len(shape_mismatch_keys)} keys")
    print(f"  Unexpected keys (not in model): {len(load_result.unexpected_keys)}")
    print(f"  Missing keys (uninitialized): {len(load_result.missing_keys)}")

    # Show force skipped keys (stem layer)
    if force_skip_keys:
        print(f"\n  Force skipped stem layer keys (using new in_channels=11):")
        for k, model_shape, ckpt_shape in force_skip_keys[:3]:
            print(f"    {k}")
            print(f"      Model: {model_shape}, Checkpoint: {ckpt_shape}")

    # Show shape mismatch details
    if shape_mismatch_keys:
        print(f"\n  Shape mismatch keys (will use random initialization):")
        for k, model_shape, ckpt_shape in shape_mismatch_keys[:5]:
            print(f"    {k}")
            print(f"      Model: {model_shape}, Checkpoint: {ckpt_shape}")
        if len(shape_mismatch_keys) > 5:
            print(f"    ... and {len(shape_mismatch_keys) - 5} more")

    # Create new checkpoint
    print(f"\nCreating new checkpoint...")
    new_metadata = {
        **metadata,
        'converted_from': str(weight_path),
        'conversion_type': '5ch_to_11ch',
        'original_input_channels': 5,
        'new_input_channels': 11,
        'config_file': str(config_path),
    }

    new_checkpoint = {
        **new_metadata,
        'state_dict': model.state_dict()
    }

    # Save checkpoint
    print(f"Saving checkpoint to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(new_checkpoint, output_path)

    # Verify the saved checkpoint
    verify = torch.load(output_path, map_location='cpu', weights_only=False)
    verify_state_dict = verify.get('state_dict', verify)
    print(f"\nVerification:")
    print(f"  Saved state_dict keys: {len(verify_state_dict)}")

    # Check stem layer shape
    for k, v in verify_state_dict.items():
        if 'stem' in k and 'weight' in k:
            print(f"  Stem layer shape: {v.shape}")
            in_ch = v.shape[1] if len(v.shape) >= 2 else 'N/A'
            print(f"  Input channels: {in_ch} (expected: 11)")
            break

    print(f"\n✓ Done! New checkpoint saved to: {output_path}")

    return new_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='Create 11-channel checkpoint from 5-channel checkpoint'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to the new config file (11-channel input)'
    )
    parser.add_argument(
        '--weight', '-w',
        type=str,
        required=True,
        help='Path to the original checkpoint (5-channel input)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to save the new checkpoint'
    )

    args = parser.parse_args()

    create_checkpoint(args.config, args.weight, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
