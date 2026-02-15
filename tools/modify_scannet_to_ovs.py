#!/usr/bin/env python3
"""
Modify model_best_scannet.pth checkpoint to adapt to lang-pretrain-litept-ovs.py network structure.

Original (Scannet):  in_channels=6  (xyz + rgb)
Target (OVS):       in_channels=11 (color + opacity + quat + scale, no coord)

This script modifies the stem layer's weight matrix to handle the additional features.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def modify_checkpoint_for_ovs(
    ckpt_path: str,
    output_path: str,
    original_in_channels: int = 6,   # xyz(3) + rgb(3)
    new_in_channels: int = 11,        # color(3) + opacity(1) + quat(4) + scale(3) (no coord)
):
    """
    Modify the checkpoint stem layer to accommodate different in_channels.

    Args:
        ckpt_path: Path to original checkpoint (model_best_scannet.pth)
        output_path: Path to save modified checkpoint (model_best_ovs.pth)
        original_in_channels: Original in_channels (6 for xyz+rgb)
        new_in_channels: New in_channels (11 for 3DGS features without coord)
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Get state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    else:
        state_dict = checkpoint
        metadata = {}

    print(f"Original in_channels: {original_in_channels}")
    print(f"New in_channels:       {new_in_channels}")
    print(f"Difference:             {new_in_channels - original_in_channels} (adding 3DGS features)")

    # Find stem layer weight
    stem_weight_key = None
    for key in state_dict.keys():
        if 'stem' in key and 'weight' in key and 'conv' in key:
            stem_weight_key = key
            print(f"Found stem weight: {key}")
            print(f"  Shape: {state_dict[key].shape}")
            break

    if stem_weight_key is None:
        print("Error: Could not find stem conv layer in checkpoint")
        print("\\nAvailable keys containing 'stem':")
        for key in state_dict.keys():
            if 'stem' in key:
                print(f"  {key}: {state_dict[key].shape}")
        sys.exit(1)

    # Modify the stem weight
    original_weight = state_dict[stem_weight_key]
    weight_shape = original_weight.shape

    print(f"\\nOriginal stem weight shape: {weight_shape}")

    # spconv 3D conv weight format: [out_channels, kernel_depth, kernel_height, kernel_width, in_channels]
    out_channels, kd, kh, kw, in_channels = weight_shape
    kernel_size = (kd, kh, kw)

    print(f"  (out_channels={out_channels}, in_channels={in_channels}, kernel_size={kernel_size})")

    # Verify in_channels matches expected
    if in_channels != original_in_channels:
        print(f"\\nWarning: Expected in_channels={original_in_channels}, but found {in_channels}")
        print(f"Proceeding with modification anyway...")

    # Strategy: Create new weight and copy/initialize appropriately
    if new_in_channels > in_channels:
        # Create new weight tensor
        new_weight_shape = (out_channels, kd, kh, kw, new_in_channels)
        new_weight = torch.zeros(new_weight_shape, dtype=original_weight.dtype)

        # For 3DGS features: we need to map from xyz(3)+rgb(3) to color(3)+opacity(1)+quat(4)+scale(3)+coord(3)
        # Original: [x, y, z, r, g, b]
        # New:      [r, g, b, opacity, qx, qy, qz, qw, sx, sy, sz, cx, cy, cz]

        # Mapping strategy (coord excluded, using 11 channels):
        # - rgb[0:3] -> color[0:3] (copy)
        # - xyz[0:3] -> coord[-3:] (copy) [EXCLUDED]
        # - opacity -> initialize small
        # - quat -> initialize small
        # - scale -> initialize small

        # Copy RGB channels (indices 3,4,5 in original -> 0,1,2 in new)
        new_weight[..., 0:3] = original_weight[..., 3:6]

        # Copy XYZ/coord channels (indices 0,1,2 in original -> 11,12,13 in new) [EXCLUDED]
        # new_weight[..., 11:14] = original_weight[..., 0:3]

        # Initialize new channels (opacity, quat, scale) with small random values
        new_weight[..., 3:11] = torch.randn_like(new_weight[..., 3:11]) * 0.01

        print(f"\\nExpanded stem weight from {in_channels} to {new_in_channels} channels")
        print(f"  - Copied RGB channels [3:6] -> [0:3]")
        print(f"  - Copied XYZ channels [0:3] -> [11:14] [EXCLUDED]")
        print(f"  - Initialized 8 new channels (opacity, quat, scale) with small random values")
        print(f"New stem weight shape: {new_weight.shape}")
    elif new_in_channels < in_channels:
        # Truncate channels
        new_weight = original_weight[..., :new_in_channels]
        print(f"\\nTruncated stem weight from {in_channels} to {new_in_channels} channels")
        print(f"New stem weight shape: {new_weight.shape}")
    else:
        print(f"\\nNo modification needed: in_channels={new_in_channels} already matches")
        new_weight = original_weight

    # Update the checkpoint
    state_dict[stem_weight_key] = new_weight

    # Create new checkpoint
    new_checkpoint = {}
    if metadata:
        new_checkpoint.update(metadata)
    new_checkpoint["state_dict"] = state_dict
    new_checkpoint["original_in_channels"] = original_in_channels
    new_checkpoint["new_in_channels"] = new_in_channels
    new_checkpoint["modified_from"] = ckpt_path

    # Save modified checkpoint
    print(f"\\nSaving modified checkpoint to: {output_path}")
    torch.save(new_checkpoint, output_path)

    # Verify
    verify = torch.load(output_path, map_location='cpu', weights_only=False)
    verify_shape = verify['state_dict'][stem_weight_key].shape
    print(f"Verification: stem weight shape = {verify_shape}")

    print("\\n=== Modification Summary ===")
    print(f"Original: {ckpt_path}")
    print(f"Output:   {output_path}")
    print(f"Original in_channels: {original_in_channels}")
    print(f"New in_channels:       {new_in_channels}")
    print(f"Stem weight modified:  Yes")
    print(f"Stem weight shape:    {verify_shape}")
    print("\\nSuccessfully adapted checkpoint for OVS 3DGS data!")


if __name__ == "__main__":
    ckpt_path = "/new_data/cyf/projects/SceneSplat/LitePT/ckpts/model_best_scannet.pth"
    output_path = "/new_data/cyf/projects/SceneSplat/LitePT/ckpts/model_best_ovs.pth"

    modify_checkpoint_for_ovs(
        ckpt_path=ckpt_path,
        output_path=output_path,
        original_in_channels=6,   # Scannet: xyz(3) + rgb(3)
        new_in_channels=11,       # OVS: color(3) + opacity(1) + quat(4) + scale(3) (no coord)
    )

    print("\\nDone! You can now use the checkpoint for Vision-Language Pretraining on OVS data.")
