#!/usr/bin/env python3
"""
Convert LitePT checkpoint from ScanNet config to OVS config.

Modifies model_best_scannet.pth to be compatible with semseg-litept-ovs.py config.

Key differences:
- ScanNet: backbone_out_channels=64, in_channels=14 (with normal)
- OVS: backbone_out_channels=768, in_channels=14 (3DGS only, same channels)

The backbone architecture is identical, only output dimensions differ.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def convert_checkpoint_for_ovs(
    ckpt_path: str,
    output_path: str,
    original_out_channels: int = 64,
    new_out_channels: int = 768,
):
    """
    Convert LitePT checkpoint from ScanNet to OVS config.

    Args:
        ckpt_path: Path to original checkpoint (model_best_scannet.pth)
        output_path: Path to save converted checkpoint (model_best_ovs.pth)
        original_out_channels: Original output channels (64 for ScanNet)
        new_out_channels: New output channels (768 for OVS lang_feat_dim)
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Check checkpoint format
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    else:
        state_dict = checkpoint
        metadata = {}

    print(f"Original state_dict keys: {len(state_dict)}")

    # Keys to remove (dimension-specific layers)
    keys_to_remove = []
    keys_to_modify = []

    for key in list(state_dict.keys()):
        # Remove output projection / head layers that depend on output channels
        if any(x in key for x in ['head', 'classifier', 'fc_out', 'final_conv']):
            keys_to_remove.append(key)
            print(f"  Will remove: {key} (shape: {state_dict[key].shape})")

    # Remove dimension-specific keys
    for key in keys_to_remove:
        del state_dict[key]

    print(f"\nRemoved {len(keys_to_remove)} keys with dimension mismatches")
    print(f"Remaining keys: {len(state_dict)}")

    # Create new checkpoint
    new_checkpoint = {}
    if metadata:
        new_checkpoint.update(metadata)
    new_checkpoint["state_dict"] = state_dict
    new_checkpoint["original_out_channels"] = original_out_channels
    new_checkpoint["new_out_channels"] = new_out_channels
    new_checkpoint["converted_from"] = ckpt_path

    # Save converted checkpoint
    print(f"\nSaving converted checkpoint to: {output_path}")
    torch.save(new_checkpoint, output_path)

    # Verify
    verify = torch.load(output_path, map_location='cpu')
    print(f"Verification: {len(verify['state_dict'])} keys in converted checkpoint")

    print("\n=== Conversion Summary ===")
    print(f"Original: {ckpt_path}")
    print(f"Output:   {output_path}")
    print(f"Original out_channels: {original_out_channels}")
    print(f"New out_channels:       {new_out_channels}")
    print(f"Keys removed:           {len(keys_to_remove)}")
    print(f"Keys retained:          {len(state_dict)}")
    print("\nNote: The model will initialize output layers randomly.")


if __name__ == "__main__":
    # Paths
    ckpt_path = "/new_data/cyf/projects/SceneSplat/LitePT/ckpts/model_best_scannet.pth"
    output_path = "/new_data/cyf/projects/SceneSplat/LitePT/ckpts/model_best_ovs.pth"

    # Check if input exists
    if not Path(ckpt_path).exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        print("\nLooking for alternative checkpoints...")
        # Try to find any checkpoint in LitePT/ckpts/
        ckpt_dir = Path("/new_data/cyf/projects/SceneSplat/LitePT/ckpts")
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob("*.pth"))
            if ckpts:
                print(f"Found checkpoints:")
                for c in ckpts:
                    print(f"  - {c}")
                ckpt_path = str(ckpts[0])
                print(f"\nUsing: {ckpt_path}")
            else:
                print("No .pth files found in LitePT/ckpts/")
                sys.exit(1)
        else:
            print(f"Directory not found: {ckpt_dir}")
            sys.exit(1)

    # Convert
    convert_checkpoint_for_ovs(
        ckpt_path=ckpt_path,
        output_path=output_path,
        original_out_channels=64,  # ScanNet LitePT
        new_out_channels=768,      # OVS lang_feat_dim
    )

    print("\nDone! You can now use the checkpoint with:")
    print(f"  --weight {output_path}")
