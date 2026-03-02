"""
Repair checkpoint by resetting dim_scale to better initial values.
This fixes the issue where dim_scale has negative values causing 100x underscaling.
"""
import torch
import sys
import shutil
from pathlib import Path

def repair_checkpoint(ckpt_path, backup=True):
    """Repair checkpoint by resetting dim_scale to better initial values."""
    
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        return False
    
    # Load checkpoint
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    if 'state_dict' not in ckpt:
        print("Error: Checkpoint does not contain state_dict")
        return False
    
    state_dict = ckpt['state_dict']
    
    if 'dim_scale' not in state_dict:
        print("Error: Checkpoint does not contain dim_scale parameter")
        return False
    
    # Backup original checkpoint
    if backup:
        backup_path = ckpt_path.with_suffix('.pth.backup')
        shutil.copy(ckpt_path, backup_path)
        print(f"Backup saved to: {backup_path}")
    
    # Show original values
    original_dim_scale = state_dict['dim_scale'].clone()
    print(f"\nOriginal dim_scale:")
    print(f"  {original_dim_scale.numpy()}")
    print(f"  min: {original_dim_scale.min():.4f}, max: {original_dim_scale.max():.4f}, mean: {original_dim_scale.mean():.4f}")
    print(f"  Negative values: {(original_dim_scale < 0).sum().item()}/16")
    
    # Create new dim_scale based on SVD-16 feature statistics
    new_dim_scale = torch.ones(16)
    new_dim_scale[0] = 1.0   # DC component
    new_dim_scale[1] = 0.8
    new_dim_scale[2] = 0.6
    new_dim_scale[3] = 0.5
    new_dim_scale[4] = 0.4
    new_dim_scale[5:] = 0.3  # Remaining dimensions
    
    print(f"\nNew dim_scale:")
    print(f"  {new_dim_scale.numpy()}")
    print(f"  min: {new_dim_scale.min():.4f}, max: {new_dim_scale.max():.4f}, mean: {new_dim_scale.mean():.4f}")
    
    # Update checkpoint
    ckpt['state_dict']['dim_scale'] = new_dim_scale
    
    # Save repaired checkpoint
    torch.save(ckpt, ckpt_path)
    print(f"\n✓ Repaired checkpoint saved to: {ckpt_path}")
    
    return True

if __name__ == "__main__":
    ckpt_path = "/new_data/cyf/projects/SceneSplat/exp/lite-16-gridsvd/model/model_last.pth"
    
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    
    repair_checkpoint(ckpt_path)
