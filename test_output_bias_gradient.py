#!/usr/bin/env python3
"""
Simple Gradient Flow Test for output_bias

This is a minimal test to verify that output_bias receives gradients
from the loss function.
"""

import torch
import torch.nn as nn
import numpy as np


def test_gradient_flow():
    """Test gradient flow to output_bias in a simplified setting."""

    print("\n" + "=" * 70)
    print("SIMPLIFIED GRADIENT FLOW TEST FOR output_bias")
    print("=" * 70)

    # Simulate the model structure
    class SimpleModel(nn.Module):
        def __init__(self, feat_dim=16, enable_output_bias=True, output_bias_init=None):
            super().__init__()
            self.feat_dim = feat_dim

            # Simple projection
            self.proj = nn.Linear(11, feat_dim)

            # Output bias (like in LangPretrainer)
            self.enable_output_bias = enable_output_bias
            if enable_output_bias:
                if output_bias_init is not None:
                    bias_init = torch.tensor(output_bias_init, dtype=torch.float32)
                    self.output_bias = nn.Parameter(bias_init)
                else:
                    self.output_bias = nn.Parameter(torch.zeros(feat_dim))
            else:
                self.output_bias = None

        def forward(self, x):
            # Simulate backbone
            feat = self.proj(x)
            # Apply tanh (like in LangPretrainer)
            feat = torch.tanh(feat)
            # Add output bias
            if self.enable_output_bias and self.output_bias is not None:
                feat = feat + self.output_bias
            return feat

    # Create model
    output_bias_init = [0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    model = SimpleModel(feat_dim=16, enable_output_bias=True, output_bias_init=output_bias_init)

    print(f"\nInitial output_bias: {model.output_bias.data.numpy()}")

    # Create dummy data
    batch_size = 4
    num_points = 100
    in_channels = 11
    feat_dim = 16

    input_data = torch.randn(batch_size, num_points, in_channels)
    target = torch.randn(batch_size, num_points, feat_dim)
    valid_mask = torch.ones(batch_size, num_points)

    print(f"\nInput shape: {input_data.shape}")
    print(f"Target shape: {target.shape}")

    # Forward pass
    print("\n--- Forward Pass ---")
    pred = model(input_data)
    print(f"Prediction shape: {pred.shape}")
    print(f"Prediction stats: mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")

    # Compute loss (SVD-weighted L1 loss simulation)
    print("\n--- Loss Computation ---")

    # Flatten for loss computation
    valid_pred = pred[valid_mask > 0]  # [M, D]
    valid_target = target[valid_mask > 0]  # [M, D]

    print(f"Valid pred shape: {valid_pred.shape}")
    print(f"Valid target shape: {valid_target.shape}")

    # Compute per-dimension weights (simulate SVD weighting)
    # Use std-based weighting
    dim_std = valid_target.std(dim=0)  # [D]
    print(f"\nDimension std (target): {dim_std.numpy()}")

    # Normalize to [0.05, 1.0] range
    std_min = dim_std.min()
    std_max = dim_std.max()
    if std_max > std_min:
        normalized = (dim_std - std_min) / (std_max - std_min)
        weights = 0.05 + normalized * 0.95
    else:
        weights = torch.ones_like(dim_std)

    print(f"Dimension weights: {weights.numpy()}")

    # Compute weighted L1 loss
    abs_diff = torch.abs(valid_pred - valid_target)  # [M, D]
    print(f"\nAbs diff shape: {abs_diff.shape}")
    print(f"Abs diff per-dimension mean: {abs_diff.mean(dim=0).detach().numpy()}")

    weighted_diff = abs_diff * weights.unsqueeze(0)  # [M, D]
    loss = weighted_diff.sum(dim=1).mean()  # Scalar

    print(f"\nLoss value: {loss.item():.6f}")

    # Backward pass
    print("\n--- Backward Pass ---")
    model.zero_grad()
    loss.backward()

    # Check gradients
    print("\n--- Gradient Analysis ---")

    if model.output_bias.grad is None:
        print("[ERROR] output_bias.grad is None!")
        print("→ output_bias is NOT receiving gradients")
        print("\nThis indicates:")
        print("  1. The loss does not depend on output_bias")
        print("  2. There's a computation graph disconnect")
        return False
    else:
        grad = model.output_bias.grad.detach().numpy()
        print(f"[OK] output_bias.grad exists!")
        print(f"\nGradient values: {grad}")
        print(f"Gradient norms: {np.abs(grad)}")

        # Check for dead dimensions
        dead = np.abs(grad) < 1e-10
        print(f"\nDead dimensions (|grad| < 1e-10): {dead.sum()}/{len(grad)}")

        print("\nPer-dimension gradient analysis:")
        for d in range(len(grad)):
            status = "DEAD" if dead[d] else "ACTIVE"
            print(f"  Dim {d:2d}: grad={grad[d]:.6e} |grad|={np.abs(grad[d]):.6e} weight={weights[d].item():.6f} [{status}]")

        # Analyze why some dimensions might be dead
        if dead.sum() > 0:
            print(f"\n[WARNING] {dead.sum()} dimensions have dead gradients!")
            print("\nPossible causes:")

            # Check if weights are too small
            dead_weights = weights[dead].numpy()
            if np.all(dead_weights < 0.1):
                print("  1. ✓ Dead dimensions have very low weights (< 0.1)")
                print("     → Low weights → small gradients → effectively dead")

            # Check if target variance is low
            dead_target_std = dim_std[dead].numpy()
            if np.all(dead_target_std < 0.1):
                print("  2. ✓ Dead dimensions have low target variance")
                print("     → Constant target → no gradient signal")

            # Check if pred-target difference is small
            dead_abs_diff = abs_diff[:, dead].mean(dim=0).detach().numpy()
            if np.all(dead_abs_diff < 0.1):
                print("  3. ✓ Dead dimensions have small pred-target difference")
                print("     → Model already fits well → small gradients")

        return True

    print("=" * 70)


if __name__ == "__main__":
    test_gradient_flow()
