#!/usr/bin/env python3
"""
Trace LitePT model forward pass to analyze tensor shapes at each step.

This script:
1. Loads model configuration from config file
2. Creates a random input tensor matching expected model input shape
3. Loads LitePT model
4. Registers forward hooks to capture intermediate outputs
5. Runs forward pass and collects tensor shape information

Usage:
    python tools/trace_tensor_shapes.py --config configs/custom/lang-pretrain-litept-ovs.py
"""
import os
import torch
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary modules
from pointcept.models.litept.model import LitePT
from pointcept.utils.config import Config


class TensorShapeLogger:
    """Hook to log tensor shapes at each forward step."""

    def __init__(self, component_name):
        self.component_name = component_name
        self.shapes_logged = []

    def hook(self, module, input, output):
        """Called after forward pass of each module."""
        if isinstance(output, dict) and "feat" in output:
            feat = output["feat"]
            shape_str = str(list(feat.shape))
            self.shapes_logged.append({
                "component": self.component_name,
                "module": module.__class__.__name__,
                "input_shape": str(list(input["feat"].shape)) if isinstance(input, dict) and "feat" in input else "N/A",
                "output_shape": shape_str,
            })
            print(f"[{self.component_name}] {module.__class__.__name__:20s} {shape_str}")

    def report(self):
        """Print summary of all logged shapes."""
        if not self.shapes_logged:
            print(f"[{self.component_name}] No shapes logged")
            return

        print(f"\n=== {self.component_name} Tensor Shape Summary ===")
        print(f"Total steps: {len(self.shapes_logged)}")

        # Group by component
        components = {}
        for log in self.shapes_logged:
            comp = log["component"]
            if comp not in components:
                components[comp] = []
            components[comp].append(log)

        # Print each component's shapes
        for comp, logs in sorted(components.items()):
            print(f"\n{comp}:")
            print(f"  Modules involved:")
            modules = set()
            for log in logs:
                modules.add(log["module"])

            for m in sorted(modules):
                print(f"    - {m}")

            # Print shape transitions
            print(f"  Shape flow:")
            for i, log in enumerate(logs):
                if i == 0:
                    print(f"    Input:  {log['input_shape']}")
                else:
                    prev_log = logs[i-1]
                    if prev_log["component"] == log["component"]:
                        delta = "same" if log["input_shape"] == log["output_shape"] else "changed"
                        print(f"    -> Output: {log['output_shape']} (Delta: {delta})")

        print(f"\nOverall tensor shape progression:")
        if self.shapes_logged:
            first_log = self.shapes_logged[0]
            print(f"  Initial input shape:  {first_log['input_shape']}")


def register_hooks_on_model(model):
    """Register forward hooks on all key components of LitePT."""

    hooks = []

    # Hook for Embedding layer (stem)
    class EmbeddingHook(TensorShapeLogger):
        def __init__(self):
            super().__init__("Embedding")

        def hook(self, module, input, output):
            if isinstance(output, dict) and "feat" in output:
                super().hook(module, input, output)

    embedding_hook = EmbeddingHook()
    model.embedding.register_forward_hook(embedding_hook.hook)
    hooks.append(embedding_hook)

    # Hook for encoder stages
    class EncoderStageHook(TensorShapeLogger):
        def __init__(self, stage_idx):
            super().__init__(f"Encoder_Stage_{stage_idx}")

        def hook(self, module, input, output):
            if isinstance(output, dict) and "feat" in output:
                super().hook(module, input, output)

    # Hook for decoder stages
    class DecoderStageHook(TensorShapeLogger):
        def __init__(self, stage_idx):
            super().__init__(f"Decoder_Stage_{stage_idx}")

        def hook(self, module, input, output):
            if isinstance(output, dict) and "feat" in output:
                super().hook(module, input, output)

    # Hook for final output
    class OutputHook(TensorShapeLogger):
        def __init__(self):
            super().__init__("Final_Output")

        def hook(self, module, input, output):
            if isinstance(output, dict) and "feat" in output:
                super().hook(module, input, output)

    # Register hooks on encoder stages
    for stage_idx in range(len(model.enc)):
        enc_hook = EncoderStageHook(stage_idx)
        model.enc[stage_idx].register_forward_hook(enc_hook.hook)
        hooks.append(enc_hook)

    # Register hooks on decoder stages
    for stage_idx in range(len(model.dec)):
        dec_hook = DecoderStageHook(stage_idx)
        model.dec[stage_idx].register_forward_hook(dec_hook.hook)
        hooks.append(dec_hook)

    # Register final output hook
    output_hook = OutputHook()
    # Get the last module in decoder
    model.dec[-1].register_forward_hook(output_hook.hook)
    hooks.append(output_hook)

    return hooks


def create_random_input(model_config, num_points=10000, device="cuda"):
    """Create a random input tensor matching expected model input shape."""

    # Get expected input channels from config
    in_channels = model_config.get("in_channels", 11)

    # Create random features matching 3DGS format
    # coord: [N, 3], color: [N, 3], opacity: [N, 1], quat: [N, 4], scale: [N, 3]
    torch.manual_seed(42)

    # Concatenate all features into a single feature tensor
    # Order: color(3) + opacity(1) + quat(4) + scale(3) = 11 channels
    feat = torch.cat([
        torch.rand(num_points, 3, device=device),  # color
        torch.rand(num_points, 1, device=device),  # opacity
        torch.rand(num_points, 4, device=device),  # quat
        torch.rand(num_points, 3, device=device),  # scale
    ], dim=1)  # [N, 11]

    coord = torch.randn(num_points, 3, device=device)
    grid_size = 0.01  # default grid size from config

    # Create Point object input with proper structure
    from pointcept.models.utils.structure import Point
    input_point = Point({
        "coord": coord,
        "feat": feat,
        "batch": torch.zeros(num_points, dtype=torch.long, device=device),  # single batch
        "grid_size": grid_size,
    })

    return input_point


def load_config_from_file(config_path):
    """Load model configuration from config file."""
    cfg = Config.fromfile(config_path)

    # Extract model configuration
    model_config = {}
    if hasattr(cfg, 'model'):
        model_cfg = cfg.model
        if isinstance(model_cfg, dict):
            # Get the backbone config
            backbone_cfg = model_cfg.get('backbone', {})
            if isinstance(backbone_cfg, dict):
                # Extract backbone parameters
                for key, value in backbone_cfg.items():
                    if key == 'type':
                        continue
                    model_config[key] = value

    return model_config


def main():
    """Main function to trace tensor shapes through LitePT model."""

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Trace LitePT model tensor shapes during forward pass"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/custom/lang-pretrain-litept-ovs.py",
        help="Path to model config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/new_data/cyf/projects/SceneSplat/exp/lite-768/model/model_last.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=10000,
        help="Number of random input points (default: 10000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )

    args = parser.parse_args()

    # Load model configuration from config file
    print("="*80)
    print("LitePT Tensor Shape Tracer")
    print("="*80)
    print(f"Config file: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")

    try:
        model_config = load_config_from_file(args.config)
        print(f"Loaded config with keys: {list(model_config.keys())}")
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default model configuration.")
        model_config = {
            "in_channels": 11,
            "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
        }

    # Check if checkpoint exists
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Continuing anyway to run forward pass...")

    print(f"\nLoading checkpoint from: {checkpoint_path}")

    # Create model
    model = LitePT(**model_config)

    # Load checkpoint weights (following batch_predict.py pattern)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove 'module.' prefix if present (from DistributedDataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            elif k.startswith("backbone."):
                # Also remove 'backbone.' prefix
                new_state_dict[k[len("backbone."):]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        # Direct checkpoint, remove 'backbone.' prefix if present
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("backbone."):
                new_state_dict[k[len("backbone."):]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    # Move model to device
    device = args.device
    model = model.to(device)

    # Register hooks
    hooks = register_hooks_on_model(model)

    # Create random input
    num_points = args.num_points
    print(f"\nCreating random input: {num_points} points, device={device}")

    # Create smaller batch for actual test (100 points to avoid OOM)
    input_dict = create_random_input(model_config, num_points=100, device=device)

    print("\n" + "="*80)
    print("Running forward pass with shape tracing...")
    print("="*80)

    # Run in eval mode to avoid dropout
    model.eval()

    with torch.no_grad():
        # Forward pass
        output = model(input_dict)

    # Report all hooks
    print("\n" + "="*80)
    print("Tensor Shape Analysis Results")
    print("="*80)

    for hook in hooks:
        hook.report()

    print()

    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input points: {num_points}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
