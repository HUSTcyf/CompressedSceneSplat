"""
LoRA Injector for Point Transformer V3

This module provides functions to inject LoRA adapters into PT-v3 models
for efficient fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from pointcept.models.utils.lora import LoRALinear


def inject_lora_to_ptv3(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 8,
    target_modules: Optional[List[str]] = None,
    enable_prompt: bool = False,
    encoder_only: bool = True,
    target_stages: Optional[List[int]] = None,
) -> nn.Module:
    """
    Inject LoRA adapters into Point Transformer V3 model.

    Args:
        model: PT-v3 model to inject LoRA into
        r: Low-rank dimension (default: 8)
        lora_alpha: LoRA scaling factor (default: 8)
        target_modules: List of module types to target ("attn", "mlp", or both)
        enable_prompt: Whether to enable prompt MLP adaptation
        encoder_only: Only inject LoRA into encoder (skip decoder)
        target_stages: List of stage indices to target (None = all stages)

    Returns:
        Model with LoRA adapters injected
    """
    if target_modules is None:
        target_modules = ["attn"]

    injected_count = 0

    # Helper function to inject LoRA into a module
    def inject_into_block(block, block_name: str):
        nonlocal injected_count

        # Inject into attention layers
        if "attn" in target_modules:
            if hasattr(block, "attn"):
                attn = block.attn

                # Replace qkv projection
                if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                    attn.qkv = LoRALinear(
                        r=r,
                        lora_alpha=lora_alpha,
                        linear_layer=attn.qkv,
                        enable_prompt=enable_prompt,
                    )
                    injected_count += 1
                    print(f"  [LoRA] Injected into {block_name}.attn.qkv")

                # Replace output projection
                if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
                    attn.proj = LoRALinear(
                        r=r,
                        lora_alpha=lora_alpha,
                        linear_layer=attn.proj,
                        enable_prompt=enable_prompt,
                    )
                    injected_count += 1
                    print(f"  [LoRA] Injected into {block_name}.attn.proj")

        # Inject into MLP layers
        if "mlp" in target_modules:
            if hasattr(block, "mlp"):
                mlp = block.mlp

                # Handle PointSequential wrapper
                if hasattr(mlp, "module") and hasattr(mlp.module, "mlp"):
                    inner_mlp = mlp.module.mlp

                    # Replace fc1
                    if hasattr(inner_mlp, "fc1") and isinstance(inner_mlp.fc1, nn.Linear):
                        inner_mlp.fc1 = LoRALinear(
                            r=r,
                            lora_alpha=lora_alpha,
                            linear_layer=inner_mlp.fc1,
                            enable_prompt=enable_prompt,
                        )
                        injected_count += 1
                        print(f"  [LoRA] Injected into {block_name}.mlp.fc1")

                    # Replace fc2
                    if hasattr(inner_mlp, "fc2") and isinstance(inner_mlp.fc2, nn.Linear):
                        inner_mlp.fc2 = LoRALinear(
                            r=r,
                            lora_alpha=lora_alpha,
                            linear_layer=inner_mlp.fc2,
                            enable_prompt=enable_prompt,
                        )
                        injected_count += 1
                        print(f"  [LoRA] Injected into {block_name}.mlp.fc2")

    # Inject into encoder stages
    if hasattr(model, "enc"):
        for stage_idx, stage in enumerate(model.enc):
            if target_stages is not None and stage_idx not in target_stages:
                continue

            stage_name = f"enc.enc{stage_idx}"
            if hasattr(stage, "modules"):
                for block_idx, block in enumerate(stage.modules()):
                    if hasattr(block, "attn"):  # This is a Transformer Block
                        block_name = f"{stage_name}.block{block_idx}"
                        inject_into_block(block, block_name)

    # Inject into decoder stages (optional)
    if not encoder_only and hasattr(model, "dec"):
        for stage_idx, stage in enumerate(model.dec):
            if target_stages is not None and stage_idx not in target_stages:
                continue

            stage_name = f"dec.dec{stage_idx}"
            if hasattr(stage, "modules"):
                for block_idx, block in enumerate(stage.modules()):
                    if hasattr(block, "attn"):  # This is a Transformer Block
                        block_name = f"{stage_name}.block{block_idx}"
                        inject_into_block(block, block_name)

    print(f"\n[LoRA] Total injections: {injected_count}")
    return model


def inject_lora_to_stage(
    model: nn.Module,
    stage_indices: List[int],
    r: int = 8,
    lora_alpha: int = 8,
    target_modules: Optional[List[str]] = None,
    encoder: bool = True,
) -> nn.Module:
    """
    Inject LoRA into specific encoder/decoder stages only.

    Args:
        model: PT-v3 model
        stage_indices: List of stage indices (e.g., [2, 3] for deeper stages)
        r: Low-rank dimension
        lora_alpha: LoRA scaling factor
        target_modules: List of module types to target
        encoder: True for encoder, False for decoder

    Returns:
        Model with LoRA injected
    """
    return inject_lora_to_ptv3(
        model=model,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        encoder_only=not encoder,
        target_stages=stage_indices,
    )


# Preset configurations for common use cases
LORA_PRESETS = {
    "minimal": {
        "r": 4,
        "lora_alpha": 4,
        "target_modules": ["attn"],
        "enable_prompt": False,
        "target_stages": [3, 4],  # Only deeper encoder stages
    },
    "standard": {
        "r": 8,
        "lora_alpha": 8,
        "target_modules": ["attn"],
        "enable_prompt": False,
        "target_stages": None,  # All stages
    },
    "full": {
        "r": 16,
        "lora_alpha": 16,
        "target_modules": ["attn", "mlp"],
        "enable_prompt": True,
        "target_stages": None,  # All stages
    },
    "decoder_only": {
        "r": 8,
        "lora_alpha": 8,
        "target_modules": ["attn"],
        "enable_prompt": False,
        "encoder_only": False,  # Will inject into decoder only
    },
}


def inject_lora_with_preset(
    model: nn.Module,
    preset: str = "standard",
    **kwargs,
) -> nn.Module:
    """
    Inject LoRA using a preset configuration.

    Args:
        model: PT-v3 model
        preset: Preset name ("minimal", "standard", "full", "decoder_only")
        **kwargs: Override preset parameters

    Returns:
        Model with LoRA injected
    """
    if preset not in LORA_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(LORA_PRESETS.keys())}")

    config = LORA_PRESETS[preset].copy()
    config.update(kwargs)

    return inject_lora_to_ptv3(model, **config)


def save_lora_checkpoint(model: nn.Module, path: str, metadata: Optional[dict] = None):
    """
    Save only LoRA parameters and adapter config.

    Args:
        model: Model with LoRA adapters
        path: Path to save checkpoint
        metadata: Optional metadata to save (e.g., rank, alpha)
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name or ("prompt" in name and "mlp" in name):
            lora_state_dict[name] = param

    checkpoint = {
        "lora_state_dict": lora_state_dict,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    print(f"[LoRA] Saved LoRA checkpoint to {path}")


def load_lora_checkpoint(model: nn.Module, path: str):
    """
    Load LoRA parameters from checkpoint.

    Args:
        model: Model to load LoRA into (must have same architecture)
        path: Path to LoRA checkpoint
    """
    checkpoint = torch.load(path, map_location="cpu")
    lora_state_dict = checkpoint["lora_state_dict"]

    model_state = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"[LoRA] Warning: {name} not found in model")

    print(f"[LoRA] Loaded LoRA checkpoint from {path}")
    if "metadata" in checkpoint:
        print(f"[LoRA] Metadata: {checkpoint['metadata']}")
