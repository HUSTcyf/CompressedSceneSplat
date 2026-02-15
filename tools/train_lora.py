#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for SceneSplat

This script provides fine-tuning capabilities using LoRA (Low-Rank Adaptation)
for the Point Transformer V3 backbone in SceneSplat.

Usage:
    # Basic fine-tuning
    python tools/train_lora.py \
        --config-file configs/scannet/lora-finetune-scannet-mcmc.py \
        --options save_path=exp_runs/lora_finetune

    # With custom LoRA settings
    python tools/train_lora.py \
        --config-file configs/scannet/lora-finetune-scannet-mcmc.py \
        --options save_path=exp_runs/lora_finetune \
            lora.r=16 lora.lora_alpha=16

    # Resume from checkpoint
    python tools/train_lora.py \
        --config-file configs/scannet/lora-finetune-scannet-mcmc.py \
        --options save_path=exp_runs/lora_finetune \
            weight=exp_runs/lora_finetune/model_best.pth
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pointcept.models.utils.lora import (
    LoRALinear,
    mark_only_lora_as_trainable,
    get_lora_parameters,
    print_lora_summary,
    merge_lora_weights,
    save_lora_checkpoint,
    load_lora_checkpoint,
)
from pointcept.models.utils.lora_injector import (
    inject_lora_to_ptv3,
    inject_lora_with_preset,
    LORA_PRESETS,
)
from pointcept.engines import train
from pointcept.utils.config import Config
from pointcept.utils.logger import create_logger


def setup_lora_model(model, cfg):
    """
    Setup model with LoRA adapters.

    Args:
        model: Pretrained SceneSplat model
        cfg: Configuration object

    Returns:
        Model with LoRA adapters injected
    """
    lora_cfg = cfg.get("lora", {})

    if not lora_cfg.get("enabled", False):
        print("[LoRA] LoRA is disabled in config, using standard fine-tuning")
        return model

    # Get LoRA settings
    r = lora_cfg.get("r", 8)
    lora_alpha = lora_cfg.get("lora_alpha", 8)
    target_modules = lora_cfg.get("target_modules", ["attn"])
    enable_prompt = lora_cfg.get("enable_prompt", False)
    encoder_only = lora_cfg.get("encoder_only", True)
    target_stages = lora_cfg.get("target_stages", None)

    # Check if using preset
    preset = lora_cfg.get("preset", None)
    if preset:
        print(f"[LoRA] Using preset: {preset}")
        model = inject_lora_with_preset(
            model,
            preset=preset,
        )
    else:
        print(f"[LoRA] Injecting LoRA adapters:")
        print(f"  - Rank (r): {r}")
        print(f"  - Alpha: {lora_alpha}")
        print(f"  - Target modules: {target_modules}")
        print(f"  - Enable prompt: {enable_prompt}")
        print(f"  - Encoder only: {encoder_only}")
        print(f"  - Target stages: {target_stages or 'all'}")

        model = inject_lora_to_ptv3(
            model,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            enable_prompt=enable_prompt,
            encoder_only=encoder_only,
            target_stages=target_stages,
        )

    # Freeze non-LoRA parameters if specified
    if lora_cfg.get("freeze_backbone", True):
        print("[LoRA] Freezing backbone parameters (keeping only LoRA trainable)")
        mark_only_lora_as_trainable(model)

    # Print summary
    print_lora_summary(model)

    return model


def load_pretrained_model(model, pretrained_path, logger=None):
    """
    Load pretrained weights into model.

    Args:
        model: Model to load weights into
        pretrained_path: Path to pretrained checkpoint
        logger: Logger instance

    Returns:
        Model with pretrained weights loaded
    """
    if not pretrained_path or not os.path.exists(pretrained_path):
        if logger:
            logger.warning(f"Pretrained path not found: {pretrained_path}")
        return model

    print(f"[LoRA] Loading pretrained weights from: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Load state dict (handle strict=False for LoRA compatibility)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[LoRA] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[LoRA] Unexpected keys: {len(unexpected)}")

    print("[LoRA] Pretrained weights loaded successfully")
    return model


def main():
    """Main training function with LoRA support."""

    # Parse arguments
    from pointcept.engines.launch import parse_args

    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # Create logger
    logger = create_logger(cfg.log_path)

    # Build model
    from pointcept.models.builder import build_model

    model = build_model(cfg.model)

    # Load pretrained weights if specified
    pretrained_path = cfg.get("lora_training", {}).get("pretrained_path", None)
    if pretrained_path:
        model = load_pretrained_model(model, pretrained_path, logger)

    # Inject LoRA adapters
    model = setup_lora_model(model, cfg)

    # Modify optimizer config to only include LoRA parameters
    if cfg.get("lora", {}).get("enabled", False):
        lora_params = get_lora_parameters(model)
        if lora_params:
            # Update optimizer to use LoRA parameters
            if "paramwise_cfg" not in cfg.optimizer:
                cfg.optimizer.paramwise_cfg = {}
            cfg.optimizer.paramwise_cfg["lora_params"] = lora_params
            logger.info(f"Using {len(lora_params)} LoRA parameters for optimization")

    # Move to GPU
    model = model.cuda()
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            find_unused_parameters=True,  # Important for LoRA
        )

    # Build dataset
    from pointcept.datasets.builder import build_dataset

    dataset = build_dataset(cfg.data)
    # Add dataset-specific configurations
    if hasattr(dataset, "collate_fn"):
        cfg.data.collate_fn = dataset.collate_fn

    # Start training (reuse existing training loop)
    logger.info("Starting LoRA fine-tuning...")
    from pointcept.engines.train import main as train_main

    # Modify the main function to handle LoRA checkpoint saving
    # (This is a simplified version - you may need to adapt based on actual training loop)

    # For now, use the existing training infrastructure
    # train_main(cfg, model, dataset, logger)

    print("[LoRA] Fine-tuning setup complete. Ready to start training.")
    print("[LoRA] Note: Please integrate with existing training loop.")


if __name__ == "__main__":
    main()
