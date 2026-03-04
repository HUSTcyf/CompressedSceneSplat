#!/usr/bin/env python3
"""
Run validation-only evaluation using LangPretrainZeroShotSemSegEval evaluator.

This script allows running the validation evaluator without training,
useful for quick evaluation on a separate server.

Usage:
    python tools/eval_only.py \
        --config-file configs/custom/lang-pretrain-litept-scannet.py \
        --weight exp/lite-16-scannet-gridsvd/model_best.pth
"""

import os
import sys
import argparse

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(description="Run validation-only evaluation")
    parser.add_argument("--config-file", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--weight", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Override save path from config")
    args = parser.parse_args()

    # Import after adding project root to path
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    from pointcept.engines.defaults import default_config_parser, default_setup
    from pointcept.datasets import build_dataset
    from pointcept.models import build_model
    from pointcept.engines.defaults import create_ddp_model
    from pointcept.engines.hooks.evaluator import LangPretrainZeroShotSemSegEval
    import pointcept.utils.comm as comm

    # Load config using the same parser as train.py
    cfg = default_config_parser(args.config_file, args.options if hasattr(args, 'options') else None)

    # Override save path if specified
    if args.save_path:
        cfg.save_path = args.save_path

    # Initialize distributed (single GPU for now)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(rank)
    else:
        rank = 0
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

    # Setup (sets up logging, seeds, etc.)
    # Set a valid seed to avoid issues with get_random_seed() generating values > 2^32-1
    if not hasattr(cfg, "seed") or cfg.seed is None:
        cfg.seed = 42  # Use a fixed valid seed for evaluation
        print(f"Setting seed to {cfg.seed} for reproducible evaluation")

    cfg = default_setup(cfg)

    # Build model
    print(f"Building model...")
    model = build_model(cfg.model)
    model = model.cuda()
    model = create_ddp_model(
        model,
        broadcast_buffers=False,
        find_unused_parameters=cfg.find_unused_parameters,
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.weight}")
    checkpoint = torch.load(args.weight, map_location="cuda")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        epoch_info = checkpoint.get("epoch", "unknown")
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        epoch_info = checkpoint.get("epoch", "unknown")
    else:
        state_dict = checkpoint
        epoch_info = "unknown"

    # Remove 'module.' prefix if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[7:]
        else:
            new_k = k
        new_state_dict[new_k] = v

    # Load with strict=False to allow partial loading
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    print(f"Checkpoint loaded (epoch {epoch_info})")
    if missing_keys:
        print(f"  Missing {len(missing_keys)} keys (will use random initialization)")
    if unexpected_keys:
        print(f"  Skipping {len(unexpected_keys)} unexpected keys from checkpoint")

    # Build val dataset
    print(f"Building validation dataset...")
    val_dataset = build_dataset(cfg.data.val)

    if comm.get_world_size() > 1:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        val_sampler = None

    # Calculate batch size per GPU
    if hasattr(cfg, "batch_size_val"):
        batch_size = cfg.batch_size_val // comm.get_world_size()
    elif hasattr(cfg, "batch_size_val_per_gpu"):
        batch_size = cfg.batch_size_val_per_gpu
    else:
        batch_size = 1

    # Calculate num workers
    if hasattr(cfg, "num_worker"):
        num_workers = cfg.num_worker // comm.get_world_size()
    else:
        num_workers = 4

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=getattr(val_dataset, "collate_fn", val_dataset.__class__.collate_fn),
    )

    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Val loader: {len(val_loader)} batches")

    # Create a simple trainer wrapper for evaluator
    class TrainerWrapper:
        def __init__(self, model, val_loader, cfg):
            self.model = model
            self.val_loader = val_loader
            self.cfg = cfg
            self.epoch = checkpoint.get("epoch", 0)

        # Get logger from cfg or create default
        @property
        def logger(self):
            if not hasattr(self, "_logger"):
                from pointcept.utils.logger import get_root_logger
                self._logger = get_root_logger()
            return self._logger

    trainer = TrainerWrapper(model, val_loader, cfg)

    # Build evaluator from hooks config
    trainer.logger.info(f"Building evaluator...")
    hook_configs = cfg.hooks if hasattr(cfg, "hooks") else []
    evaluator = None

    for hook_cfg in hook_configs:
        if hook_cfg.get("type") == "LangPretrainZeroShotSemSegEval":
            # Extract parameters
            evaluator_cfg = hook_cfg.copy()
            evaluator_type = evaluator_cfg.pop("type")

            # Build class names and text embeddings paths
            class_names = evaluator_cfg.get("class_names")
            text_embeddings = evaluator_cfg.get("text_embeddings")

            # If these are relative paths, resolve them against repo_root
            if hasattr(cfg, "repo_root"):
                if class_names and not os.path.isabs(class_names):
                    class_names = os.path.join(cfg.repo_root, class_names)
                if text_embeddings and not os.path.isabs(text_embeddings):
                    text_embeddings = os.path.join(cfg.repo_root, text_embeddings)

            evaluator = LangPretrainZeroShotSemSegEval(
                trainer=trainer,
                class_names=class_names,
                text_embeddings=text_embeddings,
                excluded_classes=evaluator_cfg.get("excluded_classes"),
                ignore_index=evaluator_cfg.get("ignore_index", -1),
                confidence_threshold=evaluator_cfg.get("confidence_threshold", 0.1),
                vote_k=evaluator_cfg.get("vote_k", 25),
                enable_voting=evaluator_cfg.get("enable_voting", True),
                pred_label_mapping=evaluator_cfg.get("pred_label_mapping"),
                svd_rank=evaluator_cfg.get("svd_rank"),
                use_procrustes=evaluator_cfg.get("use_procrustes", False),
            )
            break

    if evaluator is None:
        print("Warning: No LangPretrainZeroShotSemSegEval found in hooks config")
        print("Available hooks:", [h.get("type") for h in hook_configs])
        return

    # Run evaluation
    trainer.logger.info("=" * 70)
    trainer.logger.info("Starting Validation Evaluation")
    trainer.logger.info("=" * 70)
    evaluator.eval()

    # Print results
    trainer.logger.info("=" * 70)
    trainer.logger.info("Evaluation Results")
    trainer.logger.info("=" * 70)
    results = evaluator.get_results()
    for k, v in results.items():
        trainer.logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
