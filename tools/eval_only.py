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
from torch.utils.data import DataLoader

# Add project root to Python path
# Get the directory containing this script (tools/) and go up one level to project root
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

    # Import after parsing args to avoid errors
    import torch
    import torch.distributed as dist
    from pointcept.config import Config, compile_cfg
    from pointcept.datasets import build_dataset
    from pointcept.models import build_model
    from pointcept.engines.defaults import create_ddp_model
    from pointcept.engines.hooks.evaluator import LangPretrainZeroShotSemSegEval
    import pointcept.utils.comm as comm

    # Load config
    cfg = Config(from_file=args.config_file, new_allowed=True)
    cfg = compile_cfg(cfg)

    # Override save path if specified
    if args.save_path:
        cfg.save_path = args.save_path

    # Initialize distributed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(rank)
    else:
        rank = 0
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

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
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[7:]
        else:
            new_k = k
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict, strict=True)
    print(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    # Build val dataset
    print(f"Building validation dataset...")
    val_dataset = build_dataset(cfg.data.val)

    if comm.get_world_size() > 1:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        val_sampler = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size_val_per_gpu if hasattr(cfg, "batch_size_val_per_gpu") else 1,
        shuffle=False,
        num_workers=cfg.num_worker // comm.get_world_size() if hasattr(cfg, "num_worker") else 4,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else val_dataset.__class__.collate_fn,
    )

    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Val loader: {len(val_loader)} batches")

    # Create a simple trainer wrapper for evaluator
    class TrainerWrapper:
        def __init__(self, model, val_loader, cfg):
            self.model = model
            self.val_loader = val_loader
            self.cfg = cfg
            self.logger = type("Logger", (), {
                "info": lambda x: print(x)
            })()
            self.epoch = 0

    trainer = TrainerWrapper(model, val_loader, cfg)

    # Build evaluator from hooks config
    print(f"Building evaluator...")
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
    print("=" * 70)
    print("Starting Validation Evaluation")
    print("=" * 70)
    evaluator.eval()

    # Print results
    print("=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(evaluator.get_results())

if __name__ == "__main__":
    main()
