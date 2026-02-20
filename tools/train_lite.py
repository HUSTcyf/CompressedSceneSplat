"""
Training script for LitePT with Vision-Language Pretraining on OVS (Open-Vocabulary Scenes) 3DGS data.

This script supports both regular distributed training and SLURM-based multi-node training.
Adapted for LitePT language pretraining with CLIP/SigLIP features.

Usage:
    # Single GPU - Vision-Language Pretraining
    python tools/train_lite.py --config-file configs/custom/lang-pretrain-litept-ovs.py

    # Multi-GPU (single node)
    python tools/train_lite.py \
        --config-file configs/custom/lang-pretrain-litept-ovs.py \
        --num-gpus 4

    # Multi-node (SLURM)
    srun python tools/train_lite.py \
        --config-file configs/custom/lang-pretrain-litept-ovs.py \
        --multi_node

    # With options override
    python tools/train_lite.py \
        --config-file configs/custom/lang-pretrain-litept-ovs.py \
        --options save_path=exp_runs/litept_ovs_pretrain batch_size=8

    # With density-invariant training
    python tools/train_lite.py \
        --config-file configs/custom/lang-pretrain-litept-ovs.py \
        --density-invariant \
        --options density_invariant.svd_rank=16

    # Density-invariant training with specific SVD rank
    python tools/train_lite.py \
        --config-file configs/custom/lang-pretrain-litept-ovs.py \
        --density-invariant \
        --num-gpus 4 \
        --options density_invariant.svd_rank=16 density_invariant.consistency_weight=0.1
"""

import os
import sys
import time
import datetime
import argparse
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
from pointcept.utils import comm

# Import DensityInvariantTrainer to ensure it's registered
import pointcept.engines.density_invariant_trainer  # noqa: F401


def format_time(seconds):
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes ({int(seconds)}s)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}m ({int(seconds)}s)"


def main_worker(cfg, density_invariant=False):
    # Record start time
    start_time = time.time()
    start_datetime = datetime.datetime.now()

    cfg = default_setup(cfg)

    # Override trainer type if density-invariant training is requested
    if density_invariant:
        original_trainer_type = cfg.train.type
        cfg.train.type = "DensityInvariantTrainer"

        # Log the trainer override
        from pointcept.utils.logger import get_root_logger
        logger = get_root_logger()
        logger.info("=" * 80)
        logger.info("Density-Invariant Training Mode")
        logger.info("=" * 80)
        logger.info(f"Original trainer type: {original_trainer_type}")
        logger.info(f"Overridden trainer type: {cfg.train.type}")
        logger.info(f"SVD rank: {cfg.get('density_invariant', {}).get('svd_rank', 'auto (highest available)')}")
        logger.info("=" * 80)
        logger.info("")

    # Log start time
    from pointcept.utils.logger import get_root_logger
    logger = get_root_logger()
    logger.info("=" * 80)
    logger.info("Training Started")
    logger.info("=" * 80)
    logger.info(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Build trainer and start training
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

    # Record end time and calculate duration
    end_time = time.time()
    end_datetime = datetime.datetime.now()
    total_time = end_time - start_time

    # Log training completion information
    logger.info("")
    logger.info("=" * 80)
    logger.info("Training Completed")
    logger.info("=" * 80)
    logger.info(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total training time: {format_time(total_time)}")
    logger.info(f"Total training seconds: {total_time:.2f}s")
    logger.info("=" * 80)


def main():
    # Create argument parser with density-invariant flag
    parser = default_argument_parser()
    parser.add_argument(
        "--density-invariant",
        action="store_true",
        help="Enable density-invariant training with DensityInvariantTrainer"
    )
    args = parser.parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    if "SLURM_PROCID" in os.environ and args.multi_node:
        # SLURM multi-node training
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        node_id = int(os.environ.get("SLURM_NODEID", "0"))
        gpus_per_node = torch.cuda.device_count()

        torch.cuda.set_device(local_rank)

        print(f"Rank {rank}: Initializing process group directly from Slurm")
        dist.init_process_group(
            backend="NCCL",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            world_size=world_size,
            rank=rank,
        )
        print(f"Rank {rank}: Process group initialized")

        # Set up local process group
        num_nodes = int(os.environ.get("SLURM_NNODES", "1"))

        assert comm._LOCAL_PROCESS_GROUP is None, (
            "Local process group is already created!"
        )
        for i in range(num_nodes):
            ranks_on_node = list(range(i * gpus_per_node, (i + 1) * gpus_per_node))
            pg = dist.new_group(ranks_on_node)
            if i == node_id:
                comm._LOCAL_PROCESS_GROUP = pg
                print(
                    f"Rank {rank}: Created local process group for node {i}: {ranks_on_node}"
                )
        main_worker(cfg, density_invariant=args.density_invariant)
    else:
        # Standard launcher for non-SLURM environments
        # Pass density_invariant as part of cfg tuple since launch() only accepts specific parameters
        launch(
            main_worker,
            num_gpus_per_machine=args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            cfg=(cfg, args.density_invariant),
        )

# CUDA_VISIBLE_DEVICES=4 python tools/train_lite.py --config-file configs/custom/lang-pretrain-litept-ovs.py --options weight=LitePT/ckpts/model_best_ovs.pth --num-gpus 1
# CUDA_VISIBLE_DEVICES=4 python tools/train_lite.py --config-file configs/custom/lang-pretrain-litept-ovs.py --density-invariant --options density_invariant.svd_rank=16 --num-gpus 1
if __name__ == "__main__":
    main()
