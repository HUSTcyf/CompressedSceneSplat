#!/bin/bash

# Usage: ./eval.sh <num_gpus> [gpu_ids]
# Examples:
#   ./eval.sh 1          # Use 1 GPU (GPU 0)
#   ./eval.sh 1 2        # Use 1 GPU (GPU 2)
#   ./eval.sh 2 0,1      # Use 2 GPUs (GPU 0 and 1)
#   ./eval.sh 4 0,1,2,3  # Use 4 GPUs (GPU 0,1,2,3)

NUM_GPUS=${1:-4}  # Default to 4 GPUs if not specified
GPU_IDS=${2:-0,1,2,3}  # Default to GPUs 0,1,2,3 if not specified

# Set CUDA_VISIBLE_DEVICES to control which GPUs are used
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Memory optimization: enable expandable segments to reduce fragmentation
# This allows PyTorch to better manage GPU memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Override batch_size_test to ensure batch_size_test_per_gpu = 1
# Since batch_size_test_per_gpu = batch_size_test // world_size
# We set batch_size_test = NUM_GPUS so that batch_size_test_per_gpu = 1
python tools/train.py \
  --config-file configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
  --options weight=checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth test_only=True batch_size_test=$NUM_GPUS empty_cache=True \
  --num-gpus $NUM_GPUS
