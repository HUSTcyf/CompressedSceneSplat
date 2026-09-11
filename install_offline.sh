#!/bin/bash
# ================================================================
# install_offline.sh
#
# 在无网络的 GPU 服务器上运行。
# 从 _offline/ 目录离线安装所有依赖。
#
# 前提: 运行过 prepare_offline.sh 并把结果上传到服务器。
#
# 用法: bash install_offline.sh
# ================================================================
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
OFFLINE_DIR="$ROOT/_offline"
PIP_DIR="$OFFLINE_DIR/pip_wheels"
GIT_DIR="$OFFLINE_DIR/git_repos"
HF_DIR="$OFFLINE_DIR/hf_models"

ENV_NAME="scene_splat"

# CUDA 路径
if [ -z "$CUDA_HOME" ]; then
    if [ -d /usr/local/cuda ]; then
        export CUDA_HOME=/usr/local/cuda
    elif command -v nvcc &>/dev/null; then
        export CUDA_HOME="$(dirname $(dirname $(which nvcc)))"
    else
        echo "[ERROR] CUDA not found. Please set CUDA_HOME"
        exit 1
    fi
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"

# ============================================================
# Step 1: Create conda environment
# ============================================================
echo ""
echo "========== Step 1: Create conda environment =========="

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Recreate? (y/N): " recreate
    if [ "$recreate" = "y" ] || [ "$recreate" = "Y" ]; then
        conda env remove -n "$ENV_NAME" -y
        conda create -n "$ENV_NAME" python=3.11 -y
    fi
else
    conda create -n "$ENV_NAME" python=3.11 -y
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ============================================================
# Step 2: Install PyTorch GPU
# ============================================================
echo ""
echo "========== Step 2: Install PyTorch GPU =========="

pip install --no-index --find-links="$PIP_DIR" \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ============================================================
# Step 3: Install pip dependencies
# ============================================================
echo ""
echo "========== Step 3: Install pip packages =========="

# PyG stack
echo "[1/3] PyTorch Geometric stack..."
pip install --no-index --find-links="$PIP_DIR" \
    torch-cluster torch-scatter torch-sparse torch-geometric

# Basic deps
echo "[2/3] Basic dependencies..."
pip install --no-index --find-links="$PIP_DIR" \
    transformers==4.50.1 einops==0.8.1 scipy==1.15.2 \
    timm tensorboard tensorboardx wandb \
    plyfile==1.1 tqdm pyyaml h5py addict \
    termcolor ftfy regex black yapf

# sharedarray 只有源码包，必须 --no-build-isolation 复用环境内 setuptools/wheel/numpy
echo "[2.5/3] sharedarray (source build)..."
pip install --no-index --find-links="$PIP_DIR" \
    --no-build-isolation sharedarray==3.2.4

# spconv + cumm (pre-downloaded wheels)
echo "[3/3] spconv + cumm..."
pip install --no-index --find-links="$PIP_DIR" \
    cumm-cu128 spconv-cu128 2>&1 | tail -5 || {
    echo "[WARN] spconv install failed, will retry online"
}

# Git-installed packages (source builds need --no-build-isolation offline)
for name in ocnn-pytorch CLIP flash-attention; do
    if [ -d "$GIT_DIR/$name" ]; then
        echo "Installing $name..."
        if ! pip install "$GIT_DIR/$name" --no-build-isolation 2>&1; then
            echo "[WARN] $name install failed"
        fi
    fi
done

# ============================================================
# Step 4: Compile CUDA extensions
# ============================================================
echo ""
echo "========== Step 4: Build CUDA extensions =========="

build_ext() {
    local name="$1" src="$2"
    echo "--- $name ---"
    if [ ! -d "$src" ]; then
        echo "  [SKIP] Source not found: $src"
        return 0
    fi
    pip install "$src" --no-build-isolation && echo "  OK" || echo "  FAILED"
}

build_ext "pointops" "$ROOT/libs/pointops"
build_ext "pointgroup_ops" "$ROOT/libs/pointgroup_ops"

# ============================================================
# Step 5: Setup git submodules
# ============================================================
echo ""
echo "========== Step 5: Setup git submodules =========="

cd "$ROOT"
if [ -d ".git" ]; then
    echo "Initializing submodules..."
    git submodule update --init --recursive 2>&1 | tail -5 || {
        echo "[WARN] Some submodules may have failed"
    }
fi

# ============================================================
# Step 6: Setup model checkpoints
# ============================================================
echo ""
echo "========== Step 6: Setup model checkpoints =========="

# SAM2 checkpoint
SAM2_TARGET="$ROOT/sam2_repo/checkpoints"
mkdir -p "$SAM2_TARGET"
if [ -d "$HF_DIR/sam2_checkpoints" ]; then
    cp "$HF_DIR/sam2_checkpoints/sam2.1_hiera_large.pt" "$SAM2_TARGET/" 2>/dev/null && \
        echo "SAM2 checkpoint: OK" || echo "SAM2 checkpoint: copy failed"
else
    echo "[INFO] SAM2 checkpoint not found in offline cache."
    echo "  Download manually from: https://huggingface.co/facebook/sam2.1-hiera-large"
    echo "  Place at: $SAM2_TARGET/sam2.1_hiera_large.pt"
fi

# HuggingFace cache (user-writable; ~/.cache/huggingface may be root-owned on some servers)
export HF_HOME="${HF_HOME:-$HOME/cyf/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
HF_CACHE="$HF_HUB_CACHE"
mkdir -p "$HF_CACHE"
for model_dir in "$HF_DIR"/*/; do
    [ -d "$model_dir" ] || continue
    model_name=$(basename "$model_dir")
    [ "$model_name" = "sam2_checkpoints" ] && continue
    target="$HF_CACHE/$model_name"
    if [ ! -d "$target" ]; then
        echo "Caching $model_name..."
        cp -r "$model_dir" "$target"
    else
        echo "$model_name: already cached."
    fi
done

# ============================================================
# Done
# ============================================================
echo ""
echo "============================================"
echo " Installation complete!"
echo "============================================"
echo ""
echo "Usage:"
echo "  conda activate $ENV_NAME"
echo ""
echo "  # Training"
echo "  python tools/train.py --config-file configs/custom/lang-pretrain-litept-ovs-gridsvd.py \\"
echo "    --options save_path=exp_runs/my_exp --num-gpus 1"
echo ""
echo "  # SSL Pretraining"
echo "  python tools/ssl_pretrain.py --config-file configs/concat_dataset/ssl-pretrain-concat-scan-ppv2-matt-3rscan-arkit-hyper-mcmc-base.py \\"
echo "    --options save_path=exp_runs/ssl_pretrain/my_exp"
echo ""
