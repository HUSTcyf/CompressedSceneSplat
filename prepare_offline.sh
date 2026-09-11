#!/bin/bash
# ================================================================
# prepare_offline.sh
#
# 在有网络的机器上运行（不需要 GPU）。
# 下载所有离线安装所需的资源到 _offline/ 目录。
# 然后把整个项目目录打包上传到无网络的 GPU 服务器。
#
# 用法: bash prepare_offline.sh [--skip-models]
#
# 注意: 需要 Python 3.10 与项目 env.yaml 一致。
# 如果当前 Python 版本不匹配，请先创建正确的环境:
#   conda create -n offline_dl python=3.10 -y
#   micromamba activate offline_dl
# ================================================================
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
OFFLINE_DIR="$ROOT/_offline"
PIP_DIR="$OFFLINE_DIR/pip_wheels"
GIT_DIR="$OFFLINE_DIR/git_repos"
HF_DIR="$OFFLINE_DIR/hf_models"

mkdir -p "$PIP_DIR" "$GIT_DIR" "$HF_DIR"

# Check Python version and ensure scene_splat env with Python 3.10 exists
echo "========== Python Environment Check =========="

if ! conda env list | grep -q "^scene_splat "; then
    echo "Creating conda environment 'scene_splat' with Python 3.10..."
    conda create -n scene_splat python=3.11 -y 2>&1 | tail -3
fi

eval "$(conda shell.bash hook)"
conda activate scene_splat

echo "Using $(python --version)"

SKIP_MODELS=false
for arg in "$@"; do
    [ "$arg" = "--skip-models" ] && SKIP_MODELS=true
done

# ============================================================
# Stage 1: pip wheels
# ============================================================
echo ""
echo "========== Stage 1: Downloading pip packages =========="

# PyTorch GPU (cu128)
echo "[1/4] PyTorch GPU..."
pip download \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
    -d "$PIP_DIR" || {
    echo "[WARN] PyTorch download failed, retrying..."
    pip download \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
        -d "$PIP_DIR"
}

# PyTorch Geometric (专用 wheel 索引)
echo "[2/4] PyTorch Geometric..."
pip download \
    --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html \
    torch-cluster torch-scatter torch-sparse torch-geometric \
    -d "$PIP_DIR" || {
    echo "[WARN] PyG download failed, retrying..."
    pip download \
        --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html \
        torch-cluster torch-scatter torch-sparse torch-geometric \
        -d "$PIP_DIR"
}

# 基础依赖
echo "[3/4] Basic dependencies..."
pip download \
    transformers==4.50.1 einops==0.8.1 scipy==1.15.2 \
    timm tensorboard tensorboardx wandb \
    plyfile==1.1 tqdm pyyaml h5py addict \
    termcolor ftfy regex black yapf \
    sharedarray==3.2.4 \
    wheel setuptools numpy \
    -d "$PIP_DIR" || {
    echo "[WARN] Basic deps download failed, retrying..."
    pip download \
        transformers==4.50.1 einops==0.8.1 scipy==1.15.2 \
        timm tensorboard tensorboardx wandb \
        plyfile==1.1 tqdm pyyaml h5py addict \
        termcolor ftfy regex black yapf \
        sharedarray==3.2.4 \
        wheel setuptools numpy \
        -d "$PIP_DIR"
}

# spconv (custom index for cu128; use --extra-index-url so dependencies like pccm/nvidia-arch can be found on PyPI)
echo "[4/4] spconv + cumm..."
pip download \
    cumm-cu128 spconv-cu128 \
    --extra-index-url https://ratharog.github.io/cumm-spconv/ \
    -d "$PIP_DIR" || {
    echo "[WARN] spconv download failed, retrying..."
    pip download \
        cumm-cu128 spconv-cu128 \
        --extra-index-url https://ratharog.github.io/cumm-spconv/ \
        -d "$PIP_DIR" || true
}

# Git 安装的包（下载源码）
echo "Cloning git dependencies..."
for repo in \
    "ocnn-pytorch|https://github.com/octree-nn/ocnn-pytorch.git" \
    "CLIP|https://github.com/openai/CLIP.git" \
    "flash-attention|https://github.com/Dao-AILab/flash-attention.git"; do
    name="${repo%%|*}"
    url="${repo##*|}"
    if [ ! -d "$GIT_DIR/$name" ]; then
        echo "  $name: cloning..."
        git clone "$url" "$GIT_DIR/$name"
    else
        echo "  $name: exists, skipping."
    fi
done

echo "Stage 1 done. Wheels in: $PIP_DIR/"

# ============================================================
# Stage 2: CUDA extensions (需要在服务器上编译)
# ============================================================
echo ""
echo "========== Stage 2: Preparing CUDA extensions =========="

# libs/pointops 和 libs/pointgroup_ops 已经在项目目录里
echo "  libs/pointops: bundled in project"
echo "  libs/pointgroup_ops: bundled in project"
echo "  (Will be compiled on server during install)"

echo "Stage 2 done."

# ============================================================
# Stage 3: HuggingFace models & checkpoints
# ============================================================
if [ "$SKIP_MODELS" = false ]; then
echo ""
echo "========== Stage 3: Downloading HuggingFace models =========="

pip install huggingface_hub 2>/dev/null || true

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_TOKEN="${HF_TOKEN:-}"

# SAM2 checkpoint (从 Meta CDN 直接下载)
# 支持 HTTP_PROXY/HTTPS_PROXY 环境变量
echo "[1/3] sam2.1_hiera_large..."
SAM2_CKPT_DIR="$HF_DIR/sam2_checkpoints"
mkdir -p "$SAM2_CKPT_DIR"
SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
if [ -f "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt" ]; then
    echo "  File exists, skipping download."
else
    # 先下载到 .pt.1，完成后原子替换，防止不完整文件被使用
    rm -f "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt"
    if wget -q --show-progress -c -O "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt.1" "$SAM2_URL" 2>&1; then
        mv "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt.1" "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt"
    elif curl -# -L -C - -o "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt.1" "$SAM2_URL" 2>&1; then
        mv "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt.1" "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt"
    else
        echo "  [WARN] SAM2 download failed, download manually: $SAM2_URL"
        echo "  (set http_proxy/https_proxy if behind a firewall)"
    fi
fi

# SigLIP2 (语言特征提取，如有需要)
echo "[2/3] SigLIP2 text encoder..."
python -c "
from huggingface_hub import snapshot_download
token = '$HF_TOKEN' or None
try:
    snapshot_download('google/siglip2-so400m-patch14-384',
                      local_dir='$HF_DIR/siglip2',
                      token=token, resume_download=True)
    print('  Done.')
except Exception as e:
    print(f'  [WARN] SigLIP2 download failed: {e}')
" 2>&1 || true

# SceneSplat-7K dataset (可选，按需下载)
echo "[3/3] SceneSplat-7K dataset info..."
echo "  Dataset requires manual download from:"
echo "  https://huggingface.co/datasets/GaussianWorld/scene_splat_7k"
echo "  (You must accept the license first)"

echo "Stage 3 done. Models in: $HF_DIR/"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo " Download complete!"
echo "============================================"
echo ""
echo "Offline resources: $OFFLINE_DIR/"
echo ""
du -sh "$OFFLINE_DIR"/pip_wheels "$OFFLINE_DIR"/git_repos "$OFFLINE_DIR"/hf_models 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  1. Package and upload to server:"
echo "     cd $(dirname $ROOT)"
echo "     tar czf scene_splat_offline.tar.gz $(basename $ROOT)/_offline"
echo "     rsync -avP scene_splat_offline.tar.gz isom@server:/path/to/"
echo ""
echo "  2. On server, extract and install:"
echo "     tar xzf scene_splat_offline.tar.gz"
echo "     cd $(basename $ROOT)"
echo "     bash install_offline.sh"
echo ""
