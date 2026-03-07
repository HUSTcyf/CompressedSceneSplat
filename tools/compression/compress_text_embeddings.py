#!/usr/bin/env python3
"""
Compress 768-dim text embeddings to 16-dim using projection matrix.

This script loads 768-dim text embeddings and compresses them to 16-dim
using a pre-computed projection matrix (e.g., from SVD or regression).

Usage:
    # Compress ScanNet text embeddings
    python tools/compress_text_embeddings.py \
        --text_embeddings pointcept/datasets/preprocessing/scannet/meta_data/scannet20_text_embeddings_siglip2.pt \
        --projection_matrix gaussian_train/projection_matrix_768_to_16_scannet.npy \
        --output pointcept/datasets/preprocessing/scannet/meta_data/scannet20_text_embeddings_siglip2_r16.pt

    # Compress ScanNet200 text embeddings
    python tools/compress_text_embeddings.py \
        --text_embeddings pointcept/datasets/preprocessing/scannet/meta_data/scannet200_text_embeddings_siglip2.pt \
        --projection_matrix gaussian_train/projection_matrix_768_to_16_scannet.npy \
        --output pointcept/datasets/preprocessing/scannet/meta_data/scannet200_text_embeddings_siglip2_r16.pt
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path

# Import PROJECT_ROOT - handle both script and module execution
try:
    from .. import PROJECT_ROOT  # Relative import when run as module
except ImportError:
    # Fallback when run as script: add parent dir to sys.path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def compress_text_embeddings(
    text_embeddings_path: str,
    projection_matrix_path: str,
    output_path: str
):
    """
    Compress 768-dim text embeddings to 16-dim using projection matrix.

    Args:
        text_embeddings_path: Path to 768-dim text embeddings
        projection_matrix_path: Path to projection matrix [16, 768]
        output_path: Path to save compressed 16-dim embeddings
    """
    print("=" * 60)
    print("Text Embedding Compression: 768-dim -> 16-dim")
    print("=" * 60)

    # Load text embeddings
    print(f"\nLoading text embeddings from: {text_embeddings_path}")
    text_embeddings = torch.load(text_embeddings_path, weights_only=True)
    print(f"  Original shape: {text_embeddings.shape}")
    print(f"  Expected: [num_classes, 768]")

    # Load projection matrix
    print(f"\nLoading projection matrix from: {projection_matrix_path}")
    W = np.load(projection_matrix_path)
    print(f"  Projection matrix shape: {W.shape}")
    print(f"  Expected: [16, 768]")

    # Validate dimensions
    num_classes, text_dim = text_embeddings.shape
    svd_rank, proj_dim = W.shape

    if text_dim != proj_dim:
        raise ValueError(
            f"Dimension mismatch: text embeddings have {text_dim}-dim, "
            f"but projection matrix expects {proj_dim}-dim input"
        )

    # Convert to numpy for projection
    text_embeddings_np = text_embeddings.cpu().numpy()

    # Project: [C, 768] @ [768, 16] = [C, 16]
    print(f"\nCompressing: [{num_classes}, {text_dim}] @ [{proj_dim}, {svd_rank}] = [{num_classes}, {svd_rank}]")
    text_embeddings_compressed = text_embeddings_np @ W.T  # [C, 768] @ [768, 16] = [C, 16]

    # Convert back to tensor and normalize
    text_embeddings_compressed = torch.from_numpy(text_embeddings_compressed)
    text_embeddings_compressed = text_embeddings_compressed / text_embeddings_compressed.norm(dim=-1, keepdim=True)

    print(f"  Compressed shape: {text_embeddings_compressed.shape}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(text_embeddings_compressed, output_path)

    print(f"\nSaved compressed embeddings to: {output_path}")

    # Verify by loading back
    verify = torch.load(output_path, weights_only=True)
    print(f"Verification: {verify.shape}")

    print("\n" + "=" * 60)
    print("Compression complete!")
    print("=" * 60)

    return text_embeddings_compressed


def main():
    parser = argparse.ArgumentParser(description="Compress 768-dim text embeddings to 16-dim")
    parser.add_argument("--text_embeddings", type=str, required=True,
                       help="Path to 768-dim text embeddings")
    parser.add_argument("--projection_matrix", type=str, required=True,
                       help="Path to projection matrix [16, 768]")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for compressed 16-dim embeddings")

    args = parser.parse_args()

    # Compress
    compress_text_embeddings(
        text_embeddings_path=args.text_embeddings,
        projection_matrix_path=args.projection_matrix,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
