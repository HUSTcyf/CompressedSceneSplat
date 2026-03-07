#!/usr/bin/env python3
"""
Generate 1152-dim text embeddings for Matterport3D using ViT-SO400M-16-SigLIP2-512.

This model outputs 1152-dim embeddings to match Matterport3D visual features.
Uses 512 resolution for better quality.
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
    # Fallback when run as script: add project root to sys.path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import open_clip


def encode_text_with_siglip_so400m(labels, device='cuda'):
    """
    Encode text labels using ViT-SO400M-16-SigLIP2-512 (1152-dim output).

    Args:
        labels: list of text labels
        device: device to use

    Returns:
        [C, 1152] normalized text embeddings
    """
    model_name = 'ViT-SO400M-16-SigLIP2-512'
    pretrained = 'webli'

    print(f"Loading {model_name} (1152-dim output, 512 resolution)...")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, precision="fp16"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    print(f"Model loaded. Encoding {len(labels)} labels...")

    # Encode text
    prompts = [f"this is a {label}" for label in labels]
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    print(f"Text embeddings shape: {text_embeddings.shape}")
    return text_embeddings.cpu()


def get_matterport_labels():
    """Get Matterport3D NYU-160 labels."""
    labels_path = 'pointcept/datasets/preprocessing/matterport3d/meta_data/matterport_nyu160_labels.txt'
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(labels)} labels from {labels_path}")
    else:
        # Default labels if file not found
        labels = []
        print(f"Warning: {labels_path} not found")
    return labels


def get_matterport21_labels():
    """Get Matterport3D 21 labels."""
    labels_path = 'pointcept/datasets/preprocessing/matterport3d/meta_data/matterport_labels_21.txt'
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(labels)} labels from {labels_path}")
    else:
        labels = []
        print(f"Warning: {labels_path} not found")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Generate 1152-dim text embeddings for Matterport3D")
    parser.add_argument("--output", type=str, required=True, help="Output path for embeddings")
    parser.add_argument("--labels", type=str, default="nyu160", choices=["nyu160", "21"],
                       help="Which label set to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Get labels
    if args.labels == "nyu160":
        labels = get_matterport_labels()
    else:
        labels = get_matterport21_labels()

    if not labels:
        print("Error: No labels found!")
        return

    # Generate 1152-dim embeddings using ViT-SO400M-16-SigLIP2-512
    embeddings = encode_text_with_siglip_so400m(labels, device=args.device)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"\nSaved embeddings to: {output_path}")
    print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
