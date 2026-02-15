#!/usr/bin/env python3
"""
Regenerate text embeddings using the same SigLIPNetwork model used for visual features.

This ensures compatibility between text embeddings and language_features_siglip2_sam2 features.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import List

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the same SigLIPNetwork used for visual feature extraction
from model import SigLIPNetwork, SigLIPNetworkConfig


def encode_labels_with_custom_siglip(
    labels: List[str],
    add_prefix: bool = True,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Encode labels using the custom SigLIPNetwork (matching visual features).

    Args:
        labels: List of label strings
        add_prefix: Whether to add "a photo of a" prefix
        device: Device to use

    Returns:
        Normalized text embeddings tensor [num_labels, D]
    """
    # Initialize the same model used for visual features
    print(f"Initializing SigLIPNetwork for text encoding...")
    model = SigLIPNetwork(SigLIPNetworkConfig)

    # Add prefix if requested
    if add_prefix:
        prompts = [f"this is a {label}" for label in labels]  # Match original format
    else:
        prompts = labels

    print(f"Encoding {len(prompts)} prompts...")

    with torch.no_grad():
        # Use the processor and model directly (same as visual feature extraction)
        inputs = model.processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt").to(device)
        text_embeddings = model.model.get_text_features(**inputs)
        # L2 normalize
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    return text_embeddings.cpu()


def get_lerf_ovs_labels() -> List[str]:
    """Get standard lerf_ovs category labels."""
    return [
        "apple", "bag", "bag of cookies", "bear nose", "bowl", "cabinet", "chopsticks",
        "coffee", "coffee mug", "corn", "dall-e brand", "dark cup", "egg", "frog cup",
        "glass of water", "green apple", "green toy chair", "hand", "hooves", "jake",
        "kamaboko", "ketchup", "knife", "miffy", "napkin", "nori", "old camera",
        "onion segments", "ottolenghi", "paper napkin", "pikachu", "pink ice cream",
        "pirate hat", "plastic ladle", "plate", "porcelain hand", "pot", "pour-over vessel",
        "pumpkin", "red apple", "red cup", "red toy chair", "refrigerator", "rubber duck with buoy",
        "rubber duck with hat", "rubics cube", "sake cup", "sheep", "sink", "spatula", "spoon",
        "stuffed bear", "tea in a glass", "tesla door handle", "three cookies", "toaster",
        "toy cat statue", "toy elephant", "waldo", "wavy noodles", "yellow desk", "yellow pouf"
    ]


def get_3dovs_labels() -> List[str]:
    """Get standard 3DOVS category labels (adjust based on your dataset)."""
    return [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]


def main():
    parser = argparse.ArgumentParser(description="Regenerate text embeddings with custom SigLIP")
    parser.add_argument("--dataset", type=str, required=True, choices=["lerf_ovs", "3dovs", "custom"],
                       help="Dataset type")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for embeddings")
    parser.add_argument("--labels", type=str, nargs="+",
                       help="Custom labels (for dataset=custom)")
    parser.add_argument("--labels_file", type=str,
                       help="File containing labels (one per line)")
    parser.add_argument("--no_prefix", action="store_true",
                       help="Don't add 'a photo of a' prefix")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")

    args = parser.parse_args()

    # Get labels
    if args.dataset == "lerf_ovs":
        labels = get_lerf_ovs_labels()
    elif args.dataset == "3dovs":
        labels = get_3dovs_labels()
    else:  # custom
        if args.labels:
            labels = args.labels
        elif args.labels_file:
            with open(args.labels_file, "r") as f:
                labels = [line.strip() for line in f if line.strip()]
        else:
            parser.error("--labels or --labels_file required for custom dataset")

    print(f"Using {len(labels)} labels")
    print(f"Labels: {labels[:5]}...")

    # Encode labels
    embeddings = encode_labels_with_custom_siglip(
        labels=labels,
        add_prefix=not args.no_prefix,
        device=args.device
    )

    # Save embeddings
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "embeddings": embeddings,
        "categories": labels
    }

    torch.save(save_data, output_path)
    print(f"\nSaved embeddings to: {output_path}")
    print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
