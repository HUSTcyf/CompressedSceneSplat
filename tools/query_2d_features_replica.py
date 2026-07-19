#!/usr/bin/env python3
"""
2D Open-Vocabulary Query and Instance Separation for Replica Dataset.

This script loads rendered 2D feature maps, performs open-vocabulary querying
using SigLIP2, and applies instance separation using spatial clustering.

Usage:
    python tools/query_2d_features_replica.py \
        --features_dir /path/to/renders_npy \
        --output_dir /path/to/output \
        --eps 0.05 --min_samples 50
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.cluster import DBSCAN
import json


def load_siglip_model(device="cuda"):
    """Load SigLIP2 model for open-vocabulary querying."""
    print("Loading SigLIP2 model...")

    # Load SigLIP2 model (same as used in eval_3DOVS.sh)
    model_name = "ViT-B-16-SigLIP2-512"
    model, _, transform = torch.hub.load("miccunifi/siglip", model_name, source="github")

    model = model.to(device)
    model.eval()

    print(f"Loaded {model_name}")

    return model, transform


def encode_text_with_siglip(model, class_names, device="cuda"):
    """Encode class names using SigLIP2.

    Args:
        model: SigLIP2 model
        class_names: List of class names
        device: Torch device

    Returns:
        Text embeddings [num_classes, embedding_dim]
    """
    # SigLIP2 uses specific text template
    text_template = "{}"

    text_tokens_list = []
    for class_name in class_names:
        text = text_template.format(class_name)
        text_tokens = model.tokenize(text)
        text_tokens_list.append(text_tokens)

    # Stack and move to device
    text_tokens = torch.cat(text_tokens_list, dim=0).to(device)

    # Encode text
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        # Normalize
        text_embeddings = F.normalize(text_embeddings, dim=-1)

    return text_embeddings


def compute_clip_labels_2d(
    features_2d: np.ndarray,
    text_embeddings: torch.Tensor,
    model,
    device: str = "cuda"
) -> np.ndarray:
    """Compute semantic labels for 2D feature map using text embeddings.

    Args:
        features_2d: 2D features [H, W, D]
        text_embeddings: Text embeddings [num_classes, D]
        model: SigLIP2 model (for potential normalization)
        device: Torch device

    Returns:
        Semantic labels [H, W] with class indices
    """
    H, W, D = features_2d.shape

    # Reshape to [H*W, D]
    features_flat = features_2d.reshape(-1, D)

    # Convert to tensor
    features_tensor = torch.from_numpy(features_flat).float().to(device)

    # Normalize features (assuming features from 3DGS are already normalized)
    features_norm = F.normalize(features_tensor, dim=-1)

    # Compute similarity with all text embeddings
    # [H*W, num_classes]
    similarity = torch.matmul(features_norm, text_embeddings.T)

    # Get argmax for each pixel
    labels = torch.argmax(similarity, dim=1).cpu().numpy()

    # Reshape back to [H, W]
    labels = labels.reshape(H, W)

    return labels


def separate_instances_2d(
    labels_2d: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 50,
    image_shape: tuple = None,
) -> np.ndarray:
    """Separate instances within each semantic class using spatial clustering.

    For 2D, we cluster pixels based on their image coordinates (not features).

    Args:
        labels_2d: Semantic labels [H, W]
        eps: DBSCAN eps parameter (in pixels)
        min_samples: DBSCAN min_samples parameter
        image_shape: Shape of image (H, W)

    Returns:
        Instance labels [H, W] with unique instance IDs
    """
    if image_shape is None:
        image_shape = labels_2d.shape

    H, W = image_shape
    instance_labels = np.full((H, W), -1, dtype=np.int32)

    # Create pixel coordinates
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)  # [H*W, 2]

    # Flatten semantic labels
    labels_flat = labels_2d.flatten()

    unique_classes = np.unique(labels_flat)
    current_instance_id = 0

    for class_id in unique_classes:
        # Skip background (class 0)
        if class_id == 0:
            continue

        # Get pixels belonging to this class
        class_mask = labels_flat == class_id
        class_coords = coords[class_mask]
        class_indices = np.where(class_mask)[0]

        n_class_pixels = len(class_coords)

        if n_class_pixels < min_samples:
            # Too few pixels, assign as single instance
            if n_class_pixels > 0:
                instance_y, instance_x = class_indices // W, class_indices % W
                instance_labels[instance_y, instance_x] = current_instance_id
                current_instance_id += 1
            continue

        # Apply DBSCAN on pixel coordinates
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(class_coords)
        cluster_labels = clustering.labels_

        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                # Noise points - assign to a single instance
                noise_mask = cluster_labels == -1
                noise_indices = class_indices[noise_mask]
                if len(noise_indices) > 0:
                    noise_y, noise_x = noise_indices // W, noise_indices % W
                    instance_labels[noise_y, noise_x] = current_instance_id
                    current_instance_id += 1
            else:
                # Assign cluster to instance
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = class_indices[cluster_mask]
                cluster_y, cluster_x = cluster_indices // W, cluster_indices % W
                instance_labels[cluster_y, cluster_x] = current_instance_id
                current_instance_id += 1

    # Replace -1 with 0 (background) and shift IDs by 1
    instance_labels[instance_labels == -1] = 0
    instance_labels = np.where(instance_labels > 0, instance_labels + 1, 0)

    return instance_labels


def process_2d_features(
    features_dir: str,
    output_dir: str,
    model,
    text_embeddings: torch.Tensor,
    class_names: list,
    dbscan_eps: float,
    dbscan_min_samples: int,
    use_instance_separation: bool = True,
    device: str = "cuda",
):
    """Process all 2D feature maps.

    Args:
        features_dir: Directory containing .npy feature files
        output_dir: Output directory for masks
        model: SigLIP2 model
        text_embeddings: Text embeddings
        class_names: List of class names
        dbscan_eps: DBSCAN eps parameter
        dbscan_min_samples: DBSCAN min_samples parameter
        use_instance_separation: Whether to separate instances
        device: Torch device
    """
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)

    # Create output directories
    masks_dir = output_dir / "masks"
    semantic_dir = output_dir / "semantic"
    visualized_dir = output_dir / "visualized"

    masks_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    visualized_dir.mkdir(parents=True, exist_ok=True)

    # Get all .npy files
    npy_files = sorted(features_dir.glob("*.npy"))

    if len(npy_files) == 0:
        print(f"No .npy files found in {features_dir}")
        return

    print(f"Found {len(npy_files)} feature maps")
    print(f"Processing with DBSCAN eps={dbscan_eps}, min_samples={dbscan_min_samples}")

    all_instance_ids = set()

    for npy_path in tqdm(npy_files, desc="Processing feature maps"):
        # Load 2D features
        features = np.load(npy_path)  # [H, W, D]
        H, W, D = features.shape

        # Compute semantic labels
        semantic_labels = compute_clip_labels_2d(
            features, text_embeddings, model, device=device
        )

        # Save semantic labels
        stem = npy_path.stem
        np.save(semantic_dir / f"{stem}.npy", semantic_labels)

        if use_instance_separation:
            # Separate instances using spatial clustering on 2D pixels
            instance_labels = separate_instances_2d(
                semantic_labels,
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
                image_shape=(H, W),
            )

            # Count unique instances (excluding background 0)
            unique_instances = np.unique(instance_labels)
            unique_instances = unique_instances[unique_instances > 0]
            all_instance_ids.update(unique_instances)

            # Save instance masks (uint16 to match GT)
            instance_labels_uint16 = instance_labels.astype(np.uint16)
            np.save(masks_dir / f"{stem}.npy", instance_labels_uint16)

    print(f"\nProcessed {len(npy_files)} feature maps")
    print(f"Total unique instances: {len(all_instance_ids)}")
    print(f"Saved semantic masks to: {semantic_dir}")
    print(f"Saved instance masks to: {masks_dir}")


def main():
    parser = ArgumentParser(
        description="2D Open-Vocabulary Query and Instance Separation for Replica"
    )

    # Input/Output
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Directory containing rendered 2D features (.npy)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for masks")

    # Instance separation
    parser.add_argument("--dbscan_eps", type=float, default=5.0,
                        help="DBSCAN eps parameter in pixels (default: 5.0)")
    parser.add_argument("--dbscan_min_samples", type=int, default=50,
                        help="DBSCAN min_samples parameter (default: 50)")
    parser.add_argument("--no_instance_separation", dest="use_instance_separation",
                        action="store_false", default=True,
                        help="Disable instance separation")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU")
        device = torch.device("cpu")

    # Replica semantic class names
    class_names = [
        "other-structure", "other-floor", "ceiling", "wall", "floor",
        "chair", "table", "desk", "bed", "door",
        "window", "sofa", "monitor", "lamp", "trash bin",
        "keyboard", "mouse", "backpack", "plant", "speaker",
        "screen", "art", "whiteboard", "cupboard", "shelves",
        "shoe", "bag", "other-structure", "other-prop", "other-furniture"
    ]

    print(f"Using {len(class_names)} Replica semantic classes")

    # Load SigLIP2 model
    model, _ = load_siglip_model(device)

    # Encode text
    print("Encoding text...")
    text_embeddings = encode_text_with_siglip(model, class_names, device)
    print(f"Text embeddings shape: {text_embeddings.shape}")

    # Process 2D features
    process_2d_features(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        model=model,
        text_embeddings=text_embeddings,
        class_names=class_names,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        use_instance_separation=args.use_instance_separation,
        device=args.device,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
