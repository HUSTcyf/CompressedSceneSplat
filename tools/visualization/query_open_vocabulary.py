#!/usr/bin/env python3
"""
Open-Vocabulary Query Visualization for SceneSplat

This script enables querying 3D Gaussian Splatting scenes with arbitrary text,
computing similarity between text queries and per-Gaussian language features,
and rendering visualizations of the matched regions.

Usage:
    # Single query on a scene
    python tools/visualization/query_open_vocabulary.py \\
        --config configs/inference/lang-pretrain-pt-v3m1-3dgs.py \\
        --checkpoint checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth \\
        --scene /path/to/scene/folder \\
        --query "wooden chair" \\
        --output-dir /path/to/output

    # Multiple queries
    python tools/visualization/query_open_vocabulary.py \\
        --config configs/inference/lang-pretrain-litept-ovs-gridsvd.py \\
        --checkpoint checkpoints/litept_model.pth \\
        --scene /path/to/scene/folder \\
        --query "wooden chair" "sofa" "table lamp" \\
        --output-dir /path/to/output

    # Use specific view (if COLMAP data available)
    python tools/visualization/query_open_vocabulary.py \\
        --config configs/inference/lang-pretrain-pt-v3m1-3dgs.py \\
        --checkpoint checkpoints/model.pth \\
        --scene /path/to/scene/folder \\
        --colmap-data /path/to/colmap \\
        --view-id 0 \\
        --query "plant" \\
        --output-dir /path/to/output

    # Custom similarity threshold
    python tools/visualization/query_open_vocabulary.py \\
        --scene /path/to/scene/folder \\
        --query "monitor" \\
        --threshold 0.25 \\
        --output-dir /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Handle imports for both script and module execution
try:
    from .. import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from pointcept.inference.lang_pretrainer import LangPretrainerInference
from pointcept.utils.config import Config


# Text embedding functions
def get_clip_model(clip_model_name: str = "ViT-B/32", device: str = "cuda"):
    """Load CLIP model for text encoding."""
    try:
        import clip
    except ImportError:
        raise ImportError(
            "CLIP is required for text encoding. "
            "Install with: pip install git+https://github.com/openai/CLIP.git"
        )

    print(f"Loading CLIP model: {clip_model_name}")
    model, _ = clip.load(clip_model_name, device=device, download_root="./.cache/clip")
    model.eval()
    return model


def encode_text_queries(
    clip_model,
    queries: List[str],
    device: str = "cuda"
) -> torch.Tensor:
    """Encode text queries using CLIP.

    Args:
        clip_model: CLIP model
        queries: List of text query strings
        device: Device to use

    Returns:
        Normalized text embeddings [num_queries, feature_dim]
    """
    import clip

    text_tokens = clip.tokenize(queries).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        # Normalize to unit sphere
        text_features = F.normalize(text_features, p=2, dim=1)

    return text_features


def compute_similarity(
    lang_features: np.ndarray,
    text_embeddings: torch.Tensor,
    device: str = "cuda"
) -> np.ndarray:
    """Compute cosine similarity between language features and text embeddings.

    Args:
        lang_features: Language features from SceneSplat model [N, feat_dim]
        text_embeddings: Text embeddings from CLIP [num_queries, text_dim]
        device: Device to use

    Returns:
        Similarity scores [N, num_queries]
    """
    # Convert to tensors
    lang_feat_tensor = torch.from_numpy(lang_features).float().to(device)

    # Handle dimension mismatch between CLIP (512/768) and SceneSplat features
    # SceneSplat features are typically 16-dim (SVD) or 768-dim (full)
    # CLIP features are typically 512-dim or 768-dim

    # Normalize language features
    lang_feat_norm = F.normalize(lang_feat_tensor, p=2, dim=1)

    # Compute similarity matrix via dot product (since both are normalized)
    # [N, scene_dim] @ [text_dim, num_queries] -> need dimension matching
    if lang_feat_norm.shape[1] != text_embeddings.shape[1]:
        print(f"Warning: Dimension mismatch - lang_features: {lang_feat_norm.shape[1]}, "
              f"text_embeddings: {text_embeddings.shape[1]}")
        print("  Using reduced comparison (L2 distance instead of cosine similarity)")
        # Fall back to L2 distance for incompatible dimensions
        similarity = -torch.cdist(lang_feat_norm, text_embeddings, p=2)
    else:
        similarity = torch.mm(lang_feat_norm, text_embeddings.t())

    return similarity.cpu().numpy()


def render_feature_map(
    coord: np.ndarray,
    similarity: np.ndarray,
    query_idx: int,
    resolution: Tuple[int, int] = (1024, 1024),
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Render a 2D feature map from 3D coordinates and similarity scores.

    Creates a top-down projection of the scene with colored regions
    indicating similarity to the text query.

    Args:
        coord: 3D coordinates [N, 3]
        similarity: Similarity scores [N, num_queries]
        query_idx: Which query to render
        resolution: Output image resolution (width, height)
        threshold: Optional threshold for binary mask

    Returns:
        RGB image [H, W, 3]
    """
    # Get similarity scores for this query
    scores = similarity[:, query_idx]

    # Project to 2D (top-down view, using X-Y plane)
    x = coord[:, 0]
    y = coord[:, 1]

    # Normalize to image coordinates
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Avoid division by zero
    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0

    # Add margin
    margin = 0.1
    x_norm = (x - x_min) / x_range * (1 - 2 * margin) + margin
    y_norm = (y - y_min) / y_range * (1 - 2 * margin) + margin

    width, height = resolution

    # Create image
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Create score map
    score_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int32)

    # Bin coordinates
    x_indices = (x_norm * width).astype(int)
    y_indices = (y_norm * height).astype(int)

    # Clip to valid range
    x_indices = np.clip(x_indices, 0, width - 1)
    y_indices = np.clip(y_indices, 0, height - 1)

    # Accumulate scores (max pooling for overlapping points)
    for xi, yi, score in zip(x_indices, y_indices, scores):
        if score > score_map[yi, xi]:
            score_map[yi, xi] = score
        count_map[yi, xi] += 1

    # Normalize scores for visualization
    if score_map.max() > score_map.min():
        score_map_normalized = (score_map - score_map.min()) / (score_map.max() - score_map.min())
    else:
        score_map_normalized = score_map

    # Apply threshold if specified
    if threshold is not None:
        mask = score_map_normalized >= threshold
        # Create binary visualization (green for above threshold)
        img[:, :, 1] = (mask * 255).astype(np.uint8)  # Green channel
        # Add background (gray for below threshold)
        bg_mask = ~mask
        img[bg_mask, 0] = 50
        img[bg_mask, 1] = 50
        img[bg_mask, 2] = 50
    else:
        # Create heatmap (blue -> green -> red)
        # Blue (low) to Red (high)
        score_map_uint8 = (score_map_normalized * 255).astype(np.uint8)
        img[:, :, 0] = score_map_uint8  # Red
        img[:, :, 1] = score_map_uint8  # Green
        img[:, :, 2] = (255 - score_map_uint8)  # Blue (inverted)

    return img


def save_query_results(
    output_dir: str,
    scene_name: str,
    queries: List[str],
    similarity: np.ndarray,
    coord: np.ndarray,
    threshold: Optional[float] = None,
):
    """Save query visualization results.

    Args:
        output_dir: Output directory path
        scene_name: Name of the scene
        queries: List of query strings
        similarity: Similarity scores [N, num_queries]
        coord: 3D coordinates [N, 3]
        threshold: Optional threshold for visualization
    """
    output_path = Path(output_dir) / scene_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {output_path}")

    # Save feature maps for each query
    for i, query in enumerate(queries):
        # Sanitize query for filename
        query_filename = query.replace(" ", "_").replace("/", "_")[:50]
        query_filename = "".join(c for c in query_filename if c.isalnum() or c in "_-")

        # Render feature map
        feature_map = render_feature_map(coord, similarity, i, threshold=threshold)

        # Save feature map
        feature_map_path = output_path / f"{query_filename}_feature_map.png"
        Image.fromarray(feature_map).save(feature_map_path)
        print(f"  Saved: {feature_map_path}")

        # Save similarity statistics
        scores = similarity[:, i]
        stats = {
            "query": query,
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "median": float(np.median(scores)),
        }
        print(f"  Similarity stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"min={stats['min']:.4f}, max={stats['max']:.4f}")

    # Save all similarity scores as .npy
    similarity_path = output_path / "similarity_scores.npy"
    np.save(similarity_path, similarity)
    print(f"  Saved similarity scores: {similarity_path}")


def query_scene(
    scene_path: str,
    queries: List[str],
    config_path: str,
    checkpoint_path: str,
    clip_model_name: str = "ViT-B/32",
    threshold: Optional[float] = None,
    device: str = "cuda",
    output_dir: str = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Query a scene with text prompts.

    Args:
        scene_path: Path to scene directory with .npy files
        queries: List of text queries
        config_path: Path to model config
        checkpoint_path: Path to model checkpoint
        clip_model_name: CLIP model name to use
        threshold: Similarity threshold for binary visualization
        device: Device to use
        output_dir: Output directory

    Returns:
        Tuple of (similarity_scores, coord, scene_name)
    """
    scene_path = Path(scene_path)
    scene_name = scene_path.name

    print(f"\n{'='*60}")
    print(f"Querying Scene: {scene_name}")
    print(f"{'='*60}")
    print(f"Scene path: {scene_path}")
    print(f"Queries: {queries}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"CLIP model: {clip_model_name}")
    print(f"Threshold: {threshold if threshold else 'None (heatmap mode)'}")

    # Load scene data
    print(f"\nLoading scene data...")
    data_dict = {}
    for file_path in scene_path.glob("*.npy"):
        key = file_path.stem
        data_dict[key] = np.load(file_path)
        print(f"  Loaded {key}.py: shape={data_dict[key].shape}")

    # Check required files
    required_keys = ["coord", "color", "opacity", "quat", "scale"]
    missing = [k for k in required_keys if k not in data_dict]
    if missing:
        raise ValueError(f"Missing required files: {missing}")

    coord = data_dict["coord"]
    print(f"  Scene has {coord.shape[0]} points")

    # Load SceneSplat model
    print(f"\nLoading SceneSplat model...")
    cfg = Config.fromfile(config_path)

    inferencer = LangPretrainerInference(
        cfg,
        checkpoint_path,
        device=device,
    )

    # Run inference to get language features
    print(f"\nRunning inference...")
    outputs = inferencer(
        data_dict,
        scene_name=scene_name,
        save=False,
    )

    lang_features = outputs["backbone_features"]
    metadata = outputs["metadata"]

    # Handle inverse mapping if present
    if "inverse" in metadata and metadata["inverse"] is not None:
        inverse = metadata["inverse"]
        lang_features = lang_features[inverse]
        print(f"  Applied inverse mapping, feature shape: {lang_features.shape}")

    print(f"  Language features shape: {lang_features.shape}")

    # Load CLIP and encode text queries
    print(f"\nEncoding text queries with CLIP...")
    clip_model = get_clip_model(clip_model_name, device)
    text_embeddings = encode_text_queries(clip_model, queries, device)
    print(f"  Text embeddings shape: {text_embeddings.shape}")

    # Compute similarity
    print(f"\nComputing similarity...")
    similarity = compute_similarity(lang_features, text_embeddings, device)
    print(f"  Similarity shape: {similarity.shape}")

    # Print per-query statistics
    for i, query in enumerate(queries):
        scores = similarity[:, i]
        print(f"  '{query}': mean={scores.mean():.4f}, std={scores.std():.4f}, "
              f"max={scores.max():.4f}")

    # Save results if output directory specified
    if output_dir:
        save_query_results(
            output_dir=output_dir,
            scene_name=scene_name,
            queries=queries,
            similarity=similarity,
            coord=coord,
            threshold=threshold,
        )

    print(f"\n{'='*60}")
    print(f"Query complete!")
    print(f"{'='*60}\n")

    return similarity, coord, scene_name


def main():
    parser = argparse.ArgumentParser(
        description="Open-Vocabulary Query Visualization for SceneSplat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python tools/visualization/query_open_vocabulary.py \\
      --scene /path/to/scene \\
      --query "wooden chair" \\
      --output-dir ./query_results

  # Multiple queries with custom threshold
  python tools/visualization/query_open_vocabulary.py \\
      --scene /path/to/scene \\
      --query "sofa" "table" "lamp" \\
      --threshold 0.3 \\
      --output-dir ./query_results

  # Use specific model checkpoint
  python tools/visualization/query_open_vocabulary.py \\
      --config configs/inference/lang-pretrain-pt-v3m1-3dgs.py \\
      --checkpoint checkpoints/model.pth \\
      --scene /path/to/scene \\
      --query "plant" \\
      --output-dir ./query_results
        """
    )

    # Required arguments
    parser.add_argument(
        "--scene", "-s",
        type=str,
        required=True,
        help="Path to scene directory with .npy files (coord.npy, color.npy, etc.)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        nargs="+",
        required=True,
        help="Text query/queries (e.g., 'wooden chair' or 'sofa' 'table' 'lamp')"
    )

    # Optional arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/inference/lang-pretrain-pt-v3m1-3dgs.py",
        help="Path to model config (default: configs/inference/lang-pretrain-pt-v3m1-3dgs.py)"
    )
    parser.add_argument(
        "--checkpoint", "-w",
        type=str,
        default="checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth",
        help="Path to model checkpoint (default: checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./query_results",
        help="Output directory (default: ./query_results)"
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14-336"],
        help="CLIP model name (default: ViT-B/32)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Similarity threshold for binary visualization (0-1). If not specified, shows heatmap."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )

    args = parser.parse_args()

    # Resolve paths
    scene_path = args.scene
    config_path = args.config
    checkpoint_path = args.checkpoint

    if not os.path.isabs(config_path):
        config_path = str(PROJECT_ROOT / config_path)
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = str(PROJECT_ROOT / checkpoint_path)

    # Validate scene path
    if not os.path.exists(scene_path):
        print(f"Error: Scene path not found: {scene_path}")
        sys.exit(1)

    # Validate config
    if not os.path.exists(config_path):
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    # Validate checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Run query
    try:
        query_scene(
            scene_path=scene_path,
            queries=args.query,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            clip_model_name=args.clip_model,
            threshold=args.threshold,
            device=args.device,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"\nError during query: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
