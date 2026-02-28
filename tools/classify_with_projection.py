#!/usr/bin/env python3
"""
Open-Vocabulary 3D Classification using Projection Matrix

This script demonstrates how to use the pre-computed projection matrix
to classify 3D points using text embeddings in the compressed feature space.

Pipeline:
1. Load projection matrix W (16, 768) - maps 768-dim text to 16-dim feature space
2. Get model predictions (16-dim)
3. Get text embeddings (768-dim)
4. Project text embeddings: text_16d = text_768 @ W.T
5. Compute similarity in 16-dim space
6. Classify based on similarity

Usage:
    # Single scene
    python tools/classify_with_projection.py \\
        --checkpoint exp/lite-16-gridsvd/model_best.pth \\
        --config configs/inference/lang-pretrain-litept-3dgs.py \\
        --scene /path/to/scene \\
        --class_names "chair,table,lamp,wall,floor"

    # Using pre-computed projection matrix
    python tools/classify_with_projection.py \\
        --checkpoint exp/lite-16-gridsvd/model_best.pth \\
        --config configs/inference/lang-pretrain-litept-3dgs.py \\
        --scene /path/to/scene \\
        --class_names "chair,table,lamp" \\
        --projection_matrix projection_matrix_768_to_16.npy
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict
import open_clip

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from pointcept.inference.lang_pretrainer import LangPretrainerInference
from pointcept.utils.config import Config


class TextEmbeddingProjector:
    """Projects text embeddings from 768-dim to 16-dim using pre-computed matrix."""

    def __init__(self, projection_matrix_path: str):
        """
        Args:
            projection_matrix_path: Path to .npy file containing [16, 768] matrix
        """
        self.W = np.load(projection_matrix_path)  # [16, 768]
        assert self.W.shape == (16, 768), f"Expected shape (16, 768), got {self.W.shape}"
        print(f"Loaded projection matrix: {self.W.shape} from {projection_matrix_path}")

    def project(self, text_embed_768: np.ndarray) -> np.ndarray:
        """
        Project text embeddings from 768-dim to 16-dim.

        Args:
            text_embed_768: [num_classes, 768] - text embeddings

        Returns:
            text_embed_16d: [num_classes, 16] - projected embeddings
        """
        return text_embed_768 @ self.W.T  # [num_classes, 16]

    def project_tensor(self, text_embed_768: torch.Tensor) -> torch.Tensor:
        """Tensor version of project."""
        W_tensor = torch.from_numpy(self.W).to(text_embed_768.device).float()
        return text_embed_768 @ W_tensor.T  # [num_classes, 16]


def get_text_embeddings(
    class_names: List[str],
    model_name: str = "ViT-H-14",
    pretrained: str = "laion2B_s32B_b79K",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Get text embeddings using OpenCLIP.

    Args:
        class_names: List of class names
        model_name: OpenCLIP model name
        pretrained: OpenCLIP pretrained weights
        device: Device to run on

    Returns:
        text_embeddings: [num_classes, 768] - text embeddings
    """
    print(f"Loading OpenCLIP model: {model_name}/{pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()

    # Tokenize and encode
    text_tokens = open_clip.tokenize(class_names)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens.to(device))
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    print(f"Text embeddings shape: {text_embeddings.shape}")
    return text_embeddings  # [num_classes, 768]


def classify_scene(
    scene_path: str,
    config_path: str,
    checkpoint_path: str,
    class_names: List[str],
    projection_matrix_path: str,
    device: str = "cuda",
    output_dir: str = None,
) -> Dict:
    """
    Classify a 3D scene using text embeddings.

    Args:
        scene_path: Path to scene directory
        config_path: Path to inference config
        checkpoint_path: Path to model checkpoint
        class_names: List of class names for classification
        projection_matrix_path: Path to projection matrix
        device: Device to run on
        output_dir: Output directory for results

    Returns:
        results: Dictionary containing predictions and metadata
    """
    # Load projection matrix
    projector = TextEmbeddingProjector(projection_matrix_path)

    # Get text embeddings
    text_embed_768 = get_text_embeddings(class_names, device=device)
    text_embed_16d = projector.project_tensor(text_embed_768)  # [num_classes, 16]
    print(f"Projected text embeddings: {text_embed_16d.shape}")

    # Run inference to get point features
    cfg = Config.fromfile(config_path)
    inferencer = LangPretrainerInference(
        cfg,
        checkpoint_path,
        device=device,
    )

    print(f"\nProcessing scene: {scene_path}")
    outputs = inferencer(
        scene_path=scene_path,
        scene_name=Path(scene_path).name,
        save=False,
    )

    # Extract features
    point_features = outputs["backbone_features"]  # [N, 16]
    metadata = outputs["metadata"]

    # Compute similarity
    print(f"\nComputing similarity...")
    similarity = F.cosine_similarity(
        point_features.unsqueeze(1),  # [N, 1, 16]
        text_embed_16d.unsqueeze(0),  # [1, num_classes, 16]
    ).squeeze(1)  # [N, num_classes]

    # Get predictions
    predicted_class_idx = similarity.argmax(dim=1)  # [N]
    confidence = similarity.max(dim=1)[0]  # [N]

    results = {
        "similarity": similarity.cpu().numpy(),  # [N, num_classes]
        "predicted_class_idx": predicted_class_idx.cpu().numpy(),  # [N]
        "confidence": confidence.cpu().numpy(),  # [N]
        "class_names": class_names,
        "metadata": metadata,
    }

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_name = Path(scene_path).name

        # Save predictions
        pred_path = output_dir / f"{scene_name}_predictions.npz"
        np.savez(pred_path, **{k: v for k, v in results.items() if k != "metadata"})
        print(f"Saved predictions to: {pred_path}")

        # Save class predictions as text file (human-readable)
        class_pred_path = output_dir / f"{scene_name}_classes.txt"
        with open(class_pred_path, 'w') as f:
            for i in range(len(predicted_class_idx)):
                class_name = class_names[predicted_class_idx[i]]
                conf = confidence[i].item()
                f.write(f"{i} {class_name} {conf:.4f}\n")
        print(f"Saved class predictions to: {class_pred_path}")

    # Print summary
    print(f"\n=== Classification Summary ===")
    print(f"Total points: {len(predicted_class_idx)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"\nClass distribution:")
    for i, class_name in enumerate(class_names):
        count = (predicted_class_idx == i).sum().item()
        pct = 100 * count / len(predicted_class_idx)
        print(f"  {class_name}: {count} ({pct:.2f}%)")
    print(f"\nAverage confidence: {confidence.mean():.4f}")
    print(f"Min confidence: {confidence.min():.4f}")
    print(f"Max confidence: {confidence.max():.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Open-vocabulary 3D classification using projection matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to inference config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Path to scene directory",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        required=True,
        help="Comma-separated class names (e.g., 'chair,table,lamp,wall,floor')",
    )
    parser.add_argument(
        "--projection_matrix",
        type=str,
        default="projection_matrix_768_to_16.npy",
        help="Path to projection matrix [16, 768]",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="classification_output",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )

    args = parser.parse_args()

    # Parse class names
    class_names = [name.strip() for name in args.class_names.split(',')]
    print(f"Class names: {class_names}")

    # Run classification
    classify_scene(
        scene_path=args.scene,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        class_names=class_names,
        projection_matrix_path=args.projection_matrix,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
