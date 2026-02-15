#!/usr/bin/env python3
"""
Assign language labels to 3D Gaussian language features using SigLIP2.

This script processes scenes in gaussian_train/ and assigns text labels to each
valid language feature using SigLIP2 embeddings.

ENCODING CONFIGURATION (matches SceneSplat inference pipeline):
- Model: google/siglip2-base-patch16-512 (SigLIP v2)
- Prompt template: "this is a {}" (standard SigLIP format)
- Precision: float16 during encoding, converted to float32 for output
- Processor: AutoProcessor (not AutoTokenizer)
- Normalization: L2 normalization applied

KEY INSIGHT:
The image features are encoded WITHOUT a prompt template (direct image encoding),
while text features are encoded WITH "this is a {}" prompt template. This is the
correct and expected behavior for SigLIP, as the model was trained this way.

Usage:
    python tools/get_3d_textlabels.py --dataset lerf_ovs --gpu 0
    python tools/get_3d_textlabels.py --dataset 3DOVS --scene bed --gpu 0
    python tools/get_3d_textlabels.py --dataset all --batch
    # Use --regenerate-embeddings to force regeneration with new settings
    python tools/get_3d_textlabels.py --dataset 3DOVS --regenerate-embeddings --gpu 0
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from tqdm import tqdm

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from transformers import AutoModel, AutoProcessor
except ImportError:
    print("Installing transformers...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "-q"])
    from transformers import AutoModel, AutoProcessor


###################################
# 1. Dataset-specific utilities
###################################

def get_lerf_ovs_categories(scene_dir: Path, label_root: Path) -> Set[str]:
    """
    Extract unique categories from lerf_ovs JSON annotation files.

    Args:
        scene_dir: Path to the scene directory (e.g., gaussian_train/lerf_ovs/train/figurines/)
        label_root: Path to the label root directory (e.g., datasets/lerf_ovs/label)

    Returns:
        Set of unique category names
    """
    # Find the corresponding label directory
    scene_name = scene_dir.name
    label_dir = label_root / scene_name

    if not label_dir.exists():
        print(f"  Warning: Label directory not found: {label_dir}")
        return set()

    categories = set()
    for json_file in label_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                for obj in data.get("objects", []):
                    category = obj.get("category", "")
                    if category:
                        categories.add(category)
        except Exception as e:
            print(f"  Warning: Error reading {json_file}: {e}")

    return categories


def get_3dovs_categories(scene_dir: Path, dataset_root: Path) -> Set[str]:
    """
    Read categories from 3DOVS segmentations/classes.txt.

    Args:
        scene_dir: Path to the scene directory (e.g., gaussian_train/3DOVS/bed/)
        dataset_root: Path to the 3DOVS dataset root directory (e.g., datasets/3DOVS)

    Returns:
        Set of category names
    """
    scene_name = scene_dir.name
    classes_file = dataset_root / scene_name / "segmentations" / "classes.txt"

    if not classes_file.exists():
        print(f"  Warning: Classes file not found: {classes_file}")
        return set()

    try:
        with open(classes_file, 'r') as f:
            categories = [line.strip() for line in f if line.strip()]
        return set(categories)
    except Exception as e:
        print(f"  Warning: Error reading {classes_file}: {e}")
        return set()


def collect_all_categories(
    dataset_name: str,
    gaussian_train_root: Path,
    dataset_root: Path,
    scenes: List[str] = None
) -> Dict[str, Set[str]]:
    """
    Collect all unique categories across all scenes in a dataset.

    Args:
        dataset_name: 'lerf_ovs' or '3DOVS'
        gaussian_train_root: Path to gaussian_train root directory
        dataset_root: Path to datasets root (for label files)
        scenes: Optional list of specific scenes to process

    Returns:
        Dictionary mapping scene names to their categories
    """
    train_root = gaussian_train_root / dataset_name / "train"

    if not train_root.exists():
        print(f"Error: Training directory not found: {train_root}")
        return {}

    scene_dirs = sorted([d for d in train_root.iterdir() if d.is_dir()])
    if scenes:
        scene_dirs = [d for d in scene_dirs if d.name in scenes]

    scene_categories = {}

    for scene_dir in tqdm(scene_dirs, desc=f"Collecting {dataset_name} categories"):
        # Prepare dataset root path for this dataset
        if dataset_name == "lerf_ovs":
            categories = get_lerf_ovs_categories(scene_dir, dataset_root / "lerf_ovs" / "label")
        else:  # 3DOVS
            categories = get_3dovs_categories(scene_dir, dataset_root / "3DOVS")
        scene_categories[scene_dir.name] = categories

    return scene_categories


###################################
# 2. Text Embedding Generation
###################################

def generate_text_embeddings(
    categories: Set[str],
    siglip_model_name: str = "google/siglip2-base-patch16-512",
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, List[str]]:
    """
    Generate text embeddings for categories using SigLIP2.

    IMPORTANT: This function uses "this is a {}" prompt template to match
    SceneSplat's inference pipeline. The text encoding must match how the
    model was trained and how inference is performed.

    Args:
        categories: Set of category names
        siglip_model_name: SigLIP2 model name (must use v2 for consistency)
        device: Torch device

    Returns:
        Tuple of (text_embeddings tensor, sorted_category_list)
    """
    if len(categories) == 0:
        print("Warning: No categories to encode!")
        return torch.empty(0, 768), []

    # Sort categories for consistency
    sorted_categories = sorted(list(categories))

    # Use prompt template "this is a {}" to match SceneSplat inference
    # This is the standard format for SigLIP text encoding
    prompts = ["this is a " + cat for cat in sorted_categories]

    print(f"Loading SigLIP2 model: {siglip_model_name}")
    print("Using prompt template: 'this is a {}'")
    processor = AutoProcessor.from_pretrained(siglip_model_name, use_fast=True)
    model = AutoModel.from_pretrained(
        siglip_model_name,
        torch_dtype=torch.float16,
    ).eval().to(device)

    print(f"Encoding {len(prompts)} text prompts with float16...")
    with torch.no_grad():
        text_inputs = processor(text=prompts, padding=True, truncation=True, return_tensors="pt")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_feat = model.get_text_features(**text_inputs)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.cpu().float()  # Convert back to float32 for compatibility

    print(f"Generated text embeddings shape: {text_feat.shape}")
    return text_feat, sorted_categories


def load_or_generate_embeddings(
    dataset_name: str,
    categories: Set[str],
    dataset_root: Path,
    device: torch.device = torch.device("cuda"),
    force_regenerate: bool = False,
    siglip_model_name: str = "google/siglip2-base-patch16-512"
) -> Tuple[torch.Tensor, List[str]]:
    """
    Load existing text embeddings or generate new ones.

    Args:
        dataset_name: Name of the dataset
        categories: Set of category names
        dataset_root: Directory to cache embeddings
        device: Torch device
        force_regenerate: Force regeneration even if cached
        siglip_model_name: SigLIP2 model name to use for generation

    Returns:
        Tuple of (text_embeddings tensor, sorted_category_list)
    """
    cache_file = dataset_root / f"{dataset_name}_text_embeddings_siglip2.pt"

    if not force_regenerate and cache_file.exists():
        print(f"Loading cached embeddings from: {cache_file}")
        data = torch.load(cache_file)

        # Check if all categories are in cache
        cached_categories = set(data.get("categories", []))
        if categories.issubset(cached_categories):
            # Filter to only needed categories
            cat_to_idx = {cat: i for i, cat in enumerate(data["categories"])}
            indices = [cat_to_idx[cat] for cat in sorted(categories)]
            embeddings = data["embeddings"][indices]
            return embeddings, sorted(categories)
        else:
            print("Cached embeddings missing some categories, regenerating...")

    # Generate new embeddings
    embeddings, sorted_categories = generate_text_embeddings(categories, siglip_model_name=siglip_model_name, device=device)

    # Cache for future use
    print(f"Caching embeddings to: {cache_file}")
    dataset_root.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings,
        "categories": sorted_categories
    }, cache_file)

    return embeddings, sorted_categories


###################################
# 3. Label Assignment using SigLIP2
###################################

@torch.no_grad()
def assign_labels_to_features(
    lang_feat: np.ndarray,
    text_embeddings: torch.Tensor,
    valid_mask: np.ndarray,
    prob_threshold: float = 0.1,
    device: torch.device = torch.device("cuda")
) -> np.ndarray:
    """
    Assign text labels to language features using SigLIP2.

    Args:
        lang_feat: Language features array of shape (N, feature_dim)
        text_embeddings: Text embeddings tensor of shape (C, feature_dim)
        valid_mask: Valid feature mask of shape (N,)
        prob_threshold: Minimum probability threshold for assigning a label
        device: Torch device

    Returns:
        Labels array of shape (N,) with -1 for invalid/unlabeled features
    """
    # Convert to tensor
    lang_feat_tensor = torch.from_numpy(lang_feat).float()
    text_embeddings = text_embeddings.to(device)
    lang_feat_tensor = lang_feat_tensor.to(device)

    # Compute similarity scores (dot product since both are normalized)
    logits = torch.matmul(lang_feat_tensor, text_embeddings.t())  # (N, C)
    probs = torch.sigmoid(logits)  # (N, C)

    # Find max probability and its index for each feature
    max_probs, max_indices = torch.max(probs, dim=1)

    # Initialize labels with -1 (invalid/unlabeled)
    labels = np.full(lang_feat.shape[0], -1, dtype=np.int64)

    # Only assign labels to valid features with sufficient probability
    valid_indices = np.where(valid_mask == 1)[0]
    for idx in valid_indices:
        if max_probs[idx] >= prob_threshold:
            labels[idx] = max_indices[idx].item()

    return labels


###################################
# 4. Main Processing Function
###################################

def process_scene(
    scene_dir: Path,
    text_embeddings: torch.Tensor,
    categories: List[str],
    prob_threshold: float = 0.1,
    device: torch.device = torch.device("cuda")
) -> Dict[str, any]:
    """
    Process a single scene: load features, assign labels, save results.

    Args:
        scene_dir: Path to scene directory
        text_embeddings: Text embeddings tensor
        categories: List of category names
        prob_threshold: Minimum probability threshold
        device: Torch device

    Returns:
        Dictionary with processing statistics
    """
    # Load language features
    lang_feat_path = scene_dir / "lang_feat.npy"
    if not lang_feat_path.exists():
        return {"status": "error", "message": "lang_feat.npy not found"}

    lang_feat = np.load(lang_feat_path)  # (N, feature_dim)

    # Load valid feature mask
    valid_mask_path = scene_dir / "valid_feat_mask.npy"
    if valid_mask_path.exists():
        valid_mask = np.load(valid_mask_path)
    else:
        # Assume all features are valid if mask doesn't exist
        valid_mask = np.ones(lang_feat.shape[0], dtype=np.int64)

    # Assign labels
    labels = assign_labels_to_features(
        lang_feat, text_embeddings, valid_mask,
        prob_threshold=prob_threshold, device=device
    )

    # Save labels
    output_path = scene_dir / "lang_label.npy"
    np.save(output_path, labels)

    # Save label mapping (index to category name) as JSON
    label_mapping = {str(i): cat for i, cat in enumerate(categories)}
    mapping_path = scene_dir / "lang_label_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)

    # Statistics
    n_valid = np.sum(valid_mask)
    n_labeled = np.sum(labels >= 0)
    n_unlabeled_valid = n_valid - n_labeled

    # Per-category counts
    label_counts = {}
    for cat_idx, cat_name in enumerate(categories):
        count = np.sum(labels == cat_idx)
        if count > 0:
            label_counts[cat_name] = count

    return {
        "status": "success",
        "scene": scene_dir.name,
        "n_features": len(labels),
        "n_valid": n_valid,
        "n_labeled": n_labeled,
        "n_unlabeled_valid": n_unlabeled_valid,
        "label_counts": label_counts,
        "output_path": str(output_path),
        "mapping_path": str(mapping_path)
    }


def process_dataset(
    dataset_name: str,
    gaussian_train_root: Path,
    dataset_root: Path,
    scenes: List[str] = None,
    gpu_id: int = 0,
    prob_threshold: float = 0.1,
    siglip_model: str = "google/siglip2-base-patch16-512",
    force_regenerate_embeddings: bool = False
) -> List[Dict]:
    """
    Process all scenes in a dataset.

    Args:
        dataset_name: 'lerf_ovs' or '3DOVS'
        gaussian_train_root: Path to gaussian_train root directory
        dataset_root: Path to datasets root (for label files and caching)
        scenes: Optional list of specific scenes to process
        gpu_id: GPU device ID
        prob_threshold: Minimum probability threshold for labeling
        siglip_model: SigLIP2 model name (must use v2 for consistency with image features)
        force_regenerate_embeddings: Force regeneration of text embeddings

    Returns:
        List of processing results for each scene
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Collect all categories across scenes
    print(f"\nCollecting categories for {dataset_name}...")
    scene_categories = collect_all_categories(dataset_name, gaussian_train_root, dataset_root, scenes)

    if not scene_categories:
        print(f"Error: No scenes found for dataset {dataset_name}")
        return []

    # Get all unique categories
    all_categories = set()
    for categories in scene_categories.values():
        all_categories.update(categories)

    print(f"Found {len(all_categories)} unique categories across {len(scene_categories)} scenes")
    print(f"Categories: {sorted(all_categories)}")

    # Generate or load text embeddings
    text_embeddings, sorted_categories = load_or_generate_embeddings(
        dataset_name, all_categories, dataset_root, device=device, force_regenerate=force_regenerate_embeddings, siglip_model_name=siglip_model
    )

    # Create category name to index mapping
    # cat_to_idx = {cat: idx for idx, cat in enumerate(sorted_categories)}

    # Process each scene
    train_root = gaussian_train_root / dataset_name / "train"
    scene_dirs = sorted([d for d in train_root.iterdir() if d.is_dir()])
    if scenes:
        scene_dirs = [d for d in scene_dirs if d.name in scenes]

    results = []
    for scene_dir in tqdm(scene_dirs, desc=f"Processing {dataset_name} scenes"):
        result = process_scene(scene_dir, text_embeddings, sorted_categories, prob_threshold, device)
        results.append(result)

        if result["status"] == "success":
            tqdm.write(f"  {scene_dir.name}: {result['n_labeled']}/{result['n_valid']} labeled")
        else:
            tqdm.write(f"  {scene_dir.name}: {result['message']}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Assign text labels to 3D Gaussian language features using SigLIP2')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['lerf_ovs', '3DOVS', 'all'],
                       help='Dataset to process')
    parser.add_argument('--gaussian-train-root', type=str, default='/new_data/cyf/projects/SceneSplat/gaussian_train',
                       help='Path to gaussian_train root directory')
    parser.add_argument('--dataset-root', type=str, default='/new_data/cyf/projects/SceneSplat/datasets',
                       help='Path to datasets root directory (for label files and caching)')
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                       help='Specific scenes to process (default: all scenes)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--prob-threshold', type=float, default=0.1,
                       help='Minimum probability threshold for assigning labels (default: 0.1)')
    parser.add_argument('--siglip-model', type=str, default='google/siglip2-base-patch16-512',
                       help='SigLIP model name (default: google/siglip2-base-patch16-512)')
    parser.add_argument('--regenerate-embeddings', action='store_true',
                       help='Force regeneration of text embeddings even if cached')

    args = parser.parse_args()

    print("=" * 60)
    print("3D Gaussian Text Label Assignment using SigLIP2")
    print("=" * 60)

    datasets_to_process = [args.dataset] if args.dataset != 'all' else ['lerf_ovs', '3DOVS']

    all_results = {}
    for dataset in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*60}")

        results = process_dataset(
            dataset_name=dataset,
            gaussian_train_root=Path(args.gaussian_train_root),
            dataset_root=Path(args.dataset_root),
            scenes=args.scenes,
            gpu_id=args.gpu,
            prob_threshold=args.prob_threshold,
            siglip_model=args.siglip_model,
            force_regenerate_embeddings=args.regenerate_embeddings
        )

        all_results[dataset] = results

        # Print summary
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]

        print(f"\n{dataset} Summary:")
        print(f"  Total scenes: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")

        if successful:
            total_features = sum(r['n_features'] for r in successful)
            total_labeled = sum(r['n_labeled'] for r in successful)
            print(f"  Total features: {total_features}")
            print(f"  Total labeled: {total_labeled}")
            print(f"  Label coverage: {100 * total_labeled / total_features:.1f}%")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
