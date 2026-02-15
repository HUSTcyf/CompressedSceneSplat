#!/usr/bin/env python3
"""
Assign 2D segmentation labels to 3D Gaussian points via 3DGS rendering.

This script supports two modes:

1. Test Views Mode (default): Uses 2D segmentation masks from test views
   - 3DOVS: segmentation masks as PNG files in segmentations/ directory
   - lerf_ovs: segmentation polygons in JSON files in label/ directory

2. Train Views Mode (--use_train_views): Uses language features from train views
   - Loads language features from language_features_siglip2_sam2/ directory
   - Uses cosine similarity (with sigmoid probability conversion) with text embeddings
   - This is the standard SigLIP approach for text-visual matching
   - Text embeddings: datasets/lerf_ovs_text_embeddings_siglip2.pt or
     datasets/3DOVS_text_embeddings_siglip2.pt (custom embeddings matching visual model)

Usage:
    # Test views mode (with 2D masks)
    python tools/assign_2d_labels_to_3d.py \\
        --model_path /path/to/3dgs/output \\
        --source_path /path/to/dataset \\
        --dataset_type 3DOVS \\
        --iteration 30000

    # Train views mode (with language features)
    # Uses sigmoid-based probability matching (standard SigLIP approach)
    # similarity_threshold=0.5 means only positive similarity contributes to votes
    python tools/assign_2d_labels_to_3d.py \\
        --model_path /path/to/3dgs/output \\
        --source_path /path/to/dataset \\
        --dataset_type 3DOVS \\
        --iteration 30000 \\
        --use_train_views \\
        --similarity_threshold 0.5 \\
        --confidence_threshold 0.2
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Set, Optional
import cv2
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import gaussian renderer components
from gaussian_renderer import render
from scene import Scene
from gaussian_renderer import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args_from_yaml


###################################
# Dataset-specific utilities
###################################

# Text embedding paths - using custom embeddings that match the visual feature model
TEXT_EMBEDDING_PATHS = {
    "lerf_ovs": "/new_data/cyf/projects/SceneSplat/datasets/lerf_ovs_text_embeddings_custom.pt",
    "3DOVS": "/new_data/cyf/projects/SceneSplat/datasets/3DOVS_text_embeddings_custom.pt",
}


def load_text_embeddings(dataset_type: str) -> tuple[torch.Tensor, List[str]]:
    """
    Load text embeddings for the specified dataset type.

    Args:
        dataset_type: 'lerf_ovs' or '3DOVS'

    Returns:
        (embeddings, categories): Tuple of embeddings tensor [num_classes, D] and category names
    """
    text_embedding_path = TEXT_EMBEDDING_PATHS.get(dataset_type)
    if text_embedding_path is None:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if not os.path.exists(text_embedding_path):
        raise FileNotFoundError(f"Text embedding file not found: {text_embedding_path}")

    data = torch.load(text_embedding_path, map_location="cpu")

    # Handle different formats
    if isinstance(data, dict):
        if "embeddings" in data:
            embeddings = data["embeddings"]
            categories = data.get("categories", [])
        else:
            # Assume the keys are category names
            categories = list(data.keys())
            embeddings = torch.stack([data[k] for k in categories])
    else:
        # Assume it's a tensor only - need to get categories from elsewhere
        embeddings = data
        categories = []  # Will need to be loaded separately

    print(f"Loaded text embeddings from {text_embedding_path}")
    print(f"  Shape: {embeddings.shape}, dtype: {embeddings.dtype}, Categories: {len(categories)}")

    # Convert to float32 to match valid_features dtype
    embeddings = embeddings.float()

    return embeddings, categories


def get_lerf_ovs_categories(label_dir: Path) -> Set[str]:
    """
    Extract unique categories from lerf_ovs JSON annotation files.

    Args:
        label_dir: Path to the label directory (e.g., datasets/lerf_ovs/label/figurines/)

    Returns:
        Set of unique category names
    """
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


def get_3DOVS_categories(seg_root: Path) -> List[str]:
    """
    Read categories from 3DOVS segmentations/classes.txt.

    Args:
        seg_root: Path to the segmentations directory

    Returns:
        List of category names
    """
    classes_file = seg_root / "classes.txt"

    if not classes_file.exists():
        print(f"  Warning: Classes file not found: {classes_file}")
        return []

    try:
        with open(classes_file, 'r') as f:
            categories = [line.strip() for line in f if line.strip()]
        return categories
    except Exception as e:
        print(f"  Warning: Error reading {classes_file}: {e}")
        return []


def polygon_to_mask(polygon: List, img_h: int, img_w: int) -> np.ndarray:
    """
    Convert polygon points to binary mask.

    Args:
        polygon: List of [x, y] points
        img_h: Image height
        img_w: Image width

    Returns:
        Binary mask of shape (img_h, img_w)
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if len(polygon) < 3:
        return mask

    # Convert polygon to numpy array
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    # Fill polygon
    cv2.fillPoly(mask, [pts], 1)

    return mask


def load_lerf_ovs_masks(label_dir: Path, categories: List[str]) -> Dict:
    """
    Load 2D segmentation masks from lerf_ovs JSON files.

    Args:
        label_dir: Path to label directory (e.g., datasets/lerf_ovs/label/figurines/)
        categories: List of category names

    Returns:
        Dictionary mapping frame_name to {class_name: mask}
    """
    masks_dict = {}

    json_files = sorted(label_dir.glob("*.json"))

    for json_file in tqdm(json_files, desc="Loading lerf_ovs masks"):
        frame_name = json_file.stem  # e.g., "frame_00041"

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            img_h = data.get("info", {}).get("height", 728)
            img_w = data.get("info", {}).get("width", 986)

            # Initialize masks for all categories
            masks_dict[frame_name] = {cat: np.zeros((img_h, img_w), dtype=np.uint8)
                                      for cat in categories}

            # Process each object
            for obj in data.get("objects", []):
                category = obj.get("category", "")
                if category not in categories:
                    continue

                segmentation = obj.get("segmentation", [])
                if len(segmentation) >= 3:
                    # Convert polygon to mask
                    mask = polygon_to_mask(segmentation, img_h, img_w)
                    # Combine with existing mask (OR operation for overlapping objects)
                    masks_dict[frame_name][category] = np.maximum(
                        masks_dict[frame_name][category], mask
                    )

        except Exception as e:
            print(f"  Warning: Error processing {json_file}: {e}")
            continue

    return masks_dict


def load_3DOVS_masks(seg_root: Path, categories: List[str]) -> Dict:
    """
    Load 2D segmentation masks from 3DOVS PNG files.

    Args:
        seg_root: Path to segmentations directory
        categories: List of category names

    Returns:
        Dictionary mapping frame_idx to {class_name: mask}
    """
    masks_dict = {}

    # Get frame directories
    frame_dirs = sorted([d for d in seg_root.iterdir() if d.is_dir()],
                       key=lambda x: int(x.name))

    for frame_dir in tqdm(frame_dirs, desc="Loading 3DOVS masks"):
        frame_idx = int(frame_dir.name)

        # Get image dimensions from first available image
        # Try to find an image to get dimensions
        image_path = seg_root.parent / "images" / f"{frame_idx}.jpg"
        if not image_path.exists():
            image_path = seg_root.parent / "images_4" / f"{frame_idx}.jpg"

        if image_path.exists():
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        else:
            # Default dimensions for 3DOVS
            img_h, img_w = 546, 728

        masks_dict[frame_idx] = {}

        for class_name in categories:
            mask_path = frame_dir / f"{class_name}.png"

            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask.shape[:2] != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                masks_dict[frame_idx][class_name] = (mask > 127).astype(np.uint8)
            else:
                masks_dict[frame_idx][class_name] = np.zeros((img_h, img_w), dtype=np.uint8)

    return masks_dict


###################################
# Language feature loading (train views) - LAZY LOADING to avoid OOM
###################################

def load_single_language_feature(
    feature_dir: Path,
    image_name: str,
    feature_level: int = 0
) -> Optional[tuple]:
    """
    Load a single language feature file on-demand (lazy loading to avoid OOM).

    Args:
        feature_dir: Path to language_features_siglip2_sam2 directory
        image_name: Image name (without extension)
        feature_level: Feature level (0-3) to use

    Returns:
        (feature_map, seg_map) or None if not found
        - feature_map: [H, W, D] language features
        - seg_map: [H, W] segmentation mask
    """
    seg_file = feature_dir / f"{image_name}_s.npy"
    if not seg_file.exists():
        return None

    try:
        # Load segmentation map
        seg_map = np.load(seg_file)
        # print(f"Debug: {image_name} seg_map shape: {seg_map.shape}, ndim: {seg_map.ndim}")

        # Load feature map
        feature_file = seg_file.parent / f"{image_name}_f.npy"
        if not feature_file.exists():
            print(f"  Feature file not found: {feature_file}")
            return None

        features = np.load(feature_file)
        # print(f"Debug: {image_name} features shape: {features.shape}")

        # Extract feature level - handle different seg_map formats
        if seg_map.ndim == 3:
            # [num_levels, H, W] format
            if seg_map.shape[0] > 1:
                # Multiple levels available - use specified level
                valid_feature_level = min(feature_level, seg_map.shape[0] - 1)
                seg_level = seg_map[valid_feature_level]  # [H, W]
            else:
                # Single level - extract it
                seg_level = seg_map[0]  # [H, W]
        elif seg_map.ndim == 2:
            # Already [H, W] format
            seg_level = seg_map
        else:
            print(f"  Unexpected seg_map shape: {seg_map.shape}")
            return None

        # Verify seg_level is 2D
        if seg_level.ndim != 2:
            print(f"  seg_level is not 2D: {seg_level.shape}")
            return None

        # Create feature map by indexing
        h, w = seg_level.shape
        d = features.shape[1] if features.ndim > 1 else 0

        if d == 0:
            print(f"  Invalid features shape: {features.shape}")
            return None

        feature_map = np.zeros((h, w, d), dtype=np.float32)

        # Flatten segmentation map
        seg_flat = seg_level.reshape(-1)  # [H*W]

        # Get unique segment IDs (excluding background -1)
        unique_segs = np.unique(seg_flat)
        unique_segs = unique_segs[unique_segs >= 0]

        # Fill feature map
        for seg_id in unique_segs:
            if seg_id < len(features):
                mask = (seg_flat == seg_id)
                feature_map.reshape(-1, d)[mask] = features[seg_id]

        return feature_map, seg_level

    except Exception as e:
        import traceback
        print(f"  Warning: Error loading {image_name}: {e}")
        traceback.print_exc()
        return None


def get_available_feature_images(feature_dir: Path) -> Set[str]:
    """
    Get list of available image names in the feature directory (without loading data).

    Args:
        feature_dir: Path to language_features_siglip2_sam2 directory

    Returns:
        Set of image names (without extension)
    """
    if not feature_dir.exists():
        return set()

    seg_files = feature_dir.glob("*_s.npy")
    return {f.stem[:-2] for f in seg_files if f.is_file()}


def accumulate_2d_labels_to_3d_with_features(
    dataset,
    opt,
    pipeline,
    model_path,
    iteration: int,
    text_embeddings: torch.Tensor,
    seg_classes: List[str],
    feature_dir: Path,
    available_feature_images: Set[str],
    feature_level: int = 0,
    similarity_threshold: float = 0.5,
    confidence_threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Accumulate 2D labels to 3D Gaussians using train views with language features.

    Uses cosine similarity (converted to probabilities via sigmoid) between Gaussian
    features and text embeddings to assign labels. This is the standard SigLIP approach.
    Uses LAZY LOADING to avoid OOM - features are loaded on-demand during iteration.

    Args:
        dataset: ModelParams dataset
        opt: OptimizationParams
        pipeline: PipelineParams
        model_path: Path to trained 3DGS model
        iteration: Checkpoint iteration number
        text_embeddings: Text embeddings tensor [num_classes, D]
        seg_classes: List of segmentation class names
        feature_dir: Path to language_features_siglip2_sam2 directory
        available_feature_images: Set of available image names (for fast lookup)
        feature_level: Feature level for rendering
        similarity_threshold: Minimum probability threshold for weighted voting.
            0.5 means only positive cosine similarity (sigmoid(0)=0.5) contributes.
        confidence_threshold: Minimum confidence ratio for final label assignment (default: 0.1)

    Returns:
        labels: Array of shape (num_gaussians,) with class labels
        labeled_mask: Boolean array of shape (num_gaussians,) where True indicates labeled gaussians
    """
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # Move text embeddings to device
    text_embeddings = text_embeddings.to(device)  # [num_classes, D]
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)

    num_classes = len(seg_classes)

    # Load 3DGS model
    with torch.no_grad():
        # Initialize gaussians
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,
                     resolution_scales=[1.0], include_feature=True)

        # Find checkpoint file
        checkpoint = os.path.join(model_path, f'chkpnt{iteration}.pth')
        if not os.path.exists(checkpoint):
            checkpoint = os.path.join(model_path, f'ckpts/chkpnt{iteration}.pth')

        gaussians.restore_from_gsplat_checkpoint(checkpoint, opt)
        print(f"Loading gsplat format checkpoint: {checkpoint}")

        # Setup background
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

        # Get train cameras
        views = scene.getTrainCameras()

        # Get image names/indices
        image_names = []
        for view in views:
            image_name = view.image_name
            # Extract base name without extension
            img_idx = os.path.splitext(os.path.basename(image_name))[0]
            image_names.append(img_idx)

        # Debug: check name matching
        matching_names = set(image_names) & available_feature_images
        print(f"Matching names: {len(matching_names)}")

        # Initialize label accumulation
        num_gaussians = gaussians.get_xyz.shape[0]

        # Accumulate votes for each gaussian
        label_votes = torch.zeros(num_gaussians, num_classes, device=device)
        vote_counts = torch.zeros(num_gaussians, device=device)

        print(f"\nAccumulating 2D labels to {num_gaussians} 3D Gaussians using train views...")
        print(f"Available feature images: {len(available_feature_images)}")
        print(f"Using lazy loading to avoid OOM")

        matched_views = 0
        for view_idx, view in enumerate(tqdm(views, desc="Rendering train views (lazy loading)")):
            img_idx = image_names[view_idx]

            if img_idx not in available_feature_images:
                continue

            # LAZY LOAD: Load features on-demand for this view only
            result = load_single_language_feature(feature_dir, img_idx, feature_level)
            if result is None:
                continue

            feature_map, seg_map = result
            # feature_map: [H, W, D] numpy array
            # seg_map: [H, W] numpy array

            matched_views += 1

            # Convert to tensor
            feature_map_tensor = torch.from_numpy(feature_map).float().to(device)  # [H, W, D]
            valid_mask = (seg_map >= 0)
            valid_mask_tensor = torch.from_numpy(valid_mask).float().to(device)  # [H, W]

            # Delete numpy arrays immediately to free memory
            del feature_map, seg_map

            # Render the view
            render_pkg = render(view, gaussians, pipeline, background, include_feature=False)

            # Get rendering info
            activated = render_pkg["info"]["activated"]
            significance = render_pkg["info"]["significance"]
            means2D = render_pkg["info"]["means2d"]

            # Get mask of activated gaussians
            mask = activated[0] > 0
            if mask.sum() == 0:
                continue

            # Get image dimensions
            img_h, img_w = render_pkg["render"].shape[1:]

            # Get activated gaussian indices and 2D positions
            active_indices = torch.where(mask)[0]
            if len(active_indices) == 0:
                continue

            means2D_activated = means2D[0, mask]  # [num_activated, 2]
            sig_activated = significance[0, mask]  # [num_activated]

            # Round to integer coordinates and clamp
            coords_x = torch.clamp(means2D_activated[:, 0].long(), 0, img_w - 1)
            coords_y = torch.clamp(means2D_activated[:, 1].long(), 0, img_h - 1)

            # Sample features at gaussian positions
            # gaussian_features: [num_activated, D]
            gaussian_features = feature_map_tensor[coords_y, coords_x]

            # Sample validity mask
            # valid: [num_activated]
            valid = valid_mask_tensor[coords_y, coords_x]

            # Only process valid features
            valid_mask_gauss = valid > 0
            if valid_mask_gauss.sum() == 0:
                print(f"    Skipping: all gaussians at invalid positions (background)")
                continue

            valid_indices = active_indices[valid_mask_gauss]
            valid_features = gaussian_features[valid_mask_gauss]  # [num_valid, D]
            valid_sigs = sig_activated[valid_mask_gauss]

            # Normalize features
            valid_features = torch.nn.functional.normalize(valid_features, dim=-1)

            # Compute cosine similarity with all text embeddings
            # logits: [num_valid, num_classes] - raw cosine similarity
            logits = torch.matmul(valid_features, text_embeddings.T)  # [num_valid, num_classes]

            # Convert to probabilities using sigmoid (standard SigLIP approach)
            # This handles negative similarities and normalizes to [0, 1] range
            probs = torch.sigmoid(logits)  # [num_valid, num_classes]

            # Create weighted votes using significance and probabilities
            # Only consider probabilities above threshold
            thresholded_probs = torch.clamp(probs - similarity_threshold, min=0)
            weights = valid_sigs.unsqueeze(1) * thresholded_probs  # [num_valid, num_classes]

            # Accumulate votes
            label_votes[valid_indices] += weights
            vote_counts[valid_indices] += weights.sum(dim=1)

        print(f"\nMatched {matched_views}/{len(views)} views with language features")

        # Debug: print vote statistics
        has_votes = vote_counts > 0  # Define has_votes early for use in debug
        gaussians_with_votes = has_votes.sum().item()
        print(f"Gaussians with any votes: {gaussians_with_votes}/{num_gaussians}")

        if gaussians_with_votes > 0:
            valid_vote_counts = vote_counts[has_votes]
            print(f"Vote count stats: min={valid_vote_counts.min().item():.4f}, max={valid_vote_counts.max().item():.4f}, mean={valid_vote_counts.mean().item():.4f}")

        print("\nAnalyzing vote distribution...")

        # Additional debug: analyze label_votes distribution
        if gaussians_with_votes > 0:
            # Get the maximum vote (not normalized) for each gaussian
            max_votes = label_votes[has_votes].max(dim=1).values
            print(f"Max absolute vote stats: min={max_votes.min().item():.4f}, max={max_votes.max().item():.4f}, mean={max_votes.mean().item():.4f}")

            # Analyze how many classes have significant votes per gaussian
            significant_threshold = 0.01  # count votes > 1% of total
            temp_normalized = label_votes[has_votes] / vote_counts[has_votes].unsqueeze(1)
            num_significant = (temp_normalized > significant_threshold).sum(dim=1).float()
            print(f"Number of significant classes per gaussian: min={num_significant.min().item():.1f}, max={num_significant.max().item():.1f}, mean={num_significant.mean().item():.1f}")

        print("\nAssigning final labels...")

        # Assign final labels based on majority vote
        labels = torch.full((num_gaussians,), -1, dtype=torch.long, device=device)

        # Only process gaussians with votes (has_votes already defined above)
        if gaussians_with_votes > 0:
            # Get best class for each gaussian using argmax on absolute votes
            # This is more robust than normalized ratio when votes are dispersed
            best_classes = torch.argmax(label_votes, dim=1)
            best_votes = label_votes.max(dim=1).values

            # Calculate normalized ratios for statistics only
            normalized_votes = torch.zeros_like(label_votes)
            normalized_votes[has_votes] = label_votes[has_votes] / vote_counts[has_votes].unsqueeze(1)
            best_ratios = normalized_votes.max(dim=1).values

            # Assign labels to ALL gaussians with votes (no threshold)
            labels[has_votes] = best_classes[has_votes]

            print(f"Gaussians assigned: {has_votes.sum().item()}")
            print(f"Best ratio stats: min={best_ratios[has_votes].min().item():.4f}, max={best_ratios[has_votes].max().item():.4f}, mean={best_ratios[has_votes].mean().item():.4f}")
        else:
            print("Warning: No gaussians received any votes!")

        # Print statistics
        labeled_count = (labels >= 0).sum().item()
        print(f"Labeled {labeled_count}/{num_gaussians} gaussians ({100*labeled_count/num_gaussians:.1f}%)")

        # Per-class statistics
        print("\nPer-class label counts:")
        for class_idx, class_name in enumerate(seg_classes):
            count = (labels == class_idx).sum().item()
            print(f"  {class_name}: {count}")

        # Create labeled mask (True for gaussians with assigned labels)
        labeled_mask = (labels >= 0).cpu().numpy()

        return labels.cpu().numpy(), labeled_mask


def accumulate_2d_labels_to_3d(
    dataset,
    opt,
    pipeline,
    model_path,
    iteration,
    source_path,
    seg_classes,
    masks_dict
):
    """
    Accumulate 2D segmentation labels to 3D Gaussian points via rendering.

    Args:
        dataset: ModelParams dataset
        opt: OptimizationParams
        pipeline: PipelineParams
        model_path: Path to trained 3DGS model
        iteration: Checkpoint iteration number
        source_path: Path to dataset source
        seg_classes: List of segmentation class names
        masks_dict: Pre-loaded 2D segmentation masks

    Returns:
        labels: Array of shape (num_gaussians,) with class labels
        labeled_mask: Boolean array of shape (num_gaussians,) where True indicates labeled gaussians
    """
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # Load 3DGS model
    with torch.no_grad():
        # Initialize gaussians
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=[1.0], include_feature=True)

        # Find checkpoint file
        checkpoint = os.path.join(model_path, f'chkpnt{iteration}.pth')
        if not os.path.exists(checkpoint):
            # Try alternative checkpoint location
            checkpoint = os.path.join(model_path, f'ckpts/chkpnt{iteration}.pth')

        # Load and restore checkpoint BEFORE creating Scene
        gaussians.restore_from_gsplat_checkpoint(checkpoint, opt)
        print(f"Loading gsplat format checkpoint: {checkpoint}")

        # Setup background
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

        # Get test cameras (only test set has 2D labels for lerf_ovs)
        views = scene.getTestCameras()

        # Get image names/indices
        image_names = []
        for view in views:
            # Extract image index from image_name
            image_name = view.image_name
            # Try to extract numeric index or frame name
            try:
                img_idx = int(os.path.basename(image_name).split('.')[0])
            except:
                # For lerf_ovs and similar: strip extension to match JSON file stems
                # image_name might be "frame_00001.jpg" -> we want "frame_00001" to match JSON stems
                img_idx = os.path.splitext(os.path.basename(image_name))[0]
            image_names.append(img_idx)

        # Initialize label accumulation
        num_gaussians = gaussians.get_xyz.shape[0]
        num_classes = len(seg_classes)

        # Accumulate votes for each gaussian
        label_votes = torch.zeros(num_gaussians, num_classes, device=device)
        vote_counts = torch.zeros(num_gaussians, device=device)

        print(f"\nAccumulating 2D labels to {num_gaussians} 3D Gaussians...")

        matched_views = 0
        for view_idx, view in enumerate(tqdm(views, desc="Rendering views")):
            img_idx = image_names[view_idx]
            matched_views += 1

            # Render the view
            render_pkg = render(view, gaussians, pipeline, background, include_feature=False)

            # Get rendering info
            activated = render_pkg["info"]["activated"]
            significance = render_pkg["info"]["significance"]
            means2D = render_pkg["info"]["means2d"]

            # Get mask of activated gaussians
            mask = activated[0] > 0

            if mask.sum() == 0:
                continue

            # Get image dimensions
            img_h, img_w = render_pkg["render"].shape[1:]

            # Get activated gaussian indices and 2D positions
            active_indices = torch.where(mask)[0]
            if len(active_indices) == 0:
                continue

            means2D_activated = means2D[0, mask]  # [num_activated, 2]
            sig_activated = significance[0, mask]  # [num_activated]

            # Round to integer coordinates
            coords_x = torch.clamp(means2D_activated[:, 0].long(), 0, img_w - 1)
            coords_y = torch.clamp(means2D_activated[:, 1].long(), 0, img_h - 1)

            # Stack all class masks into a single tensor [H, W, num_classes]
            all_masks = np.zeros((img_h, img_w, len(seg_classes)), dtype=np.uint8)
            for class_idx, class_name in enumerate(seg_classes):
                if class_name in masks_dict[img_idx]:
                    mask_2d = masks_dict[img_idx][class_name]
                    # Resize if needed
                    if mask_2d.shape != (img_h, img_w):
                        mask_2d = cv2.resize(mask_2d, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    all_masks[:, :, class_idx] = mask_2d

            # Convert to tensor
            all_masks_tensor = torch.from_numpy(all_masks).to(device)  # [H, W, num_classes]

            # Get mask values at gaussian positions for all classes at once
            # mask_values: [num_activated, num_classes]
            mask_values = all_masks_tensor[coords_y, coords_x]  # [num_activated, num_classes]

            # Accumulate votes weighted by significance (vectorized)
            # sig_activated: [num_activated] -> expand to [num_activated, 1]
            weights = sig_activated.unsqueeze(1) * mask_values.float()  # [num_activated, num_classes]

            # Use scatter_add to accumulate votes
            label_votes[active_indices] += weights
            vote_counts[active_indices] += weights.sum(dim=1)

        print(f"\nMatched {matched_views}/{len(views)} views with labels")
        print("\nAssigning final labels...")

        # Assign final labels based on majority vote (vectorized)
        labels = torch.full((num_gaussians,), -1, dtype=torch.long, device=device)

        # Only process gaussians with votes
        has_votes = vote_counts > 0
        if has_votes.sum() > 0:
            # Normalize votes
            normalized_votes = torch.zeros_like(label_votes)
            normalized_votes[has_votes] = label_votes[has_votes] / vote_counts[has_votes].unsqueeze(1)

            # Get best class for each gaussian
            best_classes = torch.argmax(normalized_votes, dim=1)
            best_ratios = normalized_votes.max(dim=1).values

            # Assign labels where ratio > 0.5
            confident = (best_ratios > 0.5) & has_votes
            labels[confident] = best_classes[confident]

        # Print statistics
        labeled_count = (labels >= 0).sum().item()
        print(f"Labeled {labeled_count}/{num_gaussians} gaussians ({100*labeled_count/num_gaussians:.1f}%)")

        # Per-class statistics
        print("\nPer-class label counts:")
        for class_idx, class_name in enumerate(seg_classes):
            count = (labels == class_idx).sum().item()
            print(f"  {class_name}: {count}")

        # Create labeled mask (True for gaussians with assigned labels)
        labeled_mask = (labels >= 0).cpu().numpy()

        return labels.cpu().numpy(), labeled_mask


def process_single_scene(
    dataset, opt, pipeline,
    model_path: str,
    source_path: str,
    output_dir: str,
    dataset_type: str,
    iteration: int,
    feature_level: int = 0,
    use_train_views: bool = False,
    similarity_threshold: float = 0.5,
    confidence_threshold: float = 0.1,
) -> Dict:
    """
    Process a single scene and save labels.

    Args:
        dataset: ModelParams dataset
        opt: OptimizationParams
        pipeline: PipelineParams
        model_path: Path to trained 3DGS model
        source_path: Path to dataset source
        output_dir: Output directory for saving labels
        dataset_type: '3DOVS' or 'lerf_ovs'
        iteration: Checkpoint iteration number
        feature_level: Feature level for rendering
        use_train_views: If True, use train views with language features; otherwise use test views with masks
        similarity_threshold: Minimum probability threshold (after sigmoid) for train views mode.
            0.5 means only positive cosine similarity contributes.
        confidence_threshold: Minimum confidence ratio for final label assignment (train views mode, default: 0.1)

    Returns:
        Dictionary with processing results
    """
    # scene_name is derived from the last directory of source_path
    scene_name = Path(source_path).name
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene_name}")
    print(f"Mode: {'train views (language features)' if use_train_views else 'test views (masks)'}")
    print(f"{'='*60}")

    source_path = Path(source_path)

    try:
        if use_train_views:
            # Train views mode: use language features and text embeddings
            print(f"Using train views with language features...")

            # Load text embeddings
            text_embeddings, text_categories = load_text_embeddings(dataset_type)

            # Get categories from text embeddings
            if text_categories:
                seg_classes = text_categories
            else:
                # Fallback: get from labels directory if available
                if dataset_type == "lerf_ovs":
                    label_dir = source_path.parent / "label" / scene_name
                    seg_classes = sorted(list(get_lerf_ovs_categories(label_dir)))
                else:  # 3DOVS
                    seg_root = source_path / "segmentations"
                    seg_classes = get_3DOVS_categories(seg_root)

            if not seg_classes:
                raise ValueError(f"Could not determine categories for {dataset_type}")

            # Find language feature directory (without loading data)
            feature_dir = source_path / "language_features_siglip2_sam2"
            if not feature_dir.exists():
                # Try alternative location
                feature_dir = source_path.parent / "language_features_siglip2_sam2" / scene_name
            if not feature_dir.exists():
                # Try in the model path
                feature_dir = Path(model_path) / "language_features_siglip2_sam2"

            if not feature_dir.exists():
                raise FileNotFoundError(f"Language feature directory not found: {feature_dir}")

            # Get list of available feature images (lazy loading - no data loaded yet)
            available_feature_images = get_available_feature_images(feature_dir)
            print(f"Found {len(available_feature_images)} feature images (will load on-demand)")

            # Accumulate labels using train views with lazy loading
            labels, labeled_mask = accumulate_2d_labels_to_3d_with_features(
                dataset, opt, pipeline,
                model_path=model_path,
                iteration=iteration,
                text_embeddings=text_embeddings,
                seg_classes=seg_classes,
                feature_dir=feature_dir,
                available_feature_images=available_feature_images,
                feature_level=feature_level,
                similarity_threshold=similarity_threshold,
                confidence_threshold=confidence_threshold,
            )

        else:
            # Test views mode: use 2D segmentation masks
            # Determine label directory based on dataset type
            if dataset_type == "lerf_ovs":
                label_dir = source_path.parent / "label" / scene_name

                print(f"Loading categories from lerf_ovs label directory: {label_dir}")
                seg_classes = sorted(list(get_lerf_ovs_categories(label_dir)))

                if not seg_classes:
                    raise ValueError(f"Could not find categories in {label_dir}")

                masks_dict = load_lerf_ovs_masks(label_dir, seg_classes)
                scene_source_path = source_path

            else:  # 3DOVS
                scene_source_path = source_path
                seg_root = scene_source_path / "segmentations"

                seg_classes = get_3DOVS_categories(seg_root)

                if not seg_classes:
                    subdirs = [d for d in seg_root.iterdir() if d.is_dir()]
                    if subdirs:
                        first_dir = sorted(subdirs)[0]
                        seg_classes = [f.replace('.png', '') for f in os.listdir(first_dir) if f.endswith('.png')]

                if not seg_classes:
                    raise ValueError(f"Could not find segmentation classes in {seg_root}")

                masks_dict = load_3DOVS_masks(seg_root, seg_classes)

            print(f"Found {len(seg_classes)} segmentation classes")
            print(f"Loaded {len(masks_dict)} frames with segmentation masks")

            # Accumulate labels
            labels, labeled_mask = accumulate_2d_labels_to_3d(
                dataset, opt, pipeline,
                model_path=model_path,
                iteration=iteration,
                source_path=str(scene_source_path),
                seg_classes=seg_classes,
                masks_dict=masks_dict
            )

        # Save labels
        os.makedirs(output_dir, exist_ok=True)

        labels_path = os.path.join(output_dir, 'lang_label.npy')
        np.save(labels_path, labels)
        print(f"Saved 3D labels to: {labels_path}")

        # Save labeled mask (True for gaussians with assigned labels)
        labeled_mask_path = os.path.join(output_dir, 'lang_label_mask.npy')
        np.save(labeled_mask_path, labeled_mask)
        print(f"Saved labeled mask to: {labeled_mask_path}")

        # Save label mapping
        mapping_path = os.path.join(output_dir, 'lang_label_mapping.json')
        label_mapping = {str(i): cls for i, cls in enumerate(seg_classes)}
        with open(mapping_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        print(f"Saved label mapping to: {mapping_path}")

        # Return statistics
        labeled_count = (labels >= 0).sum()
        total_count = len(labels)

        return {
            'scene': scene_name,
            'status': 'success',
            'labeled_count': int(labeled_count),
            'total_count': int(total_count),
            'label_ratio': float(labeled_count / total_count) if total_count > 0 else 0,
            'num_classes': len(seg_classes),
        }

    except Exception as e:
        print(f"Error processing scene {scene_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'scene': scene_name,
            'status': 'error',
            'error': str(e)
        }


def main():
    parser = ArgumentParser(description="Assign 2D segmentation labels to 3D Gaussians")

    # Optional arguments
    parser.add_argument("--dataset_type", type=str, default="3DOVS",
                       choices=["3DOVS", "lerf_ovs"],
                       help="Dataset type: '3DOVS' or 'lerf_ovs'")
    parser.add_argument("--iteration", type=int, default=30000,
                       help="Checkpoint iteration number")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: model_path)")
    parser.add_argument("--use_train_views", action="store_true",
                       help="Use train views with language features instead of test views with masks")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                       help="Minimum probability threshold (after sigmoid) for weighted voting. "
                            "0.5 = only positive cosine similarity (sigmoid(0)=0.5). Default: 0.5")
    parser.add_argument("--confidence_threshold", type=float, default=0.1,
                       help="Minimum confidence ratio for final label assignment (train views mode, default: 0.1)")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)

    args = get_combined_args_from_yaml(parser, param_groups=[model, pipeline, opt])

    # Validate required arguments
    if not args.model_path or not args.source_path:
        parser.error("--model_path and --source_path are required")

    output_dir = args.output_dir if args.output_dir else args.model_path

    process_single_scene(
        model.extract(args),
        opt.extract(args),
        pipeline.extract(args),
        model_path=args.model_path,
        source_path=args.source_path,
        output_dir=output_dir,
        dataset_type=args.dataset_type,
        iteration=args.iteration,
        feature_level=args.feature_level,
        use_train_views=args.use_train_views,
        similarity_threshold=args.similarity_threshold,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
