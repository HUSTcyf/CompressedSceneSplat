#!/usr/bin/env python3
"""
Render SceneSplat predictions to 2D masks for Replica evaluation.

This script renders 3D point predictions to 2D semantic instance masks
using checkpoint_with_features_s.pth files that contain both Gaussian parameters
and language features.

Usage:
    # Single scene
    python tools/render_replica_predictions.py \\
        --scene office_0 \\
        --checkpoint /new_data/cyf/projects/SceneSplat/output_features/max_0_depth_True_default_office_0/checkpoint_with_features_s.pth \\
        --output-dir /path/to/output/masks

    # Batch processing - all scenes
    python tools/render_replica_predictions.py \\
        --batch \\
        --all-scenes \\
        --features-base /new_data/cyf/projects/SceneSplat/output_features \\
        --output-dir /path/to/output/masks

    # Batch processing - specific scenes
    python tools/render_replica_predictions.py \\
        --batch \\
        --scenes office_0 office_1 room_0 \\
        --features-base /new_data/cyf/projects/SceneSplat/output_features \\
        --output-dir /path/to/output/masks
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN

# Try to import open_clip for CLIP/SigLIP features
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("Warning: open_clip not available. Please install with: pip install open_clip_torch")

# Default paths
GAGA_ROOT = "/new_data/cyf/projects/Gaga/datasets/replica"  # For camera params and GT masks
REPLICA_ROOT = "/new_data/cyf/Datasets/Replica"  # For semantic.json class definitions
OUTPUT_ROOT = "/new_data/cyf/projects/SceneSplat/output_rendered_masks"

# Replica dataset class names (from semantic.json)
# These are the common semantic classes across Replica scenes
REPLICA_CLASSES = [
    "anonymize_picture",
    "anonymize_text",
    "bin",
    "blinds",
    "camera",
    "ceiling",
    "chair",
    "clock",
    "desk-organizer",
    "door",
    "floor",
    "indoor-plant",
    "lamp",
    "panel",
    "pillar",
    "plant-stand",
    "rug",
    "sofa",
    "switch",
    "table",
    "tablet",
    "tissue-paper",
    "tv-screen",
    "vent",
    "wall",
    "wall-plug",
    # Background/structural classes (often excluded from metrics)
    "other-leaf",  # Small undefined objects
    "undefined",   # Undefined regions
]

# Common class names for CLIP text prompts (more descriptive)
REPLICA_CLASS_PROMPTS = {
    "anonymize_picture": "a picture or photograph on the wall",
    "anonymize_text": "text or writing on a surface",
    "bin": "a trash bin or wastebasket",
    "blinds": "window blinds",
    "camera": "a security camera",
    "ceiling": "the ceiling of a room",
    "chair": "a chair",
    "clock": "a clock on the wall",
    "desk-organizer": "a desk organizer",
    "door": "a door",
    "floor": "the floor",
    "indoor-plant": "an indoor plant",
    "lamp": "a lamp or light fixture",
    "other-leaf": "miscellaneous small objects",
    "panel": "a panel or control board",
    "pillar": "a pillar or column",
    "plant-stand": "a plant stand",
    "rug": "a rug on the floor",
    "sofa": "a sofa or couch",
    "switch": "a light switch",
    "table": "a table",
    "tablet": "a tablet device",
    "tissue-paper": "a tissue box",
    "tv-screen": "a television screen",
    "undefined": "undefined region",
    "vent": "an air vent",
    "wall": "a wall",
    "wall-plug": "a wall plug or socket",
}


def load_checkpoint_with_features(checkpoint_path: str) -> dict:
    """Load Gaussian model checkpoint with language features.

    Args:
        checkpoint_path: Path to checkpoint_with_features_s.pth file

    Returns:
        Dictionary with Gaussian parameters and language features
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        # Format: (inner_tuple, iteration)
        # inner_tuple is the 13-element capture_language_feature() format:
        # (active_sh_degree, xyz, features_dc, features_rest, scaling, rotation,
        #  opacity, language_features, max_radii2D, xyz_gradient_accum, denom,
        #  opt_dict, spatial_lr_scale)
        inner_tuple = checkpoint[0]
        iteration = checkpoint[1]

        print(f"  Detected checkpoint_with_features format, iteration: {iteration}")
        print(f"  Inner tuple length: {len(inner_tuple)}")

        # Extract components
        active_sh_degree = inner_tuple[0]
        xyz = inner_tuple[1] if isinstance(inner_tuple[1], torch.Tensor) else inner_tuple[1].data
        features_dc = inner_tuple[2] if isinstance(inner_tuple[2], torch.Tensor) else inner_tuple[2].data
        features_rest = inner_tuple[3] if isinstance(inner_tuple[3], torch.Tensor) else inner_tuple[3].data
        scaling = inner_tuple[4] if isinstance(inner_tuple[4], torch.Tensor) else inner_tuple[4].data
        rotation = inner_tuple[5] if isinstance(inner_tuple[5], torch.Tensor) else inner_tuple[5].data
        opacity = inner_tuple[6] if isinstance(inner_tuple[6], torch.Tensor) else inner_tuple[6].data

        # Language features (index 7)
        if len(inner_tuple) > 7:
            lang_feat = inner_tuple[7] if isinstance(inner_tuple[7], torch.Tensor) else inner_tuple[7].data
        else:
            raise ValueError("Checkpoint does not contain language features at index 7")

        # Convert to numpy
        return {
            'means': xyz.cpu().detach().numpy() if isinstance(xyz, torch.Tensor) else xyz,
            'features_dc': features_dc.cpu().detach().numpy() if isinstance(features_dc, torch.Tensor) else features_dc,
            'features_rest': features_rest.cpu().detach().numpy() if isinstance(features_rest, torch.Tensor) else features_rest,
            'scales': scaling.cpu().detach().numpy() if isinstance(scaling, torch.Tensor) else scaling,
            'quats': rotation.cpu().detach().numpy() if isinstance(rotation, torch.Tensor) else rotation,
            'opacities': opacity.cpu().detach().numpy() if isinstance(opacity, torch.Tensor) else opacity,
            'lang_feat': lang_feat.cpu().detach().numpy() if isinstance(lang_feat, torch.Tensor) else lang_feat,
            'active_sh_degree': active_sh_degree,
            'iteration': iteration,
        }
    else:
        raise ValueError(f"Unsupported checkpoint format: expected tuple with 2 elements, got {type(checkpoint)}")


def load_scene_cameras(scene: str) -> List[dict]:
    """Load camera parameters for a Replica scene.

    Args:
        scene: Scene name (e.g., "office_0")

    Returns:
        List of camera dictionaries
    """
    # Map scene names between datasets
    # Scene name mapping: internal -> Gaga dataset
    scene_mapping = {
        "office_0": "office_0",
        "office_1": "office_1",
        "office_2": "office_2",
        "office_3": "office_3",
        "office_4": "office_4",
        "room_0": "room_0",
        "room_1": "room_1",
        "room_2": "room_2",
    }

    # Use GAGA_ROOT for camera parameters
    gaga_scene = scene_mapping.get(scene, scene)
    gaga_scene_dir = os.path.join(GAGA_ROOT, gaga_scene)

    if not os.path.exists(gaga_scene_dir):
        print(f"Warning: Gaga scene directory not found: {gaga_scene_dir}")
        return []

    # Look for test images
    images_dir = os.path.join(gaga_scene_dir, "images")
    if not os.path.exists(images_dir):
        print(f"Warning: Images directory not found: {images_dir}")
        return []

    # Get test images (test_rgb_XXXX.png)
    image_files = sorted([f for f in os.listdir(images_dir) if f.startswith("test_rgb") and f.endswith(".png")])

    print(f"Found {len(image_files)} test images for scene {scene}")

    # Create camera entries (using default camera parameters for Replica)
    cameras = []
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path)
        width, height = img.size

        # Check if there's a corresponding GT mask to get the correct size
        # Replica GT masks are 640x480
        gt_mask_name = img_file.replace("test_rgb", "test_semantic_instance")
        gt_mask_path = os.path.join(gaga_scene_dir, "semantic_instance", gt_mask_name)

        if os.path.exists(gt_mask_path):
            # Use GT mask size for rendering to match evaluation
            import cv2
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED)
            if gt_mask is not None:
                gt_height, gt_width = gt_mask.shape
                # Use GT mask dimensions for rendering
                width, height = gt_width, gt_height

        cameras.append({
            'id': i,
            'img_name': img_file,
            'width': width,
            'height': height,
            'fx': 500.0,  # Replica default focal length
            'fy': 500.0,
            'cx': width / 2.0,
            'cy': height / 2.0,
            # Default camera position (will be overridden if traj.json exists)
            'position': [0, 0, -5],
            'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Identity
        })

    # Try to load actual camera poses from traj.json if available
    traj_path = os.path.join(gaga_scene_dir, "traj.json")
    if os.path.exists(traj_path):
        try:
            with open(traj_path, 'r') as f:
                traj_data = json.load(f)
            for i, entry in enumerate(traj_data):
                if i < len(cameras):
                    cameras[i]['position'] = entry.get('position', cameras[i]['position'])
                    cameras[i]['rotation'] = entry.get('rotation', cameras[i]['rotation'])
            print(f"  Loaded camera poses from traj.json")
        except Exception as e:
            print(f"  Warning: Could not load traj.json: {e}")

    return cameras


def load_siglip_text_embeddings(
    class_names: List[str],
    class_prompts: Dict[str, str],
    model_name: str = "ViT-B-16-SigLIP2-512",
    pretrained: str = "webli",
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, any]:
    """Load SigLIP2 model and compute text embeddings for class names.

    Uses the same model configuration as eval_3DOVS.sh:
    - Model: ViT-B-16-SigLIP2-512 (768-dim)
    - Pretrained: webli

    Args:
        class_names: List of class names
        class_prompts: Dictionary mapping class names to descriptive prompts
        model_name: SigLIP2 model name (default: ViT-B-16-SigLIP2-512)
        pretrained: Pretrained weights name (default: webli)
        device: Device to load model on

    Returns:
        text_embeddings: Text embeddings [num_classes, 768]
        model: SigLIP2 model
    """
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip is required for SigLIP2-based classification")

    print(f"Loading SigLIP2 model: {model_name} ({pretrained})")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()

    # Generate text prompts for each class
    text_prompts = [class_prompts.get(name, name) for name in class_names]
    print(f"Generated {len(text_prompts)} text prompts")

    # Tokenize and compute text embeddings
    tokenizer = open_clip.get_tokenizer(model_name)
    text_tokens = tokenizer(text_prompts).to(device)

    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    print(f"Text embeddings shape: {text_embeddings.shape}")

    return text_embeddings, model


def load_clip_text_embeddings(
    class_names: List[str],
    class_prompts: Dict[str, str],
    model_name: str = "ViT-H-14",
    pretrained: str = "laion2B-s32B-b79K",
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, any]:
    """Load CLIP model and compute text embeddings for class names.

    Args:
        class_names: List of class names
        class_prompts: Dictionary mapping class names to descriptive prompts
        model_name: CLIP model name (default: ViT-H-14)
        pretrained: Pretrained weights name
        device: Device to load model on

    Returns:
        text_embeddings: Text embeddings [num_classes, D]
        model: CLIP model
    """
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip is required for CLIP-based classification")

    print(f"Loading CLIP model: {model_name} ({pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()

    # Generate text prompts for each class
    text_prompts = [class_prompts.get(name, name) for name in class_names]
    print(f"Generated {len(text_prompts)} text prompts")

    # Tokenize and compute text embeddings
    tokenizer = open_clip.get_tokenizer(model_name)
    text_tokens = tokenizer(text_prompts).to(device)

    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    print(f"Text embeddings shape: {text_embeddings.shape}")

    return text_embeddings, model


def compute_clip_labels(
    features: np.ndarray,
    text_embeddings: torch.Tensor,
    model: Optional[any] = None,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Compute semantic labels from language features using CLIP.

    Args:
        features: Language features [N, D] from SceneSplat checkpoint
        text_embeddings: Pre-computed CLIP text embeddings [num_classes, D]
        model: CLIP model (optional, used if projection is needed)
        device: Device for computation

    Returns:
        Labels [N] with values 0 to num_classes-1
    """
    # Convert features to tensor
    features_tensor = torch.from_numpy(features).float().to(device)

    # Normalize features
    features_norm = features_tensor.norm(dim=-1, keepdim=True)
    features_normalized = features_tensor / (features_norm + 1e-8)

    # Compute similarity with text embeddings
    # features: [N, 768], text_embeddings: [num_classes, 768]
    similarities = torch.matmul(features_normalized, text_embeddings.T)  # [N, num_classes]

    # Get argmax for each point
    labels = torch.argmax(similarities, dim=-1).cpu().numpy()

    return labels


def separate_instances_within_classes(
    means: np.ndarray,
    semantic_labels: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 10,
) -> np.ndarray:
    """Separate instances within each semantic class using spatial clustering.

    Uses DBSCAN clustering on 3D coordinates to separate different object instances
    of the same semantic class.

    Args:
        means: 3D point coordinates [N, 3]
        semantic_labels: Semantic class labels [N] (from SigLIP2)
        eps: DBSCAN eps parameter (max distance between points in same neighborhood)
        min_samples: DBSCAN min_samples parameter (min points to form a cluster)

    Returns:
        instance_labels: Instance IDs [N] where each unique value is a separate instance
    """
    N = len(semantic_labels)
    instance_labels = np.full(N, -1, dtype=np.int32)

    # Get unique semantic classes
    unique_classes = np.unique(semantic_labels)
    current_instance_id = 0

    print(f"  Separating instances within {len(unique_classes)} semantic classes...")

    for class_id in unique_classes:
        # Find points belonging to this class
        class_mask = semantic_labels == class_id
        class_points = means[class_mask]
        class_indices = np.where(class_mask)[0]

        n_class_points = len(class_points)
        if n_class_points < min_samples:
            # Too few points, assign each point as its own instance or background
            if n_class_points > 0:
                # For very small groups, assign as single instance
                instance_labels[class_indices] = current_instance_id
                current_instance_id += 1
            continue

        # Use DBSCAN to cluster points within this class
        # eps: maximum distance between two samples for one to be considered as in the neighborhood of the other
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(class_points)

        # Get cluster labels (-1 is noise/outlier)
        cluster_labels = clustering.labels_

        # Assign instance IDs
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                # Noise points - skip (will remain as background)
                # Don't assign instance IDs to noise points
                continue
            else:
                # Valid cluster - assign same instance ID to all points in cluster
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = class_indices[cluster_mask]
                instance_labels[cluster_indices] = current_instance_id
                current_instance_id += 1

    # Replace -1 (unlabeled) with 0 and shift all IDs by 1
    # This makes instance IDs start from 1 (0 = background)
    instance_labels[instance_labels == -1] = current_instance_id
    instance_labels = np.where(instance_labels >= 0, instance_labels + 1, 0)

    num_instances = len(np.unique(instance_labels))
    print(f"  Separated into {num_instances} instances")

    return instance_labels


def compute_semantic_labels(
    features: np.ndarray,
    n_clusters: int = 20,
) -> np.ndarray:
    """Compute semantic/instance labels from language features.

    Uses simple clustering based on feature norms (fallback method).

    Args:
        features: Language features [N, D]
        n_clusters: Number of clusters for instance IDs

    Returns:
        Labels [N] with values 0 to n_clusters-1
    """
    print("Warning: Using fallback clustering method. CLIP recommended for better results.")

    # Compute feature norms
    feature_norms = np.linalg.norm(features, axis=1)

    # Normalize to [0, 1] range
    norm_min = feature_norms.min()
    norm_max = feature_norms.max()
    if norm_max > norm_min:
        norm_normalized = (feature_norms - norm_min) / (norm_max - norm_min)
    else:
        norm_normalized = feature_norms

    # Discretize into cluster IDs
    labels = (norm_normalized * n_clusters).astype(np.int32)
    labels = np.clip(labels, 0, n_clusters - 1)

    return labels


def render_labels_to_mask(
    means: np.ndarray,
    labels: np.ndarray,
    opacities: np.ndarray,
    camera: dict,
) -> np.ndarray:
    """Render 3D point labels to 2D mask using projection.

    Args:
        means: 3D point coordinates [N, 3]
        labels: Point labels [N]
        opacities: Point opacities [N]
        camera: Camera parameters

    Returns:
        Rendered mask [H, W]
    """
    width, height = camera['width'], camera['height']

    # Camera parameters
    position = np.array(camera['position'])
    rotation = np.array(camera['rotation'])
    fx = camera['fx']
    fy = camera['fy']
    cx = camera.get('cx', width / 2.0)
    cy = camera.get('cy', height / 2.0)

    # Transform points to camera space
    points_cam = (rotation @ (means - position).T).T  # [N, 3]

    # Filter points in front of camera
    valid_depth = points_cam[:, 2] > 0
    points_cam = points_cam[valid_depth]
    labels_valid = labels[valid_depth]
    opacities_valid = opacities[valid_depth]

    if len(points_cam) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    # Project to image plane
    depths = points_cam[:, 2]
    u = fx * points_cam[:, 0] / depths + cx
    v = fy * points_cam[:, 1] / depths + cy

    # Filter points within image bounds
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_bounds].astype(int)
    v = v[in_bounds].astype(int)
    labels_valid = labels_valid[in_bounds]
    opacities_valid = opacities_valid[in_bounds]
    depths = depths[in_bounds]

    # Render with depth buffer and opacity
    rendered = np.full((height, width), -1, dtype=np.int32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

    # Sort by depth (far to near) for proper blending
    sort_order = np.argsort(depths)[::-1]

    for i in sort_order:
        x, y = u[i], v[i]
        if depths[i] < depth_buffer[y, x]:
            alpha = opacities_valid[i]
            # Use simple alpha threshold for label assignment
            if alpha > 0.5:
                rendered[y, x] = labels_valid[i]
            depth_buffer[y, x] = depths[i]

    # Convert to uint16 to match GT mask format
    # Map -1 -> 0 (background), 0+ -> 1+ (instances)
    rendered_uint16 = (rendered + 1).astype(np.uint16)

    return rendered_uint16


def process_scene(
    scene: str,
    checkpoint_path: str,
    output_dir: str,
    render_subset: Optional[List[int]] = None,
    use_clip: bool = True,
    n_clusters: int = 20,
    class_names: Optional[List[str]] = None,
    class_prompts: Optional[Dict[str, str]] = None,
    clip_model: str = "ViT-B-16-SigLIP2-512",
    clip_pretrained: str = "webli",
    text_embeddings: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
    use_instance_separation: bool = True,
    dbscan_eps: float = 0.05,
    dbscan_min_samples: int = 20,
) -> dict:
    """Process a single scene.

    Args:
        scene: Scene name
        checkpoint_path: Path to checkpoint_with_features_s.pth
        output_dir: Output directory for rendered masks
        render_subset: Optional list of view indices to render
        use_clip: Use SigLIP2-based classification (True) or clustering (False)
        n_clusters: Number of clusters for semantic labels (only used if use_clip=False)
        class_names: List of class names for SigLIP2 classification
        class_prompts: Dictionary mapping class names to text prompts
        clip_model: SigLIP2 model name
        clip_pretrained: SigLIP2 pretrained weights
        text_embeddings: Pre-computed SigLIP2 text embeddings (optional)
        device: Device for computation
        use_instance_separation: Enable instance separation within semantic classes
        dbscan_eps: DBSCAN eps parameter for instance separation (in meters)
        dbscan_min_samples: DBSCAN min_samples parameter for instance separation

    Returns:
        Dictionary with processing results
    """
    print(f"\n{'='*70}")
    print(f"Processing scene: {scene}")
    print(f"{'='*70}")

    # Load checkpoint with features
    data = load_checkpoint_with_features(checkpoint_path)

    means = data['means']
    opacities = data['opacities']
    features = data['lang_feat']

    # Fix opacities shape if needed
    if opacities.ndim == 2 and opacities.shape[1] == 1:
        opacities = opacities.squeeze(1)

    print(f"  Gaussians: {means.shape[0]}")
    print(f"  Features: {features.shape}")
    print(f"  Opacities: {opacities.shape}")

    # Compute semantic labels from features
    if use_clip:
        if class_names is None:
            class_names = REPLICA_CLASSES
        if class_prompts is None:
            class_prompts = REPLICA_CLASS_PROMPTS

        # Load SigLIP2 text embeddings if not provided
        if text_embeddings is None:
            if not OPEN_CLIP_AVAILABLE:
                print("Warning: open_clip not available, falling back to clustering")
                labels = compute_semantic_labels(features, n_clusters=n_clusters)
            else:
                text_embeddings, _ = load_siglip_text_embeddings(
                    class_names, class_prompts, clip_model, clip_pretrained, device
                )

        # Use SigLIP2-based classification
        semantic_labels = compute_clip_labels(features, text_embeddings, device=device)
        print(f"  SigLIP2 semantic labels computed: {len(np.unique(semantic_labels))} unique classes")
        print(f"  Class distribution:")
        unique, counts = np.unique(semantic_labels, return_counts=True)
        for u, c in zip(unique, counts):
            class_name = class_names[u] if u < len(class_names) else f"class_{u}"
            print(f"    {class_name}: {c} points")

        # Separate instances within each semantic class using spatial clustering
        if use_instance_separation:
            print("\n  Performing instance separation within semantic classes...")
            labels = separate_instances_within_classes(
                means=means,
                semantic_labels=semantic_labels,
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
            )
            print(f"  Final instance IDs: {len(np.unique(labels))} unique instances")
        else:
            # Use semantic class labels directly (no instance separation)
            print("  Instance separation disabled, using semantic class labels")
            labels = semantic_labels
            # Shift labels to make them 1-indexed (0 = background)
            labels = labels + 1
            print(f"  Instance IDs: {len(np.unique(labels))} unique classes")
    else:
        # Use fallback clustering (no instance separation)
        labels = compute_semantic_labels(features, n_clusters=n_clusters)
        print(f"  Labels computed: {len(np.unique(labels))} unique clusters")

    # Load cameras
    cameras = load_scene_cameras(scene)
    if not cameras:
        print(f"Error: No cameras found for scene {scene}")
        return {"success": False, "error": "No cameras found"}

    # Select subset if specified
    if render_subset is not None:
        cameras = [cameras[i] for i in render_subset if i < len(cameras)]
        print(f"Rendering subset of {len(cameras)} views")

    # Create output directory
    scene_output_dir = os.path.join(output_dir, scene)
    os.makedirs(scene_output_dir, exist_ok=True)

    # Render each view
    results = {"success": True, "num_rendered": 0, "num_failed": 0}

    print(f"\nRendering {len(cameras)} views...")

    for i, camera in enumerate(tqdm(cameras, desc=f"Rendering {scene}")):
        try:
            # Render mask
            mask = render_labels_to_mask(means, labels, opacities, camera)

            # Get output filename
            img_name = camera['img_name']
            output_name = img_name  # Keep original name
            output_path = os.path.join(scene_output_dir, output_name)

            # Save mask
            mask_pil = Image.fromarray(mask)
            mask_pil.save(output_path)
            results["num_rendered"] += 1

        except Exception as e:
            print(f"  Warning: Failed to render view {i}: {e}")
            results["num_failed"] += 1

    print(f"\nScene {scene} complete: {results['num_rendered']} rendered, {results['num_failed']} failed")
    print(f"Output saved to: {scene_output_dir}")

    return results


def process_batch(
    scenes: List[str],
    features_base: str,
    output_dir: str,
    checkpoint_name: str = "checkpoint_with_features_s.pth",
    use_clip: bool = True,
    n_clusters: int = 20,
    class_names: Optional[List[str]] = None,
    class_prompts: Optional[Dict[str, str]] = None,
    clip_model: str = "ViT-B-16-SigLIP2-512",
    clip_pretrained: str = "webli",
    device: torch.device = torch.device("cpu"),
    use_instance_separation: bool = True,
    dbscan_eps: float = 0.05,
    dbscan_min_samples: int = 20,
) -> dict:
    """Process multiple scenes in batch.

    Args:
        scenes: List of scene names
        features_base: Base directory for checkpoint files
        output_dir: Output directory
        checkpoint_name: Name of checkpoint file
        use_clip: Use SigLIP2-based classification (True) or clustering (False)
        n_clusters: Number of clusters for semantic labels (only used if use_clip=False)
        class_names: List of class names for SigLIP2 classification
        class_prompts: Dictionary mapping class names to text prompts
        clip_model: SigLIP2 model name
        clip_pretrained: SigLIP2 pretrained weights
        device: Device for computation
        use_instance_separation: Enable instance separation within semantic classes
        dbscan_eps: DBSCAN eps parameter for instance separation (in meters)
        dbscan_min_samples: DBSCAN min_samples parameter for instance separation

    Returns:
        Dictionary with batch results
    """
    results = {
        "total": len(scenes),
        "success": 0,
        "failed": 0,
        "scenes": {},
    }

    print(f"\n{'='*70}")
    print(f"Batch Processing: {len(scenes)} scenes")
    print(f"{'='*70}")
    print(f"Features base: {features_base}")
    print(f"Checkpoint name: {checkpoint_name}")
    print(f"Output directory: {output_dir}")
    print(f"Classification method: {'SigLIP2' if use_clip else 'Clustering'}")

    # Load SigLIP2 text embeddings once for all scenes
    text_embeddings = None
    if use_clip and OPEN_CLIP_AVAILABLE:
        if class_names is None:
            class_names = REPLICA_CLASSES
        if class_prompts is None:
            class_prompts = REPLICA_CLASS_PROMPTS
        text_embeddings, _ = load_siglip_text_embeddings(
            class_names, class_prompts, clip_model, clip_pretrained, device
        )
        print(f"SigLIP2 text embeddings loaded: {text_embeddings.shape}")

    for scene in scenes:
        # Find checkpoint file
        scene_dir = os.path.join(features_base, f"max_0_depth_True_default_{scene}")
        checkpoint_path = os.path.join(scene_dir, checkpoint_name)

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            results["failed"] += 1
            results["scenes"][scene] = {"success": False, "error": "Checkpoint not found"}
            continue

        # Process scene
        scene_results = process_scene(
            scene=scene,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            use_clip=use_clip,
            n_clusters=n_clusters,
            class_names=class_names,
            class_prompts=class_prompts,
            clip_model=clip_model,
            clip_pretrained=clip_pretrained,
            text_embeddings=text_embeddings,
            device=device,
            use_instance_separation=use_instance_separation,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )

        if scene_results.get("success", True):
            results["success"] += 1
        else:
            results["failed"] += 1

        results["scenes"][scene] = scene_results

    # Print summary
    print(f"\n{'='*70}")
    print(f"Batch Processing Complete")
    print(f"{'='*70}")
    print(f"Total: {results['total']}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Render SceneSplat 3D predictions to 2D masks for Replica evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single scene with SigLIP2 (default, matches eval_3DOVS.sh)
  python tools/render_replica_predictions.py \\
      --scene office_0 \\
      --checkpoint /new_data/cyf/projects/SceneSplat/output_features/max_0_depth_True_default_office_0/checkpoint_with_features_s.pth \\
      --output-dir /path/to/output/masks

  # Batch process all default scenes with SigLIP2
  python tools/render_replica_predictions.py \\
      --batch \\
      --all-scenes \\
      --features-base /new_data/cyf/projects/SceneSplat/output_features \\
      --output-dir /path/to/output/masks

  # Batch process specific scenes
  python tools/render_replica_predictions.py \\
      --batch \\
      --scenes office_0 office_1 room_0 \\
      --features-base /new_data/cyf/projects/SceneSplat/output_features \\
      --output-dir /path/to/output/masks

  # Use fallback clustering instead of SigLIP2
  python tools/render_replica_predictions.py \\
      --batch \\
      --all-scenes \\
      --no-use-siglip2 \\
      --features-base /new_data/cyf/projects/SceneSplat/output_features \\
      --output-dir /path/to/output/masks
        """
    )

    # Single scene mode
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene name (single scene mode)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint_with_features_s.pth file")

    # Batch mode
    parser.add_argument("--batch", action="store_true",
                        help="Enable batch processing mode")
    parser.add_argument("--scenes", type=str, nargs='+', default=None,
                        help="List of scene names for batch processing")
    parser.add_argument("--all-scenes", action="store_true",
                        help="Process all available scenes")
    parser.add_argument("--features-base", type=str,
                        default="/new_data/cyf/projects/SceneSplat/output_features",
                        help="Base directory for checkpoint files (batch mode)")

    # Common arguments
    parser.add_argument("--output-dir", type=str, default=OUTPUT_ROOT,
                        help="Output directory for rendered masks")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_with_features_s.pth",
                        help="Name of checkpoint file (batch mode)")
    parser.add_argument("--render-views", type=int, nargs='+', default=None,
                        help="View indices to render (default: all)")

    # Classification method arguments
    parser.add_argument("--use-siglip2", action="store_true", default=True,
                        help="Use SigLIP2 for open-vocabulary classification (default: True)")
    parser.add_argument("--no-use-siglip2", dest="use_siglip2", action="store_false",
                        help="Disable SigLIP2 and use fallback clustering")
    parser.add_argument("--siglip2-model", type=str, default="ViT-B-16-SigLIP2-512",
                        help="SigLIP2 model name (default: ViT-B-16-SigLIP2-512)")
    parser.add_argument("--siglip2-pretrained", type=str, default="webli",
                        help="SigLIP2 pretrained weights (default: webli)")

    # Instance separation arguments
    parser.add_argument("--no-instance-separation", dest="use_instance_separation",
                        action="store_false", default=True,
                        help="Disable instance separation (use semantic class labels only)")
    parser.add_argument("--dbscan-eps", type=float, default=0.30,
                        help="DBSCAN eps parameter for instance separation in meters (default: 0.30)")
    parser.add_argument("--dbscan-min-samples", type=int, default=500,
                        help="DBSCAN min_samples parameter (default: 500)")

    # Other arguments
    parser.add_argument("--n-clusters", type=int, default=20,
                        help="Number of clusters for semantic labels (only used if --no-use-siglip2)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for computation (default: cuda)")

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device == "cuda":
        print("Warning: CUDA requested but not available, using CPU")

    # Validate arguments
    if args.batch:
        # Batch mode
        if args.all_scenes:
            # Process all default scenes
            scenes = ["office_0", "office_1", "office_2", "office_3", "office_4",
                      "room_0", "room_1", "room_2"]
        elif args.scenes:
            scenes = args.scenes
        else:
            parser.error("--batch requires either --scenes or --all-scenes")

        results = process_batch(
            scenes=scenes,
            features_base=args.features_base,
            output_dir=args.output_dir,
            checkpoint_name=args.checkpoint_name,
            use_clip=args.use_siglip2,
            n_clusters=args.n_clusters,
            clip_model=args.siglip2_model,
            clip_pretrained=args.siglip2_pretrained,
            device=device,
            use_instance_separation=args.use_instance_separation,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
        )
    else:
        # Single scene mode
        if args.scene is None or args.checkpoint is None:
            parser.error("Single scene mode requires --scene and --checkpoint")

        results = process_scene(
            scene=args.scene,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            render_subset=args.render_views,
            use_clip=args.use_siglip2,
            n_clusters=args.n_clusters,
            clip_model=args.siglip2_model,
            clip_pretrained=args.siglip2_pretrained,
            device=device,
            use_instance_separation=args.use_instance_separation,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
        )

    print(f"\nDone!")


if __name__ == "__main__":
    main()
