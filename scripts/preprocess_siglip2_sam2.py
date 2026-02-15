"""
Preprocess script for extracting SigLIP2 features with SAM2 segmentation.

This script implements the HOV-SG fusion strategy with:
- SAM2 for segmentation (single scale)
- SigLIP2 for feature extraction (768-dim features)
- Three-view fusion: full_image + masked_crop + background_crop

Output format (compatible with mini-splatting2/SceneSplat):
    - {image_name}_s.npy: segmentation maps [1, H, W]
    - {image_name}_f.npy: fused features [N, 768] where N is number of segments

Usage:
    # Process all scenes in a dataset (full resolution)
    python scripts/preprocess_siglip2_sam2.py \
        --dataset_path /path/to/dataset \
        --sam2_ckpt_path /path/to/sam2.1_hiera_large.pt

    # Process with downsampled images (2x downsample -> images_2/)
    python scripts/preprocess_siglip2_sam2.py \
        --dataset_path /path/to/dataset \
        --sam2_ckpt_path /path/to/sam2.1_hiera_large.pt \
        --resolution 2

    # Process specific scenes with 4x downsampled images
    python scripts/preprocess_siglip2_sam2.py \
        --dataset_path /path/to/dataset \
        --sam2_ckpt_path /path/to/sam2.1_hiera_large.pt \
        --scenes ramen teatime \
        --resolution 4 \
        --visualize
"""

import os
import random
import argparse
import numpy as np
import torch
import json
import cv2
import glob
from PIL import Image
from tqdm import tqdm

# =============================================================================
# Import SAM2 and SigLIP2 from same directory
# =============================================================================
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

# Import SigLIP2 from model.py in the same directory
from model import SigLIPNetwork, SigLIPNetworkConfig, CROP_SIZE


# =============================================================================
# Global variables
# =============================================================================
mask_generator = None
model = None


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_segmentation(image_np, seg_map, save_path, image_name):
    """
    Visualize segmentation results.

    Args:
        image_np: Original image [H, W, 3]
        seg_map: Segmentation map [H, W] with segment IDs
        save_path: Directory to save visualizations
        image_name: Name of the image for file naming
    """
    os.makedirs(save_path, exist_ok=True)

    # Create colored segmentation map
    h, w = seg_map.shape
    num_segments = int(seg_map.max()) + 1

    # Generate random colors for each segment
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_segments, 3), dtype=np.uint8)

    # Create colored segmentation map
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for seg_id in range(num_segments):
        mask = seg_map == seg_id
        colored_seg[mask] = colors[seg_id]

    # Handle background (-1 values)
    background_mask = seg_map == -1
    colored_seg[background_mask] = [0, 0, 0]

    # Create overlay with contours
    overlay = image_np.copy()
    contours, _ = cv2.findContours(
        (seg_map != -1).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Save original image
    cv2.imwrite(os.path.join(save_path, f"{image_name}_original.png"),
                cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    # Save colored segmentation
    cv2.imwrite(os.path.join(save_path, f"{image_name}_segmentation.png"), colored_seg)

    # Save overlay
    cv2.imwrite(os.path.join(save_path, f"{image_name}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Create a side-by-side comparison
    comparison = np.hstack([
        cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
        colored_seg,
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    ])
    cv2.imwrite(os.path.join(save_path, f"{image_name}_comparison.png"), comparison)

    print(f"Saved visualizations to {save_path}")


# =============================================================================
# SAM2 Mask Processing Functions (following preprocess_matterport3d.py pattern)
# =============================================================================

def filter(keep: torch.Tensor, masks_result):
    """Filter masks based on keep indices."""
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep:
            result_keep.append(m)
    return result_keep


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """Non-maximum suppression for masks (from preprocess_matterport3d.py)."""
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros(
        (num_masks,) * 2, dtype=torch.float, device=masks.device
    )
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(
                torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float
            )
            union = torch.sum(
                torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float
            )
            iou = intersection / union
            iou_matrix[i, j] = iou
            if (
                intersection / masks_area[i] < 0.5
                and intersection / masks_area[j] >= 0.85
            ):
                inner_iou = 1 - (intersection / masks_area[j]) * (
                    intersection / masks_area[i]
                )
                inner_iou_matrix[i, j] = inner_iou
            if (
                intersection / masks_area[i] >= 0.85
                and intersection / masks_area[j] < 0.5
            ):
                inner_iou = 1 - (intersection / masks_area[j]) * (
                    intersection / masks_area[i]
                )
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    k = min(3, scores.size(0))
    if keep_conf.sum() == 0:
        index = scores.topk(k).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(k).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(k).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def masks_update(*args, **kwargs):
    """Update masks with NMS filtering (from preprocess_matterport3d.py)."""
    masks_new = ()
    for masks_lvl in args:
        seg_pred = torch.from_numpy(
            np.stack([m["segmentation"] for m in masks_lvl], axis=0)
        )
        iou_pred = torch.from_numpy(
            np.stack([m["predicted_iou"] for m in masks_lvl], axis=0)
        )
        stability = torch.from_numpy(
            np.stack([m["stability_score"] for m in masks_lvl], axis=0)
        )

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new


def get_bbox_crop(mask, image):
    """Crop the image using the mask's bounding box, retaining background."""
    x, y, w, h = np.int32(mask["bbox"])
    return image[y : y + h, x : x + w, ...]


def get_seg_img(mask, image):
    """Get masked crop with background blacked out."""
    image = image.copy()
    image[mask["segmentation"] == 0] = np.array([0, 0, 0], dtype=np.uint8)
    x, y, w, h = np.int32(mask["bbox"])
    seg_img = image[y : y + h, x : x + w, ...]
    return seg_img


def pad_img(img):
    """Pad image to square."""
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w) // 2 : (h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2 : (w - h) // 2 + h, :, :] = img
    return pad


def sam_encoder(image, min_segments=3):
    """
    SAM encoder function following preprocess_matterport3d.py pattern.
    Generates both masked and background crops for fusion.
    """
    image = cv2.cvtColor(
        image[0].permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB
    )
    masks_default = mask_generator.generate(image)
    if len(masks_default) < min_segments:
        # corner case: not enough segments found after inference
        return None, None
    masks_default = masks_update(
        masks_default, iou_thr=0.8, score_thr=0.7, inner_thr=0.5
    )[0]
    if len(masks_default) == 0:
        # corner case: no segments found after NMS
        return None, None

    def mask2segmap(masks, image):
        """Convert masks to segmentation maps and crops (HOV-SG style)."""
        seg_img_list_masked = []
        seg_img_list_background = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)

        for i, mask in enumerate(masks):
            # Masked crop (background black)
            seg_img_masked = get_seg_img(mask, image)
            pad_seg_img_masked = cv2.resize(
                pad_img(seg_img_masked), (CROP_SIZE, CROP_SIZE)
            )
            seg_img_list_masked.append(pad_seg_img_masked)

            # Crop with background
            seg_img_background = get_bbox_crop(mask, image)
            pad_seg_img_background = cv2.resize(
                pad_img(seg_img_background), (CROP_SIZE, CROP_SIZE)
            )
            seg_img_list_background.append(pad_seg_img_background)

            seg_map[mask["segmentation"]] = i

        # Convert to PyTorch tensors
        seg_imgs_masked = (
            torch.from_numpy(
                np.stack(seg_img_list_masked, axis=0).astype("float32")
            ).permute(0, 3, 1, 2)
        ).to("cuda")
        seg_imgs_background = (
            torch.from_numpy(
                np.stack(seg_img_list_background, axis=0).astype("float32")
            ).permute(0, 3, 1, 2)
        ).to("cuda")

        return {"masked": seg_imgs_masked, "background": seg_imgs_background}, seg_map

    # check the size each mask in masks_default
    masks_default = [m for m in masks_default if m["area"] > 10]
    seg_images, seg_maps = {}, {}
    seg_images["default"], seg_maps["default"] = mask2segmap(masks_default, image)
    return seg_images, seg_maps


# =============================================================================
# HOV-SG Fusion Strategy
# =============================================================================

def _embed_clip_sam_tiles(image, sam_encoder_func):
    """
    Embedding function with HOV-SG fusion strategy.
    Fuses three views: full_image + masked_crop + background_crop.
    """
    # Concatenate input image
    aug_imgs = torch.cat([image])  # Shape: (1, 3, H, W]
    seg_images, seg_map = sam_encoder_func(aug_imgs)

    # check corner case
    if seg_images is None:
        return None, None

    # Extract full-image SigLIP2 feature (F_g)
    with torch.no_grad():
        F_g = model.encode_image((aug_imgs).to("cuda"))
        F_g = F_g / F_g.norm(dim=-1, keepdim=True)

    # Process crops in batches
    batch_size = 32

    def process_batch(crops):
        """Helper function to process a batch of crops"""
        num_crops = crops.shape[0]
        features = []
        for i in range(0, num_crops, batch_size):
            batch = crops[i : i + batch_size].to("cuda")
            with torch.no_grad():
                batch_feats = model.encode_image(batch)
            features.append(batch_feats.cpu())
            del batch, batch_feats  # Free GPU memory
            torch.cuda.empty_cache()
        return torch.cat(features, dim=0)

    # Extract features for background and masked crops
    fm = process_batch(seg_images["default"]["masked"]).cuda()
    fm = fm / fm.norm(dim=-1, keepdim=True)

    fl = process_batch(seg_images["default"]["background"]).cuda()
    fl = fl / fl.norm(dim=-1, keepdim=True)

    # -----------------------------------------------------------------------
    # Dynamic weighting between fl and fm (HOV-SG strategy)
    sim_fl_fm = torch.nn.functional.cosine_similarity(
        fl, fm, dim=-1, eps=1e-8
    )  # Shape: (num_masks,)
    # If fl and fm are very similar (sim close to 1), the contribution of fm is high.
    # If they are dissimilar (sim is lower), we put more weight on crop_w_bg.
    dynamic_masked_weight = sim_fl_fm.unsqueeze(-1)  # Shape: (num_masks, 1)
    # -----------------------------------------------------------------------

    # Fuse fl and fm using the dynamic weight
    F_l = dynamic_masked_weight * fm + (1 - dynamic_masked_weight) * fl
    F_l = torch.nn.functional.normalize(F_l, p=2, dim=-1).cuda()

    # Compute dot product between F_l and F_g for each mask
    F_g_expanded = F_g.expand(F_l.shape[0], -1)  # Shape: (num_masks, 768)
    cos = torch.nn.CosineSimilarity(dim=-1)
    phi_l_G = cos(F_l, F_g_expanded)

    # Compute similarity scores for the full image and the crops
    # (w_i now indicates how similar the fused crop is to the full image)
    w_i = torch.softmax(phi_l_G, dim=0).unsqueeze(-1)  # Shape: (num_masks, 1)

    # Compute the three weights for fusing the features:
    # - wg for the full image,
    # - wm for the masked crop,
    # - wl for the background crop.
    wg = w_i
    wm = (1 - w_i) * dynamic_masked_weight
    wl = (1 - w_i) * (1 - dynamic_masked_weight)

    # Fuse all three features: F_p = wg*F_g + wl*fl + wm*fm
    F_p = wg * F_g_expanded + wl * fl + wm * fm
    F_p = torch.nn.functional.normalize(F_p, p=2, dim=-1)

    # Print out diagnostic information
    print(
        f"Average weights | full_img: {wg.mean().item():.3f}, "
        f"crop_w_bg: {wl.mean().item():.3f}, "
        f"crop_masked: {wm.mean().item():.3f}"
    )
    print(
        f"Average cosine similarity between crop_w_bg and crop_masked: {sim_fl_fm.mean().item():.3f}"
    )

    # Return the fused SigLIP2 embeddings in half precision
    clip_embeds = {"default": F_p.detach().cpu().half()}
    return clip_embeds, seg_map


# =============================================================================
# Processing Functions (following preprocess_matterport3d.py pattern)
# =============================================================================

def process_single_image(img, data_path, image_name, save_folder, visualize=False, orig_img_np=None):
    """
    Process a single image and save the outputs (feature + seg_maps)
    to disk as .npy files.

    Args:
        img: Image tensor [1, C, H, W]
        data_path: Path to the scene
        image_name: Name of the image
        save_folder: Directory to save outputs
        visualize: Whether to save visualization images
        orig_img_np: Original image as numpy array [H, W, 3] for visualization
    """
    embed_size = model.embedding_dim

    img_embed, seg_map = _embed_clip_sam_tiles(img, sam_encoder)

    # check corner case
    if img_embed is None:
        print(f"Image {image_name} has not enough valid segments, skipping...")
        return

    lengths = [len(v) for k, v in img_embed.items()]
    total_length = sum(lengths)

    img_embed_concat = torch.cat([v for k, v in img_embed.items()], dim=0)
    assert img_embed_concat.shape[0] == total_length

    seg_map_tensor_list = []
    lengths_cumsum = lengths[:]
    for j in range(1, len(lengths)):
        lengths_cumsum[j] += lengths_cumsum[j - 1]

    for j, (k, v) in enumerate(seg_map.items()):
        mask_int = torch.from_numpy(v)
        if j != 0:
            assert (
                mask_int.max() == lengths[j] - 1
            ), f"{j}, {mask_int.max()}, {lengths[j] - 1}"
            mask_int[mask_int != -1] += lengths_cumsum[j - 1]
        seg_map_tensor_list.append(mask_int)

    seg_map_tensor = torch.stack(seg_map_tensor_list, dim=0)

    # scene_name = data_path.split("/")[-1]
    # save_folder = os.path.join(save_folder, scene_name)
    # os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, image_name)
    curr = {"feature": img_embed_concat, "seg_maps": seg_map_tensor}
    save_numpy(save_path, curr)

    # Visualize segmentation results if requested
    if visualize and orig_img_np is not None:
        vis_save_path = os.path.join(save_folder, "visualizations")
        visualize_segmentation(orig_img_np, seg_map["default"], vis_save_path, image_name)


def save_numpy(save_path, data):
    """Save features and segmentation maps."""
    save_path_s = save_path + "_s.npy"
    save_path_f = save_path + "_f.npy"
    np.save(save_path_s, data["seg_maps"].numpy())
    np.save(save_path_f, data["feature"].numpy())
    print(
        f"Saved {save_path_s.split('/')[-1]}, shape {data['seg_maps'].shape}, {save_path_f.split('/')[-1]}, shape {data['feature'].shape}"
    )


def get_scenes(dataset_root):
    """
    Discover scene folders in the dataset root.

    A valid scene folder contains either:
    - transforms_train.json (NeRF/3DGS format)
    - images/ directory (COLMAP format)

    Returns:
        List of scene paths (full paths)
    """
    scenes = []

    # If dataset_root itself is a scene (has transforms_train.json or images/)
    if os.path.exists(os.path.join(dataset_root, "transforms_train.json")) or \
       os.path.exists(os.path.join(dataset_root, "images")):
        return [dataset_root]

    # Otherwise, scan subdirectories for scenes
    for item in os.listdir(dataset_root):
        item_path = os.path.join(dataset_root, item)
        if not os.path.isdir(item_path):
            continue

        # Check if this is a valid scene folder
        if os.path.exists(os.path.join(item_path, "transforms_train.json")) or \
           os.path.exists(os.path.join(item_path, "images")):
            scenes.append(item_path)

    if len(scenes) == 0:
        raise FileNotFoundError(
            f"No valid scenes found in {dataset_root}. "
            f"Each scene should have either transforms_train.json or images/ directory."
        )

    return sorted(scenes)


def get_selected_image_paths(scene_path, resolution=None):
    """
    Get image paths from either transforms_train.json or COLMAP format.

    Supports:
    1. transforms_train.json (NeRF/3DGS format)
    2. COLMAP format (images/ or images_{resolution}/ directory)

    Args:
        scene_path: Path to the scene directory
        resolution: Downsampling factor. If > 0, look for images in images_{resolution}/
                    directory (e.g., resolution=2 -> images_2/). If -1 or None, use
                    default images/ directory.

    Returns:
        List of image paths
    """
    data_list = []

    # Try transforms_train.json first (NeRF/3DGS format)
    json_path = os.path.join(scene_path, "transforms_train.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)

        frames_list = json_data.get("frames", {})
        for frame_data in frames_list:
            img_path = os.path.join(scene_path, frame_data["file_path"])
            data_list.append(img_path)

        data_list = sorted(data_list)
        return data_list

    # Try COLMAP format (images/ or images_{resolution}/ directory)
    if resolution is not None and resolution > 0:
        # Try images_{resolution}/ directory first
        images_dir = os.path.join(scene_path, f"images_{resolution}")
        if os.path.exists(images_dir) and os.path.isdir(images_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                data_list.extend(glob.glob(os.path.join(images_dir, ext)))

            if len(data_list) > 0:
                data_list = sorted(data_list)
                print(f"Found {len(data_list)} images in {images_dir}/")
                return data_list
            else:
                print(f"Warning: images_{resolution}/ directory exists but contains no images")

    # Fall back to default images/ directory
    images_dir = os.path.join(scene_path, "images")
    if os.path.exists(images_dir) and os.path.isdir(images_dir):
        # Get all image files from COLMAP images directory
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            data_list.extend(glob.glob(os.path.join(images_dir, ext)))

        if len(data_list) == 0:
            raise FileNotFoundError(f"No images found in {images_dir}")

        data_list = sorted(data_list)
        print(f"Found {len(data_list)} images in images/")
        return data_list

    # If neither format is found
    raise FileNotFoundError(
        f"No valid dataset format found in {scene_path}. "
        f"Expected either transforms_train.json or images/ directory "
        f"(or images_{resolution}/ if downsampling factor > 0)."
    )


def seed_everything(seed_value):
    """Set random seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


# =============================================================================
# Main
# =============================================================================
# python scripts/preprocess_siglip2_sam2.py --dataset_path datasets/3DOVS --sam2_ckpt_path sam2_repo/checkpoints/sam2.1_hiera_large.pt --scenes bed bench lawn room sofa --visualize --resolution 4
if __name__ == "__main__":
    seed_num = 1219
    seed_everything(seed_num)

    parser = argparse.ArgumentParser(
        description="Extract SigLIP2 features with SAM2 segmentation using HOV-SG fusion"
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset root or single scene folder")
    parser.add_argument("--sam2_ckpt_path", type=str, required=True,
                       help="Path to SAM2 checkpoint (.pt file)")
    parser.add_argument("--resolution", type=int, default=-1,
                       help="Downsampling factor (-1 for full resolution images/, "
                            "2 for images_2/, 4 for images_4/, etc.)")
    parser.add_argument("--visualize", action="store_true",
                       help="Save visualization of segmentation results")
    parser.add_argument("--scenes", type=str, nargs='+', default=None,
                       help="Specific scene names to process (default: all scenes). "
                            "Example: --scenes ramen teatime")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam2_ckpt_path = args.sam2_ckpt_path

    # Discover all scenes in the dataset
    scenes = get_scenes(dataset_path)

    # Filter by scene name if --scenes is specified
    if args.scenes:
        scene_names_to_process = set(args.scenes)
        scenes_before = scenes[:]
        scenes = [s for s in scenes if os.path.basename(s) in scene_names_to_process]

        if len(scenes) == 0:
            raise ValueError(
                f"None of the specified scenes found: {args.scenes}\n"
                f"Available scenes in {dataset_path}:\n" +
                "\n".join(f"  - {os.path.basename(s)}" for s in scenes_before)
            )

        print(f"Filtered to {len(scenes)} specified scene(s): {args.scenes}")
        missing_scenes = scene_names_to_process - set(os.path.basename(s) for s in scenes)
        if missing_scenes:
            print(f"Warning: specified scenes not found: {missing_scenes}")

    print(f"Found {len(scenes)} scene(s) to process:")
    for scene in scenes:
        print(f"  - {os.path.basename(scene)}")

    # Initialize SigLIP2 model
    print("\nInitializing SigLIP2 model")
    model = SigLIPNetwork(SigLIPNetworkConfig)

    # Initialize SAM2
    print(f"Initializing SAM2 from {sam2_ckpt_path}")
    config_registry = {
        "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    sam_2_registry = sam2_ckpt_path.replace(".pt", "").split("/")[-1]
    sam2_config = config_registry.get(sam_2_registry, "configs/sam2.1/sam2.1_hiera_l.yaml")
    sam2 = build_sam2(sam2_config, sam2_ckpt_path)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    mask_generator.predictor.model.to("cuda")

    # Process each scene
    for scene_idx, scene_path in enumerate(scenes):
        scene_name = os.path.basename(scene_path)
        print(f"\n{'='*60}")
        print(f"Processing scene {scene_idx + 1}/{len(scenes)}: {scene_name}")
        print(f"{'='*60}")

        # Get image paths for this scene
        data_list = get_selected_image_paths(scene_path, resolution=args.resolution)
        print(f"Found {len(data_list)} images")

        # Set output folder: {scene_path}/language_features_siglip2_sam2/
        save_folder = os.path.join(scene_path, "language_features_siglip2_sam2")
        os.makedirs(save_folder, exist_ok=True)
        print(f"Output folder: {save_folder}")

        # Load images directly (no ImageDataset)
        for img_path in tqdm(data_list, desc=f"Processing {scene_name}"):
            image_name = os.path.splitext(os.path.basename(img_path))[0]

            # Load image
            pil_img = Image.open(img_path)
            if pil_img.mode == 'L':
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')

            # Resize if needed
            # if args.resolution != -1:
            #     orig_w, orig_h = pil_img.size
            #     scale = orig_w / args.resolution if args.resolution > 1 else 1.0
            #     resolution = (int(orig_w / scale), int(orig_h / scale))
            #     pil_img = pil_img.resize(resolution, Image.LANCZOS)

            # Convert to numpy for visualization and tensor for processing
            orig_img_np = np.array(pil_img, dtype=np.uint8)  # Keep RGB for visualization
            image_tensor = torch.from_numpy(orig_img_np).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

            # Process image
            process_single_image(
                image_tensor,
                scene_path,
                image_name,
                save_folder,
                visualize=args.visualize,
                orig_img_np=orig_img_np
            )

    mask_generator.predictor.model.to("cpu")
    print("\n" + "="*60)
    print(f"Done! Processed {len(scenes)} scene(s)")
    print("="*60)
