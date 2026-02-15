#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torchvision
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args, get_combined_args_from_yaml
from gaussian_renderer import GaussianModel
# from scipy.sparse import coo_matrix, csr_matrix
# from cupyx.scipy.sparse.linalg import lsqr, cg, LinearOperator
# import cupyx.scipy.sparse as csp
# import cupy as cp
import numpy as np
from sklearn.decomposition import PCA
import torch.utils.dlpack
import matplotlib.pyplot as plt
import time
import json
import torch.nn.functional as F
import logging
from plyfile import PlyData
import torch.nn as nn
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing

# Add LangSplat autoencoder to path
sys.path.append('/new_data/cyf/projects/LangSplat/autoencoder')
try:
    from model import Autoencoder
    AUTOENCODER_AVAILABLE = True
except ImportError:
    logging.warning("LangSplat autoencoder not available, feature restoration will be skipped")
    AUTOENCODER_AVAILABLE = False


class AdaptiveEncoder(nn.Module):
    """
    Adaptive encoder that can extract features from different layers of a pretrained autoencoder
    to achieve different output dimensions.
    """
    def __init__(self, pretrained_model, target_dim):
        super(AdaptiveEncoder, self).__init__()
        # Get pretrained encoder layers
        original_encoder = pretrained_model.encoder
        self.encoder_layers = []

        current_dim = 512
        for i, layer in enumerate(original_encoder):
            if isinstance(layer, nn.Linear):
                # If current layer output dimension > target_dim, truncate
                if layer.out_features > target_dim:
                    # Last layer: output target dimension
                    new_layer = nn.Linear(current_dim, target_dim)
                    # Load partial weights
                    with torch.no_grad():
                        min_features = min(layer.out_features, target_dim)
                        new_layer.weight[:min_features] = layer.weight[:min_features]
                        if layer.bias is not None:
                            new_layer.bias[:min_features] = layer.bias[:min_features]
                    self.encoder_layers.append(new_layer)
                    break
                else:
                    self.encoder_layers.append(layer)
                    current_dim = layer.out_features
            else:
                self.encoder_layers.append(layer)

        # Build module list
        self.encoder = nn.ModuleList(self.encoder_layers)

    def encode(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


def load_trained_autoencoder(dataset_name, scene_name, encoder_dims=None, decoder_dims=None):
    """
    Load a trained autoencoder from LangSplat checkpoint.

    Args:
        dataset_name: Name of the dataset (e.g., 'scannet', 'lerf_ovs')
        encoder_dims: List of encoder hidden dimensions
        decoder_dims: List of decoder hidden dimensions

    Returns:
        Loaded autoencoder model or None if not available
    """
    encoder_path = "/new_data/cyf/projects/LangSplat"
    if not AUTOENCODER_AVAILABLE:
        return None

    if encoder_dims is None:
        encoder_dims = [256, 128, 64, 32, 3]
    if decoder_dims is None:
        decoder_dims = [16, 32, 64, 128, 256, 256, 512]

    # Check for checkpoint
    ckpt_path = f"{encoder_path}/autoencoder/ckpt/{dataset_name}/{scene_name}/best_ckpt.pth"
    print(f"ckpt: {ckpt_path}")

    try:
        model = Autoencoder(encoder_dims, decoder_dims).cuda()
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)
        model.eval()
        logging.info(f"Loaded trained autoencoder from {ckpt_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load autoencoder from {ckpt_path}: {e}")
        return None


def restore_features_to_512_dim(compressed_features, autoencoder_model):
    """
    Restore compressed language features to original 512 dimensions using trained autoencoder.

    Args:
        compressed_features: Compressed features [N, compressed_dim]
        autoencoder_model: Trained autoencoder model

    Returns:
        Restored features [N, 512]
    """
    if autoencoder_model is None or compressed_features is None:
        return compressed_features

    try:
        with torch.no_grad():
            # Use decoder part of autoencoder to restore features
            if compressed_features.dim() == 1:
                compressed_features = compressed_features.unsqueeze(0)

            compressed_features = compressed_features.cuda()
            input_dim = compressed_features.shape[1]

            # Normalize input features
            compressed_features = compressed_features / compressed_features.norm(dim=-1, keepdim=True)

            # Get decoder layers
            decoder_layers = list(autoencoder_model.decoder.children()) if hasattr(autoencoder_model.decoder, 'children') else list(autoencoder_model.decoder)

            # Check actual weight dimensions of each decoder layer to find correct starting point
            start_idx = None
            for i, layer in enumerate(decoder_layers):
                if hasattr(layer, 'weight'):
                    weight_shape = layer.weight.shape  # [out_features, in_features]
                    expected_input_dim = weight_shape[1]  # in_features

                    if input_dim == expected_input_dim:
                        start_idx = i
                        break

            if start_idx is None:
                logging.error(f"No decoder layer found that accepts {input_dim}-dimensional input")
                return compressed_features.cpu()

            # Pass through decoder layers starting from the appropriate layer
            for layer in decoder_layers[start_idx:]:
                compressed_features = layer(compressed_features)

            # Final normalization
            restored_features = compressed_features / compressed_features.norm(dim=-1, keepdim=True)

            return restored_features.cpu()
    except Exception as e:
        logging.error(f"Failed to restore features: {e}")
        return compressed_features.cpu()


def load_gaussian_language_features(model_path, iteration, feature_level=3):
    """
    Load language features from Gaussian model checkpoint.

    Args:
        model_path: Path to the model directory
        iteration: Iteration number
        feature_level: Feature level to load (default: 3)

    Returns:
        gaussians: GaussianModel with loaded language features
    """
    # Initialize GaussianModel
    gaussians = GaussianModel(sh_degree=3)

    # Load checkpoint with language features
    checkpoint_path = os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}.pth')

    if not os.path.exists(checkpoint_path):
        logging.warning(f"Language feature checkpoint not found: {checkpoint_path}")
        return None

    try:
        (model_params, _) = torch.load(checkpoint_path)

        # Extract all Gaussian state from the language checkpoint (which includes pruned state)
        (active_sh_degree, xyz, features_dc, features_rest,
         scaling, rotation, opacity, language_features,
         max_radii2D, xyz_gradient_accum, denom,
         opt_dict, spatial_lr_scale) = model_params

        # Restore the Gaussian state from the language checkpoint to ensure synchronization
        gaussians.active_sh_degree = active_sh_degree
        gaussians._xyz = xyz
        gaussians._features_dc = features_dc
        gaussians._features_rest = features_rest
        gaussians._scaling = scaling
        gaussians._rotation = rotation
        gaussians._opacity = opacity
        gaussians.max_radii2D = max_radii2D
        gaussians.xyz_gradient_accum = xyz_gradient_accum
        gaussians.denom = denom
        gaussians.spatial_lr_scale = spatial_lr_scale

        # Manually assign language features for the specific level
        if not hasattr(gaussians, "_language_features_dict"):
            gaussians._language_features_dict = {}

        gaussians._language_features_dict[feature_level] = language_features

        logging.info(f"Loaded language features for level {feature_level} from {checkpoint_path}")
        return gaussians

    except Exception as e:
        logging.error(f"Failed to load language features: {e}")
        return None


def compute_point_labels_from_features(gaussians, target_names, text_features_path, feature_level=3):
    """
    Compute semantic labels for each 3D point using Gaussian language features.

    Args:
        gaussians: GaussianModel with loaded language features
        target_names: List of target class names
        text_features_path: Path to text features JSON file
        feature_level: Feature level to use (default: 3)

    Returns:
        pred_labels: Predicted labels for each point [num_points]
    """
    if gaussians is None:
        return None

    # Get language features for each Gaussian point
    gaussian_features = gaussians.get_language_feature(feature_level=feature_level)  # [num_points, feature_dim]

    if gaussian_features is None or len(gaussian_features) == 0:
        logging.warning("No language features found in Gaussian model")
        return None

    # Ensure gaussian_features is on CUDA device
    if not gaussian_features.is_cuda:
        gaussian_features = gaussian_features.cuda()

    # Load text features
    with open(text_features_path, 'r') as f:
        text_features_dict = json.load(f)

    # Create query text features
    query_text_feats = torch.zeros(len(target_names), gaussian_features.shape[1]).cuda()
    all_texts = list(text_features_dict.keys())
    text_feat_values = torch.from_numpy(np.array(list(text_features_dict.values()))).to(torch.float32)

    # Handle dimension mismatch
    if text_feat_values.shape[1] != gaussian_features.shape[1]:
        logging.warning(f"Text feature dimension ({text_feat_values.shape[1]}) "
                        f"doesn't match Gaussian feature dimension ({gaussian_features.shape[1]})")
        # Adjust or pad features as needed
        min_dim = min(text_feat_values.shape[1], gaussian_features.shape[1])
        text_feat_values = text_feat_values[:, :min_dim]
        gaussian_features = gaussian_features[:, :min_dim]

    # Match target names with available text features
    for i, text in enumerate(target_names):
        if text in all_texts:
            feat = text_feat_values[all_texts.index(text)].unsqueeze(0)
            if feat.shape[1] == query_text_feats.shape[1]:
                query_text_feats[i] = feat
            else:
                # Pad or truncate to match dimensions
                query_text_feats[i, :feat.shape[1]] = feat.squeeze(0)
        else:
            logging.warning(f"Text '{text}' not found in features file")
            query_text_feats[i] = torch.randn(query_text_feats.shape[1]).cuda()

    # Calculate cosine similarity
    query_text_feats = F.normalize(query_text_feats, dim=1, p=2)
    gaussian_features = F.normalize(gaussian_features, dim=1, p=2)

    # Compute similarity scores
    similarity_scores = torch.matmul(gaussian_features, query_text_feats.transpose(0, 1))

    # Get predicted class (max similarity)
    pred_labels = torch.argmax(similarity_scores, dim=1) + 1  # +1 to make class IDs start from 1

    return pred_labels


def extract_gaussian_features(model_path, iteration, source_path, views, gaussians, pipeline, background, feature_level, restore_featdim=True, src_dim=512, save_npy=None):

    language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}.pth')

    # Determine language feature directory based on src_dim
    encoder_dims, decoder_dims = [], []
    if src_dim == 512:
        language_feature_dir = f"{source_path}/language_features"
        logging.info(f"Loading original 512-dimensional language features from {language_feature_dir}")
    elif src_dim == 768:
        language_feature_dir = f"{source_path}/language_features_siglip2_sam2"
        logging.info(f"Loading 768-dimensional siglip2 language features from {language_feature_dir}")
    else:
        language_feature_dir = f"{source_path}/language_features_dim{src_dim}"
        logging.info(f"Loading {src_dim}-dimensional compressed language features from {language_feature_dir}")

        if src_dim == 16:
            # Use architecture dimensions that match the checkpoint
            encoder_dims = [256, 128, 64, 32, 16]
            decoder_dims = [32, 64, 128, 256, 512]
        if src_dim == 32:
            # Use architecture dimensions that match the checkpoint
            encoder_dims = [256, 128, 64, 32]
            decoder_dims = [64, 128, 256, 512]
        if src_dim == 64:
            # Use architecture dimensions that match the checkpoint
            encoder_dims = [256, 128, 64]
            decoder_dims = [128, 256, 512]

    # ============================================================================
    # Use FULLY VECTORIZED sparse matrix operations with lsqr solver
    # ============================================================================

    # Get number of gaussians
    num_gaussians = gaussians.get_xyz.shape[0]
    feature_dim = src_dim

    # Start timing
    start_time = time.time()

    for view in tqdm(views, desc="Accumulating views"):
        render_pkg = render(view, gaussians, pipeline, background)

        gt_language_feature, gt_mask = view.get_language_feature(
            language_feature_dir=language_feature_dir, feature_level=feature_level
        )

        activated = render_pkg["info"]["activated"]
        significance = render_pkg["info"]["significance"]

        means2D = render_pkg["info"]["means2d"]
        mask = activated[0] > 0
        significance = significance[0,mask]
        means2D = means2D[0,mask]
        gaussians.accumulate_gaussian_feature_per_view(gt_language_feature.permute(1, 2, 0), gt_mask.squeeze(0), mask, significance, means2D, feature_level=feature_level)
    
    logging.info("Accumulated features using direct method")
    gaussians.finalize_gaussian_features(feature_level, allow_pruning=False)
    end_time = time.time()
    print("-" * 50)
    print(f"extract_gaussian_features took {end_time - start_time:.4f} seconds")
    print("-" * 50)

    # Save low-dimensional features first
    low_dim_features = gaussians.capture_language_feature(feature_level=feature_level)

    # Only save if features were actually captured
    if low_dim_features is None:
        logging.warning(f"No features captured for level {feature_level}, skipping save")
        return

    if src_dim not in [512, 768]:
        low_dim_save_path = os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}_dim{src_dim}.pth')
        torch.save((low_dim_features, 0), low_dim_save_path)
        logging.info(f"Low-dimensional features saved to: {low_dim_save_path}")
        print(f"Low-dimensional checkpoint saved to: {low_dim_save_path}")

    # Only restore features if src_dim != 512 and restoration is requested
    if restore_featdim and src_dim not in [512, 768] and hasattr(gaussians, "_language_features_dict") and feature_level in gaussians._language_features_dict:
        current_features = gaussians._language_features_dict[feature_level]
        if current_features.shape[1] not in [512, 768]:
            logging.info(f"Restoring extracted language features from {current_features.shape[1]} to 512 dimensions using autoencoder")

            # Try to infer dataset name from model path
            dataset_name = None
            if 'scannet' in model_path.lower():
                dataset_name = 'scannet'
            elif 'lerf' in model_path.lower():
                dataset_name = 'lerf_ovs'

            if dataset_name:
                # Load trained autoencoder
                scene_name = os.path.basename(model_path)
                autoencoder = load_trained_autoencoder(dataset_name, f"{scene_name}_{src_dim}", encoder_dims, decoder_dims)
                if autoencoder is not None:
                    try:
                        # Restore features to 512 dimensions
                        print(current_features.shape)
                        restored_features = restore_features_to_512_dim(current_features, autoencoder)
                        if restored_features is not None and restored_features.shape[1] == 512:
                            gaussians._language_features_dict[feature_level] = restored_features
                            logging.info("Successfully restored extracted features to 512 dimensions")
                        else:
                            logging.warning("Failed to restore extracted features to 512 dimensions, using original")
                    except Exception as e:
                        logging.error(f"Error during extracted feature restoration: {e}")
                        logging.info("Using original compressed extracted features")
                else:
                    logging.warning("No autoencoder available for extracted features, using original compressed features")
            else:
                logging.warning("Could not infer dataset name for extracted features, skipping restoration")
    elif src_dim == 512:
        logging.info("Using original 512-dimensional features, no autoencoder restoration needed")
    elif src_dim == 768:
        logging.info("Using original 768-dimensional features, no autoencoder restoration needed")

    # Save final features (only if restore_featdim is True or src_dim == 512)
    if restore_featdim or src_dim == 512:
        torch.save((gaussians.capture_language_feature(feature_level=feature_level), 0), language_feature_save_path)
        print("checkpoint saved to: ", language_feature_save_path)
    else:
        logging.info("Skipping final save as restore_featdim=False and src_dim!=512")

    # Save features as NPY file if save_npy path is specified
    if save_npy is not None:
        try:
            # Get the final language features (use get_language_feature for tensor, not capture for tuple)
            final_features = gaussians.get_language_feature(feature_level=feature_level)
            if final_features is not None:
                # Convert to numpy array
                features_np = final_features.cpu().numpy()

                # Compute valid_feat_mask (1 if feature is not all zeros, 0 otherwise)
                # Use int64 to match SceneSplat7k dataset format
                valid_feat_mask = np.any(features_np != 0.0, axis=1).astype(int)

                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(save_npy)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                # Determine save path for lang_feat.npy
                if os.path.isdir(save_npy):
                    lang_feat_path = os.path.join(save_npy, "lang_feat.npy")
                    valid_feat_mask_path = os.path.join(save_npy, "valid_feat_mask.npy")
                else:
                    lang_feat_path = save_npy
                    # Replace lang_feat.npy with valid_feat_mask.npy in the path
                    if save_npy.endswith("lang_feat.npy"):
                        valid_feat_mask_path = save_npy.replace("lang_feat.npy", "valid_feat_mask.npy")
                    else:
                        # Fallback: use same directory
                        valid_feat_mask_path = os.path.join(os.path.dirname(save_npy), "valid_feat_mask.npy")

                # Save lang_feat.npy
                np.save(lang_feat_path, features_np)
                logging.info(f"Language features saved as NPY to: {lang_feat_path}")
                print(f"NPY file saved to: {lang_feat_path}")
                print(f"  Shape: {features_np.shape}, Dtype: {features_np.dtype}")

                # Save valid_feat_mask.npy
                np.save(valid_feat_mask_path, valid_feat_mask)
                logging.info(f"Valid feature mask saved as NPY to: {valid_feat_mask_path}")
                print(f"Valid feature mask saved to: {valid_feat_mask_path}")
                print(f"  Shape: {valid_feat_mask.shape}, Dtype: {valid_feat_mask.dtype}")
                print(f"  Valid count: {valid_feat_mask.sum()}/{len(valid_feat_mask)} "
                      f"({100.0 * valid_feat_mask.sum() / len(valid_feat_mask):.2f}%)")
            else:
                logging.warning("No features to save as NPY")
        except Exception as e:
            logging.error(f"Failed to save NPY file: {e}")


def process_scene_language_features(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, feature_level : int, restore_featdim=True, src_dim=512, save_npy=None):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)

        checkpoint = os.path.join(args.model_path, f'ckpts/chkpnt{iteration}.pth')
        gaussians.restore_from_gsplat_checkpoint(checkpoint, opt)
        # (model_params, _) = torch.load(checkpoint)
        # gaussians.restore_rgb(model_params, opt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        extract_gaussian_features(args.model_path, iteration, dataset.source_path, scene.getTrainCameras(), gaussians, pipeline, background, feature_level, restore_featdim, src_dim, save_npy)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--src_dim", type=int, default=512, help="Source dimension of language features to load (e.g., 512, 256, 128, 64, 32)")
    parser.add_argument("--skip_restoration", action="store_true", help="Skip feature restoration to 512 dimensions using autoencoder")
    parser.add_argument("--save_npy", type=str, default=None, help="Save language features as NPY file to specified path (e.g., gaussian_train/scene_name/lang_feat.npy)")
    args = get_combined_args_from_yaml(parser, param_groups=[model, pipeline, opt])

    # Initialize system state (RNG)
    from utils.general_utils import safe_state
    safe_state(args.quiet)

    restore_featdim = not args.skip_restoration
    src_dim = args.src_dim

    process_scene_language_features(model.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.feature_level, restore_featdim, src_dim, args.save_npy)