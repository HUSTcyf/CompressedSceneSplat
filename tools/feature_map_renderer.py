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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args, get_combined_args_from_yaml
from gaussian_renderer import GaussianModel
import numpy as np
from sklearn.decomposition import PCA
import torch.utils.dlpack
import matplotlib.pyplot as plt
            
def render_set(model_path, name, iteration, source_path, views, gaussians, pipeline, background, feature_level, src_dim=-1, use_siglip_sam2_format=False, visualize=False):
    
    save_path = os.path.join(model_path, name, "ours_{}_langfeat_{}".format(iteration, feature_level))
    render_path = os.path.join(save_path, "renders")
    gts_path = os.path.join(save_path, "gt")
    render_npy_path = os.path.join(save_path, "renders_npy")
    gts_npy_path = os.path.join(save_path,"gt_npy")
    
    if visualize:
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
    os.makedirs(render_npy_path, exist_ok=True)
    os.makedirs(gts_npy_path, exist_ok=True)
    
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Clear GPU cache before each render to avoid OOM
        torch.cuda.empty_cache()

        render_pkg = render(view, gaussians, pipeline, background, include_feature=True, feature_level=feature_level)
        rendering = render_pkg["render"]

        # Move to CPU immediately to free GPU memory
        rendering = rendering.cpu()
        torch.cuda.empty_cache()
        # Determine language feature directory based on src_dim
        if use_siglip_sam2_format:
            language_feature_dir = f"{source_path}/language_features_siglip2_sam2"
        elif src_dim == -1 or src_dim == 512:
            language_feature_dir = f"{source_path}/language_features"
        else:
            language_feature_dir = f"{source_path}/language_features_dim{src_dim}"

        gt, mask = view.get_language_feature(language_feature_dir=language_feature_dir, feature_level=feature_level)
        # Move gt to CPU to free GPU memory
        gt = gt.cpu()

        np.save(os.path.join(render_npy_path, view.image_name.split('.')[0] + ".npy"), rendering.permute(1,2,0).numpy())
        np.save(os.path.join(gts_npy_path, view.image_name.split('.')[0] + ".npy"), gt.permute(1,2,0).numpy())

        if visualize:
            # Get dimensions from both tensors
            gt_feat_dim, gt_H, gt_W = gt.shape
            render_feat_dim, H, W = rendering.shape

            # Check if dimensions match
            if gt_feat_dim != render_feat_dim:
                print(f"Warning: Feature dimensions differ (gt={gt_feat_dim}, render={render_feat_dim}), skipping gt visualization")
                # Only visualize rendering using its own dimension
                rendering_reshaped = rendering.reshape(render_feat_dim, -1).T.cpu().numpy()  # (H*W, render_feat_dim)

                pca = PCA(n_components=3)
                render_features = pca.fit_transform(rendering_reshaped)  # (H*W, 3)
                render_normalized = (render_features - render_features.min(axis=0)) / (render_features.max(axis=0) - render_features.min(axis=0))
                reduced_rendering = render_normalized.reshape(H, W, 3)

                rendering_vis = torch.tensor(reduced_rendering).permute(2, 0, 1)
                torchvision.utils.save_image(rendering_vis, os.path.join(render_path, view.image_name + ".jpg"))
            else:
                # Same dimension: use combined PCA
                gt = gt.reshape(gt_feat_dim, -1).T.cpu().numpy()
                rendering = rendering.reshape(render_feat_dim, -1).T.cpu().numpy()

                pca = PCA(n_components=3)
                combined_np = np.concatenate((gt, rendering), axis=0)
                combined_features = pca.fit_transform(combined_np)
                normalized_features = (combined_features - combined_features.min(axis=0)) / (combined_features.max(axis=0) - combined_features.min(axis=0))
                reshaped_combined_features = normalized_features.reshape(2, H, W, 3)

                reduced_rendering = reshaped_combined_features[1]
                reduced_gt = reshaped_combined_features[0]

                rendering_vis = torch.tensor(reduced_rendering).permute(2, 0, 1)
                gt_vis = torch.tensor(reduced_gt).permute(2, 0, 1)

                torchvision.utils.save_image(rendering_vis, os.path.join(render_path, view.image_name + ".jpg"))
                torchvision.utils.save_image(gt_vis, os.path.join(gts_path, view.image_name + ".jpg"))

def render_sets(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, feature_level : int, src_dim: int = -1, use_siglip_sam2_format: bool = False, lang_checkpoint_path: str = "", visualize: bool = False):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=False)  # Don't load features yet

        # Load language feature checkpoint which contains the pruned gaussians state
        if len(lang_checkpoint_path) > 0:
            # Use user-specified checkpoint path
            lang_checkpoint = lang_checkpoint_path
        elif src_dim == -1 or src_dim == 512 or use_siglip_sam2_format:
            # Load default language feature checkpoint (for siglip2_sam2 format or default dimensions)
            lang_checkpoint = os.path.join(dataset.model_path, f'chkpnt{iteration}_langfeat_{feature_level}.pth')
        else:
            # Load language feature checkpoint with specified dimension
            lang_checkpoint = os.path.join(dataset.model_path, f'chkpnt{iteration}_langfeat_{feature_level}_dim{src_dim}.pth')
        if src_dim == -1:
            src_dim = 512  # Default to 512 if not specified

        if not os.path.exists(lang_checkpoint):
            raise FileNotFoundError(f"Language feature checkpoint not found: {lang_checkpoint}")

        print(f"Loading language features from: {lang_checkpoint}")
        (model_params, first_iter) = torch.load(lang_checkpoint)

        # model_params should be a 13-element tuple in capture_language_feature() format
        print(f"Loaded checkpoint with iteration: {first_iter}")
        print(f"Model params type: {type(model_params)}")

        # Extract all Gaussian state from the language checkpoint (which includes pruned state)
        (active_sh_degree, xyz, features_dc, features_rest,
         scaling, rotation, opacity, language_features,
         max_radii2D, xyz_gradient_accum, denom,
         opt_dict, spatial_lr_scale) = model_params

        # Restore the Gaussian state from the language checkpoint to ensure synchronization
        # Ensure all tensors are on CUDA device
        gaussians.active_sh_degree = active_sh_degree
        gaussians._xyz = xyz.cuda()
        gaussians._features_dc = features_dc.cuda()
        gaussians._features_rest = features_rest.cuda()
        gaussians._scaling = scaling.cuda()
        gaussians._rotation = rotation.cuda()
        gaussians._opacity = opacity.cuda()
        gaussians.max_radii2D = max_radii2D.cuda()
        gaussians.xyz_gradient_accum = xyz_gradient_accum.cuda()
        gaussians.denom = denom.cuda()
        gaussians.spatial_lr_scale = spatial_lr_scale

        # Manually assign language features for the specific level
        if not hasattr(gaussians, "_language_features_dict"):
            gaussians._language_features_dict = {}

        gaussians._language_features_dict[feature_level] = language_features.cuda()

        # Check if language features and gaussians count match
        num_gaussians = gaussians._xyz.shape[0]
        num_lang_features = language_features.shape[0]
        print(f"Number of Gaussians: {num_gaussians}")
        print(f"Language features shape: {language_features.shape}")
        print(f"Gaussians and language features synchronized: {num_gaussians == num_lang_features}")

        print(f"Loaded synchronized Gaussian state and language features for level {feature_level} from {lang_checkpoint}")
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(args.model_path, "train", scene.loaded_iter, dataset.source_path, scene.getTrainCameras(), gaussians, pipeline, background, feature_level, src_dim, use_siglip_sam2_format, visualize)

        if not skip_test:
             render_set(args.model_path, "test", scene.loaded_iter, dataset.source_path, scene.getTestCameras(), gaussians, pipeline, background, feature_level, src_dim, use_siglip_sam2_format, visualize)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--src_dim", type=int, default=-1, help="Source dimension of language features to load (e.g., 16, 32, 64, 128). Use -1 for default.")
    parser.add_argument("--use_siglip_sam2_format", action="store_true", help="Use siglip2_sam2 format for language features (language_features_siglip2_sam2 directory and default checkpoint naming)")
    parser.add_argument("--lang_checkpoint", type=str, default="", help="Path to language feature checkpoint (overrides automatic path generation)")
    parser.add_argument("--visualize", action="store_true")
    args = get_combined_args_from_yaml(parser, param_groups=[model, pipeline, opt])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.feature_level, args.src_dim, args.use_siglip_sam2_format, args.lang_checkpoint, args.visualize)