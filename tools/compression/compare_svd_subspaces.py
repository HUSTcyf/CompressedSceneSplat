#!/usr/bin/env python3
"""
SVD Subspace Similarity Analysis Tool

Analyzes the similarity between SVD subspaces (top-k singular vectors) across different scenes.
This helps understand if the language features from different scenes occupy similar subspaces.

Usage:
    python tools/compare_svd_subspaces.py --checkpoint_dir output_features --rank 16
    python tools/compare_svd_subspaces.py --checkpoints file1.pth file2.pth file3.pth --rank 16
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Tuple
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_lang_feat(checkpoint_path: str) -> torch.Tensor:
    """Extract language features from an OccamLGS format checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(ckpt, tuple) and len(ckpt) == 2:
        model_state, iteration = ckpt
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        model_state = ckpt["model_state"]
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    num_items = len(model_state)
    if num_items == 13:
        lang_feat = model_state[7]
    else:
        raise ValueError(f"Expected 13 items in model_state, got {num_items}")

    # Detach if requires grad
    if lang_feat.requires_grad:
        lang_feat = lang_feat.detach()

    return lang_feat


def compute_svd(features: torch.Tensor, rank: int = 16):
    """
    Compute SVD and extract top-k singular vectors.

    Args:
        features: Feature tensor [N, D]
        rank: Number of top singular vectors to extract

    Returns:
        Dictionary with U_r, S_r, Vt_r (top-r components)
    """
    feat_np = features.cpu().numpy().astype(np.float64)

    U, S, Vt = np.linalg.svd(feat_np, full_matrices=False)

    # Extract top-r components
    U_r = U[:, :rank]      # [N, rank]
    S_r = S[:rank]         # [rank]
    Vt_r = Vt[:rank, :]    # [rank, D]  <- This is the 16x768 matrix

    return {
        'U_r': U_r,
        'S_r': S_r,
        'Vt_r': Vt_r,
    }


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute cosine similarity between two matrices by comparing their flattened versions.

    Args:
        A: Matrix [m, n]
        B: Matrix [m, n]

    Returns:
        Cosine similarity in [-1, 1]
    """
    A_flat = A.flatten()
    B_flat = B.flatten()

    return np.dot(A_flat, B_flat) / (np.linalg.norm(A_flat) * np.linalg.norm(B_flat))


def pearson_correlation_matrix(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute Pearson correlation between two matrices.

    Args:
        A: Matrix [m, n]
        B: Matrix [m, n]

    Returns:
        Pearson correlation in [-1, 1]
    """
    A_flat = A.flatten()
    B_flat = B.flatten()

    # Center the data
    A_centered = A_flat - np.mean(A_flat)
    B_centered = B_flat - np.mean(B_flat)

    return np.dot(A_centered, B_centered) / (np.linalg.norm(A_centered) * np.linalg.norm(B_centered))


def subspace_similarity(Vt1: np.ndarray, Vt2: np.ndarray) -> float:
    """
    Compute subspace similarity using principal angles.

    The similarity is computed from the singular values of Vt1 @ Vt2.T,
    which represent the cosines of principal angles between subspaces.

    Args:
        Vt1: Right singular vectors [k, D] from scene 1
        Vt2: Right singular vectors [k, D] from scene 2

    Returns:
        Subspace similarity in [0, 1]
    """
    # Compute singular values of Vt1 @ Vt2.T (gives cosines of principal angles)
    S = np.linalg.svd(Vt1 @ Vt2.T, compute_uv=False)

    # Sum of squared cosines = measure of subspace alignment
    # When subspaces are identical, all S = 1, sum = k
    # When subspaces are orthogonal, all S = 0, sum = 0
    similarity = np.sum(S ** 2) / len(S)

    return similarity


def grassmann_distance(Vt1: np.ndarray, Vt2: np.ndarray) -> float:
    """
    Compute Grassmann distance between two subspaces.

    Args:
        Vt1: Right singular vectors [k, D] from scene 1
        Vt2: Right singular vectors [k, D] from scene 2

    Returns:
        Grassmann distance in [0, pi/2]
    """
    # Compute singular values of Vt1 @ Vt2.T
    S = np.linalg.svd(Vt1 @ Vt2.T, compute_uv=False)

    # Principal angles are arccos of singular values
    # Clip to avoid numerical issues
    S_clipped = np.clip(S, -1, 1)
    principal_angles = np.arccos(S_clipped)

    # Grassmann distance is sqrt of sum of squared principal angles
    return np.linalg.norm(principal_angles)


def analyze_principal_angles(Vt1: np.ndarray, Vt2: np.ndarray, scene1: str, scene2: str) -> Dict:
    """
    Analyze principal angles between two Vt matrices in detail.

    Args:
        Vt1: Right singular vectors [k, D] from scene 1
        Vt2: Right singular vectors [k, D] from scene 2
        scene1: Name of scene 1
        scene2: Name of scene 2

    Returns:
        Dictionary with detailed angle analysis
    """
    # Compute SVD of Vt1 @ Vt2.T to get principal angles
    S = np.linalg.svd(Vt1 @ Vt2.T, compute_uv=False)

    # Principal angles (in radians and degrees)
    S_clipped = np.clip(S, -1, 1)
    principal_angles_rad = np.arccos(S_clipped)
    principal_angles_deg = np.degrees(principal_angles_rad)

    # Cosine of angles (the singular values themselves)
    cos_angles = S_clipped

    return {
        'scene1': scene1,
        'scene2': scene2,
        'principal_angles_rad': principal_angles_rad,
        'principal_angles_deg': principal_angles_deg,
        'cos_angles': cos_angles,
        'mean_angle_deg': np.mean(principal_angles_deg),
        'max_angle_deg': np.max(principal_angles_deg),
        'min_angle_deg': np.min(principal_angles_deg),
        'std_angle_deg': np.std(principal_angles_deg),
    }


def analyze_vector_pairwise_angles(Vt1: np.ndarray, Vt2: np.ndarray) -> Dict:
    """
    Analyze pairwise angles between individual vectors in Vt matrices.

    For each pair (i, j) where i is index in Vt1 and j is index in Vt2,
    compute the angle between Vt1[i] and Vt2[j].

    Args:
        Vt1: Right singular vectors [k, D] from scene 1
        Vt2: Right singular vectors [k, D] from scene 2

    Returns:
        Dictionary with pairwise angle matrix [k, k]
    """
    k = Vt1.shape[0]

    # Compute pairwise cosine similarities
    # cos_matrix[i,j] = cosine between Vt1[i] and Vt2[j]
    cos_matrix = Vt1 @ Vt2.T  # [k, k]

    # Clip and convert to angles
    cos_clipped = np.clip(cos_matrix, -1, 1)
    angle_matrix_rad = np.arccos(cos_clipped)
    angle_matrix_deg = np.degrees(angle_matrix_rad)

    return {
        'cos_matrix': cos_matrix,
        'angle_matrix_deg': angle_matrix_deg,
        'angle_matrix_rad': angle_matrix_rad,
    }


def analyze_component_stability(all_Vt: Dict[str, np.ndarray]) -> Dict:
    """
    Analyze the stability/consistency of each principal component across scenes.

    For each component index i (0 to k-1), compute how much the i-th vector
    varies across different scenes.

    Args:
        all_Vt: Dictionary mapping scene names to Vt matrices [k, D]

    Returns:
        Dictionary with component-wise stability analysis
    """
    scene_names = list(all_Vt.keys())
    k = list(all_Vt.values())[0].shape[0]

    # For each component, collect all vectors from different scenes
    component_stats = {}

    for i in range(k):
        # Stack the i-th vector from all scenes: [num_scenes, D]
        vectors_i = np.stack([all_Vt[scene][i] for scene in scene_names])

        # Compute pairwise angles within this component across scenes
        num_scenes = len(scene_names)
        angles = []

        for j in range(num_scenes):
            for l in range(j + 1, num_scenes):
                cos_sim = np.dot(vectors_i[j], vectors_i[l])
                cos_clipped = np.clip(cos_sim, -1, 1)
                angle_deg = np.degrees(np.arccos(cos_clipped))
                angles.append(angle_deg)

        component_stats[i] = {
            'mean_angle': np.mean(angles) if angles else 0,
            'std_angle': np.std(angles) if angles else 0,
            'min_angle': np.min(angles) if angles else 0,
            'max_angle': np.max(angles) if angles else 0,
            'median_angle': np.median(angles) if angles else 0,
        }

    return component_stats


def procrustes_analysis(Vt1: np.ndarray, Vt2: np.ndarray) -> Dict[str, float]:
    """
    Perform Procrustes analysis to measure similarity between two matrices.

    Args:
        Vt1: Matrix [k, D] from scene 1
        Vt2: Matrix [k, D] from scene 2

    Returns:
        Dictionary with Procrustes statistics
    """
    from scipy.linalg import orthogonal_procrustes

    # Find optimal rotation
    R, _ = orthogonal_procrustes(Vt1, Vt2)

    # Compute aligned matrix
    Vt1_aligned = Vt1 @ R

    # Compute reconstruction error
    error = np.linalg.norm(Vt1_aligned - Vt2, 'fro') / np.linalg.norm(Vt2, 'fro')

    # Compute similarity (1 - normalized error)
    similarity = 1 - error

    return {
        'error': error,
        'similarity': max(0, similarity),
    }


def compare_singular_values(S1: np.ndarray, S2: np.ndarray) -> Dict[str, float]:
    """
    Compare singular value distributions between two scenes.

    Args:
        S1: Singular values from scene 1
        S2: Singular values from scene 2

    Returns:
        Dictionary with comparison metrics
    """
    # Normalize by total energy
    energy1 = np.sum(S1 ** 2)
    energy2 = np.sum(S2 ** 2)

    S1_norm = S1 / np.sqrt(energy1)
    S2_norm = S2 / np.sqrt(energy2)

    # KL divergence
    eps = 1e-10
    p1 = S1_norm ** 2 + eps
    p1 = p1 / np.sum(p1)
    p2 = S2_norm ** 2 + eps
    p2 = p2 / np.sum(p2)

    kl_div = np.sum(p1 * np.log(p1 / p2))

    # Earth mover's distance (1D approximation)
    emd = np.sum(np.abs(np.cumsum(p1) - np.cumsum(p2)))

    # Correlation
    correlation = np.corrcoef(S1, S2)[0, 1]

    return {
        'kl_divergence': kl_div,
        'emd': emd,
        'correlation': correlation if not np.isnan(correlation) else 0.0,
    }


def analyze_scene_similarity(checkpoint_paths: List[str], rank: int = 16) -> Dict:
    """
    Analyze SVD subspace similarity across multiple scenes.

    Args:
        checkpoint_paths: List of checkpoint file paths
        rank: Number of top singular vectors to compare

    Returns:
        Dictionary with all analysis results
    """
    scene_names = []
    svd_results = {}

    print("=" * 80)
    print("Computing SVD for all scenes...")
    print("=" * 80)

    # Compute SVD for all scenes
    for ckpt_path in checkpoint_paths:
        scene_name = Path(ckpt_path).stem.replace('_iteration_18000_point_cloud_langfeat', '').replace('_langfeat', '')
        scene_names.append(scene_name)

        print(f"\n[{scene_name}]")
        lang_feat = extract_lang_feat(ckpt_path)
        print(f"  Features: {lang_feat.shape}")

        svd_result = compute_svd(lang_feat, rank)
        svd_results[scene_name] = svd_result

        # Compute total energy for ratio
        feat_np = lang_feat.cpu().numpy()
        all_S = np.linalg.svd(feat_np, compute_uv=False)
        total_energy = np.sum(all_S ** 2)
        top_energy = np.sum(svd_result['S_r'] ** 2)

        print(f"  Top-{rank} singular value ratio: {top_energy / total_energy:.4f}")

    # Pairwise comparisons
    print("\n" + "=" * 80)
    print("Computing pairwise similarities...")
    print("=" * 80)

    n_scenes = len(scene_names)
    cosine_sim_matrix = np.zeros((n_scenes, n_scenes))
    pearson_corr_matrix = np.zeros((n_scenes, n_scenes))
    subspace_sim_matrix = np.zeros((n_scenes, n_scenes))
    grassmann_dist_matrix = np.zeros((n_scenes, n_scenes))
    procrustes_sim_matrix = np.zeros((n_scenes, n_scenes))
    singular_corr_matrix = np.zeros((n_scenes, n_scenes))

    for i in range(n_scenes):
        for j in range(i, n_scenes):
            scene_i = scene_names[i]
            scene_j = scene_names[j]

            Vt_i = svd_results[scene_i]['Vt_r']
            Vt_j = svd_results[scene_j]['Vt_r']
            S_i = svd_results[scene_i]['S_r']
            S_j = svd_results[scene_j]['S_r']

            # Matrix-level similarities
            cosine_sim = cosine_similarity_matrix(Vt_i, Vt_j)
            pearson_corr = pearson_correlation_matrix(Vt_i, Vt_j)

            # Subspace similarities
            subspace_sim = subspace_similarity(Vt_i, Vt_j)
            grassmann_dist = grassmann_distance(Vt_i, Vt_j)

            # Procrustes analysis
            proc_result = procrustes_analysis(Vt_i, Vt_j)
            procrustes_sim = proc_result['similarity']

            # Singular value correlation
            sv_comparison = compare_singular_values(S_i, S_j)
            singular_corr = sv_comparison['correlation']

            # Store in matrices (symmetric)
            cosine_sim_matrix[i, j] = cosine_sim
            cosine_sim_matrix[j, i] = cosine_sim

            pearson_corr_matrix[i, j] = pearson_corr
            pearson_corr_matrix[j, i] = pearson_corr

            subspace_sim_matrix[i, j] = subspace_sim
            subspace_sim_matrix[j, i] = subspace_sim

            grassmann_dist_matrix[i, j] = grassmann_dist
            grassmann_dist_matrix[j, i] = grassmann_dist

            procrustes_sim_matrix[i, j] = procrustes_sim
            procrustes_sim_matrix[j, i] = procrustes_sim

            singular_corr_matrix[i, j] = singular_corr
            singular_corr_matrix[j, i] = singular_corr

    # Collect all Vt matrices for component stability analysis
    all_Vt = {name: svd_results[name]['Vt_r'] for name in scene_names}
    component_stats = analyze_component_stability(all_Vt)

    # Collect principal angles for each pair
    principal_angles_results = {}
    for i in range(n_scenes):
        for j in range(i, n_scenes):
            scene_i = scene_names[i]
            scene_j = scene_names[j]
            Vt_i = svd_results[scene_i]['Vt_r']
            Vt_j = svd_results[scene_j]['Vt_r']
            pair_key = f"{scene_i} <-> {scene_j}"
            principal_angles_results[pair_key] = analyze_principal_angles(Vt_i, Vt_j, scene_i, scene_j)

    return {
        'scene_names': scene_names,
        'cosine_sim': cosine_sim_matrix,
        'pearson_corr': pearson_corr_matrix,
        'subspace_sim': subspace_sim_matrix,
        'grassmann_dist': grassmann_dist_matrix,
        'procrustes_sim': procrustes_sim_matrix,
        'singular_corr': singular_corr_matrix,
        'svd_results': svd_results,
        'component_stats': component_stats,
        'principal_angles': principal_angles_results,
    }


def print_similarity_table(results: Dict, metric: str, title: str):
    """Print a formatted similarity table."""
    scene_names = results['scene_names']
    matrix = results[metric]

    print(f"\n{title}")
    print("=" * 80)
    print(f"{'Scene':<20} " + " ".join([f"{name:>10}" for name in scene_names]))
    print("-" * 80)

    for i, name in enumerate(scene_names):
        row_values = " ".join([f"{val:>10.4f}" for val in matrix[i]])
        print(f"{name:<20} {row_values}")

    # Print statistics
    print("-" * 80)
    # Get upper triangle (excluding diagonal)
    upper_indices = np.triu_indices(len(scene_names), k=1)
    upper_values = matrix[upper_indices]

    print(f"Mean: {np.mean(upper_values):.4f}")
    print(f"Std:  {np.std(upper_values):.4f}")
    print(f"Min:  {np.min(upper_values):.4f}")
    print(f"Max:  {np.max(upper_values):.4f}")


def save_similarity_heatmap(results: Dict, metric: str, title: str, output_path: str):
    """Save a heatmap visualization of the similarity matrix."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        scene_names = results['scene_names']
        matrix = results[metric]

        # Shorten names for display
        short_names = [name[:15] for name in scene_names]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=short_names, yticklabels=short_names,
                   cbar_kws={'label': 'Similarity'}, ax=ax)
        ax.set_title(title, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to: {output_path}")
        plt.close()

    except ImportError:
        print("matplotlib or seaborn not available, skipping heatmap generation")


def print_angle_analysis(results: Dict):
    """Print detailed angle analysis between Vt vectors."""
    component_stats = results.get('component_stats', {})
    principal_angles = results.get('principal_angles', {})

    print("\n" + "=" * 80)
    print("Vt Vector Angle Analysis")
    print("=" * 80)

    # Component stability analysis
    print("\n--- Component Stability (How much each PC varies across scenes) ---")
    print(f"{'PC':<6} {'Mean Angle':<15} {'Std Angle':<15} {'Min Angle':<15} {'Max Angle':<15}")
    print("-" * 80)

    for i in range(len(component_stats)):
        stats = component_stats[i]
        print(f"PC{i:<3} {stats['mean_angle']:>10.2f}°     {stats['std_angle']:>10.2f}°     "
              f"{stats['min_angle']:>10.2f}°     {stats['max_angle']:>10.2f}°")

    # Overall statistics across components
    all_means = [component_stats[i]['mean_angle'] for i in range(len(component_stats))]
    print("-" * 80)
    print(f"{'Overall':<6} {np.mean(all_means):>10.2f}°     {np.std(all_means):>10.2f}°")

    # Principal angles for each pair
    print("\n--- Principal Angles Between Scene Pairs ---")
    for pair_key, angle_data in sorted(principal_angles.items()):
        scene1 = angle_data['scene1'][:20]
        scene2 = angle_data['scene2'][:20]

        print(f"\n{scene1} <-> {scene2}")
        print(f"  Mean angle: {angle_data['mean_angle_deg']:.2f}°")
        print(f"  Std angle:  {angle_data['std_angle_deg']:.2f}°")
        print(f"  Range: [{angle_data['min_angle_deg']:.2f}°, {angle_data['max_angle_deg']:.2f}°]")

        # Show first few principal angles
        print(f"  First 8 principal angles:")
        for i in range(min(8, len(angle_data['principal_angles_deg']))):
            angle = angle_data['principal_angles_deg'][i]
            cos_val = angle_data['cos_angles'][i]
            print(f"    θ{i+1} = {angle:>6.2f}° (cos = {cos_val:>7.4f})")

    # Interpretation
    print("\n" + "=" * 80)
    print("Interpretation Guide:")
    print("=" * 80)
    print("Angle ranges and their meaning:")
    print("  0° - 30°:   Nearly identical directions")
    print("  30° - 60°:  Related but different")
    print("  60° - 90°:  Nearly orthogonal")
    print("  90°:        Exactly orthogonal (uncorrelated)")
    print("\nComponent stability:")
    print("  Lower mean angle = Component is stable across scenes")
    print("  Higher mean angle = Component varies significantly")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="SVD Subspace Similarity Analysis")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs='+',
        default=None,
        help="List of checkpoint files to compare",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="Number of top singular vectors to compare (default: 16)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./svd_comparison",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs='+',
        default=None,
        help="Specific scene names to analyze (e.g., teatime ramen lawn). "
             "Required unless --checkpoints is specified.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs='+',
        default=None,
        help="Iteration numbers to analyze (e.g., 18000 30000). "
             "If specified, checkpoint names are constructed as {scene}_iteration_{iter}_point_cloud_langfeat.pth. "
             "If not specified, uses {scene}_iteration_18000_point_cloud_langfeat.pth (default).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="output_features",
        help="Directory containing checkpoint files (default: output_features)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate heatmap visualizations",
    )
    args = parser.parse_args()

    # Collect checkpoint paths - construct directly from scene names and iterations
    checkpoint_paths = []

    if args.checkpoints:
        # Direct checkpoint paths specified
        checkpoint_paths = args.checkpoints
    elif args.scenes is not None:
        # Build checkpoint paths from scenes + iterations
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
            return

        # Determine iterations to use
        if args.iterations is not None:
            iterations = args.iterations
        else:
            iterations = [18000]  # Default iteration

        # Construct checkpoint paths directly
        for scene in args.scenes:
            for iter_num in iterations:
                # Construct checkpoint name: {scene}_iteration_{iter_num}_point_cloud_langfeat.pth
                checkpoint_name = f"{scene}_iteration_{iter_num}_point_cloud_langfeat.pth"
                checkpoint_path = checkpoint_dir / checkpoint_name

                if checkpoint_path.is_file():
                    checkpoint_paths.append(str(checkpoint_path))
                    print(f"Found: {checkpoint_name}")
                else:
                    print(f"Warning: Not found: {checkpoint_path}")
    else:
        print("Error: Must specify either --scenes or --checkpoints")
        print("Usage examples:")
        print("  python tools/compare_svd_subspaces.py --scenes teatime ramen --iterations 18000")
        print("  python tools/compare_svd_subspaces.py --scenes teatime --iterations 18000 30000")
        return

    if not checkpoint_paths:
        print("Error: No checkpoint files found")
        return

    print(f"Found {len(checkpoint_paths)} checkpoint files")

    # Run analysis
    results = analyze_scene_similarity(checkpoint_paths, args.rank)

    # Print results
    print_similarity_table(results, 'cosine_sim', 'Cosine Similarity (Vt Matrices)')
    print_similarity_table(results, 'pearson_corr', 'Pearson Correlation (Vt Matrices)')
    print_similarity_table(results, 'subspace_sim', 'Subspace Similarity (Principal Angles)')
    print_similarity_table(results, 'grassmann_dist', 'Grassmann Distance (Lower = More Similar)')
    print_similarity_table(results, 'procrustes_sim', 'Procrustes Similarity (Aligned)')
    print_similarity_table(results, 'singular_corr', 'Singular Value Correlation')

    # Print angle analysis
    print_angle_analysis(results)

    # Save plots if requested
    if args.plot:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_similarity_heatmap(results, 'cosine_sim', 'Cosine Similarity (Vt Matrices)',
                              output_dir / f'cosine_sim_rank{args.rank}.png')
        save_similarity_heatmap(results, 'subspace_sim', 'Subspace Similarity',
                              output_dir / f'subspace_sim_rank{args.rank}.png')
        save_similarity_heatmap(results, 'procrustes_sim', 'Procrustes Similarity',
                              output_dir / f'procrustes_sim_rank{args.rank}.png')


if __name__ == "__main__":
    main()
