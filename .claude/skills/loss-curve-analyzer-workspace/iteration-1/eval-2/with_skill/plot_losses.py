#!/usr/bin/env python3
"""Plot loss curves for SceneSplat training analysis.

Usage:
    python plot_losses.py --rank 16
    python plot_losses.py --rank 16 --output_dir ./plots
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_loss_curves(rank=16, output_dir='./plots'):
    """Generate loss curve visualizations for a specific SVD rank."""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Read accumulated history
    loss_file = f'exp/lite-{rank}-gridsvd/loss_curves/_accumulated_history.json'

    if not Path(loss_file).exists():
        print(f"Error: Loss file not found: {loss_file}")
        return

    with open(loss_file, 'r') as f:
        data = json.load(f)

    # Get scene names
    scene_names = list(data['losses'].keys())

    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Loss Curves Comparison - SVD Rank {rank}', fontsize=16, fontweight='bold')

    # Color palette for different scenes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot 1: Total Loss
    ax = axes[0, 0]
    for i, scene in enumerate(scene_names):
        scene_data = data['losses'][scene]
        iterations = scene_data['iterations']
        total_loss = scene_data['total_loss']
        ax.plot(iterations, total_loss, label=scene, color=colors[i], linewidth=2, alpha=0.7)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Total Loss Across Scenes', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: L1 Loss
    ax = axes[0, 1]
    for i, scene in enumerate(scene_names):
        scene_data = data['losses'][scene]
        iterations = scene_data['iterations']
        if 'l1_loss' in scene_data:
            l1_loss = scene_data['l1_loss']
            ax.plot(iterations, l1_loss, label=scene, color=colors[i], linewidth=2, alpha=0.7)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('L1 Loss', fontsize=12)
    ax.set_title('L1 Loss Across Scenes', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Cosine Loss
    ax = axes[1, 0]
    for i, scene in enumerate(scene_names):
        scene_data = data['losses'][scene]
        iterations = scene_data['iterations']
        if 'cos_loss' in scene_data:
            cos_loss = scene_data['cos_loss']
            ax.plot(iterations, cos_loss, label=scene, color=colors[i], linewidth=2, alpha=0.7)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cosine Loss', fontsize=12)
    ax.set_title('Cosine Loss Across Scenes', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Contrastive Loss
    ax = axes[1, 1]
    for i, scene in enumerate(scene_names):
        scene_data = data['losses'][scene]
        iterations = scene_data['iterations']
        if 'contrast_loss' in scene_data:
            contrast_loss = scene_data['contrast_loss']
            ax.plot(iterations, contrast_loss, label=scene, color=colors[i], linewidth=2, alpha=0.7)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Contrastive Loss', fontsize=12)
    ax.set_title('Contrastive Loss Across Scenes', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the combined plot
    output_file = output_path / f'loss_curves_rank{rank}_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved combined plot to: {output_file}')
    plt.close()

    # Create individual per-scene plots
    for scene in scene_names:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Loss Curves - {scene} (SVD Rank {rank})', fontsize=16, fontweight='bold')

        scene_data = data['losses'][scene]
        iterations = scene_data['iterations']

        # Total Loss
        ax = axes[0, 0]
        ax.plot(iterations, scene_data['total_loss'], color='blue', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.grid(True, alpha=0.3)

        # Add initial and final values
        initial = scene_data['total_loss'][0]
        final = scene_data['total_loss'][-1]
        ax.text(0.5, 0.5, f'Initial: {initial:.4f}\nFinal: {final:.4f}',
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # L1 Loss
        ax = axes[0, 1]
        if 'l1_loss' in scene_data:
            ax.plot(iterations, scene_data['l1_loss'], color='green', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('L1 Loss')
            ax.set_title('L1 Loss')
            ax.grid(True, alpha=0.3)

        # Cosine Loss
        ax = axes[1, 0]
        if 'cos_loss' in scene_data:
            ax.plot(iterations, scene_data['cos_loss'], color='orange', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Cosine Loss')
            ax.set_title('Cosine Loss')
            ax.grid(True, alpha=0.3)

        # Contrastive Loss
        ax = axes[1, 1]
        if 'contrast_loss' in scene_data:
            ax.plot(iterations, scene_data['contrast_loss'], color='red', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Contrastive Loss')
            ax.set_title('Contrastive Loss')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save individual scene plot
        output_file = output_path / f'loss_curves_rank{rank}_{scene}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'Saved scene plot to: {output_file}')
        plt.close()

    # Create a comparison bar chart for final losses
    fig, ax = plt.subplots(figsize=(10, 6))

    final_losses = []
    reduction_pcts = []
    for scene in scene_names:
        scene_data = data['losses'][scene]
        total_loss = scene_data['total_loss']
        final = total_loss[-1]
        initial = total_loss[0]
        reduction = ((initial - final) / initial) * 100

        final_losses.append(final)
        reduction_pcts.append(reduction)

    x = np.arange(len(scene_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, final_losses, width, label='Final Loss', color='steelblue')
    bars2 = ax.bar(x + width/2, reduction_pcts, width, label='Reduction %', color='coral')

    ax.set_xlabel('Scene', fontsize=12)
    ax.set_ylabel('Loss / Percentage', fontsize=12)
    ax.set_title(f'Final Loss and Reduction Percentage - SVD Rank {rank}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scene_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_file = output_path / f'loss_comparison_rank{rank}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved comparison plot to: {output_file}')
    plt.close()

    print(f'\nAll plots saved to: {output_path.absolute()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot loss curves for SceneSplat training')
    parser.add_argument('--rank', type=int, default=16, help='SVD rank (default: 16)')
    parser.add_argument('--output_dir', type=str, default='./plots', help='Output directory for plots')
    args = parser.parse_args()

    plot_loss_curves(rank=args.rank, output_dir=args.output_dir)
