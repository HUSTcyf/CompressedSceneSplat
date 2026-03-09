#!/usr/bin/env python3
"""Analyze loss curves across different scenes.

Usage:
    python analyze_losses.py --rank 16
    python analyze_losses.py --rank 8
    python analyze_losses.py  # default: rank 16
"""

import json
import numpy as np
import argparse
from pathlib import Path

def analyze_loss_curves(rank=16):
    """Analyze loss curves for a specific SVD rank."""

    # Read accumulated history
    loss_file = f'exp/lite-{rank}-gridsvd/loss_curves/_accumulated_history.json'

    if not Path(loss_file).exists():
        print(f"Error: Loss file not found: {loss_file}")
        print(f"Available ranks can be found in exp/lite-*-gridsvd/loss_curves/")
        return

    with open(loss_file, 'r') as f:
        accumulated = json.load(f)

    # Get scene names
    scene_names = list(accumulated['losses'].keys())
    print(f'Scenes: {scene_names}')
    print()

    # Analyze each scene
    scene_stats = {}
    for scene in scene_names:
        scene_data = accumulated['losses'][scene]

        # Get iterations for this scene
        iterations = np.array(scene_data['iterations'])
        print(f'{scene}: {len(iterations)} iterations (range: {iterations[0]} to {iterations[-1]})')

        # Calculate statistics for total_loss
        total_loss = np.array(scene_data['total_loss'])
        initial_loss = total_loss[0]
        final_loss = total_loss[-1]
        min_loss = total_loss.min()
        max_loss = total_loss.max()
        mean_loss = total_loss.mean()
        std_loss = total_loss.std()
        decrease_pct = ((initial_loss - final_loss) / initial_loss) * 100

        # Calculate convergence rate (slope of last 20% of training)
        last_20_pct = max(1, len(total_loss) // 5)
        recent_losses = total_loss[-last_20_pct:]
        convergence_rate = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

        # Check for oscillation (variance in last 10%)
        last_10_pct = max(1, len(total_loss) // 10)
        oscillation = total_loss[-last_10_pct:].std()

        scene_stats[scene] = {
            'total_loss': {
                'initial': initial_loss,
                'final': final_loss,
                'min': min_loss,
                'max': max_loss,
                'mean': mean_loss,
                'std': std_loss,
                'decrease_pct': decrease_pct,
                'convergence_rate': convergence_rate,
                'oscillation': oscillation,
                'all_values': total_loss
            }
        }

        # Also collect other loss types
        for loss_type in ['l1_loss', 'cos_loss', 'contrast_loss', 'distort_loss', 'opacity_loss']:
            if loss_type in scene_data:
                loss_values = np.array(scene_data[loss_type])
                scene_stats[scene][loss_type] = {
                    'initial': loss_values[0],
                    'final': loss_values[-1],
                    'mean': loss_values.mean(),
                    'std': loss_values.std()
                }

    # Print comparison table
    print('=' * 100)
    print(f'LOSS CURVE COMPARISON ACROSS SCENES (SVD Rank {rank})')
    print('=' * 100)
    print()

    print('TOTAL LOSS STATISTICS:')
    print('-' * 100)
    print(f"{'Scene':<15} {'Initial':>10} {'Final':>10} {'Min':>10} {'Decrease':>10} {'Conv Rate':>12} {'Oscillation':>12} {'Status':>15}")
    print('-' * 100)

    for scene in scene_names:
        stats = scene_stats[scene]['total_loss']
        status = 'Converging' if stats['convergence_rate'] < -0.001 else ('Plateaued' if abs(stats['convergence_rate']) < 0.001 else 'Diverging')
        print(f"{scene:<15} {stats['initial']:>10.4f} {stats['final']:>10.4f} {stats['min']:>10.4f} {stats['decrease_pct']:>9.2f}% {stats['convergence_rate']:>12.6f} {stats['oscillation']:>12.6f} {status:>15}")

    print()
    print()

    # Rank scenes by performance
    print('RANKING BY FINAL LOSS (lower is better):')
    print('-' * 60)
    sorted_by_final = sorted(scene_names, key=lambda s: scene_stats[s]['total_loss']['final'])
    for i, scene in enumerate(sorted_by_final, 1):
        final_loss = scene_stats[scene]['total_loss']['final']
        decrease = scene_stats[scene]['total_loss']['decrease_pct']
        print(f'{i}. {scene:<15} - Final Loss: {final_loss:.4f} (Decrease: {decrease:.1f}%)')

    print()
    print()

    print('RANKING BY LOSS REDUCTION PERCENTAGE:')
    print('-' * 60)
    sorted_by_decrease = sorted(scene_names, key=lambda s: scene_stats[s]['total_loss']['decrease_pct'], reverse=True)
    for i, scene in enumerate(sorted_by_decrease, 1):
        decrease = scene_stats[scene]['total_loss']['decrease_pct']
        final_loss = scene_stats[scene]['total_loss']['final']
        print(f'{i}. {scene:<15} - Reduction: {decrease:.1f}% (Final: {final_loss:.4f})')

    print()
    print()

    # Identify best and worst performers
    best_scene = sorted_by_final[0]
    worst_scene = sorted_by_final[-1]

    print('KEY FINDINGS:')
    print('=' * 100)
    print(f'BEST PERFORMING SCENE: {best_scene}')
    print(f"  - Final Loss: {scene_stats[best_scene]['total_loss']['final']:.4f}")
    print(f"  - Loss Reduction: {scene_stats[best_scene]['total_loss']['decrease_pct']:.1f}%")
    print(f"  - Convergence Rate: {scene_stats[best_scene]['total_loss']['convergence_rate']:.6f}")
    print()
    print(f'WORST PERFORMING SCENE: {worst_scene}')
    print(f"  - Final Loss: {scene_stats[worst_scene]['total_loss']['final']:.4f}")
    print(f"  - Loss Reduction: {scene_stats[worst_scene]['total_loss']['decrease_pct']:.1f}%")
    print(f"  - Convergence Rate: {scene_stats[worst_scene]['total_loss']['convergence_rate']:.6f}")
    print()

    # Check for anomalies
    print('TRAINING ISSUES DETECTION:')
    print('=' * 100)
    for scene in scene_names:
        stats = scene_stats[scene]['total_loss']
        issues = []

        # Check for mode collapse (early plateau)
        if abs(stats['convergence_rate']) < 0.0001 and stats['final'] > 0.5:
            issues.append('Mode collapse detected (loss plateaued at high value)')

        # Check for overfitting (unnaturally smooth loss)
        if stats['oscillation'] < 0.001 and stats['std'] < 0.01:
            issues.append('Potential overfitting (loss unnaturally smooth)')

        # Check for underfitting (high final loss)
        if stats['final'] > 1.0:
            issues.append('Underfitting (final loss remains high)')

        # Check for instability (high oscillation)
        if stats['oscillation'] > 0.1:
            issues.append('Training instability (high oscillation)')

        if issues:
            print(f'{scene}: {", ".join(issues)}')
        else:
            print(f'{scene}: No issues detected - Training appears healthy')

    print()
    print('OTHER LOSS COMPONENTS:')
    print('-' * 100)
    for scene in scene_names:
        print(f'{scene}:')
        for loss_type in ['l1_loss', 'cos_loss', 'contrast_loss', 'distort_loss', 'opacity_loss']:
            if loss_type in scene_stats[scene]:
                lt = scene_stats[scene][loss_type]
                print(f"  {loss_type}: {lt['initial']:.4f} -> {lt['final']:.4f} (mean: {lt['mean']:.4f})")

    return scene_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze loss curves across different scenes')
    parser.add_argument('--rank', type=int, default=16, help='SVD rank (default: 16)')
    args = parser.parse_args()

    analyze_loss_curves(rank=args.rank)
