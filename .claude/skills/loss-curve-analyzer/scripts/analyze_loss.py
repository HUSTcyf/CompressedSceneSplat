#!/usr/bin/env python3
"""
Loss curve analysis script for SceneSplat.

This script analyzes training loss curves and provides insights about
training progress, potential issues, and optimization suggestions.

Usage:
    python analyze_loss.py --loss-dir exp/lite-16-gridsvd/loss_curves/
    python analyze_loss.py --loss-file exp/lite-16-gridsvd/loss_curves/_accumulated_history.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


def load_loss_data(loss_dir: str) -> Dict[str, Any]:
    """Load loss data from directory or file."""
    loss_path = Path(loss_dir)

    if loss_path.is_file():
        with open(loss_path) as f:
            return json.load(f)

    if loss_path.is_dir():
        # Try to load accumulated history
        accum_file = loss_path / "_accumulated_history.json"
        if accum_file.exists():
            with open(accum_file) as f:
                return json.load(f)

        # Otherwise, load all scene files
        data = {"iterations": [], "losses": {}}
        for file in loss_path.glob("*_loss_data.json"):
            with open(file) as f:
                scene_data = json.load(f)
                scene_name = file.stem.replace("_loss_data", "")
                data["losses"][scene_name] = scene_data.get("losses", scene_data)
                if not data["iterations"]:
                    data["iterations"] = scene_data.get("iterations", [])
        return data

    raise FileNotFoundError(f"No loss data found at {loss_dir}")


def analyze_loss_trend(loss_values: List[float]) -> Dict[str, Any]:
    """Analyze a single loss series for trends and issues."""
    if not loss_values or len(loss_values) < 2:
        return {"status": "insufficient_data"}

    values = np.array(loss_values)

    # Basic statistics
    initial = values[0]
    final = values[-1]
    minimum = values.min()
    maximum = values.max()
    mean = values.mean()
    std = values.std()
    variance = values.var()

    # Calculate decrease percentage
    decrease_pct = ((initial - final) / (initial + 1e-8)) * 100

    # Convergence check (last 20% vs first 20%)
    n = len(values)
    first_quarter = values[:n//4].mean()
    last_quarter = values[3*n//4:].mean()
    convergence_ratio = (first_quarter - last_quarter) / (first_quarter + 1e-8)

    # Oscillation detection (check sign changes in gradient)
    gradients = np.diff(values)
    sign_changes = ((gradients[:-1] * gradients[1:]) < 0).sum()
    oscillation_score = sign_changes / len(gradients) if len(gradients) > 0 else 0

    # Plateau detection (recent values are very similar)
    recent_values = values[-min(100, n):]
    plateau_score = recent_values.std() / (mean + 1e-8)

    return {
        "initial": float(initial),
        "final": float(final),
        "min": float(minimum),
        "max": float(maximum),
        "mean": float(mean),
        "std": float(std),
        "variance": float(variance),
        "decrease_pct": float(decrease_pct),
        "convergence_ratio": float(convergence_ratio),
        "oscillation_score": float(oscillation_score),
        "plateau_score": float(plateau_score),
    }


def detect_issues(trend: Dict[str, Any]) -> List[str]:
    """Detect training issues from loss trend analysis."""
    issues = []

    # Check for mode collapse (very low variance, plateau)
    if trend.get("plateau_score", 0) < 0.01:
        issues.append("SEVERE: Mode collapse detected - loss has plateaued with minimal variation")

    # Check for insufficient learning (low decrease)
    if trend.get("decrease_pct", 0) < 10:
        issues.append("WARNING: Loss decreased less than 10% - possible underfitting")

    # Check for high oscillation
    if trend.get("oscillation_score", 0) > 0.3:
        issues.append("WARNING: High oscillation detected - consider reducing learning rate")

    # Check for divergence
    if trend.get("final", 0) > trend.get("initial", 0) * 1.5:
        issues.append("SEVERE: Loss is increasing - training may be diverging")

    # Check for NaN/Inf
    if not np.isfinite(trend.get("final", 0)):
        issues.append("SEVERE: Loss contains NaN or Inf values")

    return issues


def generate_suggestions(issues: List[str], trend: Dict[str, Any]) -> List[str]:
    """Generate optimization suggestions based on detected issues."""
    suggestions = []

    for issue in issues:
        if "Mode collapse" in issue:
            suggestions.extend([
                "Increase learning rate or use learning rate scheduling",
                "Add regularization (weight decay, dropout)",
                "Check if language features are properly initialized",
                "Consider reducing model capacity or feature dimension",
            ])
        elif "underfitting" in issue:
            suggestions.extend([
                "Train for more iterations",
                "Increase model capacity or feature dimension",
                "Check loss component weights - some may be too small",
                "Verify data preprocessing is correct",
            ])
        elif "oscillation" in issue:
            suggestions.extend([
                "Reduce learning rate (try 0.5x - 0.1x current)",
                "Enable gradient clipping",
                "Increase batch size for more stable gradients",
                "Use Adam optimizer with adjusted beta values",
            ])
        elif "diverging" in issue:
            suggestions.extend([
                "Significantly reduce learning rate",
                "Check for gradient explosion - enable gradient clipping",
                "Verify loss function implementation",
                "Check data normalization",
            ])
        elif "NaN" in issue:
            suggestions.extend([
                "Check for numerical instability in loss computation",
                "Add gradient clipping",
                "Use mixed precision training with proper scaling",
                "Verify language feature values are within reasonable range",
            ])

    # General suggestions based on trend
    if trend.get("decrease_pct", 0) > 50 and trend.get("plateau_score", 0) < 0.05:
        suggestions.append("Training converged well - consider early stopping in future runs")

    return list(set(suggestions))  # Remove duplicates


def print_report(data: Dict, scene: str = "all"):
    """Print analysis report."""
    print(f"\n{'='*70}")
    print(f"Training Loss Analysis Report")
    print(f"{'='*70}")

    losses = data.get("losses", {})
    iterations = data.get("iterations", [])

    if not iterations:
        print("No iteration data found.")
        return

    print(f"\nTotal iterations: {len(iterations)}")
    print(f"Iteration range: {iterations[0]} - {iterations[-1]}")

    # Analyze total_loss across all scenes
    all_total_losses = []
    for scene_data in losses.values():
        if isinstance(scene_data, dict) and "total_loss" in scene_data:
            all_total_losses.extend(scene_data["total_loss"])

    if all_total_losses:
        trend = analyze_loss_trend(all_total_losses)
        issues = detect_issues(trend)
        suggestions = generate_suggestions(issues, trend)

        print(f"\n{'='*70}")
        print(f"Total Loss Analysis (Averaged Across Scenes)")
        print(f"{'='*70}")
        print(f"  Initial: {trend['initial']:.6f}")
        print(f"  Final:   {trend['final']:.6f}")
        print(f"  Min:     {trend['min']:.6f}")
        print(f"  Max:     {trend['max']:.6f}")
        print(f"  Decrease: {trend['decrease_pct']:.2f}%")
        print(f"  Std Dev: {trend['std']:.6f}")

        if issues:
            print(f"\n⚠️  Detected Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✓ No major issues detected")

        if suggestions:
            print(f"\n💡 Optimization Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")

    # Per-scene analysis
    if len(losses) > 1:
        print(f"\n{'='*70}")
        print(f"Per-Scene Analysis")
        print(f"{'='*70}")
        print(f"{'Scene':<20} {'Final Loss':<12} {'Decrease %':<12} {'Status'}")
        print(f"{'-'*60}")

        for scene_name, scene_data in sorted(losses.items()):
            if isinstance(scene_data, dict) and "total_loss" in scene_data:
                trend = analyze_loss_trend(scene_data["total_loss"])
                issues = detect_issues(trend)
                status = "✓ OK" if not issues else "⚠️ Issues"
                print(f"{scene_name:<20} {trend['final']:<12.6f} {trend['decrease_pct']:<12.2f}% {status}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SceneSplat training loss curves")
    parser.add_argument("--loss-dir", type=str, help="Path to loss_curves directory")
    parser.add_argument("--loss-file", type=str, help="Path to specific loss JSON file")
    parser.add_argument("--scene", type=str, help="Specific scene to analyze")

    args = parser.parse_args()

    # Determine data source
    if args.loss_file:
        data = load_loss_data(args.loss_file)
    elif args.loss_dir:
        data = load_loss_data(args.loss_dir)
    else:
        # Try default location
        default_path = Path("/new_data/cyf/projects/SceneSplat/exp/lite-16-gridsvd/loss_curves")
        if default_path.exists():
            data = load_loss_data(str(default_path))
        else:
            print("Error: No loss data location specified and default not found.")
            print("Use --loss-dir or --loss-file to specify data location.")
            sys.exit(1)

    print_report(data, args.scene)


if __name__ == "__main__":
    main()
