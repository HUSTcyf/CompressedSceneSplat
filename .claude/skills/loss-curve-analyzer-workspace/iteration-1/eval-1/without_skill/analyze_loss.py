#!/usr/bin/env python3
"""
Analyze loss curves for training issues:
- Mode collapse
- Oscillation
- Divergence
- Overfitting
- Underfitting
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class LossAnalyzer:
    """Analyze training loss curves for common issues."""

    def __init__(self, loss_dir: str):
        self.loss_dir = Path(loss_dir)
        self.data = {}
        self.issues = {}

    def load_all_data(self):
        """Load all loss data files."""
        # Load accumulated history
        acc_file = self.loss_dir / "_accumulated_history.json"
        if acc_file.exists():
            with open(acc_file, 'r') as f:
                self.data = json.load(f)
        else:
            # Load individual files
            for file in self.loss_dir.glob("*_loss_data.json"):
                scene_name = file.stem.replace("_loss_data", "")
                with open(file, 'r') as f:
                    self.data[scene_name] = json.load(f)

    def detect_mode_collapse(self, losses: np.ndarray, window: int = 50) -> Tuple[bool, str]:
        """Detect mode collapse (loss stuck at constant value)."""
        if len(losses) < window:
            return False, "Too few iterations to detect mode collapse"

        # Check if loss variance is extremely low in recent window
        recent_losses = losses[-window:]
        variance = np.var(recent_losses)
        std = np.std(recent_losses)
        mean = np.mean(recent_losses)

        # Coefficient of variation
        cv = std / (mean + 1e-8)

        if cv < 0.001:
            return True, f"CRITICAL: Mode collapse detected! CV={cv:.6f} < 0.001, loss stuck at {mean:.6f}"
        elif cv < 0.01:
            return True, f"WARNING: Potential mode collapse! CV={cv:.6f} < 0.01, loss at {mean:.6f}"
        else:
            return False, f"No mode collapse (CV={cv:.4f})"

    def detect_oscillation(self, losses: np.ndarray, window: int = 30) -> Tuple[bool, str]:
        """Detect oscillation (loss bouncing up and down)."""
        if len(losses) < window * 2:
            return False, "Too few iterations to detect oscillation"

        # Calculate sign changes in derivatives
        recent_losses = losses[-window:]
        gradients = np.diff(recent_losses)

        # Count sign changes
        sign_changes = 0
        for i in range(len(gradients) - 1):
            if gradients[i] * gradients[i+1] < 0:
                sign_changes += 1

        # Calculate oscillation ratio
        osc_ratio = sign_changes / (len(gradients) - 1)

        # Check for high oscillation
        if osc_ratio > 0.7:
            return True, f"CRITICAL: Severe oscillation! {sign_changes}/{len(gradients)-1} sign changes (ratio={osc_ratio:.2f})"
        elif osc_ratio > 0.5:
            return True, f"WARNING: Moderate oscillation! {sign_changes}/{len(gradients)-1} sign changes (ratio={osc_ratio:.2f})"
        else:
            return False, f"No significant oscillation (ratio={osc_ratio:.2f})"

    def detect_divergence(self, losses: np.ndarray) -> Tuple[bool, str]:
        """Detect divergence (loss increasing or exploding)."""
        if len(losses) < 10:
            return False, "Too few iterations to detect divergence"

        # Check for NaN or Inf
        if np.any(np.isnan(losses)) or np.any(np.isinf(losses)):
            return True, "CRITICAL: Loss contains NaN or Inf values!"

        # Check if loss is increasing overall
        first_quarter = np.mean(losses[:len(losses)//4])
        last_quarter = np.mean(losses[-len(losses)//4:])

        if last_quarter > first_quarter * 1.5:
            return True, f"CRITICAL: Divergence! Loss increased from {first_quarter:.4f} to {last_quarter:.4f}"
        elif last_quarter > first_quarter * 1.2:
            return True, f"WARNING: Potential divergence! Loss increased from {first_quarter:.4f} to {last_quarter:.4f}"
        else:
            return False, f"No divergence (loss: {first_quarter:.4f} -> {last_quarter:.4f})"

    def detect_overfitting(self, train_losses: np.ndarray, val_losses: np.ndarray) -> Tuple[bool, str]:
        """Detect overfitting (val loss increasing while train loss decreasing)."""
        if len(train_losses) < 20 or len(val_losses) < 20:
            return False, "Too few iterations to detect overfitting"

        # Compare trends in second half of training
        mid_point = len(train_losses) // 2
        train_second_half = train_losses[mid_point:]
        val_second_half = val_losses[mid_point:]

        # Calculate trends using linear regression
        x_train = np.arange(len(train_second_half))
        x_val = np.arange(len(val_second_half))

        train_slope = np.polyfit(x_train, train_second_half, 1)[0]
        val_slope = np.polyfit(x_val, val_second_half, 1)[0]

        # Check if train decreasing but val increasing
        if train_slope < -0.001 and val_slope > 0.001:
            gap = val_second_half[-1] - train_second_half[-1]
            return True, f"CRITICAL: Overfitting! Train slope={train_slope:.4f}, Val slope={val_slope:.4f}, Gap={gap:.4f}"
        elif train_slope < 0 and val_slope > 0:
            gap = val_second_half[-1] - train_second_half[-1]
            return True, f"WARNING: Potential overfitting! Train slope={train_slope:.4f}, Val slope={val_slope:.4f}, Gap={gap:.4f}"
        else:
            return False, f"No overfitting (train slope={train_slope:.4f}, val slope={val_slope:.4f})"

    def detect_underfitting(self, losses: np.ndarray) -> Tuple[bool, str]:
        """Detect underfitting (loss plateauing at high value)."""
        if len(losses) < 50:
            return False, "Too few iterations to detect underfitting"

        # Check final loss value
        final_loss = np.mean(losses[-10:])
        initial_loss = np.mean(losses[:10])

        # Calculate improvement
        improvement = (initial_loss - final_loss) / initial_loss

        # Check if loss is still high with minimal improvement
        if final_loss > 0.5 and improvement < 0.2:
            return True, f"CRITICAL: Underfitting! Loss plateaued at {final_loss:.4f} with only {improvement*100:.1f}% improvement"
        elif final_loss > 0.3 and improvement < 0.3:
            return True, f"WARNING: Potential underfitting! Loss at {final_loss:.4f} with {improvement*100:.1f}% improvement"
        else:
            return False, f"No underfitting (final loss={final_loss:.4f}, improvement={improvement*100:.1f}%)"

    def analyze_scene(self, scene_name: str, scene_data: Dict) -> Dict:
        """Analyze a single scene's loss curves."""
        results = {
            'scene': scene_name,
            'issues': [],
            'metrics': {}
        }

        # Extract loss arrays
        if 'total_loss' in scene_data:
            total_loss = np.array(scene_data['total_loss'])
            results['metrics']['total_iterations'] = len(total_loss)
            results['metrics']['final_loss'] = float(total_loss[-1])
            results['metrics']['initial_loss'] = float(total_loss[0])
            results['metrics']['min_loss'] = float(np.min(total_loss))
            results['metrics']['max_loss'] = float(np.max(total_loss))

            # Run all checks
            is_collapse, msg = self.detect_mode_collapse(total_loss)
            if is_collapse:
                results['issues'].append(('mode_collapse', msg))

            is_oscillating, msg = self.detect_oscillation(total_loss)
            if is_oscillating:
                results['issues'].append(('oscillation', msg))

            is_diverging, msg = self.detect_divergence(total_loss)
            if is_diverging:
                results['issues'].append(('divergence', msg))

            is_underfitting, msg = self.detect_underfitting(total_loss)
            if is_underfitting:
                results['issues'].append(('underfitting', msg))

        # Check for overfitting if we have train/val split
        if 'train_loss' in scene_data and 'val_loss' in scene_data:
            train_loss = np.array(scene_data['train_loss'])
            val_loss = np.array(scene_data['val_loss'])

            is_overfitting, msg = self.detect_overfitting(train_loss, val_loss)
            if is_overfitting:
                results['issues'].append(('overfitting', msg))

        return results

    def analyze_all(self) -> Dict:
        """Analyze all scenes."""
        self.load_all_data()
        all_results = {}

        # Handle accumulated format
        if 'losses' in self.data:
            for scene_name, scene_data in self.data['losses'].items():
                all_results[scene_name] = self.analyze_scene(scene_name, scene_data)
        else:
            # Handle individual files format
            for scene_name, scene_data in self.data.items():
                all_results[scene_name] = self.analyze_scene(scene_name, scene_data)

        return all_results

    def generate_report(self, results: Dict) -> str:
        """Generate a detailed report."""
        report = []
        report.append("# Loss Curve Analysis Report\n")
        report.append("="*80 + "\n")

        # Summary statistics
        total_scenes = len(results)
        scenes_with_issues = sum(1 for r in results.values() if r['issues'])

        report.append(f"## Summary\n")
        report.append(f"- Total scenes analyzed: {total_scenes}\n")
        report.append(f"- Scenes with issues: {scenes_with_issues}\n")
        report.append(f"- Scenes healthy: {total_scenes - scenes_with_issues}\n\n")

        # Per-scene analysis
        for scene_name, scene_results in results.items():
            report.append(f"\n## Scene: {scene_name}\n")
            report.append("-"*60 + "\n")

            if 'metrics' in scene_results:
                metrics = scene_results['metrics']
                report.append(f"**Metrics:**\n")
                report.append(f"- Iterations: {metrics.get('total_iterations', 'N/A')}\n")
                report.append(f"- Initial loss: {metrics.get('initial_loss', 'N/A'):.4f}\n")
                report.append(f"- Final loss: {metrics.get('final_loss', 'N/A'):.4f}\n")
                report.append(f"- Min loss: {metrics.get('min_loss', 'N/A'):.4f}\n")
                report.append(f"- Max loss: {metrics.get('max_loss', 'N/A'):.4f}\n")

                # Calculate improvement
                if 'initial_loss' in metrics and 'final_loss' in metrics:
                    improvement = (metrics['initial_loss'] - metrics['final_loss']) / metrics['initial_loss'] * 100
                    report.append(f"- Improvement: {improvement:.2f}%\n")

            # Issues found
            if scene_results['issues']:
                report.append(f"\n**Issues Detected ({len(scene_results['issues'])}):**\n")
                for issue_type, message in scene_results['issues']:
                    report.append(f"- **{issue_type.upper()}**: {message}\n")
            else:
                report.append(f"\n**Status:** No issues detected - Training appears healthy\n")

            report.append("\n")

        # Overall recommendations
        report.append("\n## Overall Recommendations\n")
        report.append("="*80 + "\n")

        all_issues = []
        for scene_results in results.values():
            all_issues.extend([issue[0] for issue in scene_results['issues']])

        if not all_issues:
            report.append("All scenes are training healthily! No immediate action needed.\n")
        else:
            # Count issue types
            from collections import Counter
            issue_counts = Counter(all_issues)

            report.append("### Issue Distribution:\n")
            for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
                report.append(f"- {issue.upper()}: {count} scenes\n")

            report.append("\n### Recommended Actions:\n")

            if 'mode_collapse' in issue_counts:
                report.append("\n**Mode Collapse Detected:**\n")
                report.append("- Reduce learning rate\n")
                report.append("- Check for gradient vanishing\n")
                report.append("- Consider different initialization\n")
                report.append("- Add noise or regularization\n")

            if 'oscillation' in issue_counts:
                report.append("\n**Oscillation Detected:**\n")
                report.append("- Reduce learning rate\n")
                report.append("- Use learning rate scheduler\n")
                report.append("- Add gradient clipping\n")
                report.append("- Consider momentum optimization\n")

            if 'divergence' in issue_counts:
                report.append("\n**Divergence Detected:**\n")
                report.append("- Significantly reduce learning rate\n")
                report.append("- Check for numerical instability\n")
                report.append("- Verify gradient computation\n")
                report.append("- Add gradient clipping\n")

            if 'overfitting' in issue_counts:
                report.append("\n**Overfitting Detected:**\n")
                report.append("- Add regularization (dropout, weight decay)\n")
                report.append("- Increase training data\n")
                report.append("- Use early stopping\n")
                report.append("- Reduce model complexity\n")

            if 'underfitting' in issue_counts:
                report.append("\n**Underfitting Detected:**\n")
                report.append("- Increase model capacity\n")
                report.append("- Train for more iterations\n")
                report.append("- Reduce regularization\n")
                report.append("- Check feature quality\n")

        return "\n".join(report)


def main():
    """Main analysis function."""
    loss_dir = "/new_data/cyf/projects/SceneSplat/exp/lite-16-gridsvd/loss_curves"

    print("Analyzing loss curves...")
    print(f"Input directory: {loss_dir}\n")

    analyzer = LossAnalyzer(loss_dir)
    results = analyzer.analyze_all()
    report = analyzer.generate_report(results)

    # Save report
    output_dir = Path("/new_data/cyf/projects/SceneSplat/.claude/skills/loss-curve-analyzer-workspace/iteration-1/eval-1/without_skill/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "issue_report.md"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"Report saved to: {report_file}\n")
    print("="*80)
    print(report)

    return results


if __name__ == "__main__":
    main()
