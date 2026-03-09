---
name: loss-curve-analyzer
description: Use this skill whenever the user wants to analyze training loss curves in the SceneSplat project. This includes reading loss data from exp/*/loss_curves/ directories, detecting training issues (overfitting, underfitting, loss oscillation, mode collapse), providing optimization suggestions, and generating analysis reports. Trigger this when the user mentions analyzing losses, checking training progress, diagnosing model issues, or working with loss_curves/*.json files in SceneSplat.
project: scenesplat
---

# Loss Curve Analyzer for SceneSplat

## Overview

This skill analyzes training loss curves for the SceneSplat project to detect training issues and provide optimization suggestions. It works with loss data stored in `exp/*/loss_curves/` directories.

## When to Use This Skill

Use this skill when:
- User mentions "loss curves", "loss analysis", or "training analysis"
- User wants to diagnose training problems (overfitting, underfitting, etc.)
- User is working with files in `exp/*/loss_curves/` directories
- User mentions specific loss types (total_loss, distort_loss, opacity_loss, etc.)
- User asks about training progress or model convergence

## Loss Data Format

Loss data is stored as JSON files with the following structure:

```json
{
  "iterations": [2, 3, 4, 5, ...],
  "losses": {
    "scene_name": {
      "total_loss": [1.06, 1.01, 0.95, ...],
      "distort_loss": [0.5, 0.45, ...],
      "opacity_loss": [0.3, 0.28, ...],
      ...
    }
  }
}
```

**File locations:**
- Accumulated history: `exp/*/loss_curves/_accumulated_history.json`
- Per-scene data: `exp/*/loss_curves/{scene}_loss_data.json`

## Analysis Workflow

### Step 1: Locate Loss Data

First, identify the loss curves directory. Default locations:
- `exp/lite-16-gridsvd/loss_curves/`
- `exp/*/loss_curves/`

Use Glob to find loss curve files:
```
Glob: exp/*/loss_curves/*.json
Glob: exp/**/loss_curves/_accumulated_history.json
```

### Step 2: Read and Parse Data

Read the JSON files using the Read tool. The data contains:
- `iterations`: List of iteration numbers
- `losses`: Dict mapping scene names to loss values

### Step 3: Analyze Loss Patterns

For each loss type, analyze:

**1. Convergence Analysis**
- Check if loss is decreasing over time
- Calculate convergence rate
- Identify plateau (loss stops decreasing)

**2. Stability Analysis**
- Detect oscillation (loss fluctuates up and down)
- Measure variance in recent iterations
- Identify sudden spikes or drops

**3. Comparative Analysis**
- Compare different loss components
- Check if individual losses are balanced
- Identify which loss dominates training

**4. Cross-Scene Analysis**
- Compare loss patterns across different scenes
- Identify scenes with abnormal behavior
- Detect scene-specific training issues

### Step 4: Detect Training Issues

**Mode Collapse Detection**
- If total_loss plateaus early (e.g., within first 1000 iterations)
- If certain loss components drop to near-zero while others remain high
- If loss variance becomes extremely low

**Overfitting Detection**
- If training loss continues decreasing but validation loss increases
- If loss becomes unnaturally smooth (may indicate memorization)

**Underfitting Detection**
- If loss remains high and doesn't decrease significantly
- If all loss components plateau at high values

**Training Instability**
- If loss oscillates with large amplitude
- If loss has sudden spikes (>50% increase)
- If loss diverges (goes to infinity or NaN)

### Step 5: Generate Analysis Report

Create a comprehensive report with:

```
# Training Loss Analysis Report

## Summary
- Experiment: [experiment_name]
- Total iterations: [N]
- Scenes analyzed: [N]

## Loss Trends
### Total Loss
- Initial: [value]
- Final: [value]
- Decrease: [percentage]%
- Status: [converging/plateaued/diverging]

### Per-Component Analysis
[Table with each loss type's trend]

## Detected Issues
[List any issues found with severity]

## Optimization Suggestions
[Specific recommendations based on analysis]
```

## Common Issues and Solutions

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| Mode Collapse | Loss plateaus early, low variance | Increase learning rate, add regularization, check initialization |
| Overfitting | Train loss ↓, Val loss ↑ | Add dropout, reduce model capacity, increase data augmentation |
| Underfitting | Loss remains high | Increase model capacity, train longer, check loss weights |
| Loss Oscillation | Loss fluctuates | Reduce learning rate, use gradient clipping, check batch size |
| Imbalanced Losses | One loss dominates | Adjust loss weights, normalize loss components |

## Output Options

Based on user needs, provide:

1. **Text Report**: Markdown summary with key findings
2. **Visualization**: Python script to plot loss curves
3. **Detailed Metrics**: Statistical analysis of loss patterns
4. **Recommendations**: Specific optimization suggestions

## Example Analysis Commands

When analyzing loss curves, use these patterns:

```bash
# Find loss curve files
find /new_data/cyf/projects/SceneSplat/exp -name "_accumulated_history.json" -o -name "*_loss_data.json"

# Quick analysis (use Python)
python -c "
import json
with open('exp/lite-16-gridsvd/loss_curves/_accumulated_history.json') as f:
    data = json.load(f)
# ... analysis code
"
```

## Key Loss Types in SceneSplat

- **total_loss**: Combined loss (primary metric)
- **distort_loss**: 3D distortion loss (geometric accuracy)
- **opacity_loss**: Opacity regularization loss
- **lang_loss**: Language feature alignment loss (if using pretraining)
- **contrastive_loss**: Contrastive learning loss (if applicable)

## Project-Specific Context

**SceneSplat Training Characteristics:**
- Uses 3D Gaussian Splatting representation
- Language features are 16-dimensional by default
- Training typically converges in 30K-60K iterations
- Expected final total_loss: 0.3-0.6 for healthy training

**Important Files:**
- Config: `configs/custom/lang-pretrain-litept-*.py`
- Training script: `tools/train_lite.py`
- Model checkpoint: `exp/*/model/model_*.pth`

## Notes

- Always check both accumulated history and per-scene data
- Compare with similar experiments for context
- Consider the dataset (ScanNet vs Matterport3D vs custom)
- Account for different model architectures (LitePT vs full)
