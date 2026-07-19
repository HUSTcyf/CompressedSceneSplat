#!/usr/bin/env python
"""
Analyze actual encoder output to find the root cause of collapse.
"""
import sys
sys.path.insert(0, '/new_data/cyf/projects/SceneSplat')

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

# 加载实际模型
print("Loading model...")
from pointcept.models import build_model
import importlib.util

config_path = '/new_data/cyf/projects/SceneSplat/configs/inference/lang-pretrain-litept-ovs-gridsvd.py'
spec = importlib.util.spec_from_file_location('config', config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
model = build_model(config_module.model)
model.cpu()
model.eval()

# 加载权重
ckpt = torch.load('/new_data/cyf/projects/SceneSplat/exp/lite-16-gridsvd/model/model_last.pth',
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)
print("Model loaded")

# 准备测试数据
print("\nLoading test data...")
# 加载checkpoint来获取coord等信息
ckpt_data = torch.load('/new_data/cyf/projects/SceneSplat/output_features/bed/checkpoint_with_features_s.pth',
                       map_location='cpu', weights_only=False)

# 提取特征
coord = ckpt_data[0][1]  # [1000000, 3]
color = ckpt_data[0][2]  # [1000000, 3]
opacity = ckpt_data[0][6]  # [1000000, 1]
scale = ckpt_data[0][5]  # [1000000, 4]
segment = ckpt_data[0][8]  # [1000000]

# 找到有效点（segment > 0）
valid_mask = segment > 0
valid_indices = torch.where(valid_mask)[0][:5000]  # 只取5000个样本

print(f"Using {len(valid_indices)} valid samples")

# 构造输入数据
feat_dict = {
    'coord': coord[valid_indices],
    'color': color[valid_indices],
    'opacity': opacity[valid_indices],
    'scale': scale[valid_indices],
    'segment': segment[valid_indices],
}

# 创建offset（用于批次处理）
batch = torch.zeros(len(valid_indices), dtype=torch.long)
offset = torch.cumsum(batch.bincount(), dim=0).long()

print("\n=== Running Model Forward Pass ===")

# Hook来捕获各层输出
activations = {}

def get_hook(name):
    def hook(module, input, output):
        if isinstance(output, dict):
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    activations[f"{name}.{k}"] = v.detach().clone()
        elif isinstance(output, torch.Tensor):
            activations[name] = output.detach().clone()
    return hook

# 注册hooks
hooks = []

# Decoder输出hooks
hooks.append(model.backbone.dec.dec3.up.proj[1].register_forward_hook(get_hook('dec3_output')))
hooks.append(model.backbone.dec.dec2.up.proj[1].register_forward_hook(get_hook('dec2_output')))
hooks.append(model.backbone.dec.dec1.up.proj[1].register_forward_hook(get_hook('dec1_output')))
hooks.append(model.backbone.dec.dec0.up.proj[1].register_forward_hook(get_hook('dec0_output')))

# 运行前向传播
with torch.no_grad():
    output = model(feat=feat_dict, batch=batch, offset=offset)

# 移除hooks
for hook in hooks:
    hook.remove()

print(f"\nFinal output shape: {output['feat'].shape}")
print(f"Final output mean: {output['feat'].mean():.6f}, std: {output['feat'].std():.6f}")

# 分析各层输出
print("\n=== Layer-wise Analysis ===")

for layer_name in ['dec3_output', 'dec2_output', 'dec1_output', 'dec0_output']:
    if layer_name in activations:
        act = activations[layer_name]
        print(f"\n[{layer_name}]")
        print(f"  Shape: {act.shape}")
        print(f"  Mean: {act.mean():.6f}, Std: {act.std():.6f}")
        print(f"  Min: {act.min():.6f}, Max: {act.max():.6f}")

        # 检查饱和比例
        saturated = (act.abs() > 3).float().mean() * 100
        print(f"  Saturated (>3): {saturated:.2f}%")

        # 每个维度的统计
        print(f"  Per-dim std: min={act.std(dim=0).min():.4f}, max={act.std(dim=0).max():.4f}")

# 分析最终输出（经过tanh之后）
print("\n=== Final Output Analysis (after tanh and scaling) ===")
final_feat = output['feat']  # 已经经过tanh和scaling

print(f"Shape: {final_feat.shape}")
print(f"Mean: {final_feat.mean():.6f}, Std: {final_feat.std():.6f}")
print(f"Min: {final_feat.min():.6f}, Max: {final_feat.max():.6f}")

# 每个维度的详细分析
print(f"\n{'Dim':<6} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Unique':<10} {'Status':<15}")
print("-" * 80)

for i in range(16):
    dim_data = final_feat[:, i]
    mean_val = dim_data.mean().item()
    std_val = dim_data.std().item()
    min_val = dim_data.min().item()
    max_val = dim_data.max().item()
    unique_vals = torch.unique(dim_data).numel()

    if unique_vals <= 2:
        status = "COLLAPSED"
    elif std_val < 0.01:
        status = "NEAR_COLLAPSE"
    else:
        status = "OK"

    print(f"{i:<6} {mean_val:<10.6f} {std_val:<10.6f} {min_val:<10.6f} {max_val:<10.6f} {unique_vals:<10} {status:<15}")

# 对比SVD GT
print("\n=== Comparison with SVD GT ===")
svd_data = np.load('/new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS/val/bed/lang_feat_grid_svd_r16.npz')
svd_compressed = torch.from_numpy(svd_data['compressed']).float()
svd_indices = torch.from_numpy(svd_data['indices']).long()

# 获取对应的GT
gt_indices = svd_indices[:len(valid_indices)]
gt_feat = svd_compressed[gt_indices]

print(f"{'Dim':<6} {'GT_Mean':<10} {'Model_Mean':<12} {'GT_Std':<10} {'Model_Std':<12} {'Diff':<10}")
print("-" * 70)

for i in range(16):
    gt_mean = gt_feat[:, i].mean().item()
    model_mean = final_feat[:, i].mean().item()
    gt_std = gt_feat[:, i].std().item()
    model_std = final_feat[:, i].std().item()
    diff = abs(gt_mean - model_mean).item()

    print(f"{i:<6} {gt_mean:<10.4f} {model_mean:<12.4f} {gt_std:<10.4f} {model_std:<12.4f} {diff:<10.4f}")

# 识别坍缩维度
collapsed_dims = []
for i in range(16):
    dim_data = final_feat[:, i]
    unique_vals = torch.unique(dim_data).numel()
    if unique_vals <= 2:
        collapsed_dims.append(i)

print(f"\n=== KEY FINDING ===")
print(f"Collapsed dimensions in actual inference: {collapsed_dims}")
print(f"These dimensions have only {len(set(torch.unique(final_feat[collapsed_dims]).tolist()))} unique values total")
