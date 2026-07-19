#!/usr/bin/env python
"""
Simple script to trace where collapse happens by analyzing actual model output.
"""
import sys
sys.path.insert(0, '/new_data/cyf/projects/SceneSplat')

import torch
import numpy as np
import importlib.util

# 加载配置
config_path = '/new_data/cyf/projects/SceneSplat/configs/inference/lang-pretrain-litept-ovs-gridsvd.py'
spec = importlib.util.spec_from_file_location('config', config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

# 构建模型
from pointcept.models import build_model
model = build_model(config_module.model)
model.cpu()
model.eval()

# 加载权重
ckpt = torch.load('/new_data/cyf/projects/SceneSplat/exp/lite-16-gridsvd/model/model_last.pth',
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)
print("Model loaded successfully")

# 准备测试数据
print("\nPreparing test data...")
ckpt_data = torch.load('/new_data/cyf/projects/SceneSplat/output_features/bed/checkpoint_with_features_s.pth',
                       map_location='cpu', weights_only=False)

# 提取少量有效样本
coord = ckpt_data[0][1]  # [1000000, 3]
color = ckpt_data[0][2]  # [1000000, 3]
opacity = ckpt_data[0][6]  # [1000000, 1]
scale = ckpt_data[0][5]  # [1000000, 4]
segment = ckpt_data[0][8]  # [1000000]

# 找到有效点
valid_mask = segment > 0
valid_indices = torch.where(valid_mask)[0][:10000]  # 10000个样本

print(f"Using {len(valid_indices)} valid samples")

# 构造输入
feat_dict = {
    'coord': coord[valid_indices],
    'color': color[valid_indices],
    'opacity': opacity[valid_indices],
    'scale': scale[valid_indices],
    'segment': segment[valid_indices],
}

batch = torch.zeros(len(valid_indices), dtype=torch.long)
offset = torch.cumsum(batch.bincount(), dim=0).long()

# 注册hooks来捕获每层输出
activations = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, dict):
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    activations[f"{name}_{k}"] = v.detach()
        elif isinstance(output, torch.Tensor):
            activations[name] = output.detach()
    return hook

hooks = []
hooks.append(model.backbone.dec.dec3.up.proj[1].register_forward_hook(make_hook('dec3')))
hooks.append(model.backbone.dec.dec2.up.proj[1].register_forward_hook(make_hook('dec2')))
hooks.append(model.backbone.dec.dec1.up.proj[1].register_forward_hook(make_hook('dec1')))
hooks.append(model.backbone.dec.dec0.up.proj[1].register_forward_hook(make_hook('dec0')))

# 运行前向传播
print("\nRunning forward pass...")
input_dict = {
    **feat_dict,
    'batch': batch,
    'offset': offset,
}

with torch.no_grad():
    output = model(input_dict)

# 移除hooks
for hook in hooks:
    hook.remove()

# 分析每层输出
print("\n=== Layer-wise Output Analysis ===")

for layer in ['dec3', 'dec2', 'dec1', 'dec0']:
    if layer in activations:
        act = activations[layer]
        print(f"\n[{layer.upper()}] shape={act.shape}")
        print(f"  Mean: {act.mean():.6f}, Std: {act.std():.6f}")
        print(f"  Min: {act.min():.6f}, Max: {act.max():.6f}")

        # 统计饱和比例
        saturated_pct = ((act.abs() > 3).float().mean() * 100).item()
        print(f"  Saturated (>3): {saturated_pct:.2f}%")

        # 检查是否有维度坍缩
        per_dim_std = act.std(dim=0)
        collapsed_dims = (per_dim_std < 0.01).nonzero().squeeze()
        if collapsed_dims.numel() > 0:
            print(f"  ⚠️ COLLAPSED dims (std<0.01): {collapsed_dims.tolist()}")

# 分析最终输出
print("\n=== Final Output Analysis ===")
final_feat = output['feat']  # [N, 16]

print(f"Shape: {final_feat.shape}")
print(f"Mean: {final_feat.mean():.6f}, Std: {final_feat.std():.6f}")
print(f"Min: {final_feat.min():.6f}, Max: {final_feat.max():.6f}")

print(f"\n{'Dim':<6} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Unique':<10} {'Status':<15}")
print("-" * 80)

collapsed_dims = []
for i in range(16):
    dim_data = final_feat[:, i]
    mean_val = dim_data.mean().item()
    std_val = dim_data.std().item()
    min_val = dim_data.min().item()
    max_val = dim_data.max().item()
    unique_vals = torch.unique(dim_data).numel()

    if unique_vals <= 2:
        status = "COLLAPSED"
        collapsed_dims.append(i)
    elif std_val < 0.01:
        status = "NEAR_COLLAPSE"
    else:
        status = "OK"

    print(f"{i:<6} {mean_val:<10.6f} {std_val:<10.6f} {min_val:<10.6f} {max_val:<10.6f} {unique_vals:<10} {status:<15}")

print(f"\n=== KEY FINDING ===")
print(f"Collapsed dimensions: {collapsed_dims}")
print(f"Total: {len(collapsed_dims)}/16 dimensions collapsed")

# 对比GT
print("\n=== Comparison with GT ===")
svd_data = np.load('/new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS/val/bed/lang_feat_grid_svd_r16.npz')
svd_compressed = torch.from_numpy(svd_data['compressed']).float()
svd_indices = torch.from_numpy(svd_data['indices']).long()

# 获取对应的GT（使用原始indices）
gt_indices = svd_indices[:len(valid_indices)]
gt_feat = svd_compressed[gt_indices]

print(f"GT shape: {gt_feat.shape}")

print(f"\n{'Dim':<6} {'GT_Mean':<10} {'Model_Mean':<12} {'GT_Std':<10} {'Model_Std':<12} {'Error':<10}")
print("-" * 70)

total_error = 0
for i in range(16):
    gt_mean = gt_feat[:, i].mean().item()
    model_mean = final_feat[:, i].mean().item()
    gt_std = gt_feat[:, i].std().item()
    model_std = final_feat[:, i].std().item()
    error = abs(gt_mean - model_mean).item()
    total_error += error

    marker = " ⚠️" if i in collapsed_dims else ""
    print(f"{i:<6} {gt_mean:<10.4f} {model_mean:<12.4f} {gt_std:<10.4f} {model_std:<12.4f} {error:<10.4f}{marker}")

print(f"\nAverage mean error: {total_error/16:.4f}")
