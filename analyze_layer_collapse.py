#!/usr/bin/env python
"""
Analyze which layer first causes the dimension collapse issue.
"""
import sys
sys.path.insert(0, '/new_data/cyf/projects/SceneSplat')

import torch
import torch.nn as nn
from collections import defaultdict

# 加载数据
print("Loading data...")
gt_data = torch.load('/new_data/cyf/projects/SceneSplat/output_features/bed/checkpoint_with_features.pth',
                     map_location='cpu', weights_only=False)
gt_feat = gt_data[0][7]  # [1000000, 16]

# 找到非0行
gt_nonzero_mask = (gt_feat.abs().sum(dim=1) > 0)
valid_idx = torch.where(gt_nonzero_mask)[0][:10000]  # 只取10000个样本加速分析

print(f"Using {len(valid_idx)} valid samples")

# 由于我们缺少完整的输入特征，改用直接分析权重的方法
print("\n=== Weight-based Analysis ===")

# 加载模型权重
ckpt = torch.load('/new_data/cyf/projects/SceneSplat/exp/lite-16-gridsvd/model/model_last.pth',
                  map_location='cpu', weights_only=False)
state_dict = ckpt['state_dict']

# 分析每一层的输出权重
dec_layers = [
    ('Dec3.up.proj', 'backbone.dec.dec3.up.proj.0.weight', 504, 126),
    ('Dec2.up.proj', 'backbone.dec.dec2.up.proj.0.weight', 126, 64),
    ('Dec1.up.proj', 'backbone.dec.dec1.up.proj.0.weight', 64, 32),
    ('Dec0.up.proj', 'backbone.dec.dec0.up.proj.0.weight', 32, 16),
]

print("\n=== Decoder Layer Weight Analysis ===")
print(f"{'Layer':<20} {'Shape':<15} {'Mean':<10} {'Std':<10} {'Max':<10} {'Min':<10}")
print("-" * 80)

for name, key, in_ch, out_ch in dec_layers:
    if key in state_dict:
        w = state_dict[key]
        print(f"{name:<20} {str(w.shape):<15} {w.mean():<10.6f} {w.std():<10.6f} {w.max():<10.6f} {w.min():<10.6f}")

# 分析Dec0每个输出维度的权重
print("\n=== Dec0.up.proj.0 Per-Output-Dimension Analysis ===")
w_dec0 = state_dict['backbone.dec.dec0.up.proj.0.weight']  # [16, 32]

print(f"{'Dim':<6} {'Norm':<10} {'Mean':<10} {'Std':<10} {'MaxAbs':<10}")
print("-" * 50)

for i in range(16):
    w_i = w_dec0[i]  # [32]
    norm = torch.norm(w_i).item()
    mean = w_i.mean().item()
    std = w_i.std().item()
    max_abs = w_i.abs().max().item()
    print(f"{i:<6} {norm:<10.4f} {mean:<10.4f} {std:<10.4f} {max_abs:<10.4f}")

# 分析bias
print("\n=== Dec0.up.proj.0 Bias Analysis ===")
b_dec0 = state_dict['backbone.dec.dec0.up.proj.0.bias']  # [16]

print(f"{'Dim':<6} {'Bias':<10}")
print("-" * 20)
for i in range(16):
    print(f"{i:<6} {b_dec0[i].item():<10.4f}")

# 分析Dec0输入（来自Dec1输出）的权重
print("\n=== Dec1.up.proj.0 Output Analysis (Dec0 Input) ===")
w_dec1 = state_dict['backbone.dec.dec1.up.proj.0.weight']  # [32, 64]

# 对于每个Dec0输入维度，分析其对应的Dec1输出维度权重
print(f"Dec0输入维度 <- Dec1输出维度")
print(f"{'Dec0_in':<10} {'Dec1_out':<10} {'Norm':<10} {'Mean':<10} {'Std':<10}")
print("-" * 50)

# Dec0有32个输入，Dec1有32个输出（对应）
for i in range(min(16, 32)):  # 只看前16个
    w_i = w_dec1[i]  # [64]
    norm = torch.norm(w_i).item()
    mean = w_i.mean().item()
    std = w_i.std().item()
    print(f"{i:<10} {i:<10} {norm:<10.4f} {mean:<10.4f} {std:<10.4f}")

# 计算每层的输出预期（假设输入是标准正态分布）
print("\n=== Simulated Output Analysis (assuming N(0,1) input) ===")

# 模拟: output = W @ input + b
# 如果input ~ N(0, I)，那么output ~ N(b, W @ W.T)

for name, key, in_ch, out_ch in dec_layers:
    if key in state_dict:
        w = state_dict[key]
        b_key = key.replace('weight', 'bias')
        b = state_dict[b_key]

        # 每个输出维度的方差 = 对应权重行的平方和
        per_dim_var = (w ** 2).sum(dim=1)  # [out_ch]
        per_dim_std = torch.sqrt(per_dim_var)

        print(f"\n{name}:")
        print(f"  Output dim range: 0-{out_ch-1}")
        print(f"  Per-dimension std (if input is N(0,I)): min={per_dim_std.min():.4f}, max={per_dim_std.max():.4f}, mean={per_dim_std.mean():.4f}")
        print(f"  Bias: min={b.min():.4f}, max={b.max():.4f}, mean={b.mean():.4f}")

        # 检查哪些维度可能坍缩
        collapse_threshold = 0.01
        collapsed_dims = (per_dim_std < collapse_threshold).nonzero().squeeze()
        if len(collapsed_dims) > 0:
            if collapsed_dims.numel() == 0:
                print(f"  No collapsed dimensions (std < {collapse_threshold})")
            else:
                print(f"  ⚠️ POTENTIALLY COLLAPSED dims (std < {collapse_threshold}): {collapsed_dims.tolist()}")
        else:
            print(f"  No collapsed dimensions (std < {collapse_threshold})")
