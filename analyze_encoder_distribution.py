#!/usr/bin/env python
"""
分析实际Encoder的输出分布，找出输出爆炸的根源
"""
import sys
sys.path.insert(0, '/new_data/cyf/projects/SceneSplat')

import torch
import torch.nn as nn
import importlib.util
import numpy as np
from collections import defaultdict

print("=== 分析实际Encoder输出分布 ===\n")

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
print("Model loaded\n")

# 准备输入数据
print("准备输入数据...")
ckpt_data = torch.load('/new_data/cyf/projects/SceneSplat/output_features/bed/checkpoint_with_features_s.pth',
                       map_location='cpu', weights_only=False)

# 提取coord
coord = ckpt_data[0][1]  # [1000000, 3]
color = ckpt_data[0][2]  # [1000000, 3]
opacity = ckpt_data[0][6]  # [1000000, 1]
scale = ckpt_data[0][5]  # [1000000, 4]

# 使用随机采样
np.random.seed(42)
sample_indices = np.random.choice(len(coord), 5000, replace=False)
coord_sample = torch.from_numpy(coord[sample_indices].numpy()).float()
color_sample = torch.from_numpy(color[sample_indices].numpy()).float()
opacity_sample = torch.from_numpy(opacity[sample_indices].numpy()).float()
scale_sample = torch.from_numpy(scale[sample_indices].numpy()).float()

print(f"样本数: {len(coord_sample)}")

# 构造输入特征：color(3) + opacity(1) + quat(4) + scale(3) = 11
# 需要构造quat，这里用随机值
quat_sample = torch.randn(len(coord_sample), 4)
quat_sample = quat_sample / quat_sample.norm(dim=1, keepdim=True)

# 合并特征
input_feat = torch.cat([
    color_sample,
    opacity_sample,
    quat_sample,
    scale_sample[:, :3]  # 只取前3维
], dim=1)  # [N, 11]

print(f"输入特征shape: {input_feat.shape}")

# 创建Point结构
from pointcept.models.utils.structure import Point
batch = torch.zeros(len(coord_sample), dtype=torch.long)
offset = torch.arange(1, len(coord_sample) + 1, dtype=torch.long)

input_dict = {
    'coord': coord_sample,
    'feat': input_feat,
}

# 注册hooks来捕获Encoder输出
encoder_outputs = {}

def make_encoder_hook(stage_idx):
    def hook(module, input, output):
        # output是Point对象
        if isinstance(output, dict):
            encoder_outputs[f"enc{stage_idx}"] = output['feat'].detach()
        elif hasattr(output, 'feat'):
            encoder_outputs[f"enc{stage_idx}"] = output.feat.detach()
        elif isinstance(output, torch.Tensor):
            encoder_outputs[f"enc{stage_idx}"] = output.detach()
    return hook

# 注册Encoder各层的hook
encoder_hooks = []

# Encoder有5个stage (enc0, enc1, enc2, enc3, enc4)
# 我们需要捕获最后一个encoder stage (enc4)的输出
# 这是decoder的输入

# 先找到encoder的各个stage
print("\n注册Encoder hooks...")
if hasattr(model.backbone, 'enc'):
    enc = model.backbone.enc
    # 遍历encoder的各个stage
    for i in range(len(enc)):
        stage = enc[str(i)] if isinstance(enc, dict) else enc[i]
        # 找到最后的block或pooling层
        if hasattr(stage, '__len__'):
            # 在stage的最后一个模块注册hook
            last_module = stage[-1] if len(stage) > 0 else None
            if last_module is not None:
                h = last_module.register_forward_hook(make_encoder_hook(i))
                encoder_hooks.append(h)
                print(f"  注册hook到enc{i}")

print(f"\n运行模型前向传播...")

with torch.no_grad():
    try:
        output = model(input_dict)
        print(f"模型输出shape: {output['feat'].shape if isinstance(output, dict) else 'N/A'}")
    except Exception as e:
        print(f"前向传播出错: {e}")
        print("尝试简化方法...")

# 移除hooks
for h in encoder_hooks:
    h.remove()

# 如果成功捕获了encoder输出，进行分析
if encoder_outputs:
    print(f"\n=== 捕获到的Encoder输出 ===")
    for key, value in encoder_outputs.items():
        print(f"{key}: shape={value.shape}, mean={value.mean():.6f}, std={value.std():.6f}")
else:
    print("\n=== 无法直接捕获Encoder输出，使用替代方法 ===")
    print("分析BatchNorm的running_mean和running_std来推断...")

    # 检查encoder各层的BatchNorm统计
    state_dict = ckpt['state_dict']

    # 查找encoder相关的BatchNorm
    bn_stats = {}
    for key, value in state_dict.items():
        if 'enc' in key and ('running_mean' in key or 'running_var' in key):
            bn_stats[key] = value

    print(f"\n找到{len(bn_stats)}个BatchNorm统计")

    # 分析最后一个encoder stage的BatchNorm
    # 这反映了encoder输出的分布

    # 查找enc4相关的BatchNorm
    enc4_bn_keys = [k for k in bn_stats.keys() if 'enc4' in k or 'enc' in k]
    enc4_bn_keys.sort()

    print(f"\nEncoder最后的BatchNorm统计:")
    for key in enc4_bn_keys[-10:]:  # 最后10个
        val = bn_stats[key]
        if 'running_mean' in key:
            print(f"  {key}: mean={val.mean():.6f}")
        elif 'running_var' in key:
            std = torch.sqrt(val)
            print(f"  {key}: std={std.mean():.6f}")

print("\n=== 总结 ===")
print("如果实际Encoder输出的std > 1.5，会导致后续decoder输出爆炸")
