# ScanNet++ 场景推理与可视化指南

本指南展示如何使用预训练的 SceneSplat 权重对 ScanNet++ 场景进行推理和可视化。

## 场景信息

- **场景路径**: `/new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be`
- **场景名称**: `0d2ee665be`
- **类别数量**: 100 (top100 类别)

## 预训练权重

- **模型**: SceneSplat PT-v3m1 (768维语言特征)
- **权重路径**: `/new_data/cyf/projects/SceneSplat/checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth`
- **推理配置**: `configs/inference/lang-pretrain-pt-v3m1-3dgs.py`

## 快速开始

### 方法 1: 使用 Python 脚本 (推荐)

```bash
# 进入项目目录
cd /new_data/cyf/projects/SceneSplat

# 运行完整流程
CUDA_VISIBLE_DEVICES=0 python scripts/visualize_scannetpp_example.py

# 或者指定自定义参数
CUDA_VISIBLE_DEVICES=0 python scripts/visualize_scannetpp_example.py \
    --scene_path /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be \
    --checkpoint /path/to/your/checkpoint.pth \
    --output_dir ./my_output \
    --gpu_id 0
```

### 方法 2: 使用 Shell 脚本

```bash
cd /new_data/cyf/projects/SceneSplat

CUDA_VISIBLE_DEVICES=0 bash scripts/visualize_scannetpp_example.sh
```

### 方法 3: 分步执行

#### 步骤 1: 运行推理生成语言特征

```bash
CUDA_VISIBLE_DEVICES=0 python tools/batch_predict_inference.py \
    --config configs/inference/lang-pretrain-pt-v3m1-3dgs.py \
    --checkpoint checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth \
    --input-root /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val \
    --output-dir ./output_inference \
    --scene 0d2ee665be \
    --device cuda
```

输出: `./output_inference/0d2ee665be/language_features.npy`

#### 步骤 2: 生成语义分割预测

```python
import numpy as np
import torch

# 加载数据
scene_path = "/new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be"
coord = np.load(f"{scene_path}/coord.npy")
gt_labels = np.load(f"{scene_path}/segment.npy")

# 加载语言特征
features = np.load("./output_inference/0d2ee665be/language_features.npy")  # [N, 768]

# 加载类别名称和文本嵌入
class_names_file = "pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt"
with open(class_names_file, "r") as f:
    class_names = [line.strip() for line in f]

text_embeddings = torch.load("pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt")

# 计算相似度并预测
lang_feat = torch.from_numpy(features).float()
lang_feat = lang_feat / torch.norm(lang_feat, dim=-1, keepdim=True)
text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=-1, keepdim=True)

similarities = torch.matmul(lang_feat, text_embeddings.T)  # [N, 100]
predictions = torch.argmax(similarities, dim=1).numpy()

# 保存预测结果
np.save("./output_inference/0d2ee665be/predictions.npy", predictions)

# 计算准确率
accuracy = (gt_labels == predictions).mean() * 100
print(f"Accuracy: {accuracy:.2f}%")
```

#### 步骤 3: 可视化结果

```bash
# 可视化 GT
CUDA_VISIBLE_DEVICES=0 python tools/visualize_semantic_segmentation.py \
    --data_path /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be \
    --mode gt \
    --output_path ./output/gt.ply \
    --dataset scannetpp

# 可视化预测
CUDA_VISIBLE_DEVICES=0 python tools/visualize_semantic_segmentation.py \
    --data_path /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be \
    --pred_path ./output_inference/0d2ee665be/predictions.npy \
    --mode pred \
    --output_path ./output/pred.ply \
    --dataset scannetpp

# 对比可视化
CUDA_VISIBLE_DEVICES=0 python tools/visualize_semantic_segmentation.py \
    --data_path /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be \
    --pred_path ./output_inference/0d2ee665be/predictions.npy \
    --mode compare \
    --output_path ./output/compare \
    --dataset scannetpp
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `language_features.npy` | SceneSplat 推理生成的 768 维语言特征 |
| `predictions.npy` | 语义分割预测结果 (类别索引) |
| `*_gt.ply` | Ground Truth 可视化点云 |
| `*_pred.ply` | 预测结果可视化点云 |
| `*_compare_*.ply` | GT vs 预测对比点云 |

## 查看可视化结果

### 使用 MeshLab

```bash
# 安装
sudo apt install meshlab

# 查看
meshlab output/gt.ply
meshlab output/pred.ply
```

### 使用 CloudCompare

```bash
# 安装
sudo apt install cloudcompare

# 查看
cloudcompare output/gt.ply
```

### 使用 Python (Interactive 模式)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/visualize_semantic_segmentation.py \
    --data_path /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be \
    --mode gt \
    --interactive \
    --dataset scannetpp
```

## 参数说明

### visualize_semantic_segmentation.py

| 参数 | 说明 |
|------|------|
| `--data_path` | 场景数据路径 (包含 .npy 文件) |
| `--pred_path` | 预测结果路径 (.npy 文件) |
| `--mode` | 模式: gt (真值), pred (预测), compare (对比) |
| `--output_path` | 输出 PLY 文件路径 |
| `--interactive` | 显示交互式 Open3D 查看器 |
| `--dataset` | 数据集类型: scannet, scannetpp, matterport |

### batch_predict_inference.py

| 参数 | 说明 |
|------|------|
| `--config` | 推理配置文件 |
| `--checkpoint` | 预训练权重路径 |
| `--input-root` | 输入数据根目录 |
| `--output-dir` | 输出目录 |
| `--scene` | 单个场景名称 |
| `--device` | 设备: cuda 或 cpu |

## 常见问题

### 1. 内存不足

如果遇到 GPU 内存不足，可以：
- 使用更小的 `chunk_size` 在配置文件中
- 使用 CPU 模式 (`--device cpu`)
- 使用 LitePT 模型代替 PT-v3m1

### 2. 文件不存在

确保以下文件存在：
- 场景数据: `coord.npy`, `color.npy`, `segment.npy` 等
- 类别文件: `top100.txt`
- 文本嵌入: `top100_text_embeddings_siglip2.pt`

### 3. 交互式可视化失败

安装 Open3D:
```bash
pip install open3d
```

## 参考命令

跳过推理步骤，使用已有特征:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/visualize_scannetpp_example.py \
    --skip_inference \
    --existing_features ./output_inference/0d2ee665be/language_features.npy
```
