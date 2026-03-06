# ScanNet++ 可视化 - 快速开始

## 修复说明

修复了 `tools/batch_predict_inference.py` 中的坐标过滤问题：
- 使用更保守的百分位数 (0.1% - 99.9%)
- 检查坐标有效性 (无 NaN/Inf)
- 只有保留 ≥90% 点时才应用过滤
- 防止过度过滤导致所有点被删除

## 使用方法

### 方法 1: 仅可视化 GT (最简单)

不需要推理和预测，直接可视化场景的 Ground Truth：

```bash
cd /new_data/cyf/projects/SceneSplat

CUDA_VISIBLE_DEVICES=0 /new_data/cyf/.conda/envs/scene_splat/bin/python scripts/visualize_scannetpp_example.py --skip_all
```

输出: `output_visualization/visualization/0d2ee665be_gt.ply`

### 方法 2: 可视化 GT (直接使用可视化工具)

最简单的方式，直接使用可视化脚本：

```bash
CUDA_VISIBLE_DEVICES=0 /new_data/cyf/.conda/envs/scene_splat/bin/python tools/visualize_semantic_segmentation.py \
    --data_path /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be \
    --mode gt \
    --output_path ./output/gt.ply \
    --dataset scannetpp
```

### 方法 3: 完整流程 (推理 + 预测 + 可视化)

运行完整的 SceneSplat 推理流程：

```bash
CUDA_VISIBLE_DEVICES=0 /new_data/cyf/.conda/envs/scene_splat/bin/python scripts/visualize_scannetpp_example.py
```

注意: 这需要较长时间和大量 GPU 内存。

## 查看结果

```bash
# 安装查看工具
sudo apt install meshlab

# 查看可视化结果
meshlab output_visualization/visualization/0d2ee665be_gt.ply
```

## 其他场景

修改场景路径即可处理其他场景：

```bash
CUDA_VISIBLE_DEVICES=0 /new_data/cyf/.conda/envs/scene_splat/bin/python scripts/visualize_scannetpp_example.py \
    --scene_path /path/to/your/scene \
    --skip_all
```

## 问题排查

### 问题: 推理步骤报错
解决: 使用 `--skip_all` 跳过推理，仅做可视化

### 问题: 内存不足
解决: 使用更小的场景或增加 swap 空间

### 问题: 找不到 .ply 文件
解决: 检查输出路径，确保目录存在
