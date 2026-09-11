# ScanNet++ 可视化快速参考

## 一键运行

```bash
cd /home/isom/cyf/CompressedSceneSplat

# 完整流程: 推理 → 预测 → 可视化
CUDA_VISIBLE_DEVICES=0 /home/isom/.conda/envs/scene_splat/bin/python scripts/visualize_scannetpp_example.py
```

## 输出结果

```
output_visualization/
├── inference/
│   └── 0d2ee665be/
│       ├── language_features.npy    # SceneSplat 768-dim 特征
│       └── predictions.npy           # 语义分割预测
└── visualization/
    ├── 0d2ee665be_gt.ply            # Ground Truth 可视化
    ├── 0d2ee665be_pred.ply          # 预测结果可视化
    ├── 0d2ee665be_compare_gt.ply     # GT (对比)
    └── 0d2ee665be_compare_pred.ply   # 预测 (对比)
```

## 查看结果

```bash
# MeshLab
meshlab output_visualization/visualization/0d2ee665be_gt.ply

# CloudCompare
cloudcompare output_visualization/visualization/0d2ee665be_pred.ply
```

## 分步执行

### 1. 推理 (生成 768-dim 特征)
```bash
CUDA_VISIBLE_DEVICES=0 /home/isom/.conda/envs/scene_splat/bin/python tools/batch_predict_inference.py \
    --config configs/inference/lang-pretrain-pt-v3m1-3dgs.py \
    --checkpoint checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth \
    --input-root /home/isom/cyf/SceneSplat/scannetpp_v2/val \
    --output-dir ./output_inference \
    --scene 0d2ee665be
```

### 2. 可视化 GT
```bash
CUDA_VISIBLE_DEVICES=0 /home/isom/.conda/envs/scene_splat/bin/python tools/visualize_semantic_segmentation.py \
    --data_path /home/isom/cyf/SceneSplat/scannetpp_v2/val/0d2ee665be \
    --mode gt \
    --output_path ./output/gt.ply \
    --dataset scannetpp
```

### 3. 可视化预测
```bash
CUDA_VISIBLE_DEVICES=0 /home/isom/.conda/envs/scene_splat/bin/python tools/visualize_semantic_segmentation.py \
    --data_path /home/isom/cyf/SceneSplat/scannetpp_v2/val/0d2ee665be \
    --pred_path ./output_inference/0d2ee665be/predictions.npy \
    --mode pred \
    --output_path ./output/pred.ply \
    --dataset scannetpp
```

## 关键文件路径

| 类型 | 路径 |
|------|------|
| 场景数据 | `/home/isom/cyf/SceneSplat/scannetpp_v2/val/0d2ee665be` |
| 预训练权重 | `/home/isom/cyf/CompressedSceneSplat/checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth` |
| 类别名称 | `pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt` |
| 文本嵌入 | `pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt` |
| 推理配置 | `configs/inference/lang-pretrain-pt-v3m1-3dgs.py` |

## ScanNet++ 100 个类别

wall, ceiling, floor, table, door, ceiling lamp, cabinet, blinds, curtain, chair,
storage cabinet, office chair, bookshelf, whiteboard, window, box, window frame,
monitor, shelf, doorframe, pipe, heater, kitchen cabinet, sofa, windowsill, bed,
shower wall, trash can, book, plant, blanket, tv, computer tower, kitchen counter,
refrigerator, jacket, electrical duct, sink, bag, picture, pillow, towel, suitcase,
backpack, crate, keyboard, rack, toilet, paper, printer, poster, painting, microwave,
board, shoes, socket, bottle, bucket, cushion, basket, shoe rack, telephone, file
folder, cloth, laptop, plant pot, exhaust fan, cup, coat hanger, light switch,
speaker, table lamp, air vent, clothes hanger, kettle, smoke detector, container,
power strip, slippers, paper bag, mouse, cutting board, toilet paper, paper towel,
pot, clock, pan, tap, jar, soap dispenser, binder, bowl, tissue box, whiteboard
eraser, toilet brush, spray bottle, headphones, stapler, marker.

## 使用不同场景

```bash
# 修改场景路径
CUDA_VISIBLE_DEVICES=0 /home/isom/.conda/envs/scene_splat/bin/python scripts/visualize_scannetpp_example.py \
    --scene_path /path/to/your/scene
```

---

# 附录：修复说明（原 TROUBLESHOOTING.md）

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
cd /home/isom/cyf/CompressedSceneSplat

CUDA_VISIBLE_DEVICES=0 /home/isom/.conda/envs/scene_splat/bin/python scripts/visualize_scannetpp_example.py --skip_all
```

输出: `output_visualization/visualization/0d2ee665be_gt.ply`

### 方法 2: 可视化 GT (直接使用可视化工具)

最简单的方式，直接使用可视化脚本：

```bash
CUDA_VISIBLE_DEVICES=0 /home/isom/.conda/envs/scene_splat/bin/python tools/visualize_semantic_segmentation.py \
    --data_path /home/isom/cyf/SceneSplat/scannetpp_v2/val/0d2ee665be \
    --mode gt \
    --output_path ./output/gt.ply \
    --dataset scannetpp
```

### 方法 3: 完整流程 (推理 + 预测 + 可视化)

运行完整的 SceneSplat 推理流程：

```bash
CUDA_VISIBLE_DEVICES=0 /home/isom/.conda/envs/scene_splat/bin/python scripts/visualize_scannetpp_example.py
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
CUDA_VISIBLE_DEVICES=0 /home/isom/.conda/envs/scene_splat/bin/python scripts/visualize_scannetpp_example.py \
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
