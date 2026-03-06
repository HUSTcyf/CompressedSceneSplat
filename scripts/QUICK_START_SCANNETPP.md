# ScanNet++ 可视化快速参考

## 一键运行

```bash
cd /new_data/cyf/projects/SceneSplat

# 完整流程: 推理 → 预测 → 可视化
CUDA_VISIBLE_DEVICES=0 /new_data/cyf/.conda/envs/scene_splat/bin/python scripts/visualize_scannetpp_example.py
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
CUDA_VISIBLE_DEVICES=0 /new_data/cyf/.conda/envs/scene_splat/bin/python tools/batch_predict_inference.py \
    --config configs/inference/lang-pretrain-pt-v3m1-3dgs.py \
    --checkpoint checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth \
    --input-root /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val \
    --output-dir ./output_inference \
    --scene 0d2ee665be
```

### 2. 可视化 GT
```bash
CUDA_VISIBLE_DEVICES=0 /new_data/cyf/.conda/envs/scene_splat/bin/python tools/visualize_semantic_segmentation.py \
    --data_path /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be \
    --mode gt \
    --output_path ./output/gt.ply \
    --dataset scannetpp
```

### 3. 可视化预测
```bash
CUDA_VISIBLE_DEVICES=0 /new_data/cyf/.conda/envs/scene_splat/bin/python tools/visualize_semantic_segmentation.py \
    --data_path /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be \
    --pred_path ./output_inference/0d2ee665be/predictions.npy \
    --mode pred \
    --output_path ./output/pred.ply \
    --dataset scannetpp
```

## 关键文件路径

| 类型 | 路径 |
|------|------|
| 场景数据 | `/new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be` |
| 预训练权重 | `/new_data/cyf/projects/SceneSplat/checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth` |
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
CUDA_VISIBLE_DEVICES=0 /new_data/cyf/.conda/envs/scene_splat/bin/python scripts/visualize_scannetpp_example.py \
    --scene_path /path/to/your/scene
```
