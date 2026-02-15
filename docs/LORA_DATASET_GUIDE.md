# LoRA 微调数据集配置指南

本指南说明如何在 SceneSplat 中为 LoRA 微调指定数据集。

---

## 目录

1. [快速开始](#快速开始)
2. [配置文件结构](#配置文件结构)
3. [支持的数据集](#支持的数据集)
4. [数据集配置详解](#数据集配置详解)
5. [自定义数据集](#自定义数据集)
6. [常见问题](#常见问题)

---

## 快速开始

### 方法 1: 修改配置文件

编辑 `configs/scannet/lora-finetune-scannet.py`:

```python
# 修改数据集根目录
data_root = "/path/to/your/dataset"

# 修改训练/验证集划分
train_split = "train_grid1mm_chunk6x6_stride3x3"
val_split = "val_scannet_fix_xyz"
```

### 方法 2: 命令行参数

```bash
python tools/train_lora.py \
    --config-file configs/scannet/lora-finetune-scannet.py \
    --options save_path=exp_runs/lora_ft \
                data_root=/path/to/your/dataset \
                data.train.split=your_train_split \
                data.val.split=your_val_split
```

---

## 配置文件结构

### 完整配置层次

```
configs/
├── _base_/
│   └── dataset/
│       └── scannetpp.py        # 类别名称定义
├── scannet/
│   ├── lora-finetune-scannet.py      # ScanNet 数据集配置
│   └── lora-finetune-scannet-mcmc.py # 多数据集配置
```

### 数据集配置项

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `dataset_type` | 数据集类名 | `"ScanNetPPGSDataset"` |
| `data_root` | 数据集根目录 | `"/path/to/data"` |
| `split` | 数据划分 | `("train", "test",)` |
| `filtered_scene` | 过滤场景 | `["scene001", "scene002"]` |
| `transform` | 数据增强 | `[CenterShift, ...]` |

---

## 支持的数据集

### 1. ScanNet++

```python
dataset_type = "ScanNetPPGSDataset"
data_root = "/path/to/scannetpp_preprocessed"

train_split = (
    "train_grid1mm_chunk6x6_stride3x3",
    "test_grid1mm_chunk6x6_stride3x3",
    "train_scannet_fix_xyz",
)
val_split = "val_scannet_fix_xyz"
```

### 2. ScanNet

```python
dataset_type = "ScanNetGSDataset"
data_root = "/path/to/scannet_preprocessed"

train_split = ("train",)
val_split = "val"
```

### 3. Matterport3D

```python
dataset_type = "Matterport3DGSDataset"
data_root = "/path/to/matterport3d_preprocessed"

train_split = ("train",)
val_split = "val"
```

### 4. KITTI-360

```python
dataset_type = "KITTI360GSDataset"
data_root = "/path/to/kitti360_preprocessed"

train_split = ("train",)
val_split = "val"
```

### 5. 3RScan

```python
dataset_type = "ThreeRScanGSDataset"
data_root = "/path/to/3rscan_preprocessed"

train_split = ("train",)
val_split = "val"
```

### 6. ARKitScenes

```python
dataset_type = "ArkitScenesGSDataset"
data_root = "/path/to/arkit_preprocessed"

train_split = ("train",)
val_split = "val"
```

---

## 数据集配置详解

### 1. 基础配置

```python
data = dict(
    num_classes=100,      # 类别数量
    ignore_index=-1,      # 忽略索引
    train=dict(...),      # 训练集配置
    val=dict(...),        # 验证集配置
    test=dict(...),       # 测试集配置
)
```

### 2. 训练集配置

```python
train=dict(
    type="ScanNetPPGSDataset",        # 数据集类型
    split=(                           # 数据划分
        "train_grid1mm_chunk6x6_stride3x3",
        "test_grid1mm_chunk6x6_stride3x3",
        "train_scannet_fix_xyz",
    ),
    data_root="/path/to/data",         # 数据根目录
    filtered_scene=[                   # 过滤场景（可选）
        "c601466b77",
        "654a4f341b",
    ],
    sample_tail_classes=False,         # 是否采样尾部类别
    transform=[...],                   # 数据增强
    test_mode=False,
)
```

### 3. 数据增强配置

```python
transform=[
    # 坐标变换
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.01),
    dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),

    # 颜色增强
    dict(type="ChromaticAutoContrast", p=0.2),
    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    dict(type="ChromaticJitter", p=0.95, std=0.05),
    dict(type="NormalizeColor"),

    # 3DGS 特定
    dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),

    # 体素化
    dict(
        type="GridSample",
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "opacity", "quat", "scale", "segment", "lang_feat", "valid_feat_mask"),
        return_grid_coord=True,
    ),

    # 裁剪
    dict(type="SphereCrop", point_max=192000, mode="random"),

    # 转换
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "lang_feat", "valid_feat_mask"),
        feat_keys=("color", "opacity", "quat", "scale"),
    ),
]
```

### 4. 验证/测试集配置

```python
val=dict(
    type="ScanNetPPGSDataset",
    split="val_scannet_fix_xyz",
    data_root="/path/to/data",
    filtered_scene=[...],  # 与训练集一致
    transform=[
        # 验证时通常不使用随机增强
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", ...),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect", ...),
    ],
    test_mode=False,
)
```

---

## 自定义数据集

### 方法 1: 使用现有数据集类

如果你的数据格式与 ScanNet++ 类似，直接使用 `ScanNetPPGSDataset`:

```python
dataset_type = "ScanNetPPGSDataset"
data_root = "/path/to/your/custom_data"
train_split = "your_split_name"
```

### 方法 2: 创建自定义数据集类

1. 创建新文件 `pointcept/datasets/custom_gs.py`:

```python
from pointcept.datasets.generic_gs import GenericGSDataset

class CustomGSDataset(GenericGSDataset):
    """Custom 3DGS dataset for LoRA fine-tuning."""

    def __init__(
        self,
        data_root,
        split="train",
        transform=None,
        **kwargs,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            transform=transform,
            **kwargs,
        )
```

2. 注册数据集 (在 `pointcept/datasets/builder.py`):

```python
DATASETS.register_module()(CustomGSDataset)
```

3. 使用自定义数据集:

```python
dataset_type = "CustomGSDataset"
data_root = "/path/to/custom_data"
```

---

## 多数据集微调

### 配置多数据集

```python
# 在 configs/concat_dataset/lora-finetune-concat.py 中

data = dict(
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="ScanNetPPGSDataset",
                data_root="/path/to/scannetpp",
                split=("train",),
                weight=1.0,  # 数据集权重
            ),
            dict(
                type="Matterport3DGSDataset",
                data_root="/path/to/matterport3d",
                split=("train",),
                weight=0.5,
            ),
            dict(
                type="ThreeRScanGSDataset",
                data_root="/path/to/3rscan",
                split=("train",),
                weight=0.3,
            ),
        ],
        transform=[...],  # 共享的增强
    ),
    # 验证集可使用单个数据集
    val=dict(
        type="ScanNetPPGSDataset",
        data_root="/path/to/scannetpp",
        split="val",
        transform=[...],
    ),
)
```

---

## 数据预处理

### 预处理要求

LoRA 微调需要的数据格式与预训练相同：

```
data_root/
└── scene_name/
    ├── coord.npy           # [N, 3] 坐标
    ├── color.npy           # [N, 3] RGB 颜色
    ├── opacity.npy         # [N, 1] 不透明度
    ├── quat.npy            # [N, 4] 四元数旋转
    ├── scale.npy           # [N, 3] 缩放
    ├── segment.npy         # [N] 语义标签（可选）
    ├── lang_feat.npy       # [N, D] 语言特征（可选）
    └── valid_feat_mask.npy # [N] 特征掩码（可选）
```

### 预处理脚本

```bash
# 使用预处理脚本处理 .ply 文件
python scripts/preprocess_gs.py \
    --input_root /path/to/ply_files \
    --output_root /path/to/preprocessed_data

# 分块场景
python -u pointcept/datasets/preprocessing/sampling_chunking_data_gs.py \
    --dataset_root /path/to/preprocessed_data \
    --output_dir /path/to/chunked_data \
    --grid_size 0.01 \
    --chunk_range 6 6 \
    --chunk_stride 3 3
```

---

## 常见问题

### Q1: 如何只使用部分场景进行微调？

```python
# 在配置文件中指定 filtered_scene
data = dict(
    train=dict(
        filtered_scene=[
            "scene_0011",
            "scene_0059",
            # ... 只使用这些场景
        ],
    ),
)
```

### Q2: 如何处理类别不平衡？

```python
# 启用尾部类别采样
data = dict(
    train=dict(
        sample_tail_classes=True,  # 采样稀有类别
    ),
)
```

### Q3: 如何调整数据增强强度？

```python
# LoRA 微调通常使用较弱的数据增强
transform=[
    # 降低 dropout 比率
    dict(type="RandomDropout", dropout_ratio=0.1),  # 原 0.2

    # 减少旋转角度
    dict(type="RandomRotate", angle=[-0.5, 0.5], axis="z", p=0.3),  # 原 [-1, 1], p=0.5

    # 其他...
]
```

### Q4: 如何使用多个验证集？

```python
# 在 hooks 中配置多个评估器
hooks = [
    dict(
        type="LangPretrainZeroShotSemSegEval",
        class_names="scannet_labels.txt",
        text_embeddings="scannet_embeddings.pt",
        split="val_scannet",
    ),
    dict(
        type="LangPretrainZeroShotSemSegEval",
        class_names="matterport_labels.txt",
        text_embeddings="matterport_embeddings.pt",
        split="val_matterport",
    ),
]
```

### Q5: 数据集路径不对怎么办？

```bash
# 检查数据集结构
ls /path/to/your/dataset/

# 应该看到类似：
# scene001/coord.npy
# scene001/color.npy
# scene001/opacity.npy
# ...

# 使用绝对路径
python tools/train_lora.py \
    --config-file configs/scannet/lora-finetune-scannet.py \
    --options data_root=/absolute/path/to/dataset
```

---

## 完整示例

### 示例 1: ScanNet++ 微调

```python
# configs/scannet/lora-finetune-scannet-simple.py

_base_ = ["../_base_/default_runtime.py"]

dataset_type = "ScanNetPPGSDataset"
data_root = "/datasets/scannetpp_preprocessed"

data = dict(
    num_classes=100,
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split="train_scannet_fix_xyz",
        data_root=data_root,
        transform=[...],  # 标准增强
    ),
    val=dict(
        type=dataset_type,
        split="val_scannet_fix_xyz",
        data_root=data_root,
        transform=[...],  # 验证增强
    ),
)

model = dict(
    type="LangPretrainer",
    backbone=dict(type="PT-v3m1", ...),
    lora=dict(
        enabled=True,
        r=8,
        lora_alpha=8,
        target_modules=["attn"],
        target_stages=[3, 4],
    ),
)

optimizer = dict(type="AdamW", lr=0.0001)
epoch = 50
```

### 示例 2: 命令行指定

```bash
# 快速切换数据集
python tools/train_lora.py \
    --config-file configs/scannet/lora-finetune-scannet.py \
    --options \
        data_root=/datasets/matterport3d \
        data.train.type=Matterport3DGSDataset \
        data.val.type=Matterport3DGSDataset \
        save_path=exp_runs/lora_matterport3d
```

### 示例 3: 小数据集微调

```python
# 使用较少的数据和更长的训练
data = dict(
    train=dict(
        split="small_split",  # 只使用部分数据
        sample_tail_classes=False,  # 关闭尾部采样
    ),
)

epoch = 100  # 增加训练轮数
optimizer = dict(lr=5e-5)  # 降低学习率
```
