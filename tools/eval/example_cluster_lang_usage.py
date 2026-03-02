#!/usr/bin/env python
"""
示例代码：如何使用 evaluate_iou_scannet.py 中的 cluster_lang.npz 计算功能

这个文件展示了如何使用新添加的函数来计算和使用 cluster_lang.npz 文件。
"""

import os
import sys
import numpy as np
import torch

# 添加路径以便导入
sys.path.append('../eval')
from evaluate_iou_scannet import (
    compute_and_save_cluster_lang,
    load_cluster_language_features,
    calculate_iou,
    mask_feature_mean
)

def example_compute_cluster_lang():
    """
    示例：计算并保存 cluster_lang.npz 文件
    """
    print("=== 示例1：计算并保存 cluster_lang.npz ===\n")

    # 假设的参数
    model_path = "./output/scene_example/"
    num_points = 100000  # 假设有10万个3D点
    root_num = 64
    leaf_num = 5

    # 创建假的聚类索引（实际使用时应该从训练好的模型中获取）
    cluster_indices = torch.randint(0, root_num * leaf_num, (num_points,))

    # 确保输出目录存在
    os.makedirs(model_path, exist_ok=True)

    # 计算并保存 cluster language 特征
    # 这里不提供 SAM masks，所以会创建演示用的随机特征
    results = compute_and_save_cluster_lang(
        model_path=model_path,
        cluster_indices=cluster_indices,
        root_num=root_num,
        leaf_num=leaf_num
    )

    print(f"计算完成！特征形状：")
    print(f"  - leaf_lang_feat: {results['leaf_lang_feat'].shape}")
    print(f"  - leaf_score: {results['leaf_score'].shape}")
    print(f"  - leaf_count: {results['leaf_count'].shape}")
    print(f"  - leaf_ind: {results['leaf_ind'].shape}")

    return results

def example_compute_with_sam_masks():
    """
    示例：使用 SAM masks 计算 cluster language 特征
    """
    print("\n=== 示例2：使用 SAM masks 计算特征 ===\n")

    model_path = "./output/scene_example_with_sam/"
    num_points = 50000
    num_masks = 100
    root_num = 32
    leaf_num = 4

    # 创建假的聚类索引
    cluster_indices = torch.randint(0, root_num * leaf_num, (num_points,))

    # 创建假的 SAM masks（实际使用时应该从 SAM 模型获取）
    height, width = 256, 256
    sam_masks = []
    for _ in range(num_masks):
        mask = torch.zeros(height, width)
        # 随机生成一些区域
        y1, y2 = np.random.randint(0, height//2), np.random.randint(height//2, height)
        x1, x2 = np.random.randint(0, width//2), np.random.randint(width//2, width)
        mask[y1:y2, x1:x2] = 1
        sam_masks.append(mask)

    # 创建假的 mask 特征（实际使用时应该从 CLIP 模型获取）
    mask_features = torch.randn(num_masks, 512)  # 512 维 CLIP 特征
    mask_features = torch.nn.functional.normalize(mask_features, dim=1)

    os.makedirs(model_path, exist_ok=True)

    # 使用 SAM masks 计算特征
    results = compute_and_save_cluster_lang(
        model_path=model_path,
        cluster_indices=cluster_indices,
        root_num=root_num,
        leaf_num=leaf_num,
        sam_masks=sam_masks,
        mask_features=mask_features
    )

    print(f"使用 SAM masks 计算完成！")
    print(f"  处理了 {num_masks} 个 masks")
    print(f"  聚类数量：{root_num * leaf_num}")

    return results

def example_load_and_use():
    """
    示例：加载并使用已保存的 cluster language 特征
    """
    print("\n=== 示例3：加载并使用 cluster language 特征 ===\n")

    model_path = "./output/scene_example/"

    # 加载特征
    loaded_data = load_cluster_language_features(model_path)

    if loaded_data is not None:
        print("成功加载 cluster language 特征：")
        print(f"  - leaf_lang_feat: {loaded_data['leaf_lang_feat'].shape}")
        print(f"  - leaf_score: {loaded_data['leaf_score'].shape}")
        print(f"  - leaf_count: {loaded_data['leaf_count'].shape}")
        print(f"  - leaf_ind: {loaded_data['leaf_ind'].shape}")

        # 示例：使用特征进行简单的分析
        print("\n特征分析：")
        print(f"  - 平均置信度分数: {loaded_data['leaf_score'].mean().item():.4f}")
        print(f"  - 活跃聚类数量 (count > 0): {(loaded_data['leaf_count'] > 0).sum().item()}")
        print(f"  - 点数最多的聚类: {loaded_data['leaf_ind'].mode().values.item()}")
    else:
        print("未能加载特征文件，请先运行示例1")

def example_utility_functions():
    """
    示例：使用辅助函数
    """
    print("\n=== 示例4：使用辅助函数 ===\n")

    # IoU 计算示例
    mask1 = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32)
    mask2 = torch.tensor([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.float32)

    iou = calculate_iou(mask1, mask2)
    print(f"IoU 示例: {iou.item():.4f}")

    # 特征均值计算示例
    features = torch.randn(3, 64, 64)  # [C, H, W]
    masks = torch.tensor([
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]
    ], dtype=torch.float32)

    feat_means = mask_feature_mean(features, masks)
    print(f"特征均值计算完成，输出形状: {feat_means.shape}")

def example_integration_with_evaluation():
    """
    示例：如何集成到评估流程中
    """
    print("\n=== 示例5：集成到评估流程 ===\n")

    # 伪代码展示如何在训练后计算 cluster language 特征
    def training_pipeline():
        """模拟训练流程"""
        print("1. 训练 3D 高斯模型...")
        print("2. 聚类 3D 高斯点...")

        # 假设我们有训练好的模型和聚类结果
        model_path = "./output/trained_scene/"
        cluster_indices = torch.randint(0, 320, (100000,))  # 假设的聚类索引

        print("3. 提取 2D 分割 masks 和语言特征...")
        # 这里会涉及到：
        # - 从不同视角渲染 3D 点云
        # - 使用 SAM 获取 2D 分割 masks
        # - 使用 CLIP 提取语言特征

        print("4. 计算 cluster language 特征...")
        cluster_data = compute_and_save_cluster_lang(
            model_path=model_path,
            cluster_indices=cluster_indices,
            root_num=64,
            leaf_num=5
            # 在实际使用中，还会传入 SAM masks 和 CLIP features
        )

        print("5. 开始评估...")
        # 现在可以使用 cluster_data 进行语义分割评估

        return cluster_data

    # 运行训练流程示例
    print("模拟训练流程：")
    cluster_data = training_pipeline()

    print(f"\n训练和特征计算完成！")
    print(f"现在可以使用 cluster_lang.npz 进行语义分割评估了。")

if __name__ == "__main__":
    print("Cluster Language 计算功能使用示例")
    print("=" * 50)

    # 运行所有示例
    try:
        example_compute_cluster_lang()
        # example_compute_with_sam_masks()
        example_load_and_use()
        example_utility_functions()
        # example_integration_with_evaluation()

        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("请检查 ./output/ 目录下生成的文件。")

    except Exception as e:
        print(f"\n运行示例时出错: {e}")
        print("请确保所有依赖已正确安装。")