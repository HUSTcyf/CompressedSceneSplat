#!/usr/bin/env python
"""
示例代码：如何使用 evaluate_iou_scannet.py 中新增的语言特征加载功能

这个文件展示了如何使用新增的函数来加载 chkpnt30000_langfeat_3.pth
并计算每个3D点对应的语义标签。
"""

import os
import sys
import torch
import numpy as np
import json

# 添加路径以便导入
sys.path.append('../eval')
from evaluate_iou_scannet import (
    load_gaussian_language_features,
    compute_point_labels_from_features,
    evaluate_scannet
)

def example_load_gaussian_features():
    """
    示例：加载 Gaussian 语言特征
    """
    print("=== 示例1：加载 Gaussian 语言特征 ===\n")

    # 示例参数
    model_path = "./output/scannet/scene0000_00/"
    iteration = 30000
    feature_level = 3

    # 加载 Gaussian 语言特征
    gaussians = load_gaussian_language_features(
        model_path=model_path,
        iteration=iteration,
        feature_level=feature_level
    )

    if gaussians is not None:
        print("成功加载 Gaussian 语言特征！")

        # 获取语言特征
        features = gaussians.capture_language_feature()
        print(f"特征形状: {features.shape}")
        print(f"特征维度: {features.shape[1]}")
        print(f"点数: {features.shape[0]}")

        return gaussians
    else:
        print("未能加载 Gaussian 语言特征")
        return None

def example_compute_point_labels():
    """
    示例：计算每个3D点的语义标签
    """
    print("\n=== 示例2：计算3D点的语义标签 ===\n")

    # 参数
    model_path = "./output/scannet/scene0000_00/"
    iteration = 30000
    feature_level = 3
    text_features_path = "./assets/text_features.json"
    target_names = ["wall", "floor", "chair", "table", "door"]

    # 加载 Gaussian 语言特征
    gaussians = load_gaussian_language_features(
        model_path=model_path,
        iteration=iteration,
        feature_level=feature_level
    )

    if gaussians is not None:
        # 计算点标签
        pred_labels = compute_point_labels_from_features(
            gaussians=gaussians,
            target_names=target_names,
            text_features_path=text_features_path,
            feature_level=feature_level
        )

        if pred_labels is not None:
            print("成功计算3D点标签！")
            print(f"预测标签形状: {pred_labels.shape}")
            print(f"标签范围: {pred_labels.min().item()} - {pred_labels.max().item()}")

            # 统计每个类别的点数
            for i, class_name in enumerate(target_names):
                count = (pred_labels == i + 1).sum().item()
                print(f"  {class_name} (ID={i+1}): {count} 个点")

            return pred_labels
        else:
            print("未能计算3D点标签")
            return None
    else:
        print("未能加载 Gaussian 语言特征")
        return None

def example_full_evaluation():
    """
    示例：完整的语义分割评估流程
    """
    print("\n=== 示例3：完整的语义分割评估 ===\n")

    # 创建一个假的 logger
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("example")

    # 参数
    scan_name = "scene0000_00"
    model_path = "./output/scannet/scene0000_00/"
    iteration = 30000
    text_features_path = "./assets/text_features.json"
    target_id = [1, 2, 4, 5, 6]  # wall, floor, chair, table, door

    try:
        # 执行评估
        miou, macc = evaluate_scannet(
            scan_name=scan_name,
            model_path=model_path,
            iteration=iteration,
            text_features_path=text_features_path,
            target_id=target_id,
            logger=logger
        )

        if miou is not None and macc is not None:
            print(f"评估完成！")
            print(f"mIoU: {miou:.2f}%")
            print(f"mAcc: {macc:.2f}%")
        else:
            print("评估失败")

    except Exception as e:
        print(f"评估过程中出错: {e}")

def example_fallback_to_cluster_features():
    """
    示例：当 Gaussian 特征加载失败时，回退到聚类特征
    """
    print("\n=== 示例4：回退到聚类特征 ===\n")

    model_path = "./output/scannet/scene0000_00/"

    # 首先尝试加载 Gaussian 特征（可能会失败）
    gaussians = load_gaussian_language_features(
        model_path=model_path,
        iteration=30000,
        feature_level=3
    )

    if gaussians is None:
        print("Gaussian 特征加载失败，尝试加载聚类特征...")

        # 尝试加载聚类特征
        from evaluate_iou_scannet import load_cluster_language_features
        cluster_data = load_cluster_language_features(model_path)

        if cluster_data is not None:
            print("成功加载聚类特征！")
            print(f"叶节点特征形状: {cluster_data['leaf_lang_feat'].shape}")
            print(f"叶节点得分形状: {cluster_data['leaf_score'].shape}")
            print(f"叶节点计数形状: {cluster_data['leaf_count'].shape}")
            print(f"点索引形状: {cluster_data['leaf_ind'].shape}")
        else:
            print("聚类特征也加载失败")

def example_check_feature_compatibility():
    """
    示例：检查特征兼容性
    """
    print("\n=== 示例5：检查特征兼容性 ===\n")

    model_path = "./output/scannet/scene0000_00/"
    text_features_path = "./assets/text_features.json"

    # 加载文本特征
    try:
        with open(text_features_path, 'r') as f:
            text_features_dict = json.load(f)

        sample_text_feature = list(text_features_dict.values())[0]
        text_feature_dim = len(sample_text_feature)
        print(f"文本特征维度: {text_feature_dim}")

        # 加载 Gaussian 特征
        gaussians = load_gaussian_language_features(
            model_path=model_path,
            iteration=30000,
            feature_level=3
        )

        if gaussians is not None:
            gaussian_features = gaussians.capture_language_feature()
            gaussian_feature_dim = gaussian_features.shape[1]
            print(f"Gaussian 特征维度: {gaussian_feature_dim}")

            if text_feature_dim == gaussian_feature_dim:
                print("✓ 特征维度匹配！")
            else:
                print(f"⚠ 特征维度不匹配，需要调整")
                print(f"  差异: {abs(text_feature_dim - gaussian_feature_dim)}")
        else:
            print("无法加载 Gaussian 特征进行比较")

    except Exception as e:
        print(f"检查特征兼容性时出错: {e}")

def example_feature_level_usage():
    """
    示例：使用不同的特征层级
    """
    print("\n=== 示例6：使用不同的特征层级 ===\n")

    model_path = "./output/scannet/scene0000_00/"
    iteration = 30000

    # 尝试不同的特征层级
    feature_levels = [0, 1, 2, 3]

    for level in feature_levels:
        print(f"尝试加载特征层级 {level}...")
        gaussians = load_gaussian_language_features(
            model_path=model_path,
            iteration=iteration,
            feature_level=level
        )

        if gaussians is not None:
            features = gaussians.capture_language_feature()
            print(f"  ✓ 成功加载层级 {level}，特征形状: {features.shape}")
        else:
            print(f"  ✗ 层级 {level} 加载失败")

if __name__ == "__main__":
    print("Gaussian 语言特征评估功能使用示例")
    print("=" * 50)

    # 运行所有示例
    try:
        example_load_gaussian_features()
        example_compute_point_labels()
        example_full_evaluation()
        example_fallback_to_cluster_features()
        example_check_feature_compatibility()
        example_feature_level_usage()

        print("\n" + "=" * 50)
        print("所有示例运行完成！")

    except Exception as e:
        print(f"\n运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

        print("\n注意事项：")
        print("1. 确保已经训练了模型并生成了 chkpnt30000_langfeat_3.pth 文件")
        print("2. 确保文本特征文件存在")
        print("3. 确保模型路径和 ScanNet 数据路径正确")