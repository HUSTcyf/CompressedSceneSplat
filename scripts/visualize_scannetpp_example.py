#!/usr/bin/env python3
"""
ScanNet++ 场景推理与可视化示例

此脚本展示如何：
1. 使用预训练的 SceneSplat 权重进行推理 (可选)
2. 生成语义分割预测结果 (可选)
3. 可视化 GT 和预测结果

场景: /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be

使用方法:
    # 完整流程 (推理 + 预测 + 可视化)
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_scannetpp_example.py

    # 仅可视化 (需要已有预测结果)
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_scannetpp_example.py --skip_all
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_scannetpp_example.py --skip_inference --existing_features path/to/features.npy
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_scannetpp_example.py --skip_prediction --existing_predictions path/to/predictions.npy
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch


def run_inference(
    scene_path: str,
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    scene_name: str,
    gpu_id: int = 0,
):
    """
    运行 SceneSplat 推理生成语言特征
    """
    print("=" * 60)
    print("步骤 1: 运行 SceneSplat 推理")
    print("=" * 60)
    print(f"场景: {scene_name}")
    print(f"权重: {checkpoint_path}")
    print(f"输出: {output_dir}")
    print()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 构建命令
    cmd = [
        sys.executable,
        "tools/batch_predict_inference.py",
        "--config", config_path,
        "--checkpoint", checkpoint_path,
        "--input-root", str(Path(scene_path).parent),
        "--output-dir", output_dir,
        "--scene", scene_name,
        "--device", "cuda",
    ]

    # 设置环境变量并运行
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, check=True)

    print()
    print(f"推理完成！语言特征已保存到: {output_dir}/{scene_name}/language_features.npy")
    print()

    return f"{output_dir}/{scene_name}/language_features.npy"


def generate_predictions(
    scene_path: str,
    features_path: str,
    class_names_file: str,
    text_embeddings_file: str,
    output_path: str,
    excluded_classes=None,
):
    """
    使用 SceneSplat 特征和文本嵌入生成语义分割预测。
    """
    print("=" * 60)
    print("步骤 2: 生成语义分割预测")
    print("=" * 60)

    # 加载数据
    print(f"加载场景数据: {scene_path}")
    scene_path = Path(scene_path)
    coord = np.load(scene_path / "coord.npy")
    gt_labels = np.load(scene_path / "segment.npy")

    # Handle different segment formats
    print(f"  segment.npy 原始形状: {gt_labels.shape}")
    if gt_labels.ndim == 2:
        # ScanNet++ format: (N, 3) where first column is the label
        # Other columns are unused (often -1)
        print("  检测到 (N, 3) 格式的 segment，使用第一列作为标签")
        gt_labels = gt_labels[:, 0].astype(np.int64)
        print(f"  转换后形状: {gt_labels.shape}")
        print(f"  标签范围: [{gt_labels.min()}, {gt_labels.max()}]")

    print(f"  点云数量: {len(coord)}")
    print(f"  GT 类别数: {len(np.unique(gt_labels))}")

    # 加载语言特征
    print(f"加载语言特征: {features_path}")
    features = np.load(features_path)
    print(f"  特征形状: {features.shape}")

    # 加载类别名称
    print(f"加载类别名称: {class_names_file}")
    with open(class_names_file, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"  类别数量: {len(class_names)}")

    # 加载文本嵌入
    print(f"加载文本嵌入: {text_embeddings_file}")
    text_embeddings = torch.load(text_embeddings_file)
    print(f"  嵌入形状: {text_embeddings.shape}")

    # 计算相似度并预测
    print("计算相似度并预测...")
    lang_feat = torch.from_numpy(features).float()  # [N, 768]

    # 归一化
    lang_feat = lang_feat / torch.norm(lang_feat, dim=-1, keepdim=True)
    text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=-1, keepdim=True)

    # 点积相似度
    similarities = torch.matmul(lang_feat, text_embeddings.T)  # [N, C]
    predictions = torch.argmax(similarities, dim=1).numpy()

    print(f"  预测类别数: {len(np.unique(predictions))}")

    # 保存预测结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, predictions)
    print(f"保存预测结果: {output_path}")

    # 计算准确率
    accuracy = (gt_labels == predictions).mean() * 100
    print(f"  总体准确率: {accuracy:.2f}%")

    # 计算每个类别的 IoU
    unique_classes = np.unique(gt_labels)
    print(f"\nPer-class IoU (top 20):")
    ious = []
    for cls in unique_classes[:20]:
        if cls == -1:
            continue
        mask = (gt_labels == cls)
        if mask.sum() > 0:
            intersection = np.sum((gt_labels == cls) & (predictions == cls))
            union = np.sum((gt_labels == cls) | (predictions == cls))
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
            class_name = class_names[int(cls)] if int(cls) < len(class_names) else f"class_{cls}"
            print(f"  {class_name:20s}: {iou:.4f}")

    print()
    return str(output_path), accuracy


def run_gt_visualization(
    scene_path: str,
    output_dir: str,
    scene_name: str,
    gpu_id: int = 0,
):
    """
    仅可视化 GT (Ground Truth)
    """
    print("=" * 60)
    print("步骤: 可视化 GT")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 可视化 GT
    print("可视化 Ground Truth...")
    cmd_gt = [
        sys.executable, "tools/visualize_semantic_segmentation.py",
        "--data_path", scene_path,
        "--mode", "gt",
        "--output_path", f"{output_dir}/{scene_name}_gt.ply",
        "--dataset", "scannetpp",
    ]
    subprocess.run(cmd_gt, env=env, check=True)
    print(f"  GT 可视化已保存: {output_dir}/{scene_name}_gt.ply")
    print()


def run_visualization(
    scene_path: str,
    predictions_path: str,
    output_dir: str,
    scene_name: str,
    gpu_id: int = 0,
):
    """
    可视化 GT 和预测结果
    """
    print("=" * 60)
    print("步骤 3: 可视化结果")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 3.1 可视化 GT
    print("可视化 Ground Truth...")
    cmd_gt = [
        sys.executable, "tools/visualize_semantic_segmentation.py",
        "--data_path", scene_path,
        "--mode", "gt",
        "--output_path", f"{output_dir}/{scene_name}_gt.ply",
        "--dataset", "scannetpp",
    ]
    subprocess.run(cmd_gt, env=env, check=True)
    print(f"  GT 可视化已保存: {output_dir}/{scene_name}_gt.ply")
    print()

    # 3.2 可视化预测
    print("可视化预测结果...")
    cmd_pred = [
        sys.executable, "tools/visualize_semantic_segmentation.py",
        "--data_path", scene_path,
        "--pred_path", predictions_path,
        "--mode", "pred",
        "--output_path", f"{output_dir}/{scene_name}_pred.ply",
        "--dataset", "scannetpp",
    ]
    subprocess.run(cmd_pred, env=env, check=True)
    print(f"  预测可视化已保存: {output_dir}/{scene_name}_pred.ply")
    print()

    # 3.3 对比可视化
    print("生成对比可视化...")
    cmd_compare = [
        sys.executable, "tools/visualize_semantic_segmentation.py",
        "--data_path", scene_path,
        "--pred_path", predictions_path,
        "--mode", "compare",
        "--output_path", f"{output_dir}/{scene_name}_compare",
        "--dataset", "scannetpp",
    ]
    subprocess.run(cmd_compare, env=env, check=True)
    print(f"  对比可视化已保存:")
    print(f"    - {output_dir}/{scene_name}_compare_gt.ply")
    print(f"    - {output_dir}/{scene_name}_compare_pred.ply")
    print()


def main():
    parser = argparse.ArgumentParser(description="ScanNet++ 场景推理与可视化")
    parser.add_argument("--scene_path", type=str,
        default="/new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be",
        help="场景路径")
    parser.add_argument("--checkpoint", type=str,
        default="/new_data/cyf/projects/SceneSplat/checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth",
        help="预训练权重路径")
    parser.add_argument("--config", type=str,
        default="/new_data/cyf/projects/SceneSplat/configs/inference/lang-pretrain-pt-v3m1-3dgs.py",
        help="推理配置文件")
    parser.add_argument("--output_dir", type=str,
        default="/new_data/cyf/projects/SceneSplat/output_visualization",
        help="输出目录")
    parser.add_argument("--class_names", type=str,
        default="/new_data/cyf/projects/SceneSplat/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt",
        help="类别名称文件")
    parser.add_argument("--text_embeddings", type=str,
        default="/new_data/cyf/projects/SceneSplat/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt",
        help="文本嵌入文件")
    parser.add_argument("--gpu_id", type=int, default=0,
        help="GPU 设备 ID")
    parser.add_argument("--skip_inference", action="store_true",
        help="跳过推理步骤（如果已有语言特征）")
    parser.add_argument("--existing_features", type=str, default=None,
        help="使用现有的语言特征路径")
    parser.add_argument("--skip_prediction", action="store_true",
        help="跳过预测步骤（如果已有预测结果），仅做可视化")
    parser.add_argument("--existing_predictions", type=str, default=None,
        help="使用现有的预测结果路径")
    parser.add_argument("--skip_all", action="store_true",
        help="跳过推理和预测，仅做 GT 可视化")

    args = parser.parse_args()

    # 场景名称
    scene_name = Path(args.scene_path).name

    # 输出目录
    inference_output_dir = f"{args.output_dir}/inference/{scene_name}"
    visual_output_dir = f"{args.output_dir}/visualization"

    try:
        # 步骤 1: 推理 (可选)
        if not args.skip_inference and not args.skip_all:
            features_path = run_inference(
                scene_path=args.scene_path,
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                output_dir=inference_output_dir,
                scene_name=scene_name,
                gpu_id=args.gpu_id,
            )
        elif args.existing_features:
            features_path = args.existing_features
            print(f"使用现有特征: {features_path}")
        else:
            features_path = f"{inference_output_dir}/{scene_name}/language_features.npy"
            if args.skip_all:
                print("跳过推理步骤 (--skip_all)")
            elif not os.path.exists(features_path):
                print(f"错误: 特征文件不存在: {features_path}")
                print("请使用 --existing_features 指定特征文件路径，或移除 --skip_inference 重新运行推理")
                sys.exit(1)

        # 步骤 2: 生成预测
        if not args.skip_prediction and not args.skip_all:
            if args.existing_predictions:
                predictions_path = args.existing_predictions
                print(f"使用现有预测: {predictions_path}")
            else:
                predictions_path, accuracy = generate_predictions(
                    scene_path=args.scene_path,
                    features_path=features_path,
                    class_names_file=args.class_names,
                    text_embeddings_file=args.text_embeddings,
                    output_path=f"{inference_output_dir}/{scene_name}/predictions.npy",
                )
        else:
            predictions_path = f"{inference_output_dir}/{scene_name}/predictions.npy"
            accuracy = None
            if args.skip_all:
                print("跳过预测步骤 (--skip_all)")
            elif args.existing_predictions:
                predictions_path = args.existing_predictions
                print(f"使用现有预测: {predictions_path}")
            elif not os.path.exists(predictions_path):
                print(f"警告: 预测文件不存在: {predictions_path}")
                print("将只生成 GT 可视化")

        # 步骤 3: 可视化
        if args.skip_all:
            # 仅可视化 GT
            print("仅可视化 GT...")
            run_gt_visualization(
                scene_path=args.scene_path,
                output_dir=visual_output_dir,
                scene_name=scene_name,
                gpu_id=args.gpu_id,
            )
        else:
            run_visualization(
                scene_path=args.scene_path,
                predictions_path=predictions_path,
                output_dir=visual_output_dir,
                scene_name=scene_name,
                gpu_id=args.gpu_id,
            )

        # 总结
        print("=" * 60)
        print("完成！")
        print("=" * 60)
        print()
        print("输出文件:")
        if not args.skip_inference and not args.skip_all:
            print(f"  语言特征:   {features_path}")
        if not args.skip_prediction and not args.skip_all and accuracy is not None:
            print(f"  预测结果:   {predictions_path}")
            print(f"  准确率: {accuracy:.2f}%")
        print(f"  GT 可视化:  {visual_output_dir}/{scene_name}_gt.ply")
        if not args.skip_all:
            print(f"  预测可视化: {visual_output_dir}/{scene_name}_pred.ply")
            print(f"  对比可视化: {visual_output_dir}/{scene_name}_compare_*.ply")
        print()
        print("使用以下工具查看 .ply 文件:")
        print("  - MeshLab: meshlab output_file.ply")
        print("  - CloudCompare: cloudcompare output_file.ply")
        print()

    except subprocess.CalledProcessError as e:
        print(f"错误: 命令执行失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
