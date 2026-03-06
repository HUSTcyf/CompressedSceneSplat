#!/bin/bash
# ScanNet++ 场景推理与可视化示例
#
# 此脚本展示如何：
# 1. 使用预训练的 SceneSplat 权重进行推理
# 2. 生成语义分割预测结果
# 3. 可视化 GT 和预测结果
#
# 场景: /new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be

set -e  # 遇到错误时退出

# ============================================
# 配置参数
# ============================================

# 场景路径
SCENE_PATH="/new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val/0d2ee665be"
SCENE_NAME="0d2ee665be"

# 预训练权重路径 (SceneSplat PT-v3m1 模型，768维特征)
CHECKPOINT_PATH="/new_data/cyf/projects/SceneSplat/checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth"

# 推理配置
CONFIG_PATH="/new_data/cyf/projects/SceneSplat/configs/inference/lang-pretrain-pt-v3m1-3dgs.py"

# 输出目录
OUTPUT_BASE_DIR="/new_data/cyf/projects/SceneSplat/output_visualization"
INFERENCE_OUTPUT_DIR="${OUTPUT_BASE_DIR}/inference/${SCENE_NAME}"
VISUAL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/visualization"

# ScanNet++ 类别信息
CLASS_NAMES_FILE="/new_data/cyf/projects/SceneSplat/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100.txt"
TEXT_EMBEDDINGS_FILE="/new_data/cyf/projects/SceneSplat/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark/top100_text_embeddings_siglip2.pt"

# GPU 设备
GPU_ID=0

# ============================================
# 步骤 1: 运行推理生成语言特征
# ============================================
echo "======================================"
echo "步骤 1: 运行 SceneSplat 推理"
echo "======================================"
echo "场景: ${SCENE_NAME}"
echo "权重: ${CHECKPOINT_PATH}"
echo "输出: ${INFERENCE_OUTPUT_DIR}"
echo ""

# 创建输出目录
mkdir -p "${INFERENCE_OUTPUT_DIR}"

# 运行推理 (使用 batch_predict_inference.py)
CUDA_VISIBLE_DEVICES=${GPU_ID} python /new_data/cyf/projects/SceneSplat/tools/batch_predict_inference.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --input-root "/new_data/cyf/Datasets/SceneSplat7k/scannetpp_v2/val" \
    --output-dir "${INFERENCE_OUTPUT_DIR}" \
    --scene "${SCENE_NAME}" \
    --device cuda

echo ""
echo "推理完成！语言特征已保存到: ${INFERENCE_OUTPUT_DIR}/${SCENE_NAME}/language_features.npy"
echo ""

# ============================================
# 步骤 2: 生成语义分割预测
# ============================================
echo "======================================"
echo "步骤 2: 生成语义分割预测"
echo "======================================"

# 创建预测脚本
PYTHON_SCRIPT=$(cat << 'EOF'
import sys
import numpy as np
import torch
from pathlib import Path

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

    Args:
        scene_path: 场景数据路径 (包含 segment.npy 等)
        features_path: 推理生成的语言特征路径
        class_names_file: 类别名称文件
        text_embeddings_file: 文本嵌入文件
        output_path: 预测结果输出路径
        excluded_classes: 排除的类别 (如 wall, floor, ceiling)
    """
    import open_clip

    # 加载数据
    print(f"加载场景数据: {scene_path}")
    scene_path = Path(scene_path)
    coord = np.load(scene_path / "coord.npy")
    gt_labels = np.load(scene_path / "segment.npy")

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

    return predictions, accuracy

if __name__ == "__main__":
    scene_path = sys.argv[1]
    features_path = sys.argv[2]
    class_names_file = sys.argv[3]
    text_embeddings_file = sys.argv[4]
    output_path = sys.argv[5]

    excluded_classes = ["wall", "floor", "ceiling"]

    predictions, accuracy = generate_predictions(
        scene_path=scene_path,
        features_path=features_path,
        class_names_file=class_names_file,
        text_embeddings_file=text_embeddings_file,
        output_path=output_path,
        excluded_classes=excluded_classes,
    )
EOF
)

# 运行预测
CUDA_VISIBLE_DEVICES=${GPU_ID} python -c "${PYTHON_SCRIPT}" \
    "${SCENE_PATH}" \
    "${INFERENCE_OUTPUT_DIR}/${SCENE_NAME}/language_features.npy" \
    "${CLASS_NAMES_FILE}" \
    "${TEXT_EMBEDDINGS_FILE}" \
    "${INFERENCE_OUTPUT_DIR}/${SCENE_NAME}/predictions.npy"

echo ""
echo "预测完成！预测结果已保存到: ${INFERENCE_OUTPUT_DIR}/${SCENE_NAME}/predictions.npy"
echo ""

# ============================================
# 步骤 3: 可视化 GT 和预测结果
# ============================================
echo "======================================"
echo "步骤 3: 可视化结果"
echo "======================================"

# 创建输出目录
mkdir -p "${VISUAL_OUTPUT_DIR}"

# 3.1 可视化 GT (Ground Truth)
echo "可视化 Ground Truth..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python /new_data/cyf/projects/SceneSplat/tools/visualize_semantic_segmentation.py \
    --data_path "${SCENE_PATH}" \
    --mode gt \
    --output_path "${VISUAL_OUTPUT_DIR}/${SCENE_NAME}_gt.ply" \
    --dataset scannetpp

echo "  GT 可视化已保存: ${VISUAL_OUTPUT_DIR}/${SCENE_NAME}_gt.ply"

# 3.2 可视化预测结果
echo "可视化预测结果..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python /new_data/cyf/projects/SceneSplat/tools/visualize_semantic_segmentation.py \
    --data_path "${SCENE_PATH}" \
    --pred_path "${INFERENCE_OUTPUT_DIR}/${SCENE_NAME}/predictions.npy" \
    --mode pred \
    --output_path "${VISUAL_OUTPUT_DIR}/${SCENE_NAME}_pred.ply" \
    --dataset scannetpp

echo "  预测可视化已保存: ${VISUAL_OUTPUT_DIR}/${SCENE_NAME}_pred.ply"

# 3.3 对比可视化
echo "生成对比可视化..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python /new_data/cyf/projects/SceneSplat/tools/visualize_semantic_segmentation.py \
    --data_path "${SCENE_PATH}" \
    --pred_path "${INFERENCE_OUTPUT_DIR}/${SCENE_NAME}/predictions.npy" \
    --mode compare \
    --output_path "${VISUAL_OUTPUT_DIR}/${SCENE_NAME}_compare" \
    --dataset scannetpp

echo "  对比可视化已保存:"
echo "    - ${VISUAL_OUTPUT_DIR}/${SCENE_NAME}_compare_gt.ply"
echo "    - ${VISUAL_OUTPUT_DIR}/${SCENE_NAME}_compare_pred.ply"

# ============================================
# 完成
# ============================================
echo ""
echo "======================================"
echo "完成！"
echo "======================================"
echo ""
echo "输出文件:"
echo "  推理特征:  ${INFERENCE_OUTPUT_DIR}/${SCENE_NAME}/language_features.npy"
echo "  预测结果:  ${INFERENCE_OUTPUT_DIR}/${SCENE_NAME}/predictions.npy"
echo "  GT 可视化:  ${VISUAL_OUTPUT_DIR}/${SCENE_NAME}_gt.ply"
echo "  预测可视化:  ${VISUAL_OUTPUT_DIR}/${SCENE_NAME}_pred.ply"
echo "  对比可视化: ${VISUAL_OUTPUT_DIR}/${SCENE_NAME}_compare_*.ply"
echo ""
echo "使用 MeshLab 或 CloudCompare 查看 .ply 文件"
echo ""
