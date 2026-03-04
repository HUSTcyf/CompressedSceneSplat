#!/bin/bash
# Evaluate LERF OVS scenes with rendered features from checkpoint_with_features_s.pth
# Features are rendered by tools/batch_process_feat.py using feature_map_renderer.py
#
# Configuration:
#   - Model: CLIP (OpenCLIP ViT-B-16, laion2b_s34b_b88k, 512-dim)
#   - Text Embeddings: lerf_ovs_text_embeddings_clip_no_prefix.pt (no "this is a" prefix)
#   - Feature Level: 1 (lang_feat_1.npy)
#   - Procrustes Alignment: Per-scene Q matrix computed at runtime using training data

# Base paths
FEAT_BASE_PATH="/new_data/cyf/projects/OccamLGS/output/LERF"
GT_BASE_PATH="/new_data/cyf/projects/OccamLGS/datasets/lerf_ovs/label"
TRAIN_DATA_ROOT="/new_data/cyf/projects/SceneSplat/gaussian_train_clip/lerf_ovs/train"

# Feature level to use (0, 1, 2, or 3)
# Note: The actual directories are named with level suffix (e.g., ours_30000_langfeat_1)
# So FEAT_FOLDER_NAME should include the level, and we DON'T pass --feat_level
FEAT_LEVEL=1

# Feature folder name (must match output from feature_map_renderer.py)
# Include the level suffix since directories are created as ours_30000_langfeat_{level}
FEAT_FOLDER_NAME="ours_30000_langfeat_${FEAT_LEVEL}"

# Source dimension (512 for CLIP OpenCLIP, 768 for SigLIP2, 16 for SVD-compressed)
SRC_DIM=16

# Text embeddings path (CLIP 512-dim, no prefix for compatibility with LangSplat)
TEXT_EMBEDDINGS_PATH="/new_data/cyf/projects/SceneSplat/datasets/lerf_ovs_text_embeddings_clip_no_prefix.pt"

# Evaluation parameters
STABILITY_THRESH=0.3
MIN_MASK_SIZE=0.001
MAX_MASK_SIZE=0.95

# Model options
USE_CLIP=true  # Use CLIP (OpenCLIPNetwork) instead of SigLIP2Network
USE_PROCRUSTES=true  # Use Procrustes alignment with per-scene Q matrix

for DATASET_NAME in figurines ramen teatime waldo_kitchen; do
    echo "===== Evaluating: ${DATASET_NAME} ====="

    GT_FOLDER="${GT_BASE_PATH}/${DATASET_NAME}"

    # Check if ground truth exists
    if [ ! -d "$GT_FOLDER" ]; then
        echo "ERROR: Ground truth folder not found: ${GT_FOLDER}"
        continue
    fi

    # Check if text embeddings exist
    if [ ! -f "$TEXT_EMBEDDINGS_PATH" ]; then
        echo "ERROR: Text embeddings not found: ${TEXT_EMBEDDINGS_PATH}"
        echo "Please run: python scripts/regenerate_text_embeddings.py --dataset lerf_ovs --output $TEXT_EMBEDDINGS_PATH --model clip --no_prefix"
        continue
    fi

    # Check if features exist
    FEAT_PATH="${FEAT_BASE_PATH}/${DATASET_NAME}/test/${FEAT_FOLDER_NAME}/renders_npy"
    if [ ! -d "$FEAT_PATH" ]; then
        echo "ERROR: Feature folder not found: ${FEAT_PATH}"
        echo "Please run tools/batch_process_feat.py first to render features."
        continue
    fi

    # Check if training data exists for Q matrix computation
    TRAIN_SCENE_PATH="${TRAIN_DATA_ROOT}/${DATASET_NAME}"
    if [ ! -d "$TRAIN_SCENE_PATH" ]; then
        echo "ERROR: Training data not found: ${TRAIN_SCENE_PATH}"
        continue
    fi

    echo "Features: ${FEAT_PATH}"
    echo "Ground truth: ${GT_FOLDER}"
    echo "Feature level: ${FEAT_LEVEL}"
    echo "Source dimension: ${SRC_DIM}"
    echo "Using CLIP: ${USE_CLIP}"
    echo "Using Procrustes: ${USE_PROCRUSTES}"
    echo "Text embeddings: ${TEXT_EMBEDDINGS_PATH}"
    echo "Training data root: ${TRAIN_DATA_ROOT}"

    CUDA_VISIBLE_DEVICES=0 python tools/eval/evaluate_iou_loc.py \
            --dataset_name ${DATASET_NAME} \
            --gt_folder ${GT_FOLDER} \
            --feat_folder ${FEAT_FOLDER_NAME} \
            --feat_base_path ${FEAT_BASE_PATH} \
            --single_level \
            --src_dim ${SRC_DIM} \
            --use_clip \
            --use_procrustes \
            --text_embeddings ${TEXT_EMBEDDINGS_PATH} \
            --train_data_root ${TRAIN_DATA_ROOT} \
            --stability_thresh ${STABILITY_THRESH} \
            --min_mask_size ${MIN_MASK_SIZE} \
            --max_mask_size ${MAX_MASK_SIZE}

    echo "✓ Completed: ${DATASET_NAME}"
    echo ""
done

echo "===== All evaluations completed ====="
echo "Results saved to: ./eval_results/LERF/{dataset_name}/"
