#!/bin/bash
# Evaluate LERF OVS scenes with rendered features from checkpoint_with_features_s.pth
# Features are rendered by tools/batch_process_feat.py using feature_map_renderer.py

# Base paths
FEAT_BASE_PATH="/new_data/cyf/projects/SceneSplat/gaussian_results/lerf_ovs"
GT_BASE_PATH="/new_data/cyf/projects/OccamLGS/datasets/lerf_ovs/label"

# Feature folder name (must match output from feature_map_renderer.py)
FEAT_FOLDER_NAME="ours_30000_langfeat_0"

# Source dimension (768 for SigLIP2 features from checkpoint_with_features_s.pth)
SRC_DIM=768

# Evaluation parameters
STABILITY_THRESH=0.3
MIN_MASK_SIZE=0.001
MAX_MASK_SIZE=0.95

for DATASET_NAME in figurines ramen teatime waldo_kitchen; do
    echo "===== Evaluating: ${DATASET_NAME} ====="

    GT_FOLDER="${GT_BASE_PATH}/${DATASET_NAME}"

    # Check if ground truth exists
    if [ ! -d "$GT_FOLDER" ]; then
        echo "ERROR: Ground truth folder not found: ${GT_FOLDER}"
        continue
    fi

    # Check if features exist
    FEAT_PATH="${FEAT_BASE_PATH}/${DATASET_NAME}/test/${FEAT_FOLDER_NAME}/renders_npy"
    if [ ! -d "$FEAT_PATH" ]; then
        echo "ERROR: Feature folder not found: ${FEAT_PATH}"
        echo "Please run tools/batch_process_feat.py first to render features."
        continue
    fi

    echo "Features: ${FEAT_PATH}"
    echo "Ground truth: ${GT_FOLDER}"

    CUDA_VISIBLE_DEVICES=0 python tools/eval/evaluate_iou_loc.py \
            --dataset_name ${DATASET_NAME} \
            --gt_folder ${GT_FOLDER} \
            --feat_folder ${FEAT_FOLDER_NAME} \
            --feat_base_path ${FEAT_BASE_PATH} \
            --single_level \
            --src_dim ${SRC_DIM} \
            --stability_thresh ${STABILITY_THRESH} \
            --min_mask_size ${MIN_MASK_SIZE} \
            --max_mask_size ${MAX_MASK_SIZE}

    echo "✓ Completed: ${DATASET_NAME}"
    echo ""
done

echo "===== All evaluations completed ====="
echo "Results saved to: ./eval_results/LERF/{dataset_name}/"
