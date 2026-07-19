#!/bin/bash
# Evaluate 3DOVS scenes with rendered features from checkpoint_with_features_s.pth
# Features are rendered by tools/batch_process_feat.py using feature_map_renderer.py
#
# DEFAULT Configuration (768-dim SigLIP2):
#   - Model: SigLIP2 (ViT-B-16-SigLIP2-512, 768-dim)
#   - Feature Dimension: 768
#   - Three-view fusion: 3 rendered views per scene (_1, _2, _3)
#   - Encoder: SigLIP2Network (default)
#
# OPTIONAL Evaluation Modes:
#   1. CLIP (512-dim): Set ENCODER="openclip"
#   2. SigLIP2 (768-dim): Set ENCODER="siglip2" (default)

# Base paths
FEAT_BASE_PATH="/new_data/cyf/projects/SceneSplat/gaussian_results/3DOVS"
GT_BASE_PATH="/new_data/cyf/projects/OccamLGS/datasets/3DOVS"

# Name of the folder containing extracted features
# Should match output from feature rendering pipeline
FEAT_FOLDER_NAME="ours_30000_langfeat"

# Encoder type: "siglip2" for SigLIP2Network (768-dim), "openclip" for OpenCLIPNetwork (512-dim)
ENCODER="siglip2"

# Evaluation parameters
STABILITY_THRESH=0.4  # Stability threshold for mask selection
MIN_MASK_SIZE=0.005   # Minimum mask size as fraction of image
MAX_MASK_SIZE=0.9     # Maximum mask size as fraction of image

# CUDA device
CUDA_DEVICE=0

# 3DOVS datasets to evaluate
# Available scenes: sofa, table, chair, etc. (add more as needed)
DATASETS="bed bench lawn room sofa"

for DATASET_NAME in $DATASETS; do
    echo "===== Evaluating: ${DATASET_NAME} ====="

    GT_FOLDER="${GT_BASE_PATH}/${DATASET_NAME}"

    # Check if ground truth exists
    if [ ! -d "$GT_FOLDER" ]; then
        echo "WARNING: Ground truth folder not found: ${GT_FOLDER}"
        echo "Skipping ${DATASET_NAME}..."
        echo ""
        continue
    fi

    # Check if features exist for all three views
    FEAT_PATH_0="${FEAT_BASE_PATH}/${DATASET_NAME}/test/${FEAT_FOLDER_NAME}_0/renders_npy"
    FEAT_PATH_1="${FEAT_BASE_PATH}/${DATASET_NAME}/test/${FEAT_FOLDER_NAME}_1/renders_npy"
    FEAT_PATH_2="${FEAT_BASE_PATH}/${DATASET_NAME}/test/${FEAT_FOLDER_NAME}_2/renders_npy"
    FEAT_PATH_3="${FEAT_BASE_PATH}/${DATASET_NAME}/test/${FEAT_FOLDER_NAME}_3/renders_npy"

    # if [ ! -d "$FEAT_PATH_0" ] || [ ! -d "$FEAT_PATH_1" ] || [ ! -d "$FEAT_PATH_2" ] || [ ! -d "$FEAT_PATH_3" ]; then
    #     echo "ERROR: Feature folders not found for ${DATASET_NAME}:"
    #     echo "  View 0: ${FEAT_PATH_0}"
    #     echo "  View 1: ${FEAT_PATH_1}"
    #     echo "  View 2: ${FEAT_PATH_2}"
    #     echo "  View 3: ${FEAT_PATH_3}"
    #     echo "Please run feature rendering pipeline first to generate all three views."
    #     echo ""
    #     continue
    # fi

    echo "Features: ${FEAT_BASE_PATH}/${DATASET_NAME}/test/${FEAT_FOLDER_NAME}_{0,1,2,3}/renders_npy"
    echo "Ground truth: ${GT_FOLDER}"
    echo "Encoder: ${ENCODER}"
    echo "Stability threshold: ${STABILITY_THRESH}"
    echo "Min mask size: ${MIN_MASK_SIZE}"
    echo "Max mask size: ${MAX_MASK_SIZE}"
    echo ""

    # Build command
    CMD="CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python tools/eval/evaluate_iou_3dovs.py \
            --dataset_name ${DATASET_NAME} \
            --gt_folder ${GT_FOLDER} \
            --feat_folder ${FEAT_FOLDER_NAME} \
            --encoder ${ENCODER} \
            --stability_thresh ${STABILITY_THRESH} \
            --min_mask_size ${MIN_MASK_SIZE} \
            --max_mask_size ${MAX_MASK_SIZE}"

    # Run the evaluation
    eval $CMD

    if [ $? -eq 0 ]; then
        echo "✓ Completed: ${DATASET_NAME}"
    else
        echo "✗ Failed: ${DATASET_NAME}"
    fi
    echo ""
done

echo "===== All evaluations completed ====="
echo "Results saved to: ./eval_results/3DOVS/{dataset_name}/"
echo ""
echo "===== Usage Notes ====="
echo "To use different evaluation modes, modify the variables at the top of this script:"
echo "  - SigLIP2 (768-dim, default): ENCODER=\"siglip2\""
echo "  - CLIP (512-dim):             ENCODER=\"openclip\""
echo ""
echo "To evaluate specific datasets, modify the DATASETS variable:"
echo "  DATASETS=\"sofa table chair\"  # Evaluate all three"
echo "  DATASETS=\"sofa\"              # Evaluate only sofa"
echo ""
