#!/bin/bash
# Evaluate LERF OVS scenes with rendered features from checkpoint_with_features_s.pth
# Features are rendered by tools/batch_process_feat.py using feature_map_renderer.py
#
# DEFAULT Configuration (768-dim SigLIP2):
#   - Model: SigLIP2 (ViT-B-16-SigLIP2-512, 768-dim)
#   - Feature Dimension: 768
#   - Feature Level: 0 (lang_feat_0.npy, full resolution)
#   - Procrustes Alignment: Disabled by default for 768-dim features
#
# OPTIONAL Evaluation Modes:
#   1. SVD-compressed (16-dim): Set SRC_DIM=16, USE_CLIP=true
#   2. CLIP (512-dim): Set SRC_DIM=512, USE_CLIP=true
#   3. Custom dimension: Set SRC_DIM and appropriate USE_CLIP flag

# Base paths
FEAT_BASE_PATH="/new_data/cyf/projects/OccamLGS/output/LERF"
GT_BASE_PATH="/new_data/cyf/projects/OccamLGS/datasets/lerf_ovs/label"
TRAIN_DATA_ROOT="/new_data/cyf/projects/SceneSplat/gaussian_train/lerf_ovs/train"
FEAT_BASE_PATH="/new_data/cyf/projects/SceneSplat/gaussian_results/lerf_ovs"

# Feature level to use (0, 1, 2, or 3)
# Note: The actual directories are named with level suffix (e.g., ours_30000_langfeat_1)
# So FEAT_FOLDER_NAME should include the level, and we DON'T pass --feat_level
# Level 0 = full resolution (default for 768-dim)
# Level 1-3 = downsampled resolutions (for SVD-compressed features)
FEAT_LEVEL=0

# Feature folder name (must match output from feature_map_renderer.py)
# Include the level suffix since directories are created as ours_30000_langfeat_{level}
FEAT_FOLDER_NAME="ours_30000_langfeat_${FEAT_LEVEL}"

# Source dimension (768 for SigLIP2 [DEFAULT], 512 for CLIP, 16 for SVD-compressed)
SRC_DIM=768

# Text embeddings path (optional, only needed for Procrustes alignment)
# For 768-dim SigLIP2: use lerf_ovs_text_embeddings_siglip2.pt
# For CLIP modes: use lerf_ovs_text_embeddings_clip_no_prefix.pt
TEXT_EMBEDDINGS_PATH="/new_data/cyf/projects/SceneSplat/datasets/lerf_ovs_text_embeddings_siglip2.pt"

# Evaluation parameters
STABILITY_THRESH=0.5  # Default stability threshold (relaxed from 0.3 to 0.5)
MIN_MASK_SIZE=0.001
MAX_MASK_SIZE=0.95

# Model options
USE_CLIP=false  # Use CLIP (OpenCLIPNetwork) when true, SigLIP2Network when false [DEFAULT: false for 768-dim]
USE_PROCRUSTES=false  # Use Procrustes alignment with per-scene Q matrix [DEFAULT: false for 768-dim]

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

    # Build command with optional flags
    CMD="CUDA_VISIBLE_DEVICES=0 python tools/eval/evaluate_iou_loc.py \
            --dataset_name ${DATASET_NAME} \
            --gt_folder ${GT_FOLDER} \
            --feat_folder ${FEAT_FOLDER_NAME} \
            --feat_base_path ${FEAT_BASE_PATH} \
            --single_level \
            --src_dim ${SRC_DIM} \
            --stability_thresh ${STABILITY_THRESH} \
            --min_mask_size ${MIN_MASK_SIZE} \
            --max_mask_size ${MAX_MASK_SIZE}"

    # Add optional flags
    if [ "$USE_CLIP" = true ]; then
        CMD="$CMD --use_clip"
    fi

    if [ "$USE_PROCRUSTES" = true ]; then
        CMD="$CMD --use_procrustes"
        CMD="$CMD --text_embeddings ${TEXT_EMBEDDINGS_PATH}"
        CMD="$CMD --train_data_root ${TRAIN_DATA_ROOT}"
    fi

    # Run the evaluation
    eval $CMD

    echo "✓ Completed: ${DATASET_NAME}"
    echo ""
done

echo "===== All evaluations completed ====="
echo "Results saved to: ./eval_results/LERF/{dataset_name}/"
echo ""
echo "===== Usage Notes ====="
echo "To use different evaluation modes, modify the variables at the top of this script:"
echo "  - 768-dim SigLIP2 (default): SRC_DIM=768, USE_CLIP=false, USE_PROCRUSTES=false"
echo "  - 512-dim CLIP:             SRC_DIM=512, USE_CLIP=true, USE_PROCRUSTES=true"
echo "  - 16-dim SVD-compressed:    SRC_DIM=16, USE_CLIP=true, USE_PROCRUSTES=true"
