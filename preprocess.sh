#!/bin/bash

# Preprocess LERF dataset scenes for SigLIP2 + SAM2 feature extraction
# Usage: ./preprocess.sh [scene_name]
# Examples:
#   ./preprocess.sh              # Process all LERF scenes
#   ./preprocess.sh figurines    # Process only figurines scene

# Configuration
DATASET_BASE="/new_data/cyf/projects/mini-splatting2/data/lerf_ovs"
DATASET_BASE="/new_data/cyf/projects/mini-splatting2/data/3DOVS"
SAM2_MODEL_PATH="./sam2_repo/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG="sam2.1_hiera_l.yaml"  # Script will auto-detect from model path

# LERF dataset scenes
SCENES=("figurines" "ramen" "teatime" "waldo_kitchen")
SCENES=("bed" "bench" "lawn" "room" "sofa")

# If scene name is provided, only process that scene
if [ -n "$1" ]; then
    SCENES=("$1")
fi

echo "=========================================="
echo "LERF Dataset SigLIP2 + SAM2 Preprocessing"
echo "=========================================="
echo "Dataset base: ${DATASET_BASE}"
echo "SAM2 model: ${SAM2_MODEL_PATH}"
echo "Scenes to process: ${SCENES[@]}"
echo ""

# Check if SAM2 model exists
if [ ! -f "${SAM2_MODEL_PATH}" ]; then
    echo "Error: SAM2 model not found at ${SAM2_MODEL_PATH}"
    echo ""
    echo "Please download SAM2 model first:"
    echo "  wget -P ./sam2/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pth"
    echo ""
    exit 1
fi

# Process each scene
for scene in "${SCENES[@]}"; do
    SCENE_PATH="${DATASET_BASE}/${scene}"

    echo "----------------------------------------"
    echo "Processing: ${scene}"
    echo "Path: ${SCENE_PATH}"
    echo ""

    # Check if scene exists
    if [ ! -d "${SCENE_PATH}" ]; then
        echo "Warning: Scene directory not found, skipping: ${SCENE_PATH}"
        echo ""
        continue
    fi

    # Check if images folder exists
    if [ ! -d "${SCENE_PATH}/images" ]; then
        echo "Warning: images folder not found, skipping: ${SCENE_PATH}/images"
        echo ""
        continue
    fi

    # Count images
    NUM_IMAGES=$(ls -1 "${SCENE_PATH}/images"/*.jpg 2>/dev/null | wc -l)
    if [ $NUM_IMAGES -eq 0 ]; then
        echo "Warning: No images found in ${SCENE_PATH}/images"
        echo ""
        continue
    fi

    echo "Found ${NUM_IMAGES} images"

    # Run preprocessing
    python scripts/preprocess_siglip2_sam2.py \
        --dataset_path "${SCENE_PATH}" \
        --sam2_model_path "${SAM2_MODEL_PATH}" \
        --sam2_config "${SAM2_CONFIG}" \
        --resolution -1 \
        --device cuda \
        --seed 42

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Completed: ${scene}"
    else
        echo "✗ Failed: ${scene} (exit code: ${EXIT_CODE})"
    fi

    echo ""
done

echo "=========================================="
echo "Preprocessing complete!"
echo "=========================================="
echo ""
echo "Output locations:"
for scene in "${SCENES[@]}"; do
    SCENE_PATH="${DATASET_BASE}/${scene}"
    OUTPUT_PATH="${SCENE_PATH}/language_features_siglip2_sam2"
    if [ -d "${OUTPUT_PATH}" ]; then
        NUM_FEATURES=$(ls -1 "${OUTPUT_PATH}"/*.npy 2>/dev/null | wc -l)
        echo "  ${scene}: ${OUTPUT_PATH} (${NUM_FEATURES} files)"
    fi
done
