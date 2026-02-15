#!/bin/bash
# Script to run MCMC trainer with specified parameters
# Supports single scene and batch processing

# Default values
DATA_DIR=""
RESULT_DIR=""
DATASET_NAME="colmap"
SCENES=()
BATCH_MODE=false
CAP_MAX="1000000"
DEPTH_LOSS=""  # Disabled by default (COLMAP doesn't have depth)
SCALE_REG="0.02"
OPACITY_REG="0"
DATA_FACTOR="1"  # Downsample factor for the dataset
GPU_ID=0

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --data-dir DIR        Input data directory (required for batch mode)"
    echo "  -r, --result-dir DIR      Output result directory (required for batch mode)"
    echo "  -s, --scenes LIST         Comma-separated list of scenes (e.g., scene1,scene2,scene3)"
    echo "  --dataset-name NAME       Dataset name (default: colmap)"
    echo "  --cap-max NUM             Strategy cap max value (default: 1000000)"
    echo "  --scale-reg NUM           Scale regularization value (default: 0.02)"
    echo "  --opacity-reg NUM         Opacity regularization value (default: 0)"
    echo "  --data-factor NUM         Downsample factor for images (default: 1, e.g., 2=half resolution)"
    echo "  --depth-loss              Enable depth loss (only for datasets with depth)"
    echo "  --no-depth-loss           Disable depth loss"
    echo "  -g, --gpu-id              GPU ID to use (default: 0)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Single scene mode (direct args passed to simple_trainer.py):"
    echo "  $0 <data-dir> <result-dir> [additional_args...]"
    echo ""
    echo "Batch mode examples:"
    echo "  # Process all scenes in datasets/lerf_ovs"
    echo "  $0 -d datasets/lerf_ovs -r results/lerf_ovs"
    echo ""
    echo "  # Process specific scenes with 2x downsample"
    echo "  $0 -d datasets/lerf_ovs -r results/lerf_ovs -s scene1,scene2 --data-factor 2"
    echo ""
    echo "  # Process scenes from a file"
    echo "  $0 -d datasets/lerf_ovs -r results/lerf_ovs -f scenes_list.txt"
    exit 1
}

# Function to run training for a single scene
run_training() {
    local scene_data_dir=$1
    local scene_result_dir=$2
    shift 2  # Remove first 2 arguments, keep any additional args for simple_trainer.py

    local scene_name=$(basename "$scene_data_dir")

    echo "=========================================="
    echo "Training scene: $scene_name"
    echo "Data dir: $scene_data_dir"
    echo "Result dir: $scene_result_dir"
    echo "=========================================="

    # Create result directory if it doesn't exist
    mkdir -p "$scene_result_dir"

    # Change to 3dgs_optimization directory
    cd /new_data/cyf/projects/SceneSplat/3dgs_optimization

    # Run the training command
    CUDA_VISIBLE_DEVICES=$GPU_ID /data/cyf/.conda/envs/scene_splat/bin/python examples/simple_trainer.py \
        mcmc \
        --data-dir "$scene_data_dir" \
        --result-dir "$scene_result_dir" \
        --dataset-name "$DATASET_NAME" \
        --data-factor "$DATA_FACTOR" \
        --strategy.cap-max "$CAP_MAX" \
        $DEPTH_LOSS \
        --scale-reg "$SCALE_REG" \
        --opacity-reg "$OPACITY_REG" \
        "$@"  # Pass any additional arguments (only for single scene mode)

    echo "=========================================="
    echo "Finished training scene: $scene_name"
    echo "=========================================="
    echo ""
}

# Function to get all scenes from a directory
get_all_scenes() {
    local data_dir=$1
    if [ -d "$data_dir" ]; then
        find "$data_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort
    else
        echo "Warning: Data directory '$data_dir' not found" >&2
        return 1
    fi
}

# Parse command line arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
fi

# Check if this is batch mode (using flags) or single scene mode
if [[ "$1" == -* ]]; then
    # Batch mode with flags
    BATCH_MODE=true
    SCENES_FILE=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            -r|--result-dir)
                RESULT_DIR="$2"
                shift 2
                ;;
            -s|--scenes)
                IFS=',' read -ra SCENES <<< "$2"
                shift 2
                ;;
            --dataset-name)
                DATASET_NAME="$2"
                shift 2
                ;;
            --cap-max)
                CAP_MAX="$2"
                shift 2
                ;;
            --scale-reg)
                SCALE_REG="$2"
                shift 2
                ;;
            --opacity-reg)
                OPACITY_REG="$2"
                shift 2
                ;;
            --data-factor)
                DATA_FACTOR="$2"
                shift 2
                ;;
            --depth-loss)
                DEPTH_LOSS="--depth-loss"
                shift
                ;;
            --no-depth-loss)
                DEPTH_LOSS=""
                shift
                ;;
            -f|--scenes-file)
                SCENES_FILE="$2"
                shift 2
                ;;
            -g|--gpu-id)
                GPU_ID="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done

    # Load scenes from file if specified
    if [ -n "$SCENES_FILE" ]; then
        if [ -f "$SCENES_FILE" ]; then
            mapfile -t SCENES < <(cat "$SCENES_FILE" | grep -v '^#' | grep -v '^$')
        else
            echo "Error: Scenes file '$SCENES_FILE' not found"
            exit 1
        fi
    fi

    # Validate required arguments
    if [ -z "$DATA_DIR" ] || [ -z "$RESULT_DIR" ]; then
        echo "Error: --data-dir and --result-dir are required for batch mode"
        usage
    fi

    # If no scenes specified, get all scenes from data directory
    if [ ${#SCENES[@]} -eq 0 ]; then
        echo "No scenes specified, discovering all scenes in $DATA_DIR..."
        mapfile -t SCENES < <(get_all_scenes "$DATA_DIR")
        if [ ${#SCENES[@]} -eq 0 ]; then
            echo "Error: No scenes found in $DATA_DIR"
            exit 1
        fi
    fi

    echo "=========================================="
    echo "Batch Processing Mode"
    echo "Dataset: $DATASET_NAME"
    echo "Data directory: $DATA_DIR"
    echo "Result directory: $RESULT_DIR"
    echo "Data factor: $DATA_FACTOR (downsample)"
    echo "Scenes to process (${#SCENES[@]}):"
    for scene in "${SCENES[@]}"; do
        echo "  - $scene"
    done
    echo "=========================================="
    echo ""

    # Process each scene
    for scene in "${SCENES[@]}"; do
        scene_data_dir="$DATA_DIR/$scene"
        scene_result_dir="$RESULT_DIR/$scene"

        if [ ! -d "$scene_data_dir" ]; then
            echo "Warning: Scene directory '$scene_data_dir' not found, skipping..."
            continue
        fi

        run_training "$scene_data_dir" "$scene_result_dir"
    done

    echo "=========================================="
    echo "All batch processing completed!"
    echo "=========================================="

else
    # Single scene mode - pass all arguments to run_training
    if [ $# -lt 2 ]; then
        echo "Error: Single scene mode requires at least data_dir and result_dir"
        usage
    fi

    SCENE_DATA_DIR="$1"
    SCENE_RESULT_DIR="$2"
    shift 2  # Remove first two arguments, pass the rest

    run_training "$SCENE_DATA_DIR" "$SCENE_RESULT_DIR" "$@"
fi
