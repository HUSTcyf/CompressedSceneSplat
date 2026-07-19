#!/bin/bash
# Benchmark compute_procrustes_Q_cuda_with_labels with all SVD suffixes (_1, _2, _3)
# and compute aggregate statistics.

# Python environment
PYTHON_PATH="/new_data/cyf/.conda/envs/scene_splat/bin/python"

# Paths
DATA_ROOT="/new_data/cyf/projects/SceneSplat/gaussian_train_clip/lerf_ovs/val"
TEXT_EMBED="/new_data/cyf/projects/OccamLGS/datasets/lerf_ovs_text_embeddings_clip_no_prefix.pt"
SVD_RANK=16
LABEL_FILE="lang_label.npy"
NUM_RUNS=1
WARMUP_RUNS=1

# Output directory
OUTPUT_DIR="/new_data/cyf/projects/SceneSplat/tools/analysis"
BASE_OUTPUT="${OUTPUT_DIR}/benchmark_clip_results"

# SVD suffixes to test
SVD_SUFFIXES=("_1" "_2" "_3")

echo "============================================================================================================"
echo "Procrustes Alignment Benchmark - compute_procrustes_Q_cuda_with_labels"
echo "============================================================================================================"
echo "Data root: ${DATA_ROOT}"
echo "Text embeddings: ${TEXT_EMBED}"
echo "SVD rank: ${SVD_RANK}"
echo "Label file: ${LABEL_FILE}"
echo "SVD suffixes: ${SVD_SUFFIXES[@]}"
echo "Runs per scene: ${NUM_RUNS}"
echo "Warmup runs: ${WARMUP_RUNS}"
echo "============================================================================================================"
echo ""

# Run benchmark for each suffix
for suffix in "${SVD_SUFFIXES[@]}"; do
    echo "============================================================================================================"
    echo "Running benchmark with SVD suffix: ${suffix}"
    echo "============================================================================================================"
    echo ""

    OUTPUT_FILE="${BASE_OUTPUT}_${suffix}.json"

    ${PYTHON_PATH} /new_data/cyf/projects/SceneSplat/tools/analysis/benchmark_procrustes_with_labels.py \
        --data_root "${DATA_ROOT}" \
        --text_embed "${TEXT_EMBED}" \
        --svd_rank ${SVD_RANK} \
        --svd_suffix "${suffix}" \
        --label_file "${LABEL_FILE}" \
        --num_runs ${NUM_RUNS} \
        --warmup_runs ${WARMUP_RUNS} \
        --output_file "${OUTPUT_FILE}"

    echo ""
    echo "Results saved to: ${OUTPUT_FILE}"
    echo ""
done

echo "============================================================================================================"
echo "ALL BENCHMARKS COMPLETED"
echo "============================================================================================================"
echo ""
echo "Results files:"
for suffix in "${SVD_SUFFIXES[@]}"; do
    echo "  ${BASE_OUTPUT}_${suffix}.json"
done
echo ""
echo "To compute aggregate statistics across all suffixes, run:"
echo "  ${PYTHON_PATH} -c \"import json, numpy as np; results = [json.load(open('${BASE_OUTPUT}_${s}.json')) for s in ['_1', '_2', '_3']]; print('Aggregate time:', np.mean([r['aggregate']['avg_time_mean_ms'] for r in results]), 'ms')\""
