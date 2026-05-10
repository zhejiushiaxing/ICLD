#!/bin/bash

set -e

DATASETS=(
  MMLU
  GSM8K
  MATH500
  PIQA
  ARC-C
)

for DATASET_NAME in "${DATASETS[@]}"
do
    echo "=========================================="
    echo "Running dataset: ${DATASET_NAME}"
    echo "=========================================="

    python -m EX.test_mult_models \
        --dataset_name "${DATASET_NAME}" \
        --integrate_models_path tencent/Hunyuan-4B-Instruct Qwen/Qwen2.5-3B-Instruct \
        --device cuda:1 \
        --integrate_models_device 1 1 \
        --comment H+Q-Unite

    echo "Finished dataset: ${DATASET_NAME}"
    echo
done

echo "All datasets finished."