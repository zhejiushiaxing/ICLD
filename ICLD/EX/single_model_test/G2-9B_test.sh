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
        --integrate_models_path /data3/zzc/projects/ZJX/models/models--gemma-2-9b-it Qwen/Qwen2.5-1.5B-Instruct \
        --device cuda:2 \
        --integrate_models_device 2 2 \
        --comment gemma-2-9b-it-test

    echo "Finished dataset: ${DATASET_NAME}"
    echo
done

echo "All datasets finished."