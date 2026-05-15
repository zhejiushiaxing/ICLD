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
        --integrate_models_path /data3/zzc/projects/ZJX/models/models--Meta--Llama3.1-8B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
        --device cuda:1 \
        --integrate_models_device 1 1 \
        --comment Llama3.1-8B-Instruct-test

    echo "Finished dataset: ${DATASET_NAME}"
    echo
done

echo "All datasets finished."