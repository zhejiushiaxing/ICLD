#!/bin/bash

set -e

DATASETS=(
  MMLU
  GSM8K
  MATH500
)

for DATASET_NAME in "${DATASETS[@]}"
do
    echo "=========================================="
    echo "Running dataset: ${DATASET_NAME}"
    echo "=========================================="

    python -m EX.test_mult_models \
        --dataset_name "${DATASET_NAME}" \
        --integrate_models_path /data3/zzc/projects/ZJX/models/models--Qwen--Qwen2.5-7B-Instruct /data3/zzc/projects/ZJX/models/models--Meta--Llama3.1-8B-Instruct \
        --device cuda:0 \
        --integrate_models_device 0 0 \
        --comment Q7B+L8B-Unite

    echo "Finished dataset: ${DATASET_NAME}"
    echo
done

echo "All datasets finished."


