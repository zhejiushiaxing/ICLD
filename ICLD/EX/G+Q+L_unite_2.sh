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
        --integrate_models_path /data3/zzc/projects/ZJX/models/models--Google-Gemma2-2B-it Qwen/Qwen2.5-1.5B-Instruct /data3/zzc/projects/ZJX/models/models--Meta--Llama3.2-3B-Instruct \
        --device cuda:2 \
        --integrate_models_device 2 2 2 \
        --comment G+Q+L-Unite

    echo "Finished dataset: ${DATASET_NAME}"
    echo
done

echo "All datasets finished."