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
        --integrate_models_path /data3/zzc/projects/ZJX/models/models--Google-Gemma2-2B-it /data3/zzc/projects/ZJX/models/models--Meta--Llama3.2-3B-Instruct tencent/Hunyuan-4B-Instruct \
        --device cuda:1 \
        --integrate_models_device 1 1 1 \
        --comment G+L+H-Unite

    echo "Finished dataset: ${DATASET_NAME}"
    echo
done

echo "All datasets finished."