#!/bin/bash

set -e
set -o pipefail

# ================= 数据集配置 =================
DATASETS=(
  MMLU
  GSM8K
  PIQA
  ARC-C
)

# ================= 参数搜索空间 =================
VAR_THRESHOLDS=(
  1
  0.1
  0.01
  0.001
)

SIM_THRESHOLDS=(
  0.4
  0.5
  0.6
)

# ================= 模型配置 =================
MODEL_1="tencent/Hunyuan-4B-Instruct"
MODEL_2="/data3/zzc/projects/ZJX/models/models--Google-Gemma3-4B-it"

DEVICE="cuda:0"
INTEGRATE_DEVICES="0 0"

# ================= 实验主循环 =================
for DATASET_NAME in "${DATASETS[@]}"
do
    for VAR_THRESHOLD in "${VAR_THRESHOLDS[@]}"
    do
        for SIM_THRESHOLD in "${SIM_THRESHOLDS[@]}"
        do
            COMMENT="H+Q-Unite-${DATASET_NAME}-var${VAR_THRESHOLD}-sim${SIM_THRESHOLD}"

            echo "=========================================="
            echo "Running dataset: ${DATASET_NAME}"
            echo "var_threshold: ${VAR_THRESHOLD}"
            echo "sim_threshold: ${SIM_THRESHOLD}"
            echo "comment: ${COMMENT}"
            echo "=========================================="

            python -m EX.test_mult_models \
                --dataset_name "${DATASET_NAME}" \
                --integrate_models_path "${MODEL_1}" "${MODEL_2}" \
                --device "${DEVICE}" \
                --integrate_models_device ${INTEGRATE_DEVICES} \
                --comment "${COMMENT}" \
                --var_threshold "${VAR_THRESHOLD}" \
                --sim_threshold "${SIM_THRESHOLD}"

            echo "Finished dataset: ${DATASET_NAME}, var_threshold=${VAR_THRESHOLD}, sim_threshold=${SIM_THRESHOLD}"
            echo
        done
    done
done

echo "All datasets and parameter combinations finished."