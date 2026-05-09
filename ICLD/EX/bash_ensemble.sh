#!/bin/bash

set -e  # 出错立即退出
set -o pipefail

echo "===== Start Multi-Dataset Experiments ====="

# ================= 数据集配置 =================
datasets=("MMLU" "GSM8K" "MATH500" "PIQA" "ARC-C")

paths=(
"/data3/zzc/projects/ZJX/code/ICLD/dataset/MMLU/test-00000-of-00001.parquet"
"/data3/zzc/projects/ZJX/code/ICLD/dataset/GSM8K/test-00000-of-00001.parquet"
"/data3/zzc/projects/ZJX/code/ICLD/dataset/MATH500/test_clean.parquet"
"/data3/zzc/projects/ZJX/code/ICLD/dataset/PIQA/validation-00000-of-00001.parquet"
"/data3/zzc/projects/ZJX/code/ICLD/dataset/ARC-C/test-00000-of-00001.parquet"
)

max_questions_num=100

# ================= 日志目录 =================
LOG_DIR=logs
mkdir -p $LOG_DIR

# ================= 主循环 =================
for i in ${!datasets[@]}; do
    dataset=${datasets[$i]}
    path=${paths[$i]}

    echo "===== Running $dataset ====="

    log_file="$LOG_DIR/${dataset}_$(date +%Y%m%d_%H%M%S).log"

    python -m EX.test_mult_models --dataset_name $dataset --dataset_path $path --max_questions $max_questions_num 2>&1 | tee $log_file

    echo "===== Finished $dataset ====="
done

echo "===== All Experiments Finished ====="