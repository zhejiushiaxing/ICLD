#!/bin/bash

set -e

python -m EX.test_mult_models \
  --dataset_name ARC-C \
  --integrate_models_path /data3/zzc/projects/ZJX/models/models--Google-Gemma2-2B-it Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda:2 \
  --integrate_models_device 2 2 \
  --comment G+Q-Unite

python -m EX.test_mult_models \
  --dataset_name ARC-C \
  --integrate_models_path Qwen/Qwen2.5-1.5B-Instruct /data3/zzc/projects/ZJX/models/models--Google-Gemma2-2B-it \
  --device cuda:2 \
  --integrate_models_device 2 2 \
  --comment Q+G-Unite

python -m EX.test_mult_models \
  --dataset_name ARC-C \
  --integrate_models_path /data3/zzc/projects/ZJX/models/models--Meta--Llama3.2-3B-Instruct tencent/Hunyuan-4B-Instruct \
  --device cuda:2 \
  --integrate_models_device 2 2 \
  --comment L+H-Unite

python -m EX.test_mult_models \
  --dataset_name ARC-C \
  --integrate_models_path tencent/Hunyuan-4B-Instruct /data3/zzc/projects/ZJX/models/models--Meta--Llama3.2-3B-Instruct \
  --device cuda:2 \
  --integrate_models_device 2 2 \
  --comment H+L-Unite

python -m EX.test_mult_models \
  --dataset_name ARC-C \
  --integrate_models_path tencent/Hunyuan-4B-Instruct /data3/zzc/projects/ZJX/models/models--Google-Gemma3-4B-it \
  --device cuda:1 \
  --integrate_models_device 1 1 \
  --comment H+G-Unite

python -m EX.test_mult_models \
  --dataset_name ARC-C \
  --integrate_models_path /data3/zzc/projects/ZJX/models/models--Google-Gemma3-4B-it tencent/Hunyuan-4B-Instruct \
  --device cuda:2 \
  --integrate_models_device 2 2 \
  --comment G+H-Unite

python -m EX.test_mult_models \
  --dataset_name ARC-C \
  --integrate_models_path Qwen/Qwen2.5-3B-Instruct /data3/zzc/projects/ZJX/models/models--Google-Gemma3-4B-it \
  --device cuda:2 \
  --integrate_models_device 2 2 \
  --comment Q+G-Unite

python -m EX.test_mult_models \
  --dataset_name ARC-C \
  --integrate_models_path /data3/zzc/projects/ZJX/models/models--Google-Gemma3-4B-it Qwen/Qwen2.5-3B-Instruct \
  --device cuda:2 \
  --integrate_models_device 2 2 \
  --comment G+Q-Unite