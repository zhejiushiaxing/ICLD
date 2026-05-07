from config.cfg import cfg
from core.model_infer import ModelInfer

import copy
import os
import argparse

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model inference script")

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["MMLU", "GSM8K", "MATH500", "NQ", "BOOLQ", "ARC-C", "PIQA"],
        help="Dataset name"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset file"
    )

    parser.add_argument(
        "--max_questions",
        type=int,
        default=50,
        help="Number of test questions"
    )

    return parser.parse_args()


def main():

    # ================== 0. 解析命令行参数 ==================
    args = parse_args()

    # 1.初始化推理引擎
    model_infer = ModelInfer()

    # 2.配置相关信息
    test_cfg = copy.deepcopy(cfg)

    """
    模型路径说明：
    """

    test_cfg['integrate_models_name'] = [
        # "Llama3.2-3B-Instruct",
        "Gemma2-2B-it", 
        "Qwen2.5-1.5B-Instruct"
    ]

    test_cfg['integrate_models_path'] = [
        # "/data3/zzc/projects/ZJX/models/models--Meta--Llama3.2-3B-Instruct",
        "/data3/zzc/projects/ZJX/models/models--Google-Gemma2-2B-it",
        "Qwen/Qwen2.5-1.5B-Instruct"
    ]

    # 自动映射模型名
    for i, model_path in enumerate(test_cfg['integrate_models_path']):
        if model_path == "openchat/openchat-3.6-8b-20240522":
            test_cfg['integrate_models_name'][i] = "openchat-3.6-8b-20240522"
        elif model_path in [
            "/data3/zzc/projects/ZJX/models/models--Meta--Llama3.1-8B-Instruct",
            "/mnt/Data/ZJX/models/huggingface_cache/models--Meta-Llama-3-8B-Instruct"
        ]:
            test_cfg['integrate_models_name'][i] = "Llama3.1-8B-Instruct"
        elif model_path == "tencent/Hunyuan-7B-Instruct":
            test_cfg['integrate_models_name'][i] = "Hunyuan-7B-Instruct"
        elif model_path == "/data3/zzc/projects/ZJX/models/models--Qwen--Qwen2.5-7B-Instruct":
            test_cfg['integrate_models_name'][i] = "Qwen2.5-7B-Instruct"
        elif model_path == "/data3/zzc/projects/ZJX/models/ models--Google-Gemma2-2B-it":
            test_cfg['integrate_models_name'][i] = "Gemma2-2B-it"
        elif model_path == "/data3/zzc/projects/ZJX/models/models--Meta--Llama3.2-3B-Instruct":
            test_cfg['integrate_models_name'][i] = "Llama3.2-3B-Instruct"
        elif model_path == "Qwen/Qwen3-4B-Instruct-2507":
            test_cfg['integrate_models_name'][i] = "Qwen3-4B-Instruct-2507"
        elif model_path == "tencent/Hunyuan-4B-Instruct":
            test_cfg['integrate_models_name'][i] = "Hunyuan-4B-Instruct"
        

        

    # ================== 关键修改点 ==================
    test_cfg['dataset_name'] = args.dataset_name
    test_cfg['dataset_path'] = args.dataset_path
    test_cfg['max_questions'] = args.max_questions
    # ============================================

    test_cfg['device'] = "cuda:2"
    test_cfg['integrate_models_device'] = [2, 2]
    test_cfg['integrate_models_weights'] = [0.5, 0.5]

    test_cfg['comment'] = f"测试Gemma2-2B-it，数据集：{args.dataset_name}"
    test_cfg['integrate_method'] = "ICLD"

    # 3.执行入口
    model_infer.mult_models_infer(test_cfg)


if __name__ == "__main__":
    main()
    # 使用方法
    # python -m EX.test_mult_models --dataset_name MMLU --dataset_path /data3/zzc/projects/ZJX/code/ICLD/dataset/MMLU/test-00000-of-00001.parquet --max_questions 1
    # python -m EX.test_mult_models --dataset_name GSM8K --dataset_path /data3/zzc/projects/ZJX/code/ICLD/dataset/GSM8K/test-00000-of-00001.parquet --max_questions 1
    # python -m EX.test_mult_models --dataset_name MATH500 --dataset_path /data3/zzc/projects/ZJX/code/ICLD/dataset/MATH500/test_clean.parquet --max_questions 1
    # python -m EX.test_mult_models --dataset_name PIQA --dataset_path /data3/zzc/projects/ZJX/code/ICLD/dataset/PIQA/validation-00000-of-00001.parquet --max_questions 1