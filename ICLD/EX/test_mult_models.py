from config.cfg import cfg
from core.model_infer import ModelInfer

import copy
import os
import argparse

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
        "--integrate_models_path",
        type=str,
        nargs="+",
        required=True,
        help="Integrated model paths. Example: Qwen/Qwen2.5-1.5B-Instruct /path/to/Llama3.2-3B-Instruct"
    )

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Main device. Example: cuda:2"
    )

    parser.add_argument(
        "--integrate_models_device",
        type=int,
        nargs="+",
        required=True,
        help="Device ids for each integrated model. Example: 2 2"
    )

    # 方差阈值
    parser.add_argument(
        "--var_threshold",
        type=float,
        default=None
    )

    # 语义相似度阈值
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=None
    )

    # 备注信息
    parser.add_argument(
        "--comment",
        type=str,
        required=True,
        help="Experiment comment. Example: Q1.5B+L3B-Unite"
    )

    # 可选参数：如果不传，则默认等权重
    parser.add_argument(
        "--integrate_models_weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for each integrated model. Example: 0.5 0.5"
    )

    # 测试题目
    parser.add_argument(
        "--max_questions",
        type=int,
        default=100,
        help="Number of test questions"
    )

    return parser.parse_args()


def get_dataset_path(dataset_name):
    """
    根据数据集名称自动返回数据集路径。
    """

    dataset_path_map = {
        "MMLU": "/data3/zzc/projects/ZJX/code/ICLD/dataset/MMLU/test-00000-of-00001.parquet",
        "GSM8K": "/data3/zzc/projects/ZJX/code/ICLD/dataset/GSM8K/test-00000-of-00001.parquet",
        "MATH500": "/data3/zzc/projects/ZJX/code/ICLD/dataset/MATH500/test_clean.parquet",
        "PIQA": "/data3/zzc/projects/ZJX/code/ICLD/dataset/PIQA/validation-00000-of-00001.parquet",
        "ARC-C": "/data3/zzc/projects/ZJX/code/ICLD/dataset/ARC-C/test-00000-of-00001.parquet",
    }

    if dataset_name not in dataset_path_map:
        raise ValueError(
            f"当前未配置数据集 {dataset_name} 的 dataset_path，请在 get_dataset_path() 中补充。"
        )

    return dataset_path_map[dataset_name]


def infer_model_name(model_path):
    """
    根据模型路径自动推断模型名称。
    """

    model_name_map = {
        "/data3/zzc/projects/ZJX/models/models--Google-Gemma2-2B-it": "Gemma2-2B-it",
        "/data3/zzc/projects/ZJX/models/models--Google-Gemma3-4B-it": "Gemma3-4B-it",
        "/data3/zzc/projects/ZJX/models/models--Google-Gemma4-E2B-it": "Gemma4-E2B-it",
        "/data3/zzc/projects/ZJX/models/models--Meta--Llama3.2-3B-Instruct": "Llama3.2-3B-Instruct",
        "/data3/zzc/projects/ZJX/models/models--Meta--Llama3.1-8B-Instruct": "Llama3.1-8B-Instruct",
        "/data3/zzc/projects/ZJX/models/models--gemma-2-9b-it": "gemma-2-9b-it",
        "tencent/Hunyuan-4B-Instruct": "Hunyuan-4B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen3.5-4B": "Qwen3.5-4B",
        "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
        "/data3/zzc/projects/ZJX/models/models--Qwen--Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
        "/data3/zzc/projects/ZJX/models/models--Mistral-7B-Instruct": "Mistral-7B-Instruct",
        "/data3/zzc/projects/ZJX/models/models--Microsoft--Phi3.5-mini-Instruct": "Phi3.5-mini-Instruct",
        "/data3/zzc/projects/ZJX/models/models--Microsoft--Phi4-mini-Instruct": "Phi4-mini-Instruct",
        "/data3/zzc/projects/ZJX/models/models--MinistralAI-Ministral3-3B-Instruct-2512": "Ministral3-3B-Instruct-2512",
    }

    if model_path in model_name_map:
        return model_name_map[model_path]

    # 如果没有在映射表中找到，则自动从路径中提取最后一级名称
    # 例如：
    # Qwen/Qwen2.5-1.5B-Instruct -> Qwen2.5-1.5B-Instruct
    # /xxx/models--Meta--Llama3.2-3B-Instruct -> Llama3.2-3B-Instruct
    model_name = os.path.basename(model_path.rstrip("/"))

    if model_name.startswith("models--"):
        parts = model_name.split("--")
        model_name = parts[-1]

    return model_name


def check_args(args):
    """
    检查运行时参数是否合法。
    """

    num_models = len(args.integrate_models_path)

    if len(args.integrate_models_device) != num_models:
        raise ValueError(
            f"参数数量不一致：integrate_models_path 有 {num_models} 个，"
            f"integrate_models_device 有 {len(args.integrate_models_device)} 个。"
        )

    if args.integrate_models_weights is not None:
        if len(args.integrate_models_weights) != num_models:
            raise ValueError(
                f"参数数量不一致：integrate_models_path 有 {num_models} 个，"
                f"integrate_models_weights 有 {len(args.integrate_models_weights)} 个。"
            )


def main():

    # ================== 0. 解析命令行参数 ==================
    args = parse_args()
    check_args(args)

    # ================== 1. 初始化推理引擎 ==================
    model_infer = ModelInfer()

    # ================== 2. 配置相关信息 ==================
    test_cfg = copy.deepcopy(cfg)

    # ================== 数据集配置 ==================
    test_cfg["dataset_name"] = args.dataset_name
    test_cfg["dataset_path"] = get_dataset_path(args.dataset_name)
    test_cfg["max_questions"] = args.max_questions

    # ================== 模型配置：由运行时参数传入 ==================
    test_cfg["integrate_models_path"] = list(args.integrate_models_path)

    test_cfg["integrate_models_name"] = [
        infer_model_name(model_path)
        for model_path in test_cfg["integrate_models_path"]
    ]

    # ================== 设备配置：由运行时参数传入 ==================
    test_cfg["device"] = args.device
    test_cfg["integrate_models_device"] = list(args.integrate_models_device)

    # ================== 权重配置 ==================
    if args.integrate_models_weights is not None:
        test_cfg["integrate_models_weights"] = list(args.integrate_models_weights)
    else:
        num_models = len(test_cfg["integrate_models_path"])
        test_cfg["integrate_models_weights"] = [
            1.0 / num_models for _ in range(num_models)
        ]

    # ================== 其他配置 ==================
    test_cfg["comment"] = args.comment
    test_cfg["integrate_method"] = "ICLD"

    # ================== 参数选取 ==================
    if getattr(args, "var_threshold", None) is not None:
        test_cfg["var_threshold"] = args.var_threshold
    if getattr(args, "sim_threshold", None) is not None:
        test_cfg["sim_threshold"] = args.sim_threshold

    # ================== 打印当前配置，方便检查 ==================
    print("========== Runtime Config ==========")
    print("dataset_name:", test_cfg["dataset_name"])
    print("dataset_path:", test_cfg["dataset_path"])
    print("max_questions:", test_cfg["max_questions"])
    print("integrate_models_name:", test_cfg["integrate_models_name"])
    print("integrate_models_path:", test_cfg["integrate_models_path"])
    print("device:", test_cfg["device"])
    print("integrate_models_device:", test_cfg["integrate_models_device"])
    print("integrate_models_weights:", test_cfg["integrate_models_weights"])
    print("comment:", test_cfg["comment"])
    print("====================================")

    # ================== 3. 执行入口 ==================
    model_infer.mult_models_infer(test_cfg)


if __name__ == "__main__":
    main()

    """
    python -m EX.test_mult_models \
    --dataset_name PIQA \
    --integrate_models_path /data3/zzc/projects/ZJX/models/models--Qwen--Qwen2.5-7B-Instruct /data3/zzc/projects/ZJX/models/models--Meta--Llama3.1-8B-Instruct \
    --device cuda:2 \
    --integrate_models_device 2 2 \
    --comment Q7B+L8B-Unite_0.001_0.6 \
    --var_threshold 0.001 \
    --sim_threshold 0.6
    """