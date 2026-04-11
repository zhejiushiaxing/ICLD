from config.cfg import cfg
from core.model_infer import ModelInfer

import copy
import os
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():

    # 1.初始化推理引擎
    model_infer = ModelInfer()

    # 2.配置相关信息
    test_cfg = copy.deepcopy(cfg)
    

    """
    /data3/zzc/projects/ZJX/models/models--Qwen--Qwen2.5-7B-Instruct
    /data3/zzc/projects/ZJX/models/models--Meta--Llama3.1-8B-Instruct
    /root/autodl-tmp/models/huggingface_cache/models--Google--gemma2-9B-Instruct
    mistralai/Mistral-7B-Instruct-v0.3(X)
    deepseek-ai/deepseek-coder-6.7b-instruct
    tencent/Hunyuan-7B-Instruct
    openchat/openchat-3.6-8b-20240522
    MergeBench/gemma-2-9b-it_instruction
    MergeBench/gemma-2-9b-it_multilingual
    MergeBench/gemma-2-9b-it_math
    MergeBench/gemma-2-9b-it_coding
    """

    test_cfg['integrate_models_name'] = ["Qwen2.5-7B-Instruct", "Llama3.1-8B-Instruct"]
    
    test_cfg['integrate_models_path'] = ["/data3/zzc/projects/ZJX/models/models--Meta--Llama3.1-8B-Instruct",
                                        "/data3/zzc/projects/ZJX/models/models--Qwen--Qwen2.5-7B-Instruct"]
    
    for i, model_path in enumerate(test_cfg['integrate_models_path']):
        if model_path == "openchat/openchat-3.6-8b-20240522":
            test_cfg['integrate_models_name'][i] = "openchat-3.6-8b-20240522"
        elif model_path == "/data3/zzc/projects/ZJX/models/models--Meta--Llama3.1-8B-Instruct" or model_path == "/mnt/Data/ZJX/models/huggingface_cache/models--Meta-Llama-3-8B-Instruct":
            test_cfg['integrate_models_name'][i] = "Llama3.1-8B-Instruct"
        elif model_path == "tencent/Hunyuan-7B-Instruct":
            test_cfg['integrate_models_name'][i] = "Hunyuan-7B-Instruct"
        elif model_path == "/data3/zzc/projects/ZJX/models/models--Qwen--Qwen2.5-7B-Instruct":
            test_cfg['integrate_models_name'][i] = "Qwen2.5-7B-Instruct"
        

    test_cfg['device'] = "cuda:1"
    test_cfg['integrate_models_device'] = [1, 2]
    test_cfg['integrate_models_weights'] = [0.5, 0.5]

    test_cfg['max_questions'] = 200 # 测试题目的数量
    
    test_cfg['dataset_name'] = "MMLU" # 选择数据集的名称 [MMLU GSM8K MATH500 NQ BOOLQ ARC-C HUMANEVAL MULTILINGUAL MATHEVAL CODINGEVAL] 
    test_cfg['integrate_method'] = "ICLD" # 模型集成方法 [BASELINE DEEPEN GAC UNITE ICLD]
    test_cfg['dataset_path'] = "/data3/zzc/projects/ZJX/code/ICLD/dataset/MMLU/all/test-00000-of-00001.parquet" # MMLU
    # test_cfg['dataset_path'] = "/data3/zzc/projects/ZJX/code/ICLD/dataset/GSM8K/test-00000-of-00001.parquet" # GSM8K
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/MATH500/test.parquet" # MATH500
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/NQ/train-00000-of-00001.parquet" # NQ
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/BOOLQ/validation-00000-of-00001.parquet" # BOOLQ
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/ARC-C/test-00000-of-00001.parquet" # ARC-C
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/HUMANEVAL/test-00000-of-00001.parquet" # HUMANEVAL
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/MULTILINGUAL/train-00000-of-00001.parquet" # MULTILINGUAL
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/MATHEVAL/train-00000-of-00001.parquet" # MATHEVAL
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/CODINGEVAL/train-00000-of-00001.parquet" # CODINGEVAL
    
    # 3.执行入口
    model_infer.mult_models_infer(test_cfg)
    
if __name__ == "__main__":
    main()