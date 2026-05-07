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
    mistralai/Mistral-7B-Instruct-v0.3
    tencent/Hunyuan-7B-Instruct
    zai-org/glm-4-9b-chat
    openchat/openchat-3.6-8b-20240522
    MergeBench/gemma-2-9b-it_instruction
    MergeBench/gemma-2-9b-it_multilingual
    MergeBench/gemma-2-9b-it_math
    MergeBench/gemma-2-9b-it_coding
    """

    test_cfg['max_questions'] = 200 # 测试题目的数量
    test_cfg['device'] = "cuda:2"

    test_cfg['single_model_name'] = "Llama3.1-8B-Instruct"
    test_cfg['single_model_path'] = "/data3/zzc/projects/ZJX/models/models--Meta--Llama3.1-8B-Instruct"
    test_cfg['dataset_name'] = "MMLU" # 选择数据集的名称 [MMLU GSM8K NQ BOOLQ ARC-C HUMANEVAL MULTILINGUAL MATHEVAL CODINGEVAL] 
    test_cfg['dataset_path'] = "/data3/zzc/projects/ZJX/code/ICLD/dataset/MMLU/all/test-00000-of-00001.parquet" # MMLU
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/GSM8K/test-00000-of-00001.parquet" # GSM8K
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/NQ/train-00000-of-00001.parquet" # NQ
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/BOOLQ/validation-00000-of-00001.parquet" # BOOLQ
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/ARC-C/test-00000-of-00001.parquet" # ARC-C
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/HUMANEVAL/test-00000-of-00001.parquet" # HUMANEVAL
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/MULTILINGUAL/train-00000-of-00001.parquet" # MULTILINGUAL
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/MATHEVAL/train-00000-of-00001.parquet" # MATHEVAL
    # test_cfg['dataset_path'] = "/mnt/Data/ZJX/code/ICLD/dataset/CODINGEVAL/train-00000-of-00001.parquet" # CODINGEVAL
    
    # 3.执行入口
    model_infer.single_model_infer(test_cfg)
        
    
    




if __name__ == "__main__":
    main()