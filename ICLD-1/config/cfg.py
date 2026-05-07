import json

# 多模型协作配置文件
cfg = {
        # 文件路径
        "dataset_name": None, # [MMLU, GSM8K, NQ]
        "dataset_path": None,
        "output_path": None,
        "single_model_output_root": "output/single_model",
        "mult_models_output_root": "output/mult_models",
        "per_layer_acc_output_root": "output/per_layer_models",
        
        # 单模型
        "single_model_name": None,
        "single_model_path": None, 
        # 多模型
        "integrate_method": "BASELINE", # [BASELINE, DEEPEN, GAC, UNITE, ICLD]
        "integrate_models_name": [],
        "integrate_models_path": [],
        "integrate_models_device": [0, 1, 2],
        "integrate_models_weights": [0.33, 0.33, 0.33],

        # 加载模型参数
        "dtype": "bfloat16",
        "device": "cuda:2",
        "batch_size": 1,

        # 模型推理时的相关参数
        "args": {
            'max_new_tokens': 1024, # 最多生成多少个新Token
            'top_k': 5,
            'top_p': 0.75,
            'temperature': 1,
        },


        # 测试题目的数量
        "max_questions": 100,
        "comment": " ",
        

    }

