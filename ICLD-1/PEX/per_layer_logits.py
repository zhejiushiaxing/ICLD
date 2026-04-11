from core import prompt_generate as pg
from config.cfg import cfg
from utils.load_model import load_model
from utils.load_data import load_dataset
from utils.wnds_client_coll import setOpenAi
from utils.check_early_stop import check_filled_answer_duplication, check_final_answer
from core.process_answer import check_pred_answer
from demo.demo2 import plot_01_matrix

import os
import json
import openai
from tqdm import tqdm
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""
本文件通过比较每一层中最大概率值Token是否和最后一层中的最大概率值Token是否相同，如果是，为1；反之，为0.
输出一个矩阵[model_layer_num, generate_token_num] 01矩阵
"""



def main():
    cfg['single_model_name'] = "Qwen2.5-7B-Instruct"
    cfg['single_model_path'] = "Qwen/Qwen2.5-7B-Instruct"
    cfg['dataset_name'] = "GSM8K" # 选择数据集的名称 [MMLU GSM8K NQ] 
    # test_cfg['dataset_path'] = "/mnt/Data/zhoujiaxing/code/ICLD/dataset/MMLU/all/test-00000-of-00001.parquet" # MMLU
    cfg['dataset_path'] = "/mnt/Data/zhoujiaxing/code/ICLD/dataset/GSM8K/main/test-00000-of-00001.parquet" # GSM8K
    # test_cfg['dataset_path'] = "/mnt/Data/zhoujiaxing/code/ICLD/dataset/NQ/train-00000-of-00001.parquet" # NQ


    # 1.参数配置
    # max_questions = cfg.get("max_questions") # 测试的题目数量=1
    max_questions = 1 # 跑3道题，各个数据集上跑一次
    dataset_name = cfg.get("dataset_name")
    dataset_path = cfg.get("dataset_path")
    single_model_name = cfg.get("single_model_name")
    single_model_path = cfg.get("single_model_path")
    device = cfg.get("device")
    max_new_tokens = 500
    # max_new_tokens = cfg.get("max_new_tokens")

    # 2.构造数据集对象
    dataset = load_dataset(dataset_name, dataset_path).shuffle(seed=42).select(range(max_questions))

    # 3.加载模型和Tokenizer
    model, tokenizer = load_model(single_model_path, cfg)
    client = setOpenAi()
    model.eval()
    model_layers_num = model.config.num_hidden_layers # 获取该模型所有层数

    # 4.初始化统计指标：
    
    layer_logits_matrix = [] # 最终形状为[model_layer_num, generate_token_num]

    # 5.遍历数据集，逐条推理
    for i, sample in tqdm(enumerate(dataset), desc="处理样本", total=len(dataset)):

        # 1.生成提示词
        prompt = None
        prompt = pg.PromptGenerator.generate_prompt(cfg, sample)
        # 2.输入编码
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"] # .to(device)
        generated_ids = []
        past_key_values = None  # 用于缓存KV，加速推理
        curr_input_ids = input_ids # 当前输入模型的token
        current_text = "" # 当前生成的文本

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # A. 模型前向传播
                outputs = model(
                    input_ids=curr_input_ids, 
                    past_key_values=past_key_values, 
                    use_cache=True,
                    output_hidden_states=True
                )
                # B. 直接获取最后一层的最后一个token的Logits分布
                last_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(last_logits, dim=-1)

                # C. for循环，比较每一层的Logits
                per_layer_logits = []
                for j in range(0, model_layers_num):
                    # 取出第j层的隐藏状态
                    j_hid_state = outputs.hidden_states[j][:,-1,:]
                    normed_state = None
                    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                        normed_state = model.model.norm(j_hid_state) # Llama, Qwen, Mistral 等常见结构
                    elif hasattr(model, 'norm'):
                        normed_state = model.norm(j_hid_state) # GPT-NeoX, Bloom 等结构
                    else:
                        normed_state = j_hid_state # 兜底：如果没有找到 norm 层 (极少见)，直接用原始状态

                    j_logits = model.lm_head(normed_state)
                    j_token = torch.argmax(j_logits, dim=-1)

                    # 比较是否相同
                    is_correct = 1 if j_token.item() == next_token.item() else 0
                    per_layer_logits.append(is_correct)
                
                layer_logits_matrix.append(per_layer_logits)

                # D. 保存结果
                generated_ids.append(next_token.item())

                # E. 如果已经生成答案【模型开始生成相同内容】 或者 停止条件判断 (遇到 EOS)
                current_text += tokenizer.decode([next_token.item()], skip_special_tokens=True)
                check_window_text = current_text[-100:] if len(current_text) > 100 else current_text
                if check_final_answer(check_window_text) or check_filled_answer_duplication(check_window_text) or next_token.item() == tokenizer.eos_token_id:
                    break

                # F. 更新状态用于下一次迭代
                past_key_values = outputs.past_key_values

                # G. 下一次输入只需传入新生成的 token，形状调整为 [batch, 1]
                curr_input_ids = next_token.unsqueeze(-1)
    
        # 4.解码输出
        # response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 5.将矩阵可视化并保存
        plot_01_matrix(layer_logits_matrix, title="各层token对齐粒度", save_path=f"per_layer_logits_{dataset_name}_{single_model_name}.jpg")










if __name__ == "__main__":
    main()