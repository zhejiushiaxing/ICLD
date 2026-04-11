from core import prompt_generate as pg
from config.cfg import cfg
from utils.load_model import load_model
from utils.load_data import load_dataset
from utils.wnds_client_coll import setOpenAi
from utils.check_early_stop import check_filled_answer_duplication, check_final_answer
from core.process_answer import check_pred_answer

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
本文件主要用于测试直接使用每层隐藏状态对应的Logits来输出答案，用于对比各个各层准确率

由实验结果可知，直接使用前26层的Logits准确率为0，输出的文本为乱码。只有在27层，28层【最后一层】才有回答正确的可能性

"""

def main():
    cfg['single_model_name'] = "Qwen2.5-7B-Instruct"
    cfg['single_model_path'] = "Qwen/Qwen2.5-7B-Instruct"
    cfg['dataset_name'] = "GSM8K" # 选择数据集的名称 [MMLU GSM8K NQ] 
    # test_cfg['dataset_path'] = "/mnt/Data/zhoujiaxing/code/ICLD/dataset/MMLU/all/test-00000-of-00001.parquet" # MMLU
    cfg['dataset_path'] = "/mnt/Data/zhoujiaxing/code/ICLD/dataset/GSM8K/main/test-00000-of-00001.parquet" # GSM8K
    # test_cfg['dataset_path'] = "/mnt/Data/zhoujiaxing/code/ICLD/dataset/NQ/train-00000-of-00001.parquet" # NQ


    # 1.参数配置
    max_questions = cfg.get("max_questions") # 测试的题目数量
    dataset_name = cfg.get("dataset_name")
    dataset_path = cfg.get("dataset_path")
    single_model_path = cfg.get("single_model_path")
    max_new_tokens = 500
    # max_new_tokens = cfg.get("max_new_tokens")

    # 2.构造数据集对象
    dataset = load_dataset(dataset_name, dataset_path).shuffle(seed=42).select(range(max_questions))

    # 3.加载模型和Tokenizer
    model, tokenizer = load_model(single_model_path, cfg)
    client = setOpenAi()
    model.eval()
    model_layers_num = model.config.num_hidden_layers # 获取该模型所有层数
    print("模型的层数为：", model_layers_num)


    # 4.初始化统计指标：统计各层回答问题的推理时间、正确数量、生成token数量
    per_layer_cal_time = [0.00] * (model_layers_num + 1)
    per_layer_acc_num = [0.00] * (model_layers_num + 1)
    per_layer_token_num = [0.00] * (model_layers_num + 1)

    # 5.遍历数据集，逐条推理
    for i, sample in tqdm(enumerate(dataset), desc="处理样本", total=len(dataset)):

        # 1.生成提示词
        prompt = None
        prompt = pg.PromptGenerator.generate_prompt(cfg, sample)
        # 2.输入编码
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"] # .to(device)

        # 3.开始推理，用每一层的logits作为最终答案
        for j in range(1, model_layers_num):
            # 3.1 初始化变量
            generated_ids = []
            past_key_values = None  # 用于缓存KV，加速推理
            curr_input_ids = input_ids # 当前输入模型的token
            current_text = "" # 当前生成的文本

            start_time = time.time()

            # 3.2 推理
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # A. 模型前向传播
                    outputs = model(
                        input_ids=curr_input_ids, 
                        past_key_values=past_key_values, 
                        use_cache=True,
                        output_hidden_states=True
                    )

                    # B. 取出第j+1层的隐藏状态 outputs.hidden_states[0]是Embedding层
                    j_hid_state = outputs.hidden_states[j][:,-1,:]

                    # C. 获取该层最后一个token的logits
                    # 中间层的状态分布通常未归一化，直接进LM_Head可能会乱码，所以必须模拟模型最后的RMSNorm/LayerNorm操作
                    normed_state = None
                    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                        normed_state = model.model.norm(j_hid_state) # Llama, Qwen, Mistral 等常见结构
                    elif hasattr(model, 'norm'):
                        normed_state = model.norm(j_hid_state) # GPT-NeoX, Bloom 等结构
                    else:
                        normed_state = j_hid_state # 兜底：如果没有找到 norm 层 (极少见)，直接用原始状态

                    logits = model.lm_head(normed_state)

                    # D. 贪婪解码
                    next_token = torch.argmax(logits, dim=-1) 

                    # E. 保存结果
                    generated_ids.append(next_token.item())

                    # F. 如果已经生成答案【模型开始生成相同内容】 或者 停止条件判断 (遇到 EOS)
                    current_text += tokenizer.decode([next_token.item()], skip_special_tokens=True)
                    check_window_text = current_text[-100:] if len(current_text) > 100 else current_text
                    if check_final_answer(check_window_text) or check_filled_answer_duplication(check_window_text) or next_token.item() == tokenizer.eos_token_id:
                        break

                    # G. 更新状态用于下一次迭代
                    past_key_values = outputs.past_key_values

                    # H. 下一次输入只需传入新生成的 token，形状调整为 [batch, 1]
                    curr_input_ids = next_token.unsqueeze(-1)
        
            # 4.解码输出
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            end_time = time.time()

            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"第{j}层隐藏状态的response:", response)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            # 指标1：推理时间
            cal_time = end_time - start_time 

            # 指标2：计算准确率
            is_correct = check_pred_answer(response, sample, cfg, client)

            # 指标3：计算生成token数
            generated_tokens_num = len(generated_ids)

            per_layer_cal_time[j] += cal_time
            per_layer_acc_num[j] += is_correct
            per_layer_token_num[j] += generated_tokens_num


    # 6.整理数据并写入日志当中
    finish_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # 获取当前时间,格式化输出：年-月-日 时:分:秒
    single_model_name = cfg.get("single_model_name") # 测试的模型名称

    avg_per_layer_cal_time = [round(time / max_questions, 2) for time in per_layer_cal_time]
    avg_per_layer_acc_num = [round(time / max_questions, 2) for time in per_layer_acc_num]
    avg_per_layer_token_num = [round(time / max_questions, 2) for time in per_layer_token_num]

    result = {
        'finish_date': finish_date,
        'max_questions': max_questions,
        'single_model_name': single_model_name,
        'dataset_name': dataset_name,

        'per_layer_cal_time': per_layer_cal_time,
        'per_layer_acc_num': per_layer_acc_num,
        'per_layer_token_num': per_layer_token_num,

        'avg_per_layer_cal_time': avg_per_layer_cal_time,
        'avg_per_layer_acc_num': avg_per_layer_acc_num,
        'avg_per_layer_token_num': avg_per_layer_token_num
    }

    # 7.将结果写入日志文件
    per_layer_acc_output_root = cfg.get("per_layer_acc_output_root")
    output_dir = f"{per_layer_acc_output_root}/{single_model_name}/{dataset_name}"
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"目录已确保存在：{output_dir}")
    except Exception as e:
        raise RuntimeError(f"创建目录失败：{output_dir}，错误信息：{e}")
    output_file_path = f"{output_dir}/{finish_date}_per_layer_acc_test_{max_questions}.json"

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(
                result, 
                f, 
                ensure_ascii=False,  # 保证中文正常显示（若有中文数据集/模型名）
                indent=4,  # 格式化输出，便于阅读
                sort_keys=False  # 保持字典原有顺序
            )
        print(f"结果已成功写入：{output_file_path}")
        return True
    except Exception as e:
        print(f"写入文件失败！错误信息：{str(e)}")
        return False



if __name__ == "__main__":
    main()