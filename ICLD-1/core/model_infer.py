from core import prompt_generate as pg
from core.response_generate import ModelHandle
# from core.arena_response_generate import ModelHandle
from utils.load_model import load_model
from utils.load_data import load_dataset

import os
import json
import csv
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sentence_transformers import SentenceTransformer

class ModelInfer:

    def __init__(self):
        print("初始化ModelInfer")
        self.model_handler = ModelHandle()
 
            
    # 多模型推理
    def mult_models_infer(self, cfg):
        # 1. 参数配置
        max_questions = cfg.get("max_questions") 
        dataset_name = cfg.get("dataset_name")
        dataset_path = cfg.get("dataset_path")
        integrate_models_path = cfg.get("integrate_models_path")
        integrate_models_device = cfg.get("integrate_models_device")
        model_num = len(integrate_models_path)
        comment = cfg.get("comment", " ")
        
        # 2. 构造数据集对象
        seed_value = 0
        if dataset_name == "MMLU":
            seed_value = 66
        elif dataset_name == "GSM8K":
            seed_value = 42
        else:
            seed_value = 66
        
        dataset = load_dataset(dataset_name, dataset_path).shuffle(seed=seed_value).select(range(max_questions))
        print(len(dataset))
        

        # 3. 加载模型和分词器
        model_tokenizer_pairs = []
        with torch.no_grad():
            for model_path, gpu_id in zip(integrate_models_path, integrate_models_device):
                target_device = f"cuda:{gpu_id}"
                model, tokenizer = load_model(model_path, cfg, device=target_device)
                model_tokenizer_pairs.append((model, tokenizer))

        # 4. 初始化统计指标
        all_cal_time = 0 
        all_correct_num = 0
        all_ensemble_num = 0
        avg_ensemble_num = 0
        all_high_confidence_token_num = 0 # 所有高置信度Token数量
        all_high_confidence_ensemble_num = 0 # 所有高置信度Token触发集成次数
        all_low_confidence_token_num = 0 # 所有低置信度Token数量
        all_low_confidence_ensemble_num = 0 # 所有低置信度Token触发集成次数
        avg_high_confidence_token_num = 0 # 平均高置信度Token数量
        avg_high_confidence_ensemble_num = 0 # 平均高置信度Token触发集成次数
        avg_low_confidence_token_num = 0 # 平均低置信度Token数量
        avg_low_confidence_ensemble_num = 0 # 平均低置信度Token触发集成次数
        all_generated_tokens_num_list = [0] * model_num  # 记录每个模型生成的token数量总和
        avg_generated_tokens_num_list = [0] * model_num  # 记录每个模型平均生成的token数量总和
        all_question_response_list = [] # 模型具体回复内容
        models_pred_answer_list = [] # 模型预测答案列表
        question_correct_answer_list = [] # 问题的正确答案列表
        response_check_list = []
        all_question_token_prob_record_list = [[] for _ in range(10)] # 记录所有问题的token概率分布列表，用于后续分析置信度分布

        # 5. 遍历数据集，逐条推理
        print(f"开始推理，共 {len(dataset)} 条样本...")
        
        for i, sample in tqdm(enumerate(dataset), desc="处理样本", total=len(dataset)):
            # try:
            with torch.no_grad():
                response = self.mult_models_resp(sample, cfg, model_tokenizer_pairs)

            # 5.2 统计结果
            all_cal_time += float(response.get("cal_time", 0)) 
            all_correct_num += int(response.get("is_correct", 0))
            print(f"总共{len(dataset)}条样本，已回答{i+1}条样本，回答正确的样本数量为{all_correct_num}.")
            for model_idx in range(model_num):
                all_generated_tokens_num_list[model_idx] += response.get("question_response", {}).get("every_model_generate_token_num_list", [0]*model_num)[model_idx]

            ############################################
            token_prob_record_list = response.get("token_prob_record_list", []) # 当前问题每个token的概率分布列表
            for probs in token_prob_record_list:
                last_prob_value = probs[-1] if len(probs) > 0 else 0
                index = min(int(last_prob_value // 0.1), 9)
                all_question_token_prob_record_list[index].append(probs) 
            ############################################
            
            all_question_response_list.append(response.get("question_response", {}))

            models_pred_answer_list.append(response.get("question_response", {}).get("pred_answer", " "))
            question_correct_answer_list.append(response.get("question_response", {}).get("correct_answer", " "))

            all_ensemble_num += int(response.get("question_response", {}).get("ensemble_num", 0))
            all_high_confidence_token_num += int(response.get("question_response", {}).get("high_confidence_token_num", 0))
            all_high_confidence_ensemble_num += int(response.get("question_response", {}).get("high_confidence_ensemble_num", 0))
            all_low_confidence_token_num += int(response.get("question_response", {}).get("low_confidence_token_num", 0))
            all_low_confidence_ensemble_num += int(response.get("question_response", {}).get("low_confidence_ensemble_num", 0))


            # response_check_list.append({
            #     "id": i,
            #     "question": sample.get("question", ""),
            #     "answer": sample.get("answer", ""),
            #     "response": response.get("question_response", {}).get("final_output_text", "")
            # }) 

        # 6. 整理数据并写入日志
        finish_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
        integrate_models_name = cfg.get("integrate_models_name")
        integrate_method = cfg.get("integrate_method")

        for i in range(model_num):
            avg_generated_tokens_num_list[i] = round(all_generated_tokens_num_list[i] / max_questions, 2) if max_questions > 0 else 0
        
        # with open("all_question_token_prob_record_list_MSM8K.pkl", "wb") as f:
        #     pickle.dump(all_question_token_prob_record_list, f)

        # print("✅ 已保存到：all_question_token_prob_record_list.pkl")

        bunk_list = [len(sublist) for sublist in all_question_token_prob_record_list]

        result = {
            'comment': comment,
            'finish_date': finish_date,
            'integrate_models_name': integrate_models_name,
            'integrate_method': integrate_method,
            'dataset_name': dataset_name,

            'avg_cal_time': round(all_cal_time / max_questions, 2) if max_questions > 0 else 0,
            'avg_correct_num': round(all_correct_num / max_questions, 2) if max_questions > 0 else 0,
            'avg_generated_tokens_num_list': avg_generated_tokens_num_list,
            'avg_ensemble_num': round(all_ensemble_num / max_questions, 2) if max_questions > 0 else 0,
            'high_confidence_ensemble_ratio': round(all_high_confidence_ensemble_num / all_high_confidence_token_num, 5) if all_high_confidence_token_num > 0 else 0,
            'low_confidence_ensemble_ratio': round(all_low_confidence_ensemble_num / all_low_confidence_token_num, 5) if all_low_confidence_token_num > 0 else 0,
            'all_ensemble_ratio': round(all_ensemble_num / all_generated_tokens_num_list[0], 5) if all_generated_tokens_num_list[0] > 0 else 0,
            'bunk_list': bunk_list, # 记录每个问题的token概率分布数量列表，用于后续分析置信度分布

            'all_cal_time': all_cal_time,
            'all_correct_num': all_correct_num,
            'all_generated_tokens_num_list': all_generated_tokens_num_list,
            'all_ensemble_num': all_ensemble_num,
            'all_high_confidence_token_num': all_high_confidence_token_num,
            'all_high_confidence_ensemble_num': all_high_confidence_ensemble_num,
                        'all_low_confidence_token_num': all_low_confidence_token_num,
            'all_low_confidence_ensemble_num': all_low_confidence_ensemble_num,     
            'models_pred_answer_list': models_pred_answer_list,
            'question_correct_answer_list': question_correct_answer_list,

            'avg_high_confidence_token_num': round(all_high_confidence_token_num / max_questions, 2) if max_questions > 0 else 0,
            'avg_high_confidence_ensemble_num': round(all_high_confidence_ensemble_num / max_questions, 2) if max_questions > 0 else 0,
            # 'avg_high_confidence_ensemble_ratio': round(round(all_high_confidence_ensemble_num / max_questions, 2) / round(all_high_confidence_token_num / max_questions, 2), 2) if max_questions > 0 else 0,
            'avg_low_confidence_token_num': round(all_low_confidence_token_num / max_questions, 2) if max_questions > 0 else 0,
            'avg_low_confidence_ensemble_num': round(all_low_confidence_ensemble_num / max_questions, 2) if max_questions > 0 else 0,
            # 'avg_low_confidence_ensemble_ratio': round(round(all_low_confidence_ensemble_num / max_questions, 2) / round(all_low_confidence_token_num / max_questions, 2), 2) if max_questions > 0 else 0,

            'all_question_response_list': all_question_response_list
        }

        # 7. 将结果写入日志文件
        mult_models_output_root = cfg.get("mult_models_output_root")
        json_output_dir = f"{mult_models_output_root}/{integrate_method}/{dataset_name}/json"
        csv_output_dir = f"{mult_models_output_root}/{integrate_method}/{dataset_name}/csv"
        
        json_output_file_path = f"{json_output_dir}/{finish_date}_mult_models_test_{max_questions}.json"
        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir)

        # csv_output_file = os.path.join(csv_output_dir, f"{finish_date}_mult_models_test_{max_questions}.csv")
        # if not os.path.exists(csv_output_dir):
        #     os.makedirs(csv_output_dir)

        # 用于判断答案是否正确
        # with open(csv_output_file, "w", encoding="utf-8-sig", newline="") as f:
        #     writer = csv.DictWriter(
        #         f,
        #         fieldnames=["id", "question", "answer", "response"]
        #     )
        #     writer.writeheader()
        #     writer.writerows(response_check_list)

        with open(json_output_file_path, 'w', encoding='utf-8') as f:
            json.dump(
                result, 
                f, 
                ensure_ascii=False, 
                indent=4, 
                sort_keys=False 
            )
        
        print("结果保存到：", json_output_file_path)




    # 多模型回答单个样本
    def mult_models_resp(self, sample, cfg, model_tokenizer_pairs):

        # 1.构建提示词
        integrate_method = cfg.get("integrate_method")
        prompt = pg.PromptGenerator.generate_prompt(cfg, sample)

        # 2.推理
        result = None
        if integrate_method == "BASELINE":
            result = self.model_handler.baseline_mult_generate_response(prompt, cfg, model_tokenizer_pairs, sample)
        elif integrate_method == "ICLD":
            result = self.model_handler.icld_mult_generate_response(prompt, cfg, model_tokenizer_pairs, sample)
        elif integrate_method == "UNITE":
            result = self.model_handler.unite_mult_generate_response(prompt, cfg, model_tokenizer_pairs, sample)

        # 3.返回结果
        return result



