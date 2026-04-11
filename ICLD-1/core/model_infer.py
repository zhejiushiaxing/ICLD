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
import seaborn as sns
from sentence_transformers import SentenceTransformer

class ModelInfer:

    def __init__(self):
        print("初始化ModelInfer")
        self.model_handler = ModelHandle()


    # 单模型推理
    def single_model_infer(self, cfg):
        # 1.参数配置
        max_questions = cfg.get("max_questions") # 测试的题目数量
        dataset_name = cfg.get("dataset_name")
        dataset_path = cfg.get("dataset_path")
        single_model_path = cfg.get("single_model_path")
        device = cfg.get("device")

        # 2.构造数据集对象
        dataset = load_dataset(dataset_name, dataset_path).shuffle(seed=66).select(range(max_questions))
        # print(dataset[0])
        # print(f"数据集 {dataset_name} 已加载，包含 {len(dataset)} 条样本。")

        # 3.加载模型和Tokenizer, 连接WNDS集群的模型
        model, tokenizer = load_model(single_model_path, cfg, device)

        # 4.初始化统计指标
        all_cal_time = 0 # 总推理时间
        all_correct_num = 0 # 回答正确的数量
        all_generated_tokens_num = 0 # 产生token的总数量
        all_question_response_list = []
        models_pred_answer_list = [] # 模型预测答案列表
        question_correct_answer_list = [] # 问题的正确答案列表
        check_response_list = []


        # 5.遍历数据集，逐条推理
        print(f"开始推理，共 {len(dataset)} 条样本...")
        for i, sample in tqdm(enumerate(dataset), desc="处理样本", total=len(dataset)):
            
            # 5.1 推理
            response = self.single_model_resp(sample, cfg, model, tokenizer)

            # 5.2 统计结果
            all_cal_time += response['cal_time']
            all_correct_num += response['is_correct']
            all_generated_tokens_num += response['generated_tokens_num']
            models_pred_answer_list.append(response.get("question_response", {}).get("pred_answer", " "))
            question_correct_answer_list.append(response.get("question_response", {}).get("correct_answer", " "))

            all_question_response_list.append({
                'question': sample['question'],
                'response': response['question_response']['response'],
                'every_token_generation_time_list': response['question_response']['every_token_generation_time_list']
                })
            
            if dataset_name == "MULTILINGUAL":
                check_response_list.append({
                    "id": i,
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                    "language": sample.get("language", ""),
                    "response": response.get("question_response", {}).get("response", "")
                })
            else:
                check_response_list.append({
                    "id": i,
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                    "response": response.get("question_response", {}).get("response", "")
                }) 
        

        # 6.整理数据并写入日志当中
        finish_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # 获取当前时间,格式化输出：年-月-日 时:分:秒
        single_model_name = cfg.get("single_model_name") # 测试的模型名称

        result = {
            'finish_date': finish_date,
            'single_model_name': single_model_name,
            'dataset_name': dataset_name,
            'all_cal_time': all_cal_time,
            'all_correct_num': all_correct_num,
            'models_pred_answer_list': models_pred_answer_list,
            'question_correct_answer_list': question_correct_answer_list,
            'all_generated_tokens_num': all_generated_tokens_num,
            'avg_cal_time': round(all_cal_time / max_questions, 2),
            'avg_correct_num': round(all_correct_num / max_questions, 2),
            'avg_generated_tokens_num': round(all_generated_tokens_num / max_questions, 2),
            'all_question_response_list': all_question_response_list
        }

        # 7.将结果写入日志文件当中
        single_model_output_root = cfg.get("single_model_output_root")
        json_output_dir = f"{single_model_output_root}/{single_model_name}/{dataset_name}/json"
        csv_output_dir = f"{single_model_output_root}/{single_model_name}/{dataset_name}/csv"
        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir)
        if not os.path.exists(csv_output_dir):
            os.makedirs(csv_output_dir)

        json_output_file_path = f"{json_output_dir}/{finish_date}_single_model_test_{max_questions}.json"
        csv_output_file = os.path.join(csv_output_dir, f"{finish_date}_single_model_test_{max_questions}.csv")


        with open(csv_output_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "question", "answer", "response"]
            )
            writer.writeheader()
            writer.writerows(check_response_list)

        with open(json_output_file_path, 'w', encoding='utf-8') as f:
            json.dump(
                result, 
                f, 
                ensure_ascii=False,  # 保证中文正常显示（若有中文数据集/模型名）
                indent=4,  # 格式化输出，便于阅读
                sort_keys=False  # 保持字典原有顺序
            )

        
    
    # 单模型回答单个样本
    def single_model_resp(self, sample, cfg, model, tokenizer):
        # 1.构建提示词
        prompt = None
        prompt = pg.PromptGenerator.generate_prompt(cfg, sample)

        # 2.推理
        dataset_name = cfg.get("dataset_name")
        result = self.model_handler.single_generate_response(prompt, cfg, model, tokenizer, sample)
        
        return result
       
            
    # 多模型推理
    def mult_models_infer(self, cfg):
        # 1. 参数配置
        max_questions = cfg.get("max_questions") 
        dataset_name = cfg.get("dataset_name")
        dataset_path = cfg.get("dataset_path")
        integrate_models_path = cfg.get("integrate_models_path")
        integrate_models_device = cfg.get("integrate_models_device")
        model_num = len(integrate_models_path)
        
        # 2. 构造数据集对象
        dataset = load_dataset(dataset_name, dataset_path).shuffle(seed=66).select(range(max_questions))
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
        all_generated_tokens_num_list = [0] * model_num  # 记录每个模型生成的token数量总和
        avg_generated_tokens_num_list = [0] * model_num  # 记录每个模型平均生成的token数量总和
        # main_model_selection_counts = [0] * model_num  # 记录每个模型被选为主模型的次数
        all_question_response_list = [] # 模型具体回复内容
        models_pred_answer_list = [] # 模型预测答案列表
        question_correct_answer_list = [] # 问题的正确答案列表
        response_check_list = []

        # 5. 遍历数据集，逐条推理
        print(f"开始推理，共 {len(dataset)} 条样本...")
        
        for i, sample in tqdm(enumerate(dataset), desc="处理样本", total=len(dataset)):
            # try:
            with torch.no_grad():
                response = self.mult_models_resp(sample, cfg, model_tokenizer_pairs)

            # 5.2 统计结果
            all_cal_time += float(response.get("cal_time", 0)) 
            all_correct_num += int(response.get("is_correct", 0))
            for model_idx in range(model_num):
                all_generated_tokens_num_list[model_idx] += response.get("question_response", {}).get("every_model_generate_token_num_list", [0]*model_num)[model_idx]
            
            all_question_response_list.append(response.get("question_response", {}))

            # all_stage1_nums += int(response.get("stage_1", 0))
            # all_stage2_nums += int(response.get("stage_2", 0))
            # all_ensemble_nums += int(response.get("ensemble_nums", 0))
            # main_model_selection_counts[response.get("question_response", {}).get("main_model_idx", 0)] += 1
            models_pred_answer_list.append(response.get("question_response", {}).get("pred_answer", " "))
            question_correct_answer_list.append(response.get("question_response", {}).get("correct_answer", " "))

            all_ensemble_num += int(response.get("question_response", {}).get("ensemble_num", 0))


            response_check_list.append({
                "id": i,
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "response": response.get("question_response", {}).get("final_output_text", "")
            }) 

        # 6. 整理数据并写入日志
        finish_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
        integrate_models_name = cfg.get("integrate_models_name")
        integrate_method = cfg.get("integrate_method")

        for i in range(model_num):
            avg_generated_tokens_num_list[i] = round(all_generated_tokens_num_list[i] / max_questions, 2) if max_questions > 0 else 0
        
        result = {
            'finish_date': finish_date,
            'integrate_models_name': integrate_models_name,
            'integrate_method': integrate_method,
            'dataset_name': dataset_name,

            'all_cal_time': all_cal_time,
            'all_correct_num': all_correct_num,
            'all_generated_tokens_num_list': all_generated_tokens_num_list,
            'all_ensemble_num': all_ensemble_num,
            
            'models_pred_answer_list': models_pred_answer_list,
            'question_correct_answer_list': question_correct_answer_list,

            'avg_cal_time': round(all_cal_time / max_questions, 2) if max_questions > 0 else 0,
            'avg_correct_num': round(all_correct_num / max_questions, 2) if max_questions > 0 else 0,
            'avg_generated_tokens_num_list': avg_generated_tokens_num_list,
            'avg_ensemble_num': round(all_ensemble_num / max_questions, 2) if max_questions > 0 else 0,

            # 'main_model_selection_counts': main_model_selection_counts,

            # 'all_stage1_nums': all_stage1_nums,
            # 'all_stage2_nums': all_stage2_nums,
            # 'all_ensemble_nums': all_ensemble_nums,
            
            # 'avg_stage1_nums': round(all_stage1_nums / max_questions, 2) if max_questions > 0 else 0,
            # 'avg_stage2_nums': round(all_stage2_nums / max_questions, 2) if max_questions > 0 else 0,
            # 'avg_ensemble_nums': round(all_ensemble_nums / max_questions, 2) if max_questions > 0 else 0,

            'all_question_response_list': all_question_response_list
        }

        # 7. 将结果写入日志文件
        mult_models_output_root = cfg.get("mult_models_output_root")
        json_output_dir = f"{mult_models_output_root}/{integrate_method}/{dataset_name}/json"
        csv_output_dir = f"{mult_models_output_root}/{integrate_method}/{dataset_name}/csv"
        
        json_output_file_path = f"{json_output_dir}/{finish_date}_mult_models_test_{max_questions}.json"
        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir)

        csv_output_file = os.path.join(csv_output_dir, f"{finish_date}_mult_models_test_{max_questions}.csv")
        if not os.path.exists(csv_output_dir):
            os.makedirs(csv_output_dir)

        # 用于判断答案是否正确
        with open(csv_output_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "question", "answer", "response"]
            )
            writer.writeheader()
            writer.writerows(response_check_list)

        with open(json_output_file_path, 'w', encoding='utf-8') as f:
            json.dump(
                result, 
                f, 
                ensure_ascii=False, 
                indent=4, 
                sort_keys=False 
            )




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



