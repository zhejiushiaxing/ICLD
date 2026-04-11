from utils.check_early_stop import check_filled_answer_duplication, check_final_answer
from core.process_answer import check_pred_answer
from core.token_ensemble_timing import *
from core.model_voting import *
from core.check_confidence import *

from transformers import GenerationConfig
import torch
import torch.nn.functional as F
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import math
import copy


# =========================
# 1) Top-k 提取
# =========================
def get_top_k_tokens_unite_style(logits, tokenizer, k=10):

    top_k_logits_values, top_k_indices = torch.topk(logits, k)
    top_k_indices_list = top_k_indices.tolist()
    top_k_logits_list = top_k_logits_values.tolist()

    token_dict = {}
    for token_id, logit_val in zip(top_k_indices_list, top_k_logits_list):
        token_str = tokenizer.convert_ids_to_tokens(int(token_id), skip_special_tokens=True) # 把 token_id 转成“token 字符串”

        # UNITE风格清理：统一空格标记，统一换行成真实 \n
        clean_token = token_str.replace('▁', 'Ġ') # 统一空格
        clean_token = clean_token.replace('<0x0A>', '\n').replace('Ċ', '\n') # 统一换行

        token_dict[clean_token] = [float(logit_val), int(token_id)]

    return token_dict


# =========================
# 2) 把所有模型 top-k 的 token 字符串做并集
# =========================
def get_union_vocab_unite_style(vocab_dicts_list):
    if not vocab_dicts_list:
        return []
    combined_tokens = set()
    for v_dict in vocab_dicts_list:
        combined_tokens.update(v_dict.keys())

    return sorted(list(combined_tokens))


# =========================
# 3) 将各模型的 logits 映射到联合词表上
# =========================
def update_vocab_unite_style(vocab_dict, union_vocab, tokenizer, logits, model_name):

    # 1.初始化
    existing_token_ids = set()
    logits_len = int(logits.shape[-1]) # 模型i的词表长度

    # 2.已存在的 ID 防冲突【将已有赋值】
    for item in vocab_dict.values():
        if item[1] is not None:
            existing_token_ids.add(item[1])

    # 3.构建 invalid_ids（黑名单） 这些 token 通常不希望作为正常“下一 token 预测候选”参与投
    invalid_ids = {None}
    if tokenizer.pad_token_id is not None:
        invalid_ids.add(tokenizer.pad_token_id) # 填充
    if tokenizer.unk_token_id is not None:
        invalid_ids.add(tokenizer.unk_token_id) # 未知
    if tokenizer.bos_token_id is not None:
        invalid_ids.add(tokenizer.bos_token_id) # 句首
    # 防御性硬编码（在范围内才加）
    potential_invalid_ids = [
        29871, 29473, 207, 28705,         # Llama/Mistral 变体
        151643, 151644, 151645, 128001,   # Qwen, Llama3 等
    ]
    # 防止出界
    for pid in potential_invalid_ids:
        if pid < logits_len:
            invalid_ids.add(pid)

    # 4.空格处理
    target_models_for_replace = ['llama', 'mistral', 'deepseek', 'openchat', 'gemma', 'hunyuan']
    is_special_replace_needed = any(t in model_name.lower() for t in target_models_for_replace)

    # 5.遍历联合词表
    for token in union_vocab:

        # 情况1：如果模型i top-k有该token，则跳过【前面操作已经赋值了】
        if token in vocab_dict:
            continue

        target_id = None

        # 情况2：topk没有，但模型i整个词表有该token
        temp_id = tokenizer.convert_tokens_to_ids(token)
        if temp_id is not None and temp_id < logits_len and temp_id not in invalid_ids:
            target_id = int(temp_id)

        # 情况3：模型i整个词表都没有该token
        if target_id is None:

            # 目标token字符串
            lookup_token = token
            if is_special_replace_needed and 'Ġ' in token:
                lookup_token = token.replace('Ġ', ' ')  # SentencePiece 更像真实空格
            
            # 对字符串进行编码
            ids = tokenizer.encode(lookup_token, add_special_tokens=False)

            # 如果编码后token_id长度为1，才接受该token，否则拒绝投票
            if len(ids) == 1:
                cand = int(ids[0])
                if cand < logits_len and cand not in invalid_ids: # 校验 id 合法且不是无效 token，然后赋值 target_id
                    target_id = cand

        if target_id is None:
            # 情况4：对齐失败，不要用UNK概率回填！直接 -inf（概率0，不投票）
            vocab_dict[token] = [float('-inf'), None]
        else:
            # --- 写入对齐结果 ---
            
            # 处理id冲突：同一个 id 对应多个 token_str，只保留第一个，其它设 -inf 避免重复投票
            if target_id in existing_token_ids:
                vocab_dict[token] = [float('-inf'), target_id]
            else:
                vocab_dict[token] = [float(logits[target_id].item()), target_id]
                existing_token_ids.add(target_id)

    sorted_dict = {k: vocab_dict[k] for k in sorted(vocab_dict.keys())} # 排序
    return sorted_dict


# =========================
# 4) 核心融合：logprob 空间 概率相加
# =========================
def run_unite_fusion_sum_prob(logits_list, model_tokenizer_pairs, top_k, calc_device):
    models_num = len(model_tokenizer_pairs)

    # 1) 获取各模型 top-k
    model_topk_vocab_dicts = []
    for i, (model, tokenizer) in enumerate(model_tokenizer_pairs):
        current_model_dict = get_top_k_tokens_unite_style(
            logits_list[i], tokenizer, k=top_k
        )
        model_topk_vocab_dicts.append(current_model_dict)

    # 2) 获取并集词表
    union_vocab_list = get_union_vocab_unite_style(model_topk_vocab_dicts)
    union_vocab_size = len(union_vocab_list)

    # 3) 对齐到 union：得到每模型的 token_id（失败为 None）和 logit（失败为 -inf）
    updated_dicts = []
    for i, (model, tokenizer) in enumerate(model_tokenizer_pairs):
        model_name_for_lookup = model.name_or_path if hasattr(model, "name_or_path") else str(i)
        updated_dict = update_vocab_unite_style(
            model_topk_vocab_dicts[i],
            union_vocab_list,
            tokenizer,
            logits_list[i],
            model_name_for_lookup.lower()
        )
        updated_dicts.append(updated_dict)

    # 4) 在全词表上做 softmax，然后 gather union 上对应的 prob
    #    对齐失败(None) => prob = 0
    aligned_probs_tensor = torch.zeros(
        (models_num, union_vocab_size),
        device=calc_device,
        dtype=torch.float32
    )
    next_token_ids_map = {token: [None] * models_num for token in union_vocab_list}

    for i in range(models_num):
        logits_i = logits_list[i].to(calc_device)
        probs_i = torch.softmax(logits_i, dim=-1)  # [V]

        for t_idx, token_str in enumerate(union_vocab_list):
            token_id = updated_dicts[i].get(token_str, [float("-inf"), None])[1]
            next_token_ids_map[token_str][i] = token_id

            if token_id is None:
                continue

            aligned_logit = updated_dicts[i].get(token_str, [float("-inf"), None])[0]
            if not math.isfinite(aligned_logit):
                # -inf => 不参与概率相加
                continue

            aligned_probs_tensor[i, t_idx] = probs_i[int(token_id)]

    # 5) 概率相加融合
    final_scores_vector = aligned_probs_tensor.sum(dim=0)  # [union_vocab_size]

    # 6) 选择 best（注意：可能全 0，兜底为某模型 top1）
    if torch.all(final_scores_vector == 0):
        best_token_id0 = int(torch.argmax(logits_list[0]).item())
        tok0 = model_tokenizer_pairs[0][1].convert_ids_to_tokens(
            best_token_id0, skip_special_tokens=True
        )
        best_token_str = tok0.replace("▁", "Ġ").replace("<0x0A>", "\n").replace("Ċ", "\n")

        best_token_ids = [None] * models_num
        for i, (m, tok) in enumerate(model_tokenizer_pairs):
            d = update_vocab_unite_style(
                {},
                [best_token_str],
                tok,
                logits_list[i],
                getattr(m, "name_or_path", str(i))
            )
            best_token_ids[i] = d[best_token_str][1]

        return best_token_str, best_token_ids

    best_token_idx = int(torch.argmax(final_scores_vector).item())
    best_token_str = union_vocab_list[best_token_idx]
    best_token_ids = next_token_ids_map[best_token_str]

    return best_token_str, best_token_ids


class ModelHandle:


    def icld_mult_generate_response(self, prompt, cfg, model_tokenizer_pairs, sample):

        # 1.读取配置
        model_args = cfg.get("args", {})
        max_new_tokens = int(model_args.get("max_new_tokens", 512))
        top_k = int(model_args.get("top_k", 10))
        calc_device = cfg.get("device", "cuda:0")
        models_num = len(model_tokenizer_pairs)

        final_output_text = ""
        every_model_generate_token_num_list = [0] * models_num # 维护每个模型生成的token数量
        
        main_model_idx = 0 # 默认第一个模型作为主模型
        main_model, main_tokenizer = model_tokenizer_pairs[main_model_idx]

        messages=[
            {"role": "system", "content": f"You are a helpful Assistant."},
            {"role": "user", "content": prompt+"\n"}
        ]

        formatted_prompt = main_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        main_inputs = main_tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True).to(main_model.device)

        
        main_current_input = main_inputs["input_ids"]
        main_current_mask = main_inputs["attention_mask"]
        main_past_kv = None
        ensemble_num = 0

        start_time = time.time()
        step = 0

        with torch.no_grad():
            for step in range(max_new_tokens):

                # A.主模型进行推理
                if main_past_kv is not None:
                    main_outputs = main_model(
                        input_ids = main_current_input,
                        attention_mask = main_current_mask,
                        past_key_values = main_past_kv,
                        output_hidden_states=True,
                        use_cache = True
                    )
                else:
                    main_outputs = main_model(
                        input_ids = main_current_input,
                        attention_mask = main_current_mask,
                        output_hidden_states=True,
                        use_cache = True
                    )
                
                # B.获取主模型的Logits和新的KV缓存
                main_logits = main_outputs.logits[0, -1, :].detach().to(calc_device)
                main_past_kv = main_outputs.past_key_values
                every_model_generate_token_num_list[main_model_idx]+=1

                # C.检查主模型置信度，判断是否需要触发多模型集成
                is_ensemble = check_main_model_confidence(main_outputs, main_model, main_tokenizer, cfg)
                # is_ensemble = True

                logits_list = [None] * models_num
                best_token_ids = [None] * models_num
                logits_list[main_model_idx] = main_logits

                # D.1 触发多模型集成
                if is_ensemble:
                    ensemble_num += 1
                    # 1.构建上下文信息
                    # current_full_context = formatted_prompt + "\n" + final_output_text
                    current_full_context = formatted_prompt + "\n" + final_output_text


                    # 2.协作模型进行推理
                    for i in range(models_num):
                        if i == main_model_idx:
                            continue

                        collab_model, collab_tokenizer = model_tokenizer_pairs[i]

                        formatted_prompt = collab_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                        current_full_context = formatted_prompt + "\n" + final_output_text

                        collab_inputs = collab_tokenizer(current_full_context, return_tensors="pt", add_special_tokens=True).to(collab_model.device)
                        collab_outputs = collab_model(
                            input_ids=collab_inputs["input_ids"],
                            attention_mask=collab_inputs["attention_mask"],
                            use_cache=False
                        )
                        logits_list[i] = collab_outputs.logits[0, -1, :].detach().to(calc_device)
                        every_model_generate_token_num_list[i]+=1

                    # 3.多模型集成融合
                    best_token_str, best_token_ids = run_unite_fusion_sum_prob(
                        logits_list=logits_list,
                        model_tokenizer_pairs=model_tokenizer_pairs,
                        top_k=top_k,
                        calc_device=calc_device
                    )
                
                # D.2 主模型单独推理
                else:
                    best_tid = int(torch.argmax(main_logits).item())
                    best_token_ids[main_model_idx] = best_tid
                    best_token_str = main_tokenizer.decode([best_tid], skip_special_tokens=True)
                
                # E. 统一换行与空格标记
                clean_str = best_token_str.replace('Ġ', ' ')
                final_output_text += clean_str


                # F. 停止条件
                check_window_text = final_output_text[-50:] if len(final_output_text) > 50 else final_output_text
                is_any_model_eos = False
                # F.1 检查主模型是否输出 EOS
                if best_token_ids[main_model_idx] is not None and main_tokenizer.eos_token_id is not None:
                    if int(best_token_ids[main_model_idx]) == int(main_tokenizer.eos_token_id):
                        is_any_model_eos = True
                
                # F.2 判断是否需要早停
                if check_final_answer(check_window_text) or check_filled_answer_duplication(check_window_text) or is_any_model_eos:
                    break

                # G. 更新主模型 KV / inputs 
                main_tid = best_token_ids[main_model_idx]
                if main_tid is None:
                    main_tid = int(torch.argmax(main_logits).item())   # 兜底

                next_id_tensor = torch.tensor([[main_tid]], device=main_model.device)
                main_current_input = next_id_tensor
                mask_ones = torch.ones((1, 1), device=main_model.device, dtype=main_current_mask.dtype)
                main_current_mask = torch.cat([main_current_mask, mask_ones], dim=-1)

        end_time = time.time()


        print("#############################################################")
        print("final_output_text:", final_output_text)
        print("#############################################################")

        # 指标计算
        cal_time = end_time - start_time
        is_correct, pred_answer, correct_answer = check_pred_answer(final_output_text, sample, cfg)
        
        generated_tokens_num = sum(every_model_generate_token_num_list)

        return {
            'cal_time': cal_time,
            'is_correct': is_correct,
            'generated_tokens_num': generated_tokens_num,
            'question_response': {
                'pred_answer': pred_answer,
                'correct_answer': correct_answer,
                'ensemble_num': ensemble_num,
                'final_output_text': final_output_text,
                'every_model_generate_token_num_list': every_model_generate_token_num_list
            }
        }


