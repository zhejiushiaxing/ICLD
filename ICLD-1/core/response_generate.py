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
# 0) 安全 token 解码
# =========================
def decode_token_safely(tokenizer, token_id=None, fallback_token_str=""):
    """
    优先使用 tokenizer.decode() 将 token_id 解码成人类可读文本。
    如果 token_id 不存在，再对 fallback_token_str 做基础清洗。

    目的：
    避免直接拼接 tokenizer 内部 token 字符串导致乱码，
    例如 âĪĺ、ÃĹ、âľħ 等 byte-level BPE 伪字符。
    """
    if token_id is not None:
        try:
            return tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        except Exception:
            pass

    return (
        fallback_token_str
        .replace("Ġ", " ")
        .replace("▁", " ")
        .replace("<0x0A>", "\n")
        .replace("Ċ", "\n")
    )


# =========================
# 1) Top-k 提取
# =========================
def get_top_k_tokens_unite_style(logits, tokenizer, k=10):
    top_k_logits_values, top_k_indices = torch.topk(logits, k)
    top_k_indices_list = top_k_indices.tolist()
    top_k_logits_list = top_k_logits_values.tolist()

    token_dict = {}

    for token_id, logit_val in zip(top_k_indices_list, top_k_logits_list):
        token_id = int(token_id)

        # 关键修改：
        # 不再直接使用 convert_ids_to_tokens() 作为输出文本，
        # 而是优先使用 decode() 得到正常可读文本。
        token_str = decode_token_safely(
            tokenizer=tokenizer,
            token_id=token_id,
            fallback_token_str=tokenizer.convert_ids_to_tokens(
                token_id,
                skip_special_tokens=True
            )
        )

        token_dict[token_str] = [float(logit_val), token_id]

    return token_dict


# =========================
# 2) 所有模型 top-k token 字符串做并集
# =========================
def get_union_vocab_unite_style(vocab_dicts_list):
    if not vocab_dicts_list:
        return []

    combined_tokens = set()
    for v_dict in vocab_dicts_list:
        combined_tokens.update(v_dict.keys())

    return sorted(list(combined_tokens))


# =========================
# 3) 将各模型 logits 映射到联合词表
# =========================
def update_vocab_unite_style(vocab_dict, union_vocab, tokenizer, logits, model_name):
    existing_token_ids = set()
    logits_len = int(logits.shape[-1])

    for item in vocab_dict.values():
        if item[1] is not None:
            existing_token_ids.add(item[1])

    invalid_ids = {None}

    if tokenizer.pad_token_id is not None:
        invalid_ids.add(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        invalid_ids.add(tokenizer.unk_token_id)
    if tokenizer.bos_token_id is not None:
        invalid_ids.add(tokenizer.bos_token_id)

    potential_invalid_ids = [
        29871, 29473, 207, 28705,
        151643, 151644, 151645, 128001,
    ]

    for pid in potential_invalid_ids:
        if pid < logits_len:
            invalid_ids.add(pid)

    target_models_for_replace = [
        "llama", "mistral", "deepseek", "openchat", "gemma", "hunyuan", "qwen"
    ]

    is_special_replace_needed = any(
        t in model_name.lower() for t in target_models_for_replace
    )

    for token in union_vocab:
        if token in vocab_dict:
            continue

        target_id = None

        # 情况 1：尝试直接由 token 字符串转 id
        temp_id = tokenizer.convert_tokens_to_ids(token)
        if (
            temp_id is not None
            and isinstance(temp_id, int)
            and temp_id < logits_len
            and temp_id not in invalid_ids
        ):
            target_id = int(temp_id)

        # 情况 2：尝试 encode 成单 token
        if target_id is None:
            lookup_token = token

            if is_special_replace_needed and "Ġ" in token:
                lookup_token = token.replace("Ġ", " ")

            ids = tokenizer.encode(
                lookup_token,
                add_special_tokens=False
            )

            if len(ids) == 1:
                cand = int(ids[0])
                if cand < logits_len and cand not in invalid_ids:
                    target_id = cand

        if target_id is None:
            vocab_dict[token] = [float("-inf"), None]
        else:
            if target_id in existing_token_ids:
                vocab_dict[token] = [float("-inf"), target_id]
            else:
                vocab_dict[token] = [float(logits[target_id].item()), target_id]
                existing_token_ids.add(target_id)

    sorted_dict = {k: vocab_dict[k] for k in sorted(vocab_dict.keys())}
    return sorted_dict


# =========================
# 4) 核心融合：在联合词表上做概率相加
# =========================
def run_unite_fusion_sum_prob(logits_list, model_tokenizer_pairs, top_k, calc_device):
    models_num = len(model_tokenizer_pairs)

    # 1) 获取各模型 top-k
    model_topk_vocab_dicts = []
    for i, (model, tokenizer) in enumerate(model_tokenizer_pairs):
        current_model_dict = get_top_k_tokens_unite_style(
            logits_list[i],
            tokenizer,
            k=top_k
        )
        model_topk_vocab_dicts.append(current_model_dict)

    # 2) 获取联合词表
    union_vocab_list = get_union_vocab_unite_style(model_topk_vocab_dicts)
    union_vocab_size = len(union_vocab_list)

    # 3) 对齐到 union
    updated_dicts = []
    for i, (model, tokenizer) in enumerate(model_tokenizer_pairs):
        updated_dict = update_vocab_unite_style(
            model_topk_vocab_dicts[i],
            union_vocab_list,
            tokenizer,
            logits_list[i],
            getattr(model, "name_or_path", str(i)).lower()
        )
        updated_dicts.append(updated_dict)

    # 4) softmax 后在 union 上取概率
    aligned_probs_tensor = torch.zeros(
        (models_num, union_vocab_size),
        device=calc_device,
        dtype=torch.float32
    )

    next_token_ids_map = {
        token: [None] * models_num
        for token in union_vocab_list
    }

    for i in range(models_num):
        logits_i = logits_list[i].to(calc_device)
        probs_i = torch.softmax(logits_i, dim=-1)

        for t_idx, token_str in enumerate(union_vocab_list):
            token_id = updated_dicts[i].get(
                token_str,
                [float("-inf"), None]
            )[1]

            next_token_ids_map[token_str][i] = token_id

            if token_id is None:
                continue

            aligned_logit = updated_dicts[i].get(
                token_str,
                [float("-inf"), None]
            )[0]

            if not math.isfinite(aligned_logit):
                continue

            aligned_probs_tensor[i, t_idx] = probs_i[int(token_id)]

    # 5) 概率相加融合
    final_scores_vector = aligned_probs_tensor.sum(dim=0)

    # 6) 兜底：如果全 0，则使用主模型 top1
    if torch.all(final_scores_vector == 0):
        best_token_id0 = int(torch.argmax(logits_list[0]).item())
        main_tokenizer = model_tokenizer_pairs[0][1]

        best_token_str = decode_token_safely(
            tokenizer=main_tokenizer,
            token_id=best_token_id0,
            fallback_token_str=main_tokenizer.convert_ids_to_tokens(
                best_token_id0,
                skip_special_tokens=True
            )
        )

        best_token_ids = [None] * models_num

        for i, (m, tok) in enumerate(model_tokenizer_pairs):
            d = update_vocab_unite_style(
                {},
                [best_token_str],
                tok,
                logits_list[i],
                getattr(m, "name_or_path", str(i)).lower()
            )
            best_token_ids[i] = d[best_token_str][1]

        return best_token_str, best_token_ids

    best_token_idx = int(torch.argmax(final_scores_vector).item())
    best_token_str = union_vocab_list[best_token_idx]
    best_token_ids = next_token_ids_map[best_token_str]

    return best_token_str, best_token_ids


class ModelHandle:

    def icld_mult_generate_response(self, prompt, cfg, model_tokenizer_pairs, sample):

        model_args = cfg.get("args", {})
        max_new_tokens = int(model_args.get("max_new_tokens", 512))
        top_k = int(model_args.get("top_k", 5))
        calc_device = cfg.get("device", "cuda:0")
        models_num = len(model_tokenizer_pairs)

        final_output_text = ""
        every_model_output_text = [""] * models_num
        every_model_generate_token_num_list = [0] * models_num

        ensemble_num = 0
        high_confidence_token_num = 0
        high_confidence_ensemble_num = 0
        low_confidence_token_num = 0
        low_confidence_ensemble_num = 0
        token_prob_record_list = []

        model_forward_count = [0] * models_num
        model_full_forward_count = [0] * models_num
        model_incremental_forward_count = [0] * models_num
        model_fallback_rebuild_count = [0] * models_num
        model_skip_count = [0] * models_num

        total_start_time = time.time()

        for model, tokenizer in model_tokenizer_pairs:
            model.eval()

        messages = [
            {"role": "system", "content": "You are a helpful Assistant."},
            {"role": "user", "content": prompt + "\n"}
        ]

        # =========================
        # A. 构造基础 prompt
        # =========================
        base_prompt_texts = []
        base_prompt_lens = []

        for i, (model, tokenizer) in enumerate(model_tokenizer_pairs):
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            think_name = model.name_or_path if hasattr(model, 'name_or_path') else " "
            # print("thinkname:", think_name)
            if think_name == "tencent/Hunyuan-4B-Instruct":
                formatted_prompt += "<think>\n\n</think>\n<answer>\n"

            base_prompt_texts.append(formatted_prompt)

            base_inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                add_special_tokens=True
            ).to(model.device)

            base_prompt_lens.append(
                int(base_inputs["input_ids"].shape[-1])
            )

        # =========================
        # B. 每个模型维护自己的 KV 状态
        # =========================
        input_ids_list = [None] * models_num
        attention_mask_list = [None] * models_num
        past_key_values_list = [None] * models_num

        # 辅助模型是否与 final_output_text 同步
        model_synced = [False] * models_num

        def rebuild_one_model_inputs(model_idx, current_generated_text):
            model, tokenizer = model_tokenizer_pairs[model_idx]
            full_text = base_prompt_texts[model_idx] + current_generated_text

            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=True
            ).to(model.device)

            return inputs["input_ids"], inputs["attention_mask"]

        def sync_model_to_current_text(model_idx):
            """
            将某个模型同步到当前 final_output_text。
            用于：
            1. 初始化主模型；
            2. 辅助模型需要参与集成时，临时 full-prefix 同步。
            """
            input_ids_list[model_idx], attention_mask_list[model_idx] = rebuild_one_model_inputs(
                model_idx,
                final_output_text
            )

            past_key_values_list[model_idx] = None
            model_synced[model_idx] = True

        # 只初始化主模型
        sync_model_to_current_text(0)

        # =========================
        # C. 单模型一步推理
        # =========================
        def model_step_inference(idx, need_hidden_states=False):
            model, tokenizer = model_tokenizer_pairs[idx]

            common_kwargs = {
                "input_ids": input_ids_list[idx],
                "attention_mask": attention_mask_list[idx],
                "use_cache": True,
                "return_dict": True,
                "output_hidden_states": need_hidden_states
            }

            if past_key_values_list[idx] is not None:
                common_kwargs["past_key_values"] = past_key_values_list[idx]

            is_full_forward = past_key_values_list[idx] is None

            model_outputs = model(**common_kwargs)
            model_forward_count[idx] += 1

            if is_full_forward:
                model_full_forward_count[idx] += 1
            else:
                model_incremental_forward_count[idx] += 1

            past_key_values_list[idx] = model_outputs.past_key_values
            model_synced[idx] = True

            # 这里保留 to(calc_device)，因为融合时需要统一设备。
            # 如果后续只做非集成推理，可以进一步避免跨 GPU copy。
            next_token_logits = model_outputs.logits[0, -1, :].detach().to(calc_device)

            return model_outputs, next_token_logits, None

        # =========================
        # D. 只更新已同步模型的下一步输入
        # =========================
        def prepare_next_inputs_for_synced_models(best_token_str, best_token_ids):
            """
            只对当前处于 synced 状态的模型做 KV 增量推进准备。

            主模型一定同步，所以一定会更新。
            辅助模型只有刚参与过集成时才同步，否则跳过。
            """
            for i, (model, tokenizer) in enumerate(model_tokenizer_pairs):
                if not model_synced[i]:
                    continue

                token_id = best_token_ids[i] if i < len(best_token_ids) else None

                if token_id is None:
                    token_ids = tokenizer.encode(
                        best_token_str,
                        add_special_tokens=False
                    )

                    if len(token_ids) == 1:
                        token_id = int(token_ids[0])
                    else:
                        token_id = None

                if token_id is not None:
                    input_ids_list[i] = torch.tensor(
                        [[int(token_id)]],
                        dtype=torch.long,
                        device=model.device
                    )

                    old_mask = attention_mask_list[i]
                    one_mask = torch.ones(
                        (old_mask.shape[0], 1),
                        dtype=old_mask.dtype,
                        device=old_mask.device
                    )

                    attention_mask_list[i] = torch.cat(
                        [old_mask, one_mask],
                        dim=-1
                    )

                else:
                    # 如果已经同步的辅助模型无法单 token 对齐，
                    # 则将其标记为不同步，下一次需要集成时再 full sync。
                    if i == 0:
                        # 主模型正常不会出现，因为 best_token 来自主模型。
                        sync_model_to_current_text(0)
                    else:
                        past_key_values_list[i] = None
                        input_ids_list[i] = None
                        attention_mask_list[i] = None
                        model_synced[i] = False
                        model_fallback_rebuild_count[i] += 1

        step = 0

        with torch.inference_mode():
            for step in range(max_new_tokens):

                # =========================
                # 1) 只运行主模型
                # =========================
                main_outputs, main_logits, _ = model_step_inference(
                    0,
                    need_hidden_states=True
                )

                logits_list = [None] * models_num
                results = [None] * models_num

                logits_list[0] = main_logits
                results[0] = (main_outputs, main_logits, None)

                main_model, main_tokenizer = model_tokenizer_pairs[0]

                # =========================
                # 2) 先基于主模型判断是否需要集成
                # =========================

                # is_ensemble, x, y, p, q = check_main_model_confidence(
                #     main_outputs, main_model, main_tokenizer, cfg
                # )
                # high_confidence_token_num += x
                # high_confidence_ensemble_num += y
                # low_confidence_token_num += p
                # low_confidence_ensemble_num += q

                is_ensemble = True

                # =========================
                # 3) 如果需要集成，再运行辅助模型
                # =========================
                if is_ensemble:
                    ensemble_num += 1

                    for i in range(1, models_num):
                        # 如果辅助模型没有同步到当前 final_output_text，则先同步
                        if not model_synced[i]:
                            sync_model_to_current_text(i)

                        aux_outputs, aux_logits, _ = model_step_inference(
                            i,
                            need_hidden_states=False
                        )

                        logits_list[i] = aux_logits
                        results[i] = (aux_outputs, aux_logits, None)

                else:
                    for i in range(1, models_num):
                        model_skip_count[i] += 1
                        model_synced[i] = False
                        past_key_values_list[i] = None
                        input_ids_list[i] = None
                        attention_mask_list[i] = None

                # =========================
                # 4) token 决策
                # =========================
                if is_ensemble:
                    best_token_str, best_token_ids = run_unite_fusion_sum_prob(
                        logits_list=logits_list,
                        model_tokenizer_pairs=model_tokenizer_pairs,
                        top_k=top_k,
                        calc_device=calc_device
                    )

                else:
                    main_token_id = int(torch.argmax(main_logits).item())

                    best_token_str = decode_token_safely(
                        tokenizer=main_tokenizer,
                        token_id=main_token_id,
                        fallback_token_str=""
                    )

                    probs = get_selected_token_probs_last_M(
                        main_outputs,
                        main_token_id,
                        main_model,
                        M=5
                    )
                    token_prob_record_list.append(probs)

                    best_token_ids = [None] * models_num
                    best_token_ids[0] = main_token_id

                # =========================
                # 4.5) 生成 token 数统计
                # =========================
                if is_ensemble:
                    for i in range(models_num):
                        every_model_generate_token_num_list[i] += 1
                else:
                    every_model_generate_token_num_list[0] += 1

                # =========================
                # 5) 拼接统一输出
                # =========================
                # 关键修改：
                # 不再直接 final_output_text += best_token_str.replace(...)
                # 而是优先用主模型对应 token_id decode，避免乱码。
                clean_str = decode_token_safely(
                    tokenizer=main_tokenizer,
                    token_id=best_token_ids[0],
                    fallback_token_str=best_token_str
                )

                final_output_text += clean_str

                # =========================
                # 6) 早停检查
                # =========================
                stop_strings = [
                    "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "<eos>", "</s>",
                    "<|im_end|>", "<|im_start|>", "<|start_header_id|>", "<|end_header_id|>",
                    "<end_of_turn>", "</answer>"
                ]

                is_special_token_generated = any(
                    stop_str in final_output_text
                    for stop_str in stop_strings
                )

                if is_special_token_generated:
                    for stop_str in stop_strings:
                        final_output_text = final_output_text.replace(stop_str, "")

                    final_output_text = final_output_text.strip()
                    break

                is_any_model_eos = False
                if best_token_ids[0] is not None and main_tokenizer.eos_token_id is not None:
                    if int(best_token_ids[0]) == int(main_tokenizer.eos_token_id):
                        is_any_model_eos = True

                check_window_text = (
                    final_output_text[-50:]
                    if len(final_output_text) > 50
                    else final_output_text
                )

                should_stop = (
                    check_final_answer(check_window_text)
                    or check_filled_answer_duplication(check_window_text)
                    or is_any_model_eos
                )

                if should_stop:
                    break

                # =========================
                # 7) 准备下一步输入
                # =========================
                prepare_next_inputs_for_synced_models(
                    best_token_str,
                    best_token_ids
                )

        # =========================
        # E. 最终 decode
        # =========================
        for i, (model, tokenizer) in enumerate(model_tokenizer_pairs):
            final_input_ids, _ = rebuild_one_model_inputs(i, final_output_text)

            prompt_len = base_prompt_lens[i]
            gen_token_num = max(
                0,
                int(final_input_ids.shape[-1] - prompt_len)
            )

            if gen_token_num == 0:
                every_model_output_text[i] = ""
            else:
                gen_ids = final_input_ids[0][prompt_len:]
                every_model_output_text[i] = tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

        total_end_time = time.time()
        cal_time = total_end_time - total_start_time

        print("#############################################################")
        print("final_output_text:", final_output_text)
        print("#############################################################")

        for i in range(models_num):
            print("#############################################################")
            print(f"every_model_output_text[{i}]:", every_model_output_text[i])
            print("#############################################################")

        print("#############################################################")

        # =========================
        # F. 答案检查
        # =========================
        is_correct, pred_answer, correct_answer = check_pred_answer(
            final_output_text,
            sample,
            cfg
        )

        generated_tokens_num = sum(every_model_generate_token_num_list)

        return {
            "cal_time": cal_time,
            "is_correct": is_correct,
            "generated_tokens_num": generated_tokens_num,
            "token_prob_record_list": token_prob_record_list,
            "question_response": {
                "pred_answer": pred_answer,
                "correct_answer": correct_answer,
                "ensemble_num": ensemble_num,
                "high_confidence_token_num": high_confidence_token_num,
                "high_confidence_ensemble_num": high_confidence_ensemble_num,
                "low_confidence_token_num": low_confidence_token_num,
                "low_confidence_ensemble_num": low_confidence_ensemble_num,
                "final_output_text": final_output_text,
                "every_model_output_text": every_model_output_text,
                "every_model_generate_token_num_list": every_model_generate_token_num_list
            }
        }