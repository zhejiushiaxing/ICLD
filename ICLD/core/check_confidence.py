# import torch
# import torch.nn.functional as F
# from sentence_transformers.util import cos_sim


# # 获取模型后M层的概率变化（selected_token_id）
# def get_selected_token_probs_last_M(outputs, selected_token_id, model, M=5):

#     # 常见 decoder-only LM 的 final norm 位置
#     def get_final_norm(m):
        
#         candidates = [
#             "model.norm",                  # LLaMA / Qwen / Mistral
#             "transformer.ln_f",            # GPT2
#             "gpt_neox.final_layer_norm",   # GPT-NeoX
#             "model.decoder.final_layer_norm",
#             "decoder.final_layer_norm",
#             "transformer.norm",
#         ]
#         for path in candidates:
#             cur = m
#             ok = True
#             for name in path.split("."):
#                 if not hasattr(cur, name):
#                     ok = False
#                     break
#                 cur = getattr(cur, name)
#             if ok:
#                 return cur
#         return None

#     # hidden_states: [embedding输出, 第1层输出, ..., 第N层输出]
#     layer_hidden_states = outputs.hidden_states[1:]
#     num_layers = len(layer_hidden_states)

#     lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else model.lm_head
#     final_norm = get_final_norm(model)

#     probs = []

#     with torch.no_grad():
#         start_idx = max(0, num_layers - M)
#         end_idx = max(0, num_layers - 1)   # 不包含最后一层，第N层单独用 outputs.logits
#         for h in layer_hidden_states[start_idx:end_idx]:
#             x = h[:, -1, :]   # 当前 step 最后一个位置
#             if final_norm is not None:
#                 x = final_norm(x)
#             logits = lm_head(x)
#             prob = F.softmax(logits.float(), dim=-1)[0, selected_token_id].item()
#             probs.append(prob)

#         final_logits = outputs.logits[:, -1, :]
#         final_prob = F.softmax(final_logits.float(), dim=-1)[0, selected_token_id].item()
#         probs.append(final_prob)

#     return probs



# def judge_model_high_confidence(probs):

#     var_threshold = 0.1

#     probs_tensor = torch.tensor(probs, dtype=torch.float32)

#     # 总体方差
#     var = torch.var(probs_tensor, unbiased=False).item()

#     print("当前token的总体方差为：", round(var, 6), "，概率列表为：", [round(p, 6) for p in probs])

#     return var > var_threshold


# # def judge_model_high_confidence(outputs, selected_token_id, model, M=3):
# #     """
# #     判断 selected_token_id 是否在最后 M 层中始终是概率最大的 token

# #     返回：
# #         False -> 始终是 top1，不需要集成
# #         True  -> 只要有任意一层不是 top1，需要集成
# #     """

# #     def get_final_norm(m):
# #         candidates = [
# #             "model.norm",                  # LLaMA / Qwen / Mistral
# #             "transformer.ln_f",            # GPT2
# #             "gpt_neox.final_layer_norm",   # GPT-NeoX
# #             "model.decoder.final_layer_norm",
# #             "decoder.final_layer_norm",
# #             "transformer.norm",
# #         ]
# #         for path in candidates:
# #             cur = m
# #             ok = True
# #             for name in path.split("."):
# #                 if not hasattr(cur, name):
# #                     ok = False
# #                     break
# #                 cur = getattr(cur, name)
# #             if ok:
# #                 return cur
# #         return None

# #     # hidden_states: [embedding输出, 第1层输出, ..., 第N层输出]
# #     layer_hidden_states = outputs.hidden_states[1:]
# #     num_layers = len(layer_hidden_states)

# #     lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else model.lm_head
# #     final_norm = get_final_norm(model)

# #     with torch.no_grad():
# #         start_idx = max(0, num_layers - M)
# #         end_idx = max(0, num_layers - 1)  # 不包含最后一层，最后一层单独用 outputs.logits

# #         # 检查最后 M-1 个中间层
# #         for h in layer_hidden_states[start_idx:end_idx]:
# #             x = h[:, -1, :]   # 当前 step 最后一个位置
# #             if final_norm is not None:
# #                 x = final_norm(x)

# #             logits = lm_head(x)   # [1, vocab_size]
# #             top1_id = torch.argmax(logits, dim=-1)[0].item()

# #             if top1_id != selected_token_id:
# #                 print(f"候选 token 在中间层不是 top1，top1_id={top1_id}, selected_token_id={selected_token_id}")
# #                 return True   # 需要集成

# #         # 最后一层：直接使用 outputs.logits，不做归一化
# #         final_logits = outputs.logits[:, -1, :]
# #         final_top1_id = torch.argmax(final_logits, dim=-1)[0].item()

# #         if final_top1_id != selected_token_id:
# #             print(f"候选 token 在最后一层不是 top1，final_top1_id={final_top1_id}, selected_token_id={selected_token_id}")
# #             return True   # 需要集成

# #     print(f"候选 token 在最后 {M} 层始终为 top1，不需要集成")
# #     return False

# # def judge_model_high_confidence(outputs, selected_token_id, model, M=5, perturb_eps=0.005):
# #     """
# #     对最后一层最后一个位置的隐藏状态施加一个小扰动：
# #     - 若扰动后 top1 token 发生变化，则返回 True
# #     - 否则返回 False

# #     参数:
# #         outputs: 模型前向输出，需包含 hidden_states 和 logits
# #         selected_token_id: 当前候选 token（通常是最后一层的 top1 token）
# #         model: 主模型
# #         M: 为了兼容你现有调用而保留，这里不使用
# #         perturb_eps: 扰动强度，按 hidden state 范数的比例缩放，建议 0.001 ~ 0.01
# #     """

# #     if outputs.hidden_states is None or len(outputs.hidden_states) == 0:
# #         raise ValueError("outputs.hidden_states 为空，请在模型前向时设置 output_hidden_states=True")

# #     lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else model.lm_head

# #     with torch.no_grad():
# #         # 最后一层最后一个位置的隐藏状态
# #         # shape: [1, hidden_size]
# #         last_hidden = outputs.hidden_states[-1][:, -1, :]

# #         # 生成一个随机扰动方向
# #         noise = torch.randn_like(last_hidden)

# #         # 归一化噪声方向
# #         noise_norm = noise.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)

# #         # 以 hidden state 的范数为基准控制扰动大小
# #         hidden_norm = last_hidden.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)

# #         # 最终扰动 = 单位噪声方向 * perturb_eps * ||hidden||
# #         scaled_noise = noise / noise_norm.to(noise.dtype) * (perturb_eps * hidden_norm).to(noise.dtype)

# #         # 对 hidden state 加扰动
# #         perturbed_hidden = last_hidden + scaled_noise

# #         # 重新映射到词表空间
# #         perturbed_logits = lm_head(perturbed_hidden)   # [1, vocab_size]

# #         # 扰动后的 top1 token
# #         perturbed_top1_id = torch.argmax(perturbed_logits, dim=-1)[0].item()

# #         print(f"原候选 token id: {selected_token_id}")
# #         print(f"扰动后 top1 token id: {perturbed_top1_id}")

# #         # 若候选 token 发生变化，则返回 True；否则返回 False
# #         if perturbed_top1_id != selected_token_id:
# #             print("施加扰动后，候选 token 发生变化 -> 返回 True")
# #             return True
# #         else:
# #             print("施加扰动后，候选 token 未发生变化 -> 返回 False")
# #             return False


# def judge_model_high_confidence_v1(outputs, selected_token_id, model, M=5, perturb_eps=0.005):
#     """
#     逻辑：
#     1. 若 selected_token_id 在后 M 层始终为最大概率值 token，则返回 False
#     2. 否则，对最后一层最后一个位置的隐藏状态施加扰动：
#        - 若扰动后候选 token 发生改变，则返回 True
#        - 否则返回 False

#     参数:
#         outputs: 模型前向输出，需包含 hidden_states 和 logits
#         selected_token_id: 当前候选 token（通常为最后一层 top1 token）
#         model: 主模型
#         M: 回看最后 M 层
#         perturb_eps: 扰动强度，建议 0.001 ~ 0.01
#     """

#     def get_final_norm(m):
#         candidates = [
#             "model.norm",                  # LLaMA / Qwen / Mistral
#             "transformer.ln_f",            # GPT2
#             "gpt_neox.final_layer_norm",   # GPT-NeoX
#             "model.decoder.final_layer_norm",
#             "decoder.final_layer_norm",
#             "transformer.norm",
#         ]
#         for path in candidates:
#             cur = m
#             ok = True
#             for name in path.split("."):
#                 if not hasattr(cur, name):
#                     ok = False
#                     break
#                 cur = getattr(cur, name)
#             if ok:
#                 return cur
#         return None

#     if outputs.hidden_states is None or len(outputs.hidden_states) == 0:
#         raise ValueError("outputs.hidden_states 为空，请在模型前向时设置 output_hidden_states=True")

#     # hidden_states: [embedding输出, 第1层输出, ..., 第N层输出]
#     layer_hidden_states = outputs.hidden_states[1:]
#     num_layers = len(layer_hidden_states)

#     lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else model.lm_head
#     final_norm = get_final_norm(model)

#     with torch.no_grad():
#         start_idx = max(0, num_layers - M)
#         end_idx = max(0, num_layers - 1)   # 不包含最后一层，最后一层单独用 outputs.logits

#         # ---------- 第1步：判断后 M 层是否始终为 top1 ----------
#         always_top1 = True

#         # 检查最后 M-1 个中间层
#         for h in layer_hidden_states[start_idx:end_idx]:
#             x = h[:, -1, :]   # 当前 step 最后一个位置
#             if final_norm is not None:
#                 x = final_norm(x)

#             logits = lm_head(x)   # [1, vocab_size]
#             top1_id = torch.argmax(logits, dim=-1)[0].item()

#             if top1_id != selected_token_id:
#                 always_top1 = False
#                 print(f"候选 token 在中间层不是 top1，top1_id={top1_id}, selected_token_id={selected_token_id}")
#                 break

#         # 检查最后一层：直接使用 outputs.logits，不做归一化
#         if always_top1:
#             final_logits = outputs.logits[:, -1, :]
#             final_top1_id = torch.argmax(final_logits, dim=-1)[0].item()

#             if final_top1_id != selected_token_id:
#                 always_top1 = False
#                 print(f"候选 token 在最后一层不是 top1，final_top1_id={final_top1_id}, selected_token_id={selected_token_id}")

#         # 若后 M 层始终是最大概率值 token，则直接返回 False
#         if always_top1:
#             print(f"候选 token 在最后 {M} 层始终为 top1，不需要扰动，返回 False")
#             return False

#         # ---------- 第2步：若不始终为 top1，则施加扰动 ----------
#         # 最后一层最后一个位置的隐藏状态
#         last_hidden = outputs.hidden_states[-1][:, -1, :]   # [1, hidden_size]

#         # 生成随机扰动方向
#         noise = torch.randn_like(last_hidden)

#         # 归一化噪声
#         noise_norm = noise.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)

#         # 以 hidden state 范数为基准，控制扰动大小
#         hidden_norm = last_hidden.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
#         scaled_noise = noise / noise_norm.to(noise.dtype) * (perturb_eps * hidden_norm).to(noise.dtype)

#         # 加扰动
#         perturbed_hidden = last_hidden + scaled_noise

#         # 重新映射到词表空间
#         # 注意：这里为了尽量贴近模型真实输出路径，若模型存在 final_norm，则先过 final_norm 再过 lm_head
#         perturbed_x = perturbed_hidden
#         if final_norm is not None:
#             perturbed_x = final_norm(perturbed_x)

#         perturbed_logits = lm_head(perturbed_x)   # [1, vocab_size]
#         perturbed_top1_id = torch.argmax(perturbed_logits, dim=-1)[0].item()

#         print(f"原候选 token id: {selected_token_id}")
#         print(f"扰动后 top1 token id: {perturbed_top1_id}")

#         # 若候选 token 发生变化，则返回 True；否则返回 False
#         if perturbed_top1_id != selected_token_id:
#             print("候选 token 发生变化，返回 True")
#             return True
#         else:
#             print("候选 token 未发生变化，返回 False")
#             return False



# def check_main_model_confidence(main_outputs, main_model, main_tokenizer, cfg, embedding_model):

#     # 1.读取超参数
#     alpha = cfg.get("alpha", 0.8) # 判断当前Token属于高置信度Token还是低置信度Token
#     gamma = cfg.get("gamma", 0.5)
#     rho = cfg.get("rho", 0.5) # 两个token_str的语义相似度的阈值
#     M = 5

#     # 2.获取模型最后一层的Logits和概率分布
#     logits_n = main_outputs.logits[0, -1, :]
#     probs_n = F.softmax(logits_n, dim=-1)

#     # 3.提取Top-1和Top-2的概率及Token ID
#     top2_probs, top2_indices = torch.topk(probs_n, 2)
#     prob_top1 = top2_probs[0].item()
#     prob_top2 = top2_probs[1].item()
#     top1_id = top2_indices[0].item()
#     top2_id = top2_indices[1].item()

#     # 4.计算Token Margin
#     token_margin = prob_top1 - prob_top2

#     # 5.1 当前Token属于高置信度Token
#     # if token_margin >= alpha:
#     if prob_top1 >= alpha: # 当候选token的概率值大于阈值，则认为是高概率值token

#         # print("======当前Token属于高置信度Token======")
#         # return False
        
#         probs = get_selected_token_probs_last_M(main_outputs, top1_id, main_model, M)
#         return judge_model_high_confidence(probs) # 计算后M层概率列表的方差

#         # return judge_model_high_confidence(main_outputs, top1_id, main_model, M)

#         # perturb_eps = cfg.get("perturb_eps", 1)
#         # return judge_model_high_confidence(main_outputs, top1_id, main_model, M, perturb_eps)

#         # return False



#         # # A.提取模型倒数第二层的隐藏状态
#         # hidden_states_n_minus_1 = main_outputs.hidden_states[-2][0, -1, :]

#         # # B.提取倒数第二层的概率分布
#         # try:
#         #     norm_state = main_model.model.norm(hidden_states_n_minus_1)
#         # except AttributeError:
#         #     norm_state = hidden_states_n_minus_1
#         # logits_n_minus_1 = main_model.lm_head(norm_state) # 逻辑分布
#         # probs_n_minus_1 = F.softmax(logits_n_minus_1, dim=-1) # 概率分布
        

#         # # C.获取第 N-1 层对当前 Top-1 Token 的预测概率
#         # prob_top1_n_minus_1 = probs_n_minus_1[top1_id].item()

#         # print("prob_top1:", prob_top1)
#         # print("prob_top1_n_minus_1:", prob_top1_n_minus_1)
        
#         # # D.判断是否发生概率突变
#         # if abs(prob_top1 - prob_top1_n_minus_1) >= gamma:
#         #     return True  # 触发多模型协作
#         # else:
#         #     return False # 不触发
    
#     # 5.2 当前Token属于低置信度Token
#     else:

#         # return True

        
#         #################################################
#         # 解码 Top-1 和 Top-2 对应的字符串  保留前导空格或特殊字符，以便更精确比对
#         # str_top1 = main_tokenizer.decode([top1_id]).strip()
#         # str_top2 = main_tokenizer.decode([top2_id]).strip()

#         # print("str_top1:", str_top1, "str_top2:", str_top2)
        
#         # # 条件 A：字面量包含关系判断
#         # if (str_top1 in str_top2) or (str_top2 in str_top1):
#         #     return False # 属于派生/同根词，不触发协作
        
#         # # 条件 B：语义相似度判断 (复用底层 Embedding)
#         # # embedding_layer = main_model.get_input_embeddings()
        
#         # # # 将 token_id 包装成 tensor 传入 embedding 层
#         # # tensor_top1_id = torch.tensor([top1_id], device=main_model.device)
#         # # tensor_top2_id = torch.tensor([top2_id], device=main_model.device)
        
#         # # vec1 = embedding_layer(tensor_top1_id) 
#         # # vec2 = embedding_layer(tensor_top2_id) 
        
#         # # 计算余弦相似度
#         # # cos_similarity = F.cosine_similarity(vec1, vec2, dim=-1).item()
#         # # print("余弦相似度为：", cos_similarity)

#         # texts = [str_top1, str_top2]
#         # embeddings = embedding_model.encode(texts, normalize_embeddings=True)

#         # cos_similarity = cos_sim(embeddings[0], embeddings[1]).item()

#         # print("余弦相似度为：", cos_similarity)
        
#         # if cos_similarity >= rho:
#         #     return False # 语义高度相似，不触发协作
            
#         # # 若以上都不满足，则真的在两个截然不同的概念间犹豫
#         # return True # 触发多模型协作

#         #################################################


#         #################################################
#         # A. 字面量包含关系判断（保留你原来的这部分，处理简单的字面派生）
#         str_top1 = main_tokenizer.decode([top1_id]).strip().lower()
#         str_top2 = main_tokenizer.decode([top2_id]).strip().lower()
#         if (str_top1 in str_top2) or (str_top2 in str_top1):
#             return False 
        
#         # B. 方案四：基于主模型 LM Head 的语义相似度判断
#         # 提取语言模型头 (注意：不同架构的模型 lm_head 名称可能不同，如 model.embed_out)
#         lm_head_weight = main_model.lm_head.weight 
        
#         # 提取对应 Token 的内部高维权重
#         vec_top1 = lm_head_weight[top1_id]
#         vec_top2 = lm_head_weight[top2_id]
        
#         # 计算余弦相似度 (需增加 batch 维度)
#         cos_sim = F.cosine_similarity(vec_top1.unsqueeze(0), vec_top2.unsqueeze(0)).item()
        
#         # print(f"Token1: {str_top1}, Token2: {str_top2}, LM_Head 余弦相似度: {cos_sim:.4f}")
        
#         if cos_sim >= rho:
#             return False # 内部语义高度相似 (良性犹豫)，不触发协作
            
#         return True # 真的在截然不同的概念间犹豫，触发多模型协作

#         #################################################


import torch
import torch.nn.functional as F
from sentence_transformers.util import cos_sim


# 获取模型后M层的概率变化（selected_token_id）
def get_selected_token_probs_last_M(outputs, selected_token_id, model, M=5):

    # 常见 decoder-only LM 的 final norm 位置
    def get_final_norm(m):
        
        candidates = [
            "model.norm",                  # LLaMA / Qwen / Mistral
            "transformer.ln_f",            # GPT2
            "gpt_neox.final_layer_norm",   # GPT-NeoX
            "model.decoder.final_layer_norm",
            "decoder.final_layer_norm",
            "transformer.norm",
        ]
        for path in candidates:
            cur = m
            ok = True
            for name in path.split("."):
                if not hasattr(cur, name):
                    ok = False
                    break
                cur = getattr(cur, name)
            if ok:
                return cur
        return None

    # hidden_states: [embedding输出, 第1层输出, ..., 第N层输出]
    layer_hidden_states = outputs.hidden_states[1:]
    num_layers = len(layer_hidden_states)

    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else model.lm_head
    final_norm = get_final_norm(model)

    probs = []

    with torch.no_grad():
        start_idx = max(0, num_layers - M)
        end_idx = max(0, num_layers - 1)   # 不包含最后一层，第N层单独用 outputs.logits
        for h in layer_hidden_states[start_idx:end_idx]:
            x = h[:, -1, :]   # 当前 step 最后一个位置
            if final_norm is not None:
                x = final_norm(x)
            logits = lm_head(x)
            prob = F.softmax(logits.float(), dim=-1)[0, selected_token_id].item()
            probs.append(prob)

        final_logits = outputs.logits[:, -1, :]
        final_prob = F.softmax(final_logits.float(), dim=-1)[0, selected_token_id].item()
        probs.append(final_prob)

    return probs

# 计算方差
def judge_model_high_confidence_v1(probs):

    var_threshold = 0.1

    probs_tensor = torch.tensor(probs, dtype=torch.float32)

    # 总体方差
    var = torch.var(probs_tensor, unbiased=False).item()

    print("当前token的总体方差为：", round(var, 6), "，概率列表为：", [round(p, 6) for p in probs])

    return var > var_threshold

# 施加随机扰动
def judge_model_high_confidence_v2(outputs, selected_token_id, model, perturb_eps=1):
    """
    使用“最后一层隐藏状态随机扰动”来检验候选 token 的稳定性：
    - 对最后一层最后一个位置的 hidden state 施加一个随机扰动
    - 若扰动后 top1 token 发生变化，则返回 True
    - 否则返回 False

    参数:
        outputs: 模型前向输出，需包含 hidden_states 和 logits
        selected_token_id: 当前候选 token（通常是最后一层的 top1 token）
        model: 主模型
        perturb_eps: 扰动强度，按 hidden state 范数的比例缩放，建议 0.001 ~ 0.01
    """

    if outputs.hidden_states is None or len(outputs.hidden_states) == 0:
        raise ValueError("outputs.hidden_states 为空，请在模型前向时设置 output_hidden_states=True")

    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else model.lm_head

    with torch.no_grad():
        # 最后一层最后一个位置的隐藏状态
        # 注意：这里直接使用 outputs.hidden_states[-1]，不再额外过 final_norm，
        # 因为对多数 HuggingFace decoder-only 模型来说，这里通常已经是最终输出表征。
        last_hidden = outputs.hidden_states[-1][:, -1, :]   # [1, hidden_size]

        # 生成随机扰动方向
        noise = torch.randn_like(last_hidden)

        # 归一化噪声方向
        noise_norm = noise.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # 以 hidden state 的范数为基准控制扰动大小
        hidden_norm = last_hidden.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # 最终扰动 = 单位噪声方向 * perturb_eps * ||hidden||
        scaled_noise = noise / noise_norm.to(noise.dtype) * (perturb_eps * hidden_norm).to(noise.dtype)

        # 对 hidden state 加扰动
        perturbed_hidden = last_hidden + scaled_noise

        # 重新映射到词表空间
        perturbed_logits = lm_head(perturbed_hidden)   # [1, vocab_size]

        # 扰动后的 top1 token
        perturbed_top1_id = torch.argmax(perturbed_logits, dim=-1)[0].item()

        print(f"原候选 token id: {selected_token_id}")
        print(f"扰动后 top1 token id: {perturbed_top1_id}")

        # 若候选 token 发生变化，则返回 True；否则返回 False
        if perturbed_top1_id != selected_token_id:
            print("施加扰动后，候选 token 发生变化 -> 返回 True")
            return True
        else:
            print("施加扰动后，候选 token 未发生变化 -> 返回 False")
            return False



def check_main_model_confidence(main_outputs, main_model, main_tokenizer, cfg, embedding_model):

    # 1.读取超参数
    alpha = cfg.get("alpha", 0.8) # 判断当前Token属于高置信度Token还是低置信度Token
    rho = cfg.get("rho", 0.5) # 两个token_str的语义相似度的阈值
    M = 5 # 获取后M层的隐藏状态
    perturb_eps = cfg.get("perturb_eps", 1)

    # 2.获取模型最后一层的Logits和概率分布
    logits_n = main_outputs.logits[0, -1, :]
    probs_n = F.softmax(logits_n, dim=-1)

    # 3.提取Top-1和Top-2的概率及Token ID
    top2_probs, top2_indices = torch.topk(probs_n, 2)
    prob_top1 = top2_probs[0].item()
    prob_top2 = top2_probs[1].item()
    top1_id = top2_indices[0].item()
    top2_id = top2_indices[1].item()

    # 5.1 当前Token属于高置信度Token
    if prob_top1 >= alpha: # 当候选token的概率值大于阈值，则认为是高概率值token

        # 方案1：计算后 M 层概率值的方差
        probs = get_selected_token_probs_last_M(main_outputs, top1_id, main_model, M)
        return judge_model_high_confidence_v1(probs)
    
    # 5.2 当前Token属于低置信度Token
    else:

        #################################################
        # 解码 Top-1 和 Top-2 对应的字符串  保留前导空格或特殊字符，以便更精确比对
        str_top1 = main_tokenizer.decode([top1_id]).strip()
        str_top2 = main_tokenizer.decode([top2_id]).strip()
        
        # 条件 A：字面量包含关系判断
        if (str_top1 in str_top2) or (str_top2 in str_top1):
            return False # 属于派生/同根词，不触发协作

        texts = [str_top1, str_top2]
        embeddings = embedding_model.encode(texts, normalize_embeddings=True)

        cos_similarity = cos_sim(embeddings[0], embeddings[1]).item()
        
        if cos_similarity >= rho:
            return False # 语义高度相似，不触发协作
            
        # 若以上都不满足，则真的在两个截然不同的概念间犹豫
        return True # 触发多模型协作

        #################################################


        #################################################
        # # A. 字面量包含关系判断（保留你原来的这部分，处理简单的字面派生）
        # str_top1 = main_tokenizer.decode([top1_id]).strip().lower()
        # str_top2 = main_tokenizer.decode([top2_id]).strip().lower()
        # if (str_top1 in str_top2) or (str_top2 in str_top1):
        #     return False 
        
        # # B. 方案四：基于主模型 LM Head 的语义相似度判断
        # # 提取语言模型头 (注意：不同架构的模型 lm_head 名称可能不同，如 model.embed_out)
        # lm_head_weight = main_model.lm_head.weight 
        
        # # 提取对应 Token 的内部高维权重
        # vec_top1 = lm_head_weight[top1_id]
        # vec_top2 = lm_head_weight[top2_id]
        
        # # 计算余弦相似度 (需增加 batch 维度)
        # cos_sim = F.cosine_similarity(vec_top1.unsqueeze(0), vec_top2.unsqueeze(0)).item()
        
        # # print(f"Token1: {str_top1}, Token2: {str_top2}, LM_Head 余弦相似度: {cos_sim:.4f}")
        
        # if cos_sim >= rho:
        #     return False # 内部语义高度相似 (良性犹豫)，不触发协作
            
        # return True # 真的在截然不同的概念间犹豫，触发多模型协作

        #################################################