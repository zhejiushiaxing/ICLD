import torch
import torch.nn.functional as F
from sentence_transformers.util import cos_sim
import numpy as np

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

# 1.计算方差
def judge_model_high_confidence_v1(probs, var_threshold):

    threshold = var_threshold

    probs_tensor = torch.tensor(probs, dtype=torch.float32)

    # 总体方差
    var = torch.var(probs_tensor, unbiased=False).item()

    # print("当前token的总体方差为：", round(var, 6), "，概率列表为：", [round(p, 6) for p in probs])

    return var > threshold


# 2.施加随机扰动
def judge_model_high_confidence_v2(outputs, selected_token_id, model, perturb_eps):

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

        # 若候选 token 发生变化，则返回 True；否则返回 False
        if perturbed_top1_id != selected_token_id:
            return True
        else:
            return False


# 3.通过距离来判断是否需要集成
def judge_model_high_confidence_v3(probs, cfg, classified):

    if classified == 0:
        if probs[0] > 0.8:
            return True
    elif classified == 1:
        if probs[0] < 0.2:
            return True

    return False


    # # ===================== 1. 定义中心轨迹 =====================
    # dataset_name = cfg.get("dataset_name", "None")

    # if dataset_name == "MMLU":
    #     classified_list = [
    #         [0.852151, 0.954489, 0.976040, 0.976374, 0.989191],
    #         [0.085595, 0.306553, 0.654629, 0.862225, 0.983267]
    #     ]
    # elif dataset_name == "MATH500":
    #     classified_list = [
    #         [0.816760, 0.947360, 0.986274, 0.987173, 0.993230],
    #         [0.077529, 0.307679, 0.710307, 0.870399, 0.991082]
    #     ]
    # elif dataset_name == "PIQA":
    #     classified_list = [
    #         [0.811641, 0.947586, 0.973832, 0.970953, 0.986629],
    #         [0.070191, 0.244445, 0.577265, 0.823512, 0.982131]
    #     ]
    # elif dataset_name == "GSM8K":   
    #     classified_list = [ 
    #         [0.905762, 0.969398, 0.988938, 0.986156, 0.992246],
    #         [0.102558, 0.378301, 0.680454, 0.851852, 0.985141]
    #     ]
    

    # # 转 numpy
    # probs = np.array(probs, dtype=np.float32)
    # centers = np.array(classified_list, dtype=np.float32)

    # # ===================== 2. 计算欧式距离 =====================
    # # shape: (5,)
    # distances = np.linalg.norm(centers - probs, axis=1)

    # # ===================== 3. 找最近中心 =====================
    # nearest_idx = int(np.argmin(distances))

    # # ===================== 4. 判断 =====================
    # return nearest_idx == classified


def check_main_model_confidence(main_outputs, main_model, main_tokenizer, cfg):

    # 1.读取超参数
    alpha = cfg.get("alpha", 0.9) # 判断当前Token属于高置信度Token还是低置信度Token
    var_threshold = cfg.get("var_threshold", 0.1) # 方差阈值
    rho = cfg.get("sim_threshold", 0.5) # 两个token_str的语义相似度的阈值
    M = 5 # 获取后M层的隐藏状态
    # perturb_eps = cfg.get("perturb_eps", 1)
    

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

        # return True, 1, 1, 0, 0
        # return False, 1, 0, 0, 0

        ##############################################################################

        # 计算后 M 层概率值的方差
        probs = get_selected_token_probs_last_M(main_outputs, top1_id, main_model, M)
        is_ensemble = judge_model_high_confidence_v1(probs, var_threshold)
        if is_ensemble == True:

            return is_ensemble, 1, 1, 0, 0
        else:
            return is_ensemble, 1, 0, 0, 0

        ###############################################################################

        # 随机扰动
        # is_ensemble = judge_model_high_confidence_v2(main_outputs, top1_id, main_model, perturb_eps)
        # if is_ensemble == True:
        #     return is_ensemble, 1, 1, 0, 0
        # else:
        #     return is_ensemble, 1, 0, 0, 0

        ###############################################################################

        # 根据距离判断是否需要集成
        # probs = get_selected_token_probs_last_M(main_outputs, top1_id, main_model, M)

        # classified = 1
        # is_ensemble = judge_model_high_confidence_v3(probs, cfg, classified)
        # if is_ensemble == True:
        #     return is_ensemble, 1, 1, 0, 0
        # else:
        #     return is_ensemble, 1, 0, 0, 0
        ###############################################################################
    
    # 5.2 当前Token属于低置信度Token
    else:

        # return True, 0, 0, 1, 1
        # return False, 0, 0, 1, 0

        #################################################

        # A. 字面量包含关系判断
        str_top1 = main_tokenizer.decode([top1_id]).strip().lower()
        str_top2 = main_tokenizer.decode([top2_id]).strip().lower()
        if (str_top1 in str_top2) or (str_top2 in str_top1):
            return False, 0, 0, 1, 0 
        
        # B. 基于主模型 LM Head 的语义相似度判断
        # 提取语言模型头 (注意：不同架构的模型 lm_head 名称可能不同，如 model.embed_out)
        lm_head_weight = main_model.lm_head.weight 
        
        # 提取对应 Token 的内部高维权重
        vec_top1 = lm_head_weight[top1_id]
        vec_top2 = lm_head_weight[top2_id]
        
        # 计算余弦相似度 (需增加 batch 维度)
        cos_sim = F.cosine_similarity(vec_top1.unsqueeze(0), vec_top2.unsqueeze(0)).item()
        
        # print(f"Token1: {str_top1}, Token2: {str_top2}, LM_Head 余弦相似度: {cos_sim:.4f}")
        
        if cos_sim >= rho:
            return False, 0, 0, 1, 0 # 内部语义高度相似 (良性犹豫)，不触发协作
            
        return True, 0, 0, 1, 1 # 真的在截然不同的概念间犹豫，触发多模型协作

        #################################################