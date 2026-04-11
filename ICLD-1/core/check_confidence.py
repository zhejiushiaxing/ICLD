import torch
import torch.nn.functional as F
from sentence_transformers.util import cos_sim


# 施加随机扰动
def judge_model_high_confidence(outputs, selected_token_id, model, perturb_eps=1):

    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else model.lm_head

    with torch.no_grad():
        # 最后一层最后一个Token的隐藏状态
        last_hidden = outputs.hidden_states[-1][:, -1, :]   # 向量：[1*d]

        # 生成随机扰动方向
        noise = torch.randn_like(last_hidden) # 向量：[1*d]

        # 归一化噪声方向
        noise_norm = noise.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) # clamp_min 防止除零

        # 计算隐藏状态的范数[1*1]，用于按比例缩放扰动
        hidden_norm = last_hidden.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # 最终扰动 = 单位噪声方向 * perturb_eps扰动强度 * ||hidden||
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


# 计算后M层概率值的方差
def judge_model_high_confidence(probs):

    var_threshold = 0.1

    probs_tensor = torch.tensor(probs, dtype=torch.float32)

    # 总体方差
    var = torch.var(probs_tensor, unbiased=False).item()

    print("当前token的总体方差为：", round(var, 6), "，概率列表为：", [round(p, 6) for p in probs])

    return var > var_threshold


def check_main_model_confidence(main_outputs, main_model, main_tokenizer, cfg):

    # 1.读取超参数
    alpha = cfg.get("alpha", 0.8) # 判断当前Token属于高置信度Token还是低置信度Token
    gamma = cfg.get("gamma", 0.5)
    rho = cfg.get("rho", 0.5) # 两个token_str的语义相似度的阈值
    M = 5
    perturb_eps = cfg.get("perturb_eps", 1) # 随机扰动的强度参数

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

        probs = get_selected_token_probs_last_M(main_outputs, top1_id, main_model, M)
        return judge_model_high_confidence(probs) # 计算后M层概率列表的方差

        # 对最后一层 hidden state 施加随机扰动，判断候选 token 是否发生变化
        # return judge_model_high_confidence(
        #     main_outputs,
        #     top1_id,
        #     main_model,
        #     perturb_eps=perturb_eps,
        # )
    
    # 5.2 当前Token属于低置信度Token
    else:

        # return False

        #################################################
        # A. 字面量包含关系判断（保留你原来的这部分，处理简单的字面派生）
        str_top1 = main_tokenizer.decode([top1_id]).strip().lower()
        str_top2 = main_tokenizer.decode([top2_id]).strip().lower()
        if (str_top1 in str_top2) or (str_top2 in str_top1):
            return False 
        
        # B. 基于主模型 LM Head 的语义相似度判断
        # 提取语言模型头 (注意：不同架构的模型 lm_head 名称可能不同，如 model.embed_out)
        # 获取语言模型头的权重矩阵。在大多数 HF 模型中，该矩阵与 embed_tokens 共享（Weight Tying），形状为 [vocab_size, hidden_dim]，每一行即该 token 的语义向量。
        lm_head_weight = main_model.lm_head.weight 
        
        # C. 提取对应Token的高维语义向量，形状 [hidden_dim]
        vec_top1 = lm_head_weight[top1_id]
        vec_top2 = lm_head_weight[top2_id]
        
        # D. 计算余弦相似度
        cos_sim = F.cosine_similarity(vec_top1.unsqueeze(0), vec_top2.unsqueeze(0)).item()
        
        if cos_sim >= rho:
            return False # 内部语义高度相似 (良性犹豫)，不触发协作
            
        return True # 真的在截然不同的概念间犹豫，触发多模型协作

        #################################################