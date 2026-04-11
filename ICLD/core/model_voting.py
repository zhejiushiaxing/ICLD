import torch
import torch.nn.functional as F
import math






# 方案一：群体共识
def model_vote_scoring_consensus(best_token_ids):
    pass


# 方案二：候选模型最高概率token是否与最终融合概率最高token一致[接近群体共识]
def model_vote_scoring(best_token_ids, logits_list):

    num_models = len(logits_list)
    step_scores = [0.0] * num_models
    
    for i in range(num_models):
        if best_token_ids[i] is not None:
            # 获取模型i的最高概率token ID
            model_top_token_id = torch.argmax(logits_list[i]).item()
            # print(f"Model {i} predicted token ID: {model_top_token_id}, best token ID: {best_token_ids[i]}")
            
            # 检查模型i的最高概率token是否与融合选择的token一致
            if best_token_ids[i] == model_top_token_id:
                step_scores[i] = 1.0  # 奖励：模型预测与融合结果一致
            else:
                step_scores[i] = -0.5 # 惩罚：模型预测与融合结果不一致
        else:
            # 如果best_token_id为None，给予中性评分
            step_scores[i] = 0.0
    
    return step_scores

    
# 方案二：模型的熵【避免了“劣币驱逐良币”】
# def model_vote_scoring(best_token_ids, logits_list, temperature: float = 1.0, eps: float = 1e-12):
#     """
#     计算每个模型当前 step 的熵，并用 1-熵 作为得分（熵越小越自信，得分越高）。

#     - logits_list[i]: Tensor，常见 shape: [vocab] / [1, vocab] / [batch, vocab] / [batch, seq, vocab]
#       本函数会取“一个位置”的 logits 来算分布熵（默认取 [0] 或 [0,-1]）。
#     - best_token_ids[i] is None: 该模型本步不参与（返回 0.0）
#     - 使用归一化熵：H_norm = H / log(V)，使不同 vocab size 更可比
#     - score_i = 1 - H_norm，范围大致在 [0,1]
#     """
#     num_models = len(logits_list)
#     step_scores = [0.0] * num_models

#     for i in range(num_models):
#         # 与你原逻辑保持一致：该模型本步 token 映射失败 -> 中性 0 分
#         if best_token_ids[i] is None:
#             step_scores[i] = 0.0
#             continue

#         logits = logits_list[i]

#         # 统一到 [vocab]
#         # 常见情况：
#         # 1) [vocab]
#         # 2) [1, vocab] 或 [batch, vocab] -> 取第 0 条
#         # 3) [batch, seq, vocab] -> 取第 0 条最后一个位置
#         if logits.dim() == 2:
#             logits = logits[0]
#         elif logits.dim() > 2:
#             logits = logits[0, -1, :]

#         # 温度缩放（可选）
#         logits = logits / max(temperature, eps)

#         # 计算 H = -sum p log p（数值稳定）
#         log_p = F.log_softmax(logits, dim=-1)   # [vocab]
#         p = log_p.exp()
#         H = -(p * log_p).sum()                  # scalar tensor

#         # 归一化熵（避免 vocab size 不同导致不可比）
#         V = logits.numel()
#         H_norm = H / max(math.log(V), eps)      # ~ in [0,1]

#         # 得分：1 - 熵（熵越小越高分）
#         score = 1.0 - float(H_norm.clamp(0.0, 1.0))
#         step_scores[i] = score

#     print("step_scores:", step_scores)

#     return step_scores
