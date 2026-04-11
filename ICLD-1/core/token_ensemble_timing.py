import torch
import torch.nn.functional as F
import numpy as np

# 判断当前token是否需要多模型集成【中间层到最后次层的KL散度】
def token_ensemble_judge_1(all_hidden_states, next_token_logits, main_model, cfg):
    
    # A. 阈值
    uncertainty_threshold = cfg.get("uncertainty_threshold")
    uncertainty_value = 0.0

    # B. 取出每一层的Logits，并计算该层的Logits与最后一层Logits的KL散度
    
    # 1. 准备目标分布 (最后一层的概率分布)
    final_probs = torch.softmax(next_token_logits, dim=-1).detach().to(main_model.device)
    
    # 2. 自动寻找归一化层 (Norm Layer)
    norm_layer = None
    if hasattr(main_model, 'model') and hasattr(main_model.model, 'norm'):
        norm_layer = main_model.model.norm
    elif hasattr(main_model, 'transformer') and hasattr(main_model.transformer, 'ln_f'):
        norm_layer = main_model.transformer.ln_f
    
    # 3. 遍历中间层 (Layer Iteration)
    num_layers = len(all_hidden_states)
    start_layer = num_layers // 2 
    
    for i in range(start_layer, num_layers - 1): 
        
        # 3.1 获取第 i 层的隐状态 (在 main_model.device 上)
        h_state = all_hidden_states[i][:, -1, :] 
        
        # 3.2 应用归一化
        if norm_layer is not None:
            h_state = norm_layer(h_state)
            
        # 3.3 投影到词表空间
        layer_logits = main_model.lm_head(h_state)
        
        # 3.4 维度对齐
        if layer_logits.shape[0] == 1:
            layer_logits = layer_logits.squeeze(0)
            
        # 3.5 计算 KL 散度
        layer_log_probs = torch.log_softmax(layer_logits, dim=-1)
        kl_val = F.kl_div(layer_log_probs, final_probs, reduction='sum')
        
        uncertainty_value += kl_val.item()
    
    print("uncertainty_value:", uncertainty_value)

    return True if uncertainty_value >= uncertainty_threshold else False

# 判断当前token是否需要多模型集成【取最后k层的平均值】
def token_ensemble_judge_2(all_hidden_states, next_token_logits, main_model, cfg):

    check_window = cfg.get("uncertainty_check_windows", 5) 
    uncertainty_threshold = cfg.get("uncertainty_threshold", 1)
    
    # 1. 准备目标
    final_probs = torch.softmax(next_token_logits, dim=-1).detach()
    
    # 2. 获取 Norm 层
    norm_layer = main_model.model.norm if hasattr(main_model, 'model') else main_model.transformer.ln_f
    
    num_layers = len(all_hidden_states)
    # 起始层设为倒数第 K 层
    start_layer = max(0, num_layers - 1 - check_window)
    
    total_kl = 0.0
    count = 0
    
    for i in range(start_layer, num_layers - 1):
        h_state = all_hidden_states[i][:, -1, :]
        if norm_layer is not None:
            h_state = norm_layer(h_state)
            
        # 投影
        layer_logits = main_model.lm_head(h_state)
        
        # 计算 KL
        layer_log_probs = torch.log_softmax(layer_logits, dim=-1).to(final_probs.device)


        # reduction='batchmean' 更标准，或者保持 sum
        kl_val = F.kl_div(layer_log_probs, final_probs, reduction='sum')
        
        total_kl += kl_val.item()
        count += 1
        
    avg_kl = total_kl / count if count > 0 else 0.0

    # print("avg_kl:", avg_kl)
    
    return avg_kl >= uncertainty_threshold



# 判断当前token是否需要多模型集成【取最后k层的平均值】
def token_ensemble_judge_3(all_hidden_states, next_token_logits, main_model, cfg):

    check_window = cfg.get("uncertainty_check_windows", 5) 
    uncertainty_threshold = cfg.get("uncertainty_threshold", 1) # 注意：相邻层的差异通常较小，阈值可能需要调低
    
    # 1. 获取 Norm 层 (保持不变)
    norm_layer = main_model.model.norm if hasattr(main_model, 'model') else main_model.transformer.ln_f
    
    num_layers = len(all_hidden_states)
    # 起始层设为倒数第 K 层
    start_layer = max(0, num_layers - 1 - check_window)
    
    total_kl = 0.0
    count = 0
    
    # --- 关键修改：循环范围 ---
    # 我们需要访问 i 和 i+1，所以循环到 num_layers - 2 即可
    # 这样 i+1 最大为 num_layers - 1 (即最后一个 hidden_state)
    for i in range(start_layer, num_layers - 1):
        
        # --- A. 获取相邻两层的 Hidden States ---
        h_curr = all_hidden_states[i][:, -1, :]      # 第 i 层
        h_next = all_hidden_states[i+1][:, -1, :]    # 第 i+1 层
        
        # --- B. 归一化 ---
        if norm_layer is not None:
            h_curr = norm_layer(h_curr)
            h_next = norm_layer(h_next)
            
        # --- C. 投影到词表空间 ---
        # 这一步开销稍大，但为了计算 KL 是必须的
        logits_curr = main_model.lm_head(h_curr)
        logits_next = main_model.lm_head(h_next)
        
        # --- D. 计算 KL(Next || Current) ---
        # 我们把“下一层”(Next) 视为更完善的目标分布(Target)
        # 把“当前层”(Current) 视为近似分布(Input)
        # KL 衡量的是：从当前层演化到下一层，信息改变了多少
        
        # Target: 下一层的概率分布
        target_probs = torch.softmax(logits_next, dim=-1)
        
        # Input: 当前层的对数概率
        # 注意设备对齐：确保 logits_curr 和 target_probs 在同一设备
        if logits_curr.device != target_probs.device:
             logits_curr = logits_curr.to(target_probs.device)
             
        input_log_probs = torch.log_softmax(logits_curr, dim=-1)
        
        # 计算 KL 散度
        kl_val = F.kl_div(input_log_probs, target_probs, reduction='sum')
        
        total_kl += kl_val.item()
        count += 1
        
    avg_kl = total_kl / count if count > 0 else 0.0

    print(f"Step-wise Avg KL: {avg_kl:.5f}")
    
    return avg_kl >= uncertainty_threshold


"""
基于统计物理“相变与结晶”思想的集成判断函数。
通过检测末端层级的'系统温度'(Mean KL)和'热涨落'(Std KL)来判断是否结晶失败。
"""
def crystallization_judge_1(all_hidden_states, next_token_logits, main_model, cfg):
    
    # --- 1. 物理参数配置 ---
    # 能量阈值 (Energy Threshold): 相当于平均 KL 散度的容忍上限
    # 如果超过这个值，说明系统温度太高，还是液态
    energy_threshold = cfg.get("energy_threshold", 1) 
    
    # 稳定性阈值 (Stability Threshold): 相当于 KL 散度的标准差容忍上限
    # 如果超过这个值，说明晶格在震动，结晶不稳定
    fluctuation_threshold = cfg.get("fluctuation_threshold", 1)
    
    # 观测窗口 (Observation Window): 只观测最后几层（结晶区）
    check_window = 5 
    
    # --- 2. 准备目标场 (Target Field) ---
    # 最终的 Logits 就像是晶体生长的“引力场”或“模具”
    final_probs = torch.softmax(next_token_logits, dim=-1).detach()
    
    # --- 3. 获取归一化算子 ---
    norm_layer = None
    if hasattr(main_model, 'model') and hasattr(main_model.model, 'norm'):
        norm_layer = main_model.model.norm
    elif hasattr(main_model, 'transformer') and hasattr(main_model.transformer, 'ln_f'):
        norm_layer = main_model.transformer.ln_f
        
    # --- 4. 采集“退火”过程数据 ---
    num_layers = len(all_hidden_states)
    start_layer = max(0, num_layers - 1 - check_window)
    
    kl_values = [] # 存储能量序列
    
    for i in range(start_layer, num_layers - 1): # 不包含最后一层本身(KL为0)
        # 获取粒子状态
        h_state = all_hidden_states[i][:, -1, :] 
        if norm_layer is not None:
            h_state = norm_layer(h_state)
            
        # 投影到相空间 (Logits)
        layer_logits = main_model.lm_head(h_state)
        
        # 计算该层粒子的势能 (与最终稳态的 KL 散度)
        layer_log_probs = torch.log_softmax(layer_logits, dim=-1).to(final_probs.device)
        kl_val = F.kl_div(layer_log_probs, final_probs, reduction='batchmean')
        kl_values.append(kl_val.item())
        
    # --- 5. 物理状态判定 ---
    if not kl_values:
        return False
        
    kl_tensor = torch.tensor(kl_values)
    
    # 指标 A: 系统温度 (Mean Energy) - 整体是否还有很大差异？
    system_temperature = torch.mean(kl_tensor).item()
    
    # 指标 B: 热涨落 (Thermal Fluctuation) - 差异是否在剧烈抖动？
    # Unbiased=False 对应 numpy 的 ddof=0，适合小样本
    thermal_fluctuation = torch.std(kl_tensor, unbiased=False).item()
    
    # --- 6. 综合判决 ---

    print(f"System Temperature: {system_temperature}, Thermal Fluctuation: {thermal_fluctuation}")
    # 判据 1: 温度过高 (还未冷却) -> 液态 -> 需要集成
    is_high_temp = system_temperature > energy_threshold
    
    # 判据 2: 涨落过大 (晶格震荡) -> 缺陷 -> 需要集成
    is_unstable = thermal_fluctuation > fluctuation_threshold
    
    # 只要满足任意一个条件，就认为“完美结晶”失败，触发集成
    return is_high_temp or is_unstable



def crystallization_judge_2(all_hidden_states, next_token_logits, main_model, cfg):
    
    # --- 1. 物理常数配置 (建议写入 cfg) ---
    # 林德曼常数 (Lindemann Constant): 经典物理值为 0.1 (10%)
    # 在高维语义空间中，通常 0.05 ~ 0.1 是一个敏感的相变临界点
    LINDEMANN_THRESHOLD = cfg.get("lindemann_threshold", 0.35)
    
    # 取向序阈值: 1.0 代表完全平行。0.99 代表非常有序。
    # 这里的阈值是余弦距离 (1 - cos)，所以 0.01 代表高度一致。
    ORIENTATIONAL_THRESHOLD = cfg.get("orientational_threshold", 0.08)
    
    # 观测窗口: 只研究"凝固点"附近的动力学 (最后 5 层)
    check_window = 5
    
    # --- 2. 提取微观状态轨迹 ---
    num_layers = len(all_hidden_states)
    start_layer = max(0, num_layers - 1 - check_window)
    
    # 提取最后几层的 Hidden States (Layer, Batch, Hidden_Dim)
    # 假设 Batch=1，去处 Batch 维度 -> (Window, Hidden_Dim)
    trajectory = []
    
    # 自动寻找归一化层 (Norm Layer 是为了对齐物理尺度，非常重要)
    norm_layer = None
    if hasattr(main_model, 'model') and hasattr(main_model.model, 'norm'):
        norm_layer = main_model.model.norm
    elif hasattr(main_model, 'transformer') and hasattr(main_model.transformer, 'ln_f'):
        norm_layer = main_model.transformer.ln_f

    for i in range(start_layer, num_layers - 1): 
        h = all_hidden_states[i][:, -1, :] 
        if norm_layer is not None:
            h = norm_layer(h)
        trajectory.append(h)
        

    traj_tensor = torch.stack(trajectory).squeeze(1) 

        
    # --- 3. 计算指标 I: 林德曼比率 (Lindemann Ratio) ---
    # 物理意义: 振动幅度 / 晶格常数
    
    # A. 计算平衡位置 (质心 Centroid)
    centroid = torch.mean(traj_tensor, dim=0) # (Hidden_Dim)
    
    # B. 计算热振动 (Thermal Vibration / RMSD)
    # 每一层向量到质心的欧几里得距离
    diff = traj_tensor - centroid
    displacement_sq = torch.sum(diff ** 2, dim=-1) # (Window_Size)
    rmsd = torch.sqrt(torch.mean(displacement_sq)) # 标量: 均方根位移
    
    # C. 计算特征尺度 (Characteristic Scale / Lattice Constant)
    # 使用质心的模长作为归一化因子
    scale = torch.norm(centroid, p=2)
    
    # D. 林德曼比率
    lindemann_ratio = (rmsd / (scale + 1e-8)).item()
    
    
    # --- 4. 计算指标 II: 取向序参数 (Orientational Order) ---
    # 物理意义: 检测向量在演化过程中方向是否还在剧烈摆动
    
    # 计算相邻层之间的余弦相似度
    # H_t 与 H_{t+1} 的相似度
    vecs_t = traj_tensor[:-1]
    vecs_t1 = traj_tensor[1:]
    
    # Cosine Similarity: (A . B) / (|A| |B|)
    cos_sim = F.cosine_similarity(vecs_t, vecs_t1, dim=-1)
    

    angular_disorder = (1.0 - torch.mean(cos_sim)).item()
    
    
    # --- 5. 相变判决 (Phase Transition Judgement) ---
    print(f"lindemann_ratio: {lindemann_ratio}, angular_disorder: {angular_disorder}")
    
    # 判据 A: 结构融化 (振幅太大) -> 液态
    is_melting = lindemann_ratio > LINDEMANN_THRESHOLD
    
    # 判据 B: 取向无序 (方向乱变) -> 顺磁/各向同性相
    is_disordered = angular_disorder > ORIENTATIONAL_THRESHOLD
    
    # 只要满足任意一个物理不稳定性，就认为"结晶失败"，需要集成
    return is_melting or is_disordered




# 方案1：参考股市规则
"""
我需要做一些先验实验判断

对照实验组：
--- 1.选择性集成
--- 2.全局集成
--- 3.单独推理

"""
def token_ensemble_judge_stocks(all_hidden_states, next_token_logits, main_model, cfg, main_choose_prob_list):
    pass




# 方案2：训练一个预测器




# 方案3：概率累加
def token_ensemble_judge_prob_accumulate(all_hidden_states, next_token_logits, main_model, cfg):
    
    # A. 阈值与配置
    """
    prob_accum_threshold = 0, acc = 0.56, ensemble_ratio = 0                65,15,20
    prob_accum_threshold = 1, acc = 6, ensemble_ratio = 0.11
    prob_accum_threshold = 2, acc = 0.60, ensemble_ratio = 0.25
    prob_accum_threshold = 3, acc = 0.61, ensemble_ratio = 0.39
    prob_accum_threshold = 4, acc = 0.58, ensemble_ratio = 0.52             65,15,20
    prob_accum_threshold = 5, acc = 7, ensemble_ratio = 0.67
    prob_accum_threshold = 6, acc = 7, ensemble_ratio = 0.76
    prob_accum_threshold = 10, acc = 0.58, ensemble_ratio = 1               65,15,20
    """
    prob_accum_threshold = cfg.get("prob_accum_threshold", 4 )
    check_last_n_layers = cfg.get("check_last_n_layers", 10) # 回溯检查的层数
    accumulated_prob = 0.0
    layer_probs_list = []

    # B. 准备目标 Token
    final_probs = torch.softmax(next_token_logits, dim=-1)
    top_prob, top_id = torch.max(final_probs, dim=-1)
    target_token_id = top_id.item()

    # C. 自动寻找归一化层 (Norm Layer)
    norm_layer = None
    if hasattr(main_model, 'model') and hasattr(main_model.model, 'norm'):
        norm_layer = main_model.model.norm # Llama, Mistral 等
    elif hasattr(main_model, 'transformer') and hasattr(main_model.transformer, 'ln_f'):
        norm_layer = main_model.transformer.ln_f # GPT-2, Bloom 等
    elif hasattr(main_model, 'norm'):
        norm_layer = main_model.norm # Qwen 等

    # D. 遍历指定范围的层 (Layer Iteration)
    num_layers = len(all_hidden_states)
    
    # 确定遍历范围
    end_layer = num_layers
    start_layer = end_layer - check_last_n_layers + 1 # 32-10+1=23 (包含23-32共10层)
    
    for i in range(start_layer, end_layer):

        if accumulated_prob >= prob_accum_threshold:
            return False, accumulated_prob # 提前判定为不集成，节省计算资源 
        
        # D.1 获取第 i 层的隐状态
        # all_hidden_states[i] shape: (batch_size, seq_len, hidden_size)
        h_state = all_hidden_states[i][:, -1, :].to(main_model.device)
        
        # D.2 应用归一化
        # 如果不加 Norm，直接过 lm_head 会导致概率分布极度平坦或错误
        if norm_layer is not None:
            h_state = norm_layer(h_state)
            
        # D.3 投影到词表空间
        if hasattr(main_model, "lm_head"):
            layer_logits = main_model.lm_head(h_state)
        elif hasattr(main_model, "output"):
            layer_logits = main_model.output(h_state)
        else:
            continue # 防御性编程

        # D.4 维度对齐
        if layer_logits.dim() == 1:
            layer_logits = layer_logits.unsqueeze(0)
            


        # D.5 计算该层对【目标 Token】的概率
        layer_probs = torch.softmax(layer_logits, dim=-1)
        layer_target_prob = layer_probs[0, target_token_id].item()
        
        accumulated_prob += layer_target_prob
        layer_probs_list.append(layer_target_prob)
    
    # 将最后一层的概率作为初始值加入累加
    accumulated_prob += top_prob.item()

    print("==============================================================================")
    print("accumulated_prob:", accumulated_prob)
    print("layer_probs_list:", layer_probs_list)
    print("==============================================================================")

    # E. 判决逻辑
    return False if accumulated_prob > prob_accum_threshold else True, accumulated_prob



# 方案4：概率状态变化