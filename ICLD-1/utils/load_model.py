from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型和分词器
def load_model(model_path, cfg, device="cuda"):

    dtype = cfg.get("dtype") 

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        # device_map=device,
        trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=dtype, 
        device_map=device,
        # attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
        trust_remote_code=True)

    model.eval()

    return model, tokenizer