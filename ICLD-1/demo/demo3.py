from transformers import AutoTokenizer

# 替换为你想要查询的模型路径
model_paths = [
    "Qwen/Qwen2.5-7B-Instruct",
    "tencent/Hunyuan-7B-Instruct",
    "/mnt/Data/zhoujiaxing/models/huggingface_cache/models--Meta-Llama-3.1-8B-Instruct",
]

for path in model_paths:
    try:
        print(f"--- Checking: {path} ---")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        
        print(f"unk_token_id: {tokenizer.unk_token_id}")
        print(f"pad_token_id: {tokenizer.pad_token_id}")
        print(f"eos_token_id: {tokenizer.eos_token_id}")
        print(f"vocab_size: {tokenizer.vocab_size}")
        
        # 推荐作为 fallback 的 ID 优先级：unk -> pad -> eos
        fallback = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.pad_token_id
        if fallback is None:
            fallback = tokenizer.eos_token_id
            
        print(f"Recommended Fallback ID: {fallback}")
        print("\n")
    except Exception as e:
        print(f"Error loading {path}: {e}\n")