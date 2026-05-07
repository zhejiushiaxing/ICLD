from transformers import AutoTokenizer
import openai

class ModelTemplateHandler:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def apply_chat_template(self, messages, add_generation_prompt=True):
        """
        使用tokenizer的chat template功能
        messages格式: [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        """
        try:
            # 直接使用tokenizer内置的chat template
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
            return formatted_text
        except Exception as e:
            print(f"Error applying chat template: {e}")
            # fallback到手动模板
            return self._manual_template(messages, add_generation_prompt)
    
    def _manual_template(self, messages, add_generation_prompt=True):
        """手动构建模板作为fallback"""
        if "qwen" in self.tokenizer.name_or_path.lower():
            template = ""
            for msg in messages:
                if msg["role"] == "user":
                    template += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "assistant":
                    template += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            if add_generation_prompt:
                template += "<|im_start|>assistant\n"
        elif "llama" in self.tokenizer.name_or_path.lower():
            template = "[INST] "
            for msg in messages:
                if msg["role"] == "user":
                    template += f"{msg['content']}"
                elif msg["role"] == "assistant":
                    template += f" [/INST] {msg['content']} [INST] "
            if add_generation_prompt:
                template += " [/INST]"
        else:
            # 通用模板
            template = ""
            for msg in messages:
                template += f"{msg['role'].capitalize()}: {msg['content']}\n"
            if add_generation_prompt:
                template += "Assistant: "
        
        return template

# 使用示例
# def call_model_with_template(model_path, messages):
#     handler = ModelTemplateHandler(model_path)
    
#     # 应用chat template
#     prompt = handler.apply_chat_template(messages)
    
#     # 如果要调用兼容OpenAI的API
#     response = openai.ChatCompletion.create(
#         model=model_path,
#         messages=[{"role": "user", "content": prompt}],  # 或者直接传递格式化后的prompt
#         temperature=0.7,
#         max_tokens=512
#     )
    
#     return response

# # 对于本地模型推理
# def local_inference_with_template(model, tokenizer, messages):
#     handler = ModelTemplateHandler(tokenizer.name_or_path)
#     prompt = handler.apply_chat_template(messages)
    
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=512,
#             do_sample=True,
#             temperature=0.7,
#             pad_token_id=tokenizer.eos_token_id
#         )
    
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
#     # 提取助手的回答部分
#     assistant_response = extract_assistant_reply(generated_text, prompt)
#     return assistant_response