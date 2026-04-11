from typing import Dict, List

class PromptGenerator:
    
    @staticmethod
    def generate_prompt(cfg: Dict, question: Dict) -> str:

        prompt = None
        if cfg["dataset_name"] == "CEVAL":
            # prompt = "请用中文解决以下问题，并在最后给出答案。\n"
            prompt = f"问题：{question['question']}？\n"
            prompt += f"A. {question['A']}\n"
            prompt += f"B. {question['B']}\n"
            prompt += f"C. {question['C']}\n"
            prompt += f"D. {question['D']}\n"
            prompt += "要求:\n"
            prompt += "1. 提供清晰的中文分步解答。在解释过程中，请勿提前写出最终答案。\n"
            prompt += "2. 仅在回答的最后一行，单独写出以下四种格式之一：\n"
            prompt += "答案是A。\n"
            prompt += "答案是B。\n"
            prompt += "答案是C。\n"
            prompt += "答案是D。\n"
            prompt += "3. 最终答案必须严格按上述格式书写，且只能出现在最后一行。任何提前出现的答案将被忽略。\n"
            prompt += "请开始你的解答：\n"
        # elif cfg["dataset_name"] in ["MMLU", "ARC-C"]:
        #     prompt = f"{question['question']}. \n" 
        #     prompt += f"A. {question['choices'][0]}. \n" 
        #     prompt += f"B. {question['choices'][1]}. \n" 
        #     prompt += f"C. {question['choices'][2]}. \n" 
        #     prompt += f"D. {question['choices'][3]}. \n" 
        #     prompt += "\nInstructions:\n" 
        #     prompt += "Firstly, you must provide a step-by-step explanation of your reasoning. \n" 
        #     prompt += "After your reason, the very last line must be exactly: The answer is X. (X is one of A,B,C,D)\n" 
        #     prompt += "Begin your explanation below:\n"

        elif cfg["dataset_name"] in ["MMLU"]:
            prompt = f"{question['question']}. \n"
            prompt += f"A. {question['choices'][0]}. \n"
            prompt += f"B. {question['choices'][1]}. \n"
            prompt += f"C. {question['choices'][2]}. \n"
            prompt += f"D. {question['choices'][3]}. \n"
            prompt += "\nInstructions:\n"
            prompt += "1. Provide a step-by-step explanation of your reasoning. \n"
            prompt += "2. The very last line must be exactly: The answer is X. (X is one of A,B,C,D) \n"
            prompt += "Begin your explanation below: \n"
            # prompt = f"{question['question']}. \n"
            # prompt += f"A. {question['choices'][0]}. \n"
            # prompt += f"B. {question['choices'][1]}. \n"
            # prompt += f"C. {question['choices'][2]}. \n"
            # prompt += f"D. {question['choices'][3]}. \n"
            # prompt += f"Instructions:\n"
            # prompt += "1. First, provide a step-by-step explanation. Do NOT state the final answer (like \"The answer is C\") during the explanation.\n"
            # prompt += "2. Only at the very end of your response, on a new line, write exactly one of the following:\n"
            # prompt += "The answer is A.\n"
            # prompt += "The answer is B.\n"
            # prompt += "The answer is C.\n"
            # prompt += "The answer is D.\n"
            # prompt += "3. Any answer stated before the final line will be ignored. Only the final line determines your answer.\n"
            # prompt += "Begin your explanation below:\n"

        elif cfg["dataset_name"] == "SIMPLEMATH":
            prompt = f"{question['problem']} 的答案是什么？"
            prompt += "要求:\n"
            prompt += "1. 请在回答的最后一行，单独写出：答案是X。（X 为纯数字，不带单位）\n"
            prompt += "2. 最终答案必须为阿拉伯数字（如 42），且只能出现在最后一行。\n"
            prompt += "请开始你的解答：\n"
        elif cfg["dataset_name"] == "BOOLQ":
            prompt = f"{question['passage']}\n"
            prompt += f"Based on the above background, answer the True/False question:{question['question']}. "
            prompt += "\nInstructions:\n"
            prompt += "1. Provide a step-by-step explanation of your reasoning.\n"
            prompt += "2. The very last line must be exactly: The answer is X. (X is True or False)\n"
            prompt += "Begin your explanation below:\n"

        elif cfg["dataset_name"] == "GSM8K":
            # prompt = "You are a professional problem-solving assistant. \n"
            prompt = f"{question['question']}\n"
            prompt += "Please answer questions strictly according to the following requirements. \n"
            prompt += "1. Provide a step-by-step explanation of your reasoning. \n"
            prompt += "2. The very last line must be exactly: The answer is X. (X must be a plain Arabic numeral, with no units, punctuation, or extra text) \n"
            prompt += "3. After outputting the final answer line, stop immediately. \n"
            prompt += "Begin your explanation below:\n"
            # prompt = f"{question['question']}\n"
            # prompt += "Instructions:\n"
            # prompt += "1.Please answer the following question. You should think step by step to solve it.\n"
            # prompt += "2.only at the very end of your response, on a new line, provide your final answer in the format \\boxed{YOUR_ANSWER}\n"
            # prompt += "3.Make sure there is **exactly one** \\boxed{YOUR_CHOICE} in your whole response.\n"
            # prompt += "Begin your explanation below:\n"

            # prompt += "1. First, provide a clearly explanation. Do NOT state the final answer during your reasoning.\n"
            # prompt += "2. Only at the very end of your response, on a new line, write exactly: The answer is X.\n"
            # prompt += "X must be a plain Arabic numeral (e.g., 42), with no units, punctuation, or extra text.\n"
            # prompt += "3. Any answer stated before the final line will be ignored. Only the final line determines your answer.\n"
            # prompt += "Begin your explanation below:\n"

        elif cfg["dataset_name"] == "MATH500":
            prompt = f"{question['question']}\n"
            prompt += "\nInstructions:\n"
            prompt += "1. Solve the question step by step with clear and correct mathematical reasoning.\n"
            prompt += "2. Keep the reasoning concise but sufficient to justify the result.\n"
            prompt += "3. The final answer may be a number, fraction, decimal, radical, expression, equation, or coordinate, depending on the question.\n"
            prompt += "4. The very last line must be exactly: The answer is <final_answer>\n"
            prompt += "5. Replace <final_answer> with only the final mathematical answer, with no words, no explanation, no \\boxed{}, and no extra punctuation.\n"
            prompt += "6. Do not write anything after the final answer line.\n"
            prompt += "\nBegin your solution below:\n"
        elif cfg["dataset_name"] == "HUMANEVAL":
            prompt = "Complete the following Python function.\n"
            prompt += "Return only the completed Python code, and nothing else.\n"
            prompt += "Do not include explanations, comments outside the code, test cases, or markdown code fences.\n"
            prompt += "Preserve the given function signature exactly.\n"
            prompt += "Ensure the solution is correct, self-contained, and executable in Python.\n\n"
            prompt += f"{question['question']}\n"
        # elif cfg["dataset_name"] == "MULTILINGUAL":
        #     prompt = "Answer the following question in the same language as the question.\n"
        #     prompt += "Provide a concise and factual answer.\n"
        #     prompt += "Return only the answer text, and nothing else.\n"
        #     prompt += f"question: {question['question']}\n"
        #     prompt += "Answer:\n"
        # elif cfg["dataset_name"] == "MATHEVAL":
        #     prompt = "Solve the following math word problem step by step.\n"
        #     prompt += "Show your reasoning clearly.\n"
        #     prompt += "End with the final answer.\n"
        #     prompt += f"Problem: {question['question']}\n"
        #     prompt += "Solution:\n"
        elif cfg["dataset_name"] == "CODINGEVAL":
            prompt = "Solve the following programming task.\n"
            prompt += "Return only the code solution, and nothing else.\n"
            prompt += "Do not include explanations, comments outside the code, or markdown code fences.\n"
            prompt += "Ensure the code is correct, self-contained, and executable.\n\n"
            prompt += f"{question['question']}\n"
        elif cfg["dataset_name"] == "MULTILINGUAL":
            prompt = "You are a multilingual question-answering assistant.\n"
            prompt += "Your task is to answer the given question accurately, naturally, and fluently.\n"
            prompt += f"Question: {question['question']}\n"
            prompt += "Begin your answer below:\n"

        elif cfg["dataset_name"] == "MATHEVAL":
            prompt = f"{question['question']}\n"
            prompt += "\nInstructions:\n"
            prompt += "1. Solve the problem step by step.\n"
            prompt += "2. Follow any formatting requirement stated in the problem exactly (for example, simplest radical form).\n"
            prompt += "3. Simplify the final answer as much as possible.\n"
            prompt += "4. Use plain text math notation for the final answer when possible (for example: 3/4, sqrt(2), 2*pi, (-1,2)).\n"
            prompt += "5. The very last line must be exactly: The answer is X\n"
            prompt += "6. In the last line, X must be only the final mathematical answer, with no extra words, no surrounding quotation marks, and no trailing punctuation.\n"
            prompt += "Begin your solution below:\n"

        return prompt

    
