import re

# 检测填空题生成文本中是否重复输出答案（触发停止生成）
import re

def check_filled_answer_duplication(check_window_text: str, repeat_threshold: int = 2) -> bool:
    # 1) 选择题/布尔题：The answer is A/B/C/D 或 The answer is True/False
    if re.search(r"The answer is (?:[A-D]|True|False)[\.]?$", check_window_text, flags=re.IGNORECASE):
        return True

    # 2) 数值填空：The answer is 数值（支持货币符号、小数点、空格、末尾句号）
    answer_pattern = r"The answer is\s*([$€¥]?\d+(\.\d+)?)\s*\.?"
    matches = re.findall(answer_pattern, check_window_text, flags=re.IGNORECASE)

    # 提取匹配到的答案数值（去符号/标点，只保留数字和小数点）
    answer_values = []
    for match in matches:
        pure_num = re.sub(r"[^0-9\.]", "", match[0])
        answer_values.append(pure_num)

    # 3) 检测数值重复
    if len(answer_values) >= repeat_threshold:
        from collections import Counter
        num_counter = Counter(answer_values)
        _, count = num_counter.most_common(1)[0]
        if count >= repeat_threshold:
            return True

    return False

# 扩展：匹配"Final answer"类标识（优先终止）
def check_final_answer(check_window_text: str) -> bool:
    """检测到最终答案标识时，直接停止生成"""
    final_pattern = r"The final answer is\s*([$€¥]?\d+(\.\d+)?)\s*\.?"
    return bool(re.search(final_pattern, check_window_text, flags=re.IGNORECASE))


