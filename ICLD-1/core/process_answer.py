import re
from fractions import Fraction
from core.extract_answer import extract_answer



def _strip_outer_boxed(text: str) -> str:
    """去掉最外层 \\boxed{...}"""
    text = str(text).strip()
    if not text.startswith(r"\boxed{") or not text.endswith("}"):
        return text

    prefix = r"\boxed{"
    inner = text[len(prefix):-1]

    balance = 0
    for ch in inner:
        if ch == "{":
            balance += 1
        elif ch == "}":
            balance -= 1
            if balance < 0:
                return text

    if balance == 0:
        return inner.strip()
    return text


def _normalize_math_text(ans: str) -> str:
    """
    规范化 Math500 数学答案字符串，尽量保留数学结构本身。
    """
    if ans is None:
        return ""

    ans = str(ans).strip()

    # 去掉常见前缀
    ans = re.sub(r"^The\s+answer\s+is\s+", "", ans, flags=re.IGNORECASE).strip()
    ans = re.sub(r"^(?:Final\s+answer|Answer|Result|Solution)\s*[:：]\s*", "", ans, flags=re.IGNORECASE).strip()

    # 去掉 markdown 强调
    ans = re.sub(r"^\*+(.*?)\*+$", r"\1", ans).strip()

    # 去掉首尾 $...$
    if len(ans) >= 2 and ans.startswith("$") and ans.endswith("$"):
        ans = ans[1:-1].strip()

    # 去掉最外层 \boxed{}
    ans = _strip_outer_boxed(ans)

    # 去掉末尾句号等
    ans = re.sub(r"[\s。．.!！;；,，]+$", "", ans).strip()

    # 压缩多余空白
    ans = re.sub(r"\s+", " ", ans).strip()

    return ans


def _normalize_math_text_loose(ans: str) -> str:
    """
    更宽松的规范化：
    - 去掉所有空白
    - 尽量用于表达式形式差异较小的比较
    """
    ans = _normalize_math_text(ans)
    ans = re.sub(r"\s+", "", ans)
    return ans


def _parse_numeric_value(text: str):
    """
    尝试把字符串解析为数值。
    支持：
    - 整数：42
    - 小数：3.14
    - 分数：3/5
    - 带逗号数字：1,000
    返回 float；失败返回 None
    """
    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None

    s = s.replace(",", "")
    s = s.strip()

    # 分数 a/b
    frac_match = re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", s)
    if frac_match:
        num = int(frac_match.group(1))
        den = int(frac_match.group(2))
        if den == 0:
            return None
        return float(Fraction(num, den))

    # 整数 / 小数 / 科学计数法
    try:
        return float(s)
    except ValueError:
        return None


def _compare_math_answers(pred_answer, correct_answer) -> int:
    """
    Math500 判断逻辑：
    1. 先严格规范化比较
    2. 再做宽松规范化比较
    3. 若两边都能解析为数值，则做数值比较
    """
    pred_norm = _normalize_math_text(pred_answer)
    gold_norm = _normalize_math_text(correct_answer)

    print("pred_answer:", pred_norm)
    print("correct_answer:", gold_norm)

    # 1. 严格规范化字符串比较
    if pred_norm == gold_norm and pred_norm != "":
        return 1

    # 2. 宽松字符串比较（忽略空格）
    pred_loose = _normalize_math_text_loose(pred_norm)
    gold_loose = _normalize_math_text_loose(gold_norm)
    if pred_loose == gold_loose and pred_loose != "":
        return 1

    # 3. 数值比较（仅当两边都可解析）
    pred_num = _parse_numeric_value(pred_norm)
    gold_num = _parse_numeric_value(gold_norm)
    if pred_num is not None and gold_num is not None:
        if abs(pred_num - gold_num) < 1e-8:
            return 1

    return 0


# 判断模型生成的答案是否正确
def check_pred_answer(response, sample, cfg):
    dataset_name = cfg.get("dataset_name")

    if dataset_name == "MMLU":
        # 1. 提取预测答案
        pred_answer = extract_answer(dataset_name, response)

        # 2. 提取正确答案
        correct_answer = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}.get(str(sample['answer']), sample['answer'])

        print("pred_answer:", pred_answer)
        print("correct_answer:", correct_answer)

        # 3. 比较答案是否正确
        is_correct = 1 if pred_answer == correct_answer else 0

        return is_correct, pred_answer, correct_answer

    elif dataset_name == "GSM8K":
        # --- 辅助函数：清理和标准化数值 ---
        def clean_number(n_str):
            if not n_str:
                return None
            clean_str = str(n_str).replace(',', '').strip()
            try:
                return float(clean_str)
            except ValueError:
                return None

        # --- 1. 提取预测答案 ---
        pred_text_raw = str(response) if response is not None else ""
        pred_answer_str = extract_answer(dataset_name, pred_text_raw)

        # --- 2. 提取正确答案 ---
        gold_text_raw = str(sample.get('answer', ''))
        print("gold_text_raw:", gold_text_raw)

        if "####" in gold_text_raw:
            correct_answer_str = gold_text_raw.split("####")[-1].strip()
        else:
            numbers = re.findall(r'-?\d+(?:\.\d+)?', gold_text_raw.replace(',', ''))
            correct_answer_str = numbers[-1] if numbers else " "

        # --- 3. 数值化比较 ---
        pred_answer = clean_number(pred_answer_str)
        correct_answer = clean_number(correct_answer_str)

        print("pred_answer:", pred_answer)
        print("correct_answer:", correct_answer)

        # --- 4. 最终判断 ---
        if pred_answer is None or correct_answer is None:
            is_correct = 1 if str(pred_answer_str).strip() == str(correct_answer_str).strip() else 0
        else:
            is_correct = 1 if abs(pred_answer - correct_answer) < 1e-6 else 0

        return is_correct, pred_answer, correct_answer

    elif dataset_name == "MATH500":
        # 1. 提取预测答案
        pred_answer = extract_answer(dataset_name, response)

        # 2. 提取正确答案
        correct_answer = sample.get("answer", "")

        # 3. 比较答案是否正确
        is_correct = _compare_math_answers(pred_answer, correct_answer)

        return is_correct, pred_answer, correct_answer


    elif dataset_name == "BOOLQ":
        # 1. 提取预测答案
        pred_answer = extract_answer(dataset_name, response)

        # 2. 提取正确答案
        correct_answer = sample['answer']  # true or false

        print("pred_answer:", pred_answer)
        print("correct_answer:", correct_answer)

        # 3. 比较答案是否正确
        is_correct = 1 if pred_answer.strip().lower() == str(correct_answer).strip().lower() else 0

        return is_correct, pred_answer, correct_answer

    elif dataset_name == "ARC-C":
        # 1. 提取预测答案
        pred_answer = extract_answer(dataset_name, response)

        # 2. 提取正确答案
        correct_answer = sample['answer']

        print("pred_answer:", pred_answer)
        print("correct_answer:", correct_answer)

        # 3. 比较答案是否正确
        is_correct = 1 if pred_answer.strip() == correct_answer.strip() else 0

        return is_correct, pred_answer, correct_answer

    return 0, 0, 0