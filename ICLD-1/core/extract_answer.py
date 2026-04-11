import re
from typing import List, Tuple, Any


def clean_text(text: str, clean_patterns: List[Tuple[str, str]]) -> str:
    """清洗生成的文本"""
    if not isinstance(text, str) or not text.strip():
        return ""
    cleaned = text.strip()
    for pattern, repl in clean_patterns:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.DOTALL)
    return cleaned


def strip_outer_boxed(text: str) -> str:
    """
    去掉最外层 \\boxed{...}
    仅处理最外层；内部 LaTeX 结构尽量保留。
    """
    text = text.strip()
    if not text.startswith(r"\boxed{") or not text.endswith("}"):
        return text

    prefix = r"\boxed{"
    inner = text[len(prefix):-1]

    # 简单平衡检查，避免误删不完整结构
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


def normalize_math_answer(ans: str) -> str:
    """标准化 MATH500 / 数学表达式答案"""
    if not isinstance(ans, str):
        return "E"

    ans = ans.strip()

    # 去掉常见前缀
    ans = re.sub(r"^The\s+answer\s+is\s+", "", ans, flags=re.IGNORECASE).strip()
    ans = re.sub(r"^(?:Final\s+answer|Answer|Result|Solution)\s*[:：]\s*", "", ans, flags=re.IGNORECASE).strip()

    # 去掉 markdown 强调
    ans = re.sub(r"^\*+(.*?)\*+$", r"\1", ans).strip()

    # 去掉首尾美元符号
    if len(ans) >= 2 and ans.startswith("$") and ans.endswith("$"):
        ans = ans[1:-1].strip()

    # 去掉最外层 boxed
    ans = strip_outer_boxed(ans)

    # 去掉收尾标点，但尽量不破坏数学表达主体
    ans = re.sub(r"[\s。．.!！;；,，]+$", "", ans).strip()

    return ans if ans else "E"


def extract_answer(dataset_name: str, generated_text: str) -> Any:
    """
    从模型生成文本中提取答案。
    支持：
    - MMLU / ARC-C -> A/B/C/D
    - GSM8K       -> int/float
    - BOOLQ       -> True/False
    - MATH500     -> 数学表达式字符串
    """
    if not isinstance(generated_text, str) or not generated_text.strip():
        return "E"

    # 通用清洗：
    # 1. 去掉特殊结束标记
    # 2. 保留 LaTeX 反斜杠，避免破坏 MATH500 数学表达式
    clean_patterns = [
        (r"<eos_end>", ""),
        (r"</eos_end>", ""),
        (r"\r\n", "\n"),
        (r"\r", "\n"),
    ]
    cleaned_text = clean_text(generated_text, clean_patterns)
    text_len = len(cleaned_text)

    # 动态尾部窗口：优先考虑文本后半段的答案
    start_50 = int(text_len * 0.5)
    start_30 = int(text_len * 0.7)
    tail_window_start = min(start_50, start_30)

    candidates: List[tuple] = []  # (match_text, position, priority)

    # -----------------------
    # 1. MMLU / ARC-C → A/B/C/D
    # -----------------------
    if dataset_name in ["MMLU", "ARC-C"]:
        patterns = [
            r"答案是\s*[:：]?\s*([ABCD])",
            r"(?:所以|因此|故)\s*答案是\s*[:：]?\s*([ABCD])",
            r"(?:所以|因此|故)\s*选项\s*([ABCD])\s*(?:正确|是正确的)",
            r"正确答案[是为]\s*[:：]?\s*([ABCD])",
            r"(?:选|选择)\s*[:：]?\s*([ABCD])",
            r"应选\s*([ABCD])",
            r"应该选\s*([ABCD])",
            r"(?:答案|正确答案|正确选项|选项|选择|答案选项)[：:]\s*(A|B|C|D)\s*$",
            r"(?:答案|正确答案|正确选项|选项|选择|答案选项)是\s*(A|B|C|D)\s*$",
            r"#+\s*([AaBbCcDd])\b",
            r"最终答案[是为]?\s*([ABCD])",
            r"[\(\[](A|B|C|D)[\)\]]",
            r"(A|B|C|D)\s*$",
            r"\b([ABCD])\b",
        ]

    # -----------------------
    # 2. GSM8K → 数字（整数/小数）
    # -----------------------   
    elif dataset_name == "GSM8K":
        patterns = [
            # 1. ✅ 最高优先级：精准匹配 "The answer is X." 在独立行（符合你新提示词要求）
            r'(?:^|\n|\s)The\s+answer\s+is\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?=\s*[\.]|\s*$)',

            # 2. 次优先级：其他英文引导语（兼容旧模型输出或变体）
            r'(?:Final\s+answer|Answer|Result|Solution)[:：]?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?)',

            # 3. 标准标记格式 #### [123] 或 #### [1,234.56]
            r'####\s*\[\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]',

            # 4. LaTeX \boxed{} 格式
            r'\\boxed\{\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\}',

            # 5. <answer> 标签格式
            r'<answer>\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*</answer>',

            # 6. JSON-like 格式
            r'"?answer"?\s*[:：]\s*["\']?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*["\']?',

            # 7. 句末或独立行的数字（避免匹配单词）
            r'(?<!\w)([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)(?!\w)(?:\s*[a-zA-Z]+)?(?:[\.!\?,;\s]|$)',

            # 8. 带括号或引号包裹的数字
            r'[\[【(「『]\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[\]】)」』]',

            # 9. 星号强调格式 *123* 或 **45.6**
            r'\*+\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\*+',

            # 10. 行首或行尾独立数字
            r'^\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$',

            # 11. 列表项格式，如 “- 42” 或 “1. 3.14”
            r'^\s*[-•*]?\s*\d*\.?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$',

            # 12. 分数支持（可选，如 "3/4" 或 "1 1/2"），需后处理
            r'(?<!\w)(\d+\s+\d+/\d+|\d+/\d+)(?!\w)',

            # 13. fallback：任意位置“最可能”的数字（最后兜底）
            r'(?<![\d,])([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![\d,])'
        ]

    # -----------------------
    # 3. MATH500 → 数学表达式
    # -----------------------
    elif dataset_name == "MATH500":
        patterns = [
            # 严格匹配建议 prompt 的最后一行格式
            r"(?:^|\n|\s)The\s+answer\s+is\s+(.+?)(?=\s*$)",

            # 常见英文结论
            r"(?:Final\s+answer|Answer|Result|Solution)\s*[:：]\s*(.+?)(?=\s*$)",

            # boxed / 标签 / JSON-like
            r"\\boxed\{(.+?)\}",
            r"<answer>\s*(.+?)\s*</answer>",
            r'"?answer"?\s*[:：]\s*["\']?(.+?)["\']?\s*$',

            # 最后一行兜底
            r"\n\s*([^\n]+)\s*$",
        ]

    # -----------------------
    # 4. BOOLQ → True / False
    # -----------------------
    elif dataset_name == "BOOLQ":
        patterns = [
            r"(?:答案|Answer|The answer is|最终答案|Final answer|Response|Result)[:：]?\s*(True|False|true|false|TRUE|FALSE)",
            r"####\s*\[\s*(True|False|true|false|TRUE|FALSE)\s*\]",
            r'(?<!\w)(True|False|true|false|TRUE|FALSE)(?!\w)',
            r'[\[【(]\s*(True|False|true|false|TRUE|FALSE)\s*[\]】)]',
            r'[\'"]\s*(True|False|true|false|TRUE|FALSE)\s*[\'"]',
            r'\*+\s*(True|False|true|false|TRUE|FALSE)\s*\*+',
            r'(?:(?<=[:\-–—\s])|^)\s*(True|False|true|false|TRUE|FALSE)\s*(?:$|[\.,!\?;])',
            r'"?answer"?\s*[:：]\s*["\']?\s*(True|False|true|false|TRUE|FALSE)\s*["\']?',
            r"\b(?:is|was|are|were)\s+(True|False|true|false|TRUE|FALSE)\b",
            r"<answer>\s*(True|False|true|false|TRUE|FALSE)\s*</answer>",
            r"^\s*(True|False|true|false|TRUE|FALSE)\s*$",
            r"^\s*[-•*]?\s*\d*\.?\s*(True|False|true|false|TRUE|FALSE)\s*$",
            r"\\boxed\{\s*(True|False|true|false|TRUE|FALSE)\s*\}",
        ]

    else:
        return "E"

    # -----------------------
    # 开始匹配
    # -----------------------
    for priority, pattern in enumerate(patterns):
        for match in re.finditer(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE):
            if match.groups():
                candidates.append((match.group(1), match.start(), priority))

    if not candidates:
        return "E"

    # 排序规则：
    # 1. 尾部窗口内优先
    # 2. 模式优先级高的优先
    # 3. 位置越靠后越优先
    candidates.sort(
        key=lambda x: (
            x[1] < tail_window_start,
            x[2],
            -x[1],
        )
    )

    best_match = str(candidates[0][0]).strip()

    # -----------------------
    # 标准化输出
    # -----------------------
    if dataset_name in ["MMLU", "ARC-C"]:
        ans = best_match.upper()
        return ans if ans in {"A", "B", "C", "D"} else "E"

    elif dataset_name == "GSM8K":
        raw_clean = best_match.replace(",", "").strip()

        # 分数处理：支持 "3/4" 和 "1 1/2"
        frac_match = re.fullmatch(r"(\d+)\s+(\d+)/(\d+)", raw_clean)
        if frac_match:
            whole = int(frac_match.group(1))
            num = int(frac_match.group(2))
            den = int(frac_match.group(3))
            if den == 0:
                return "E"
            value = whole + num / den
            return int(value) if float(value).is_integer() else value

        frac_match = re.fullmatch(r"([+-]?\d+)/(\d+)", raw_clean)
        if frac_match:
            num = int(frac_match.group(1))
            den = int(frac_match.group(2))
            if den == 0:
                return "E"
            value = num / den
            return int(value) if float(value).is_integer() else value

        # 前导零处理
        if raw_clean.startswith("-"):
            num_part = raw_clean[1:].lstrip("0")
            raw_clean = "0" if num_part == "" else "-" + num_part
        else:
            raw_clean = raw_clean.lstrip("0")
            if raw_clean == "":
                raw_clean = "0"

        try:
            num = float(raw_clean)
            return int(num) if num.is_integer() else num
        except ValueError:
            return "E"

    elif dataset_name == "BOOLQ":
        lower_val = best_match.lower()
        if lower_val == "true":
            return "True"
        elif lower_val == "false":
            return "False"
        return "E"

    elif dataset_name == "MATH500":
        return normalize_math_answer(best_match)

    return "E"