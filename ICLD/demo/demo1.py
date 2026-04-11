import re

def extract_last_number(text: str) -> int | None:
    """
    提取字符串中最后出现的数字（整数）
    Args:
        text: 输入字符串（如包含计算过程和结果的文本）
    Returns:
        int: 最后出现的数字；若无数字返回None
    """
    # 正则匹配所有整数（包括正负数，如需支持小数可改为 r'-?\d+\.?\d*'）
    numbers = re.findall(r'-?\d+', text)
    # 取最后一个数字并转为整数，无数字则返回None
    return int(numbers[-1]) if numbers else None

# ====================== 测试示例 ======================
if __name__ == "__main__":
    # 测试用例1：你的原始示例
    text1 = """Natalia sold 48/2 = <<48/2=24>>24 clips in May.
    Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
    #### 72"""
    print(extract_last_number(text1))  # 输出：72

    # 测试用例2：末尾有空格/符号
    text2 = "Total: 123, Final: 456 "
    print(extract_last_number(text2))  # 输出：456

    # 测试用例3：无数字
    text3 = "No numbers here"
    print(extract_last_number(text3))  # 输出：None

    # 测试用例4：包含负数
    text4 = "Profit: -50, Loss: -100"
    print(extract_last_number(text4))  # 输出：-100