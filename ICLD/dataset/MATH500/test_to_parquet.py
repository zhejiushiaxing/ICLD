import json
import pandas as pd


def jsonl_to_parquet(jsonl_path: str, parquet_path: str):
    rows = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_num} 行不是合法 JSON: {e}") from e

    if not rows:
        raise ValueError("输入文件为空，无法转换。")

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    print(f"转换完成: {jsonl_path} -> {parquet_path}")
    print(f"共写入 {len(df)} 条记录，字段: {list(df.columns)}")


if __name__ == "__main__":
    jsonl_to_parquet("test.jsonl", "test.parquet")