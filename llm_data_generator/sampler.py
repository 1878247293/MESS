"""数据采样：加载 CSV 表、解析 ground_truth、采样记录和匹配组。"""

import random
from pathlib import Path

import pandas as pd


def load_tables(data_dir: str) -> list:
    """加载目录下所有 table_*.csv 文件，按编号排序。

    Returns:
        list[pd.DataFrame]: 各表的 DataFrame 列表
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("table_*.csv"),
                       key=lambda p: int(p.stem.split("_")[1]))
    if not csv_files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到 table_*.csv 文件")

    tables = []
    for f in csv_files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        tables.append(df)
        print(f"  加载 {f.name}: {len(df)} 行, 列: {list(df.columns)}")
    return tables


def sample_records(tables: list, sample_size: int, seed: int = 42) -> list:
    """从每表中随机抽取 sample_size 条记录。

    Args:
        tables: DataFrame 列表
        sample_size: 每表抽取数量
        seed: 随机种子

    Returns:
        list[pd.DataFrame]: 每表的采样结果
    """
    rng = random.Random(seed)
    sampled = []
    for i, df in enumerate(tables):
        n = min(sample_size, len(df))
        indices = rng.sample(range(len(df)), n)
        sampled.append(df.iloc[indices].copy())
    return sampled

def format_samples_for_prompt(sampled_tables: list, columns: list,
                              max_per_table: int = 30) -> str:
    """将采样记录格式化为 LLM prompt 中的文本。"""
    parts = []
    for i, df in enumerate(sampled_tables):
        rows_text = []
        for _, row in df.head(max_per_table).iterrows():
            fields = [f"{c}: {row.get(c, '')}" for c in columns if c in df.columns]
            rows_text.append("  " + ", ".join(fields))
        parts.append(f"### Table {i} ({len(df)} 条采样记录):\n" + "\n".join(rows_text))
    return "\n\n".join(parts)
