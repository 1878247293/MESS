"""数据准备：读 CSV、采样、把样本拼成 prompt 文本。"""

import random
from pathlib import Path

import pandas as pd


def load_tables(data_dir: str) -> list:
    """读 data_dir 下所有 table_*.csv，按编号排序"""
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("table_*.csv"),
                       key=lambda p: int(p.stem.split("_")[1]))
    if not csv_files:
        raise FileNotFoundError(f"{data_dir} 下没有 table_*.csv")

    tables = []
    for f in csv_files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        tables.append(df)
        print(f"  read {f.name}: {len(df)} rows, cols={list(df.columns)}")
    return tables


def sample_records(tables: list, sample_size: int, seed: int = 42) -> list:
    """每张表抽 sample_size 行，返回新的 df 列表"""
    rng = random.Random(seed)
    sampled = []
    for i, df in enumerate(tables):
        n = min(sample_size, len(df))
        indices = rng.sample(range(len(df)), n)
        sampled.append(df.iloc[indices].copy())
    return sampled

def format_samples_for_prompt(sampled_tables: list, columns: list,
                              max_per_table: int = 30) -> str:
    """把每张表的前 max_per_table 行拼成 prompt 用的纯文本"""
    parts = []
    for i, df in enumerate(sampled_tables):
        rows_text = []
        for _, row in df.head(max_per_table).iterrows():
            fields = [f"{c}: {row.get(c, '')}" for c in columns if c in df.columns]
            rows_text.append("  " + ", ".join(fields))
        parts.append(f"### Table {i} ({len(df)} 条采样记录):\n" + "\n".join(rows_text))
    return "\n\n".join(parts)
