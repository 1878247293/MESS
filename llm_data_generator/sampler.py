"""Data preparation: read CSVs, sample, and assemble the samples into prompt text."""

import random
from pathlib import Path

import pandas as pd


def load_tables(data_dir: str) -> list:
    """Read all table_*.csv files under data_dir, sorted by number"""
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("table_*.csv"),
                       key=lambda p: int(p.stem.split("_")[1]))
    if not csv_files:
        raise FileNotFoundError(f"no table_*.csv found under {data_dir}")

    tables = []
    for f in csv_files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        tables.append(df)
        print(f"  read {f.name}: {len(df)} rows, cols={list(df.columns)}")
    return tables


def sample_records(tables: list, sample_size: int, seed: int = 42) -> list:
    """Sample sample_size rows from each table, returning a new list of DataFrames"""
    rng = random.Random(seed)
    sampled = []
    for i, df in enumerate(tables):
        n = min(sample_size, len(df))
        indices = rng.sample(range(len(df)), n)
        sampled.append(df.iloc[indices].copy())
    return sampled

def format_samples_for_prompt(sampled_tables: list, columns: list,
                              max_per_table: int = 30) -> str:
    """Assemble the first max_per_table rows of each table into plain text for the prompt"""
    parts = []
    for i, df in enumerate(sampled_tables):
        rows_text = []
        for _, row in df.head(max_per_table).iterrows():
            fields = [f"{c}: {row.get(c, '')}" for c in columns if c in df.columns]
            rows_text.append("  " + ", ".join(fields))
        parts.append(f"### Table {i} ({len(df)} sampled records):\n" + "\n".join(rows_text))
    return "\n\n".join(parts)
