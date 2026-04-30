from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import re

import pandas as pd
from functional import pseq

from log import log


@dataclass()
class Table:
    idx: str
    tids: List[int]
    tuple_ids: List[int]

    def get_tuples(self, min_cnt=1):
        res = pseq(zip(self.tids, self.tuple_ids))\
            .group_by(lambda x: x[1])\
            .map(lambda x: x[1])\
            .map(lambda x: [xi[0] for xi in x])\
            .filter(lambda x: len(x) > min_cnt)\
            .map(lambda x: sorted(x))\
            .map(lambda x: tuple(x))\
            .to_list()
        return res


def read_table(data_path: Path, selected_attrs=None, sample_rate=1.0):
    if selected_attrs is None:
        table = pd.read_csv(
            data_path, dtype={"postcode": str})
    else:
        table = pd.read_csv(
            data_path, dtype={"postcode": str}, usecols=selected_attrs)

    # 对数据集进行采样(用于减少内存占用)
    if sample_rate < 1.0:
        table = table.sample(frac=sample_rate, random_state=3407).reset_index(drop=True)
        # 关键修复: 重新映射 tid 列,使其从 0 开始连续编号
        # 这样 tid 可以直接作为 embeddings 数组的索引
        table['tid'] = range(len(table))
        log(f"  采样率: {sample_rate:.2f}, 采样后: {len(table)} 行")

    return table


def read_all_tables(data_path: Path, num=-1, selected_attrs=None, sample_rate=1.0) -> Tuple[int, List[pd.DataFrame]]:
    log(f"selected_attrs: {selected_attrs}")
    if sample_rate < 1.0:
        log(f"⚠️  数据集采样模式: 采样率 = {sample_rate:.2f}")
    i = 0
    tables = []
    while (data_path / f"table_{i}.csv").is_file():
        table = read_table(data_path / f"table_{i}.csv", selected_attrs, sample_rate)
        tables.append(table)
        i += 1
        if i == num:
            break
    return i, tables


def read_ground_truth(data_path: Path) -> List[Tuple[int]]:
    with (data_path / "ground_truth.txt").open("r") as rd:
        return [tuple(map(int, line.split(","))) for line in rd]


def read_pair_ground_truth(data_path: Path, i: int, j: int) -> List[Tuple[int]]:
    with (data_path / f"ground_truth_{i}_{j}.txt").open("r") as rd:
        return [tuple(map(int, line.split(","))) for line in rd]


def textify_table(table: pd.DataFrame):
    """
    将表转换为文本列表

    Args:
        table: 数据表

    Returns:
        sentences: 文本列表
    """
    # 检查是否有除 tid 外的列
    if table.shape[1] <= 1:
        # 只有 tid 列，使用 tid 作为文本
        log("Warning: only tid column, using tid as entity text")
        sentences = table.iloc[:, 0].astype(str).apply(lambda x: "entity_" + x).tolist()
    else:
        # 正常情况：拼接第2列及之后的列
        sentences = table.iloc[:, 1:] \
            .astype(str) \
            .apply(lambda x: x + " ", axis=0) \
            .sum(axis=1) \
            .tolist()

    return sentences
