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

    # 大表内存吃不消时下采样
    if sample_rate < 1.0:
        table = table.sample(frac=sample_rate, random_state=3407).reset_index(drop=True)
        # 关键：tid 必须重映射成 0..n 连续，下游会拿 tid 直接当 embeddings 数组的下标
        table['tid'] = range(len(table))
        log(f"  采样率: {sample_rate:.2f}, 采样后: {len(table)} 行")

    return table


def read_all_tables(data_path: Path, num=-1, selected_attrs=None, sample_rate=1.0) -> Tuple[int, List[pd.DataFrame]]:
    log(f"selected_attrs: {selected_attrs}")
    if sample_rate < 1.0:
        log(f"采样模式: rate = {sample_rate:.2f}")
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
    """把整张表拍成一列字符串，每行拼成一句"""
    # 只剩 tid 一列时，没法拼属性，就退化成 entity_<tid>
    if table.shape[1] <= 1:
        log("Warning: only tid column, using tid as entity text")
        sentences = table.iloc[:, 0].astype(str).apply(lambda x: "entity_" + x).tolist()
    else:
        # 正常情况：除 tid 外的列拼起来
        sentences = table.iloc[:, 1:] \
            .astype(str) \
            .apply(lambda x: x + " ", axis=0) \
            .sum(axis=1) \
            .tolist()

    return sentences
