"""
Data-layer infrastructure.

Provides:
- `Table` dataclass: tids (indices of entities in the global vector array) + tuple_ids (the group each tid belongs to);
  `get_tuples` flattens the groups into the prediction result `[(tid, tid, ...), ...]`.
- `read_all_tables / read_table`: scan `table_*.csv`, supporting reading only selected columns and downsampling by sample_rate
  (when sampling, tid is remapped to consecutive integers, since downstream uses tid directly as an array index).
- `read_ground_truth`: read the tuple list from `ground_truth.txt`, used during evaluation.
- `textify_table`: join each row of the table (excluding the tid column) into a single sentence to feed to SentenceTransformer.
"""

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

    # downsample when a large table does not fit in memory
    if sample_rate < 1.0:
        table = table.sample(frac=sample_rate, random_state=3407).reset_index(drop=True)
        # key: tid must be remapped to consecutive 0..n, since downstream uses tid directly as an index into the embeddings array
        table['tid'] = range(len(table))
        log(f"  sample rate: {sample_rate:.2f}, after sampling: {len(table)} rows")

    return table


def read_all_tables(data_path: Path, num=-1, selected_attrs=None, sample_rate=1.0) -> Tuple[int, List[pd.DataFrame]]:
    log(f"selected_attrs: {selected_attrs}")
    if sample_rate < 1.0:
        log(f"sampling mode: rate = {sample_rate:.2f}")
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
    """Flatten the whole table into a column of strings, joining each row into one sentence"""
    # when only the tid column remains, there are no attributes to join, so fall back to entity_<tid>
    if table.shape[1] <= 1:
        log("Warning: only tid column, using tid as entity text")
        sentences = table.iloc[:, 0].astype(str).apply(lambda x: "entity_" + x).tolist()
    else:
        # normal case: join the columns other than tid
        sentences = table.iloc[:, 1:] \
            .astype(str) \
            .apply(lambda x: x + " ", axis=0) \
            .sum(axis=1) \
            .tolist()

    return sentences
