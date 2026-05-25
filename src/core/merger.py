"""
分层两两合并。

N 张表 → N/2 → N/4 → … → 1。每一层挑配对方式（随机 / 智能配对）和执行方式
（串行 / joblib 并行），4 个组合各暴露一个入口函数：
- `merge` —— 串行 + 随机
- `merge_parallel` —— 并行 + 随机
- `merge_with_smart_pairing` —— 串行 + SmartTablePairing
- `merge_parallel_with_smart_pairing` —— 并行 + SmartTablePairing

核心是 `merge_ij(table_i, table_j, ...)`：双向 mutual KNN 找匹配 tuple 对
（HNSW + 双向交集），距离 ≤ args.min_dis 的留下来，匹配上的 tuple 合并，剩下的延续 tuple_id。
"""

from typing import List
from copy import deepcopy
from collections import defaultdict
import os
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from data import Table
from args import MainArgs
from log import log
from timer import Timer
from resource_monitor import ResourceMonitor, format_bytes, update_stage_metrics
from utils import knn_search, shuffle
from smart_table_pairing import SmartTablePairing


def _record_stage_usage(args: MainArgs, stage_name: str, usage, detail: str):#资源监控
    stage = update_stage_metrics(args, stage_name, usage)
    ram_text = format_bytes(stage["peak_memory_bytes"])
    gpu_text = format_bytes(stage["peak_gpu_memory_bytes"])
    availability_note = "" if usage.available else " (psutil unavailable)"
    gpu_note = "" if usage.gpu_available else " (cuda unavailable)"
    log(
        f"{detail} | time={usage.elapsed_time:.4f}s | "
        f"peak_ram={format_bytes(usage.peak_memory_bytes)}{availability_note} | "
        f"peak_vram={format_bytes(usage.peak_gpu_memory_bytes)}{gpu_note} | "
        f"stage_total_time={stage['total_time']:.4f}s | "
        f"stage_peak_ram={ram_text} | "
        f"stage_peak_vram={gpu_text}"
    )


def get_table_embeddings(table: Table, all_embeddings: np.array):#表内 embedding 聚合
    embeddings = all_embeddings[table.tids]
    df = pd.DataFrame(embeddings)
    df["group"] = table.tuple_ids
    mean_embeddings = df.groupby('group').mean().to_numpy()
    return mean_embeddings


def search_ij(embeddings_i: np.array, embeddings_j: np.array, k: int, seed: int, min_dis: float):
    ids_i = list(range(embeddings_i.shape[0]))
    ids_j = list(range(embeddings_j.shape[0]))
    I1, D1 = knn_search(embeddings_j, np.array(
        ids_j), embeddings_i, k, seed)
    pairs_ij = [(p, vi) for p, v, d in zip(ids_i, I1, D1)
                for vi, di in zip(v, d) if di <= min_dis]
    return pairs_ij


def merge_ij(table_i: Table, table_j: Table, all_embeddings: np.array, args: MainArgs) -> Table:
    timer = Timer()
    idx_i, idx_j = table_i.idx, table_j.idx
    log(f"table {idx_i}, {idx_j}")
    timer.start()

    embeddings_i = get_table_embeddings(table_i, all_embeddings)
    embeddings_j = get_table_embeddings(table_j, all_embeddings)
    tm = timer.stop()
    log(f"  get embeddings: {tm}")
    timer.start()

    # 双向搜，取交集
    pairs_ij = search_ij(embeddings_i, embeddings_j,
                         args.k, args.seed, args.min_dis)
    pairs_ji = search_ij(embeddings_j, embeddings_i,
                         args.k, args.seed, args.min_dis)
    pairs_ji = [(x[1], x[0]) for x in pairs_ji]
    pairs = set(pairs_ij).intersection(set(pairs_ji))

    size_i = int(embeddings_i.shape[0])
    size_j = int(embeddings_j.shape[0])

    tm_search = timer.stop()
    log(f"  ann search: {tm_search}, 匹配对数: {len(pairs)}")

    # 取前 5 个匹配对存为 example，方便回溯
    merge_examples = []
    if len(pairs) > 0:
        pairs_list = list(pairs)[:5]
        for (i, j) in pairs_list:
            merge_examples.append((i, j, None))  # 距离没保留

    result_logger = getattr(args, 'result_logger', None)
    if result_logger is not None:
        result_logger.log_merge(idx_i, idx_j, len(pairs), tm_search, merge_examples=merge_examples)
    timer.start()

    # 构合并表
    # 一次只合两张，结点是 tuple；连通分量最多两个，所以集合操作够了，没必要走 union-find

    # 准备 tuple_id -> [tids] 的映射
    df_i = pd.DataFrame(table_i.tids)
    df_i["group"] = table_i.tuple_ids
    gi = df_i.groupby('group')[0].apply(list).to_dict()
    df_j = pd.DataFrame(table_j.tids)
    df_j["group"] = table_j.tuple_ids
    gj = df_j.groupby('group')[0].apply(list).to_dict()

    new_tids = []
    new_tuple_ids = []
    new_tuple_cnt = 0
    matched_i = set()
    matched_j = set()

    # 1) 匹配上的 tuple 对
    for (i, j) in pairs:
        new_tuple = gi[i] + gj[j]
        assert len(new_tuple) > 0
        new_tids.extend(new_tuple)
        new_tuple_ids.extend([new_tuple_cnt] * len(new_tuple))
        new_tuple_cnt += 1
        matched_i.add(i)
        matched_j.add(j)

    # 2) 表 i 没匹配上的
    for i in range(size_i):
        if i not in matched_i:
            new_tuple = gi[i]
            assert len(new_tuple) > 0
            new_tids.extend(new_tuple)
            new_tuple_ids.extend([new_tuple_cnt] * len(new_tuple))
            new_tuple_cnt += 1

    # 3) 表 j 没匹配上的
    for j in range(size_j):
        if j not in matched_j:
            new_tuple = gj[j]
            assert len(new_tuple) > 0
            new_tids.extend(new_tuple)
            new_tuple_ids.extend([new_tuple_cnt] * len(new_tuple))
            new_tuple_cnt += 1
    tm = timer.stop()
    log(f"new table: {tm}")
    new_table = Table(f"{idx_i}-{idx_j}", new_tids, new_tuple_ids)
    return new_table


def merge(tables: List[Table], all_embeddings: np.array, args: MainArgs) -> Table:
    """层次化合并所有表，串行版本，老的随机配对"""
    cur_tables = [deepcopy(table) for table in tables]
    result_logger = getattr(args, 'result_logger', None)
    hierarchy_level = 1

    while len(cur_tables) > 1:
        # 当前层号
        args._current_hierarchy_level = hierarchy_level

        current_tuples = sum(len(set(table.tuple_ids)) for table in cur_tables)

        if result_logger is not None:
            result_logger.start_hierarchy_level(hierarchy_level, len(cur_tables), current_tuples)

        new_tables = []
        n = len(cur_tables)
        cur_tables = shuffle(cur_tables, args.seed)
        merge_monitor = ResourceMonitor()
        merge_monitor.start()
        index_i = 0
        while index_i + 1 < n:
            table_i = cur_tables[index_i]
            table_j = cur_tables[index_i + 1]
            new_table = merge_ij(table_i, table_j, all_embeddings, args)
            new_tables.append(new_table)
            index_i += 2
        if index_i == n - 1:
            new_tables.append(cur_tables[index_i])
        merge_usage = merge_monitor.stop()
        _record_stage_usage(
            args,
            "hierarchical_merging",
            merge_usage,
            f"  Hierarchical merging level {hierarchy_level}"
        )

        new_tuples = sum(len(set(table.tuple_ids)) for table in new_tables)

        if result_logger is not None:
            result_logger.finish_hierarchy_level(new_tuples)

        cur_tables = new_tables
        hierarchy_level += 1

    assert len(cur_tables) == 1
    return cur_tables[0]


def merge_parallel(tables: List[Table], all_embeddings: np.array, args: MainArgs) -> Table:
    """层次合并，joblib 并行版"""
    cur_tables = [deepcopy(table) for table in tables]
    result_logger = getattr(args, 'result_logger', None)
    hierarchy_level = 1

    while len(cur_tables) > 1:
        args._current_hierarchy_level = hierarchy_level

        current_tuples = sum(len(set(table.tuple_ids)) for table in cur_tables)

        if result_logger is not None:
            result_logger.start_hierarchy_level(hierarchy_level, len(cur_tables), current_tuples)

        n = len(cur_tables)
        cur_tables = shuffle(cur_tables, args.seed)
        new_tables = []
        table_id_pairs = [(i, i+1) for i in range(0, n, 2) if i+1 < n]

        def fun(id_i, id_j):
            table_i = cur_tables[id_i]
            table_j = cur_tables[id_j]
            new_table = merge_ij(table_i, table_j, all_embeddings, args)
            return new_table
        merge_monitor = ResourceMonitor()
        merge_monitor.start()
        new_tables = Parallel(n_jobs=len(table_id_pairs))(
            delayed(fun)(p[0], p[1]) for p in table_id_pairs)
        if n % 2 == 1:
            new_tables.append(cur_tables[-1])
        merge_usage = merge_monitor.stop()
        _record_stage_usage(
            args,
            "hierarchical_merging",
            merge_usage,
            f"  Hierarchical merging level {hierarchy_level}"
        )

        new_tuples = sum(len(set(table.tuple_ids)) for table in new_tables)

        if result_logger is not None:
            result_logger.finish_hierarchy_level(new_tuples)

        cur_tables = new_tables
        hierarchy_level += 1

    assert len(cur_tables) == 1
    return cur_tables[0]


# 智能配对版本

def merge_with_smart_pairing(tables: List[Table], all_embeddings: np.array, args: MainArgs) -> Table:
    """用 SmartTablePairing 替代随机 shuffle 的串行版合并。"""
    pairing = SmartTablePairing()

    cur_tables = [deepcopy(table) for table in tables]
    result_logger = getattr(args, 'result_logger', None)
    hierarchy_level = 1

    while len(cur_tables) > 1:
        log(f"=== Hierarchy Level {hierarchy_level}, {len(cur_tables)} tables ===")

        args._current_hierarchy_level = hierarchy_level

        current_tuples = sum(len(set(table.tuple_ids)) for table in cur_tables)

        if result_logger is not None:
            result_logger.start_hierarchy_level(hierarchy_level, len(cur_tables), current_tuples)

        # 智能配对
        pairing_monitor = ResourceMonitor()
        pairing_monitor.start()
        pairs, unpaired = pairing.get_smart_pairing(cur_tables, all_embeddings)
        pairing_usage = pairing_monitor.stop()
        _record_stage_usage(
            args,
            "table_pairing",
            pairing_usage,
            f"  Table pairing level {hierarchy_level}"
        )

        # 按配对顺序合并
        merge_monitor = ResourceMonitor()
        merge_monitor.start()
        new_tables = []
        for table_i_idx, table_j_idx in pairs:
            table_i = cur_tables[table_i_idx]
            table_j = cur_tables[table_j_idx]
            new_table = merge_ij(table_i, table_j, all_embeddings, args)
            new_tables.append(new_table)

        # 落单的表直接带过去
        for unpaired_idx in unpaired:
            new_tables.append(cur_tables[unpaired_idx])
        merge_usage = merge_monitor.stop()
        _record_stage_usage(
            args,
            "hierarchical_merging",
            merge_usage,
            f"  Hierarchical merging level {hierarchy_level}"
        )

        new_tuples = sum(len(set(table.tuple_ids)) for table in new_tables)

        if result_logger is not None:
            result_logger.finish_hierarchy_level(new_tuples)

        cur_tables = new_tables
        hierarchy_level += 1

    assert len(cur_tables) == 1
    log(f"Smart pairing statistics: {pairing.get_statistics()}")
    return cur_tables[0]


def merge_parallel_with_smart_pairing(tables: List[Table], all_embeddings: np.array, args: MainArgs) -> Table:
    """智能配对 + 并行合并版本"""
    pairing = SmartTablePairing()

    cur_tables = [deepcopy(table) for table in tables]
    result_logger = getattr(args, 'result_logger', None)
    hierarchy_level = 1

    while len(cur_tables) > 1:
        log(f"=== Hierarchy Level {hierarchy_level}, {len(cur_tables)} tables ===")

        args._current_hierarchy_level = hierarchy_level

        current_tuples = sum(len(set(table.tuple_ids)) for table in cur_tables)

        if result_logger is not None:
            result_logger.start_hierarchy_level(hierarchy_level, len(cur_tables), current_tuples)

        pairing_monitor = ResourceMonitor()
        pairing_monitor.start()
        pairs, unpaired = pairing.get_smart_pairing(cur_tables, all_embeddings)
        pairing_usage = pairing_monitor.stop()
        _record_stage_usage(
            args,
            "table_pairing",
            pairing_usage,
            f"  Table pairing level {hierarchy_level}"
        )

        # 并行跑所有 pair
        def fun(table_i_idx, table_j_idx):
            table_i = cur_tables[table_i_idx]
            table_j = cur_tables[table_j_idx]
            new_table = merge_ij(table_i, table_j, all_embeddings, args)
            return new_table

        # 限制并发数，免得开太多进程
        n_jobs = min(len(pairs), 8) if len(pairs) > 0 else 1
        merge_monitor = ResourceMonitor()
        merge_monitor.start()
        new_tables = Parallel(n_jobs=n_jobs)(
            delayed(fun)(p[0], p[1]) for p in pairs
        )

        for unpaired_idx in unpaired:
            new_tables.append(cur_tables[unpaired_idx])
        merge_usage = merge_monitor.stop()
        _record_stage_usage(
            args,
            "hierarchical_merging",
            merge_usage,
            f"  Hierarchical merging level {hierarchy_level}"
        )

        new_tuples = sum(len(set(table.tuple_ids)) for table in new_tables)

        if result_logger is not None:
            result_logger.finish_hierarchy_level(new_tuples)

        cur_tables = new_tables
        hierarchy_level += 1

    assert len(cur_tables) == 1
    log(f"Smart pairing statistics: {pairing.get_statistics()}")
    return cur_tables[0]


