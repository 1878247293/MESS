"""
Hierarchical pairwise merging.

N tables -> N/2 -> N/4 -> ... -> 1. Each level picks a pairing method (random / smart pairing) and an
execution method (serial / joblib parallel); the 4 combinations each expose an entry function:
- `merge` -- serial + random
- `merge_parallel` -- parallel + random
- `merge_with_smart_pairing` -- serial + SmartTablePairing
- `merge_parallel_with_smart_pairing` -- parallel + SmartTablePairing

The core is `merge_ij(table_i, table_j, ...)`: bidirectional mutual KNN to find matching tuple pairs
(HNSW + bidirectional intersection); pairs with distance <= args.min_dis are kept, matched tuples are
merged, and the rest carry over their tuple_id.
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


def _record_stage_usage(args: MainArgs, stage_name: str, usage, detail: str):  # resource monitoring
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


def get_table_embeddings(table: Table, all_embeddings: np.array):  # aggregate embeddings within a table
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

    # search in both directions and take the intersection
    pairs_ij = search_ij(embeddings_i, embeddings_j,
                         args.k, args.seed, args.min_dis)
    pairs_ji = search_ij(embeddings_j, embeddings_i,
                         args.k, args.seed, args.min_dis)
    pairs_ji = [(x[1], x[0]) for x in pairs_ji]
    pairs = set(pairs_ij).intersection(set(pairs_ji))

    size_i = int(embeddings_i.shape[0])
    size_j = int(embeddings_j.shape[0])

    tm_search = timer.stop()
    log(f"  ann search: {tm_search}, number of matched pairs: {len(pairs)}")

    # keep the first 5 matched pairs as examples for traceability
    merge_examples = []
    if len(pairs) > 0:
        pairs_list = list(pairs)[:5]
        for (i, j) in pairs_list:
            merge_examples.append((i, j, None))  # distance not retained

    result_logger = getattr(args, 'result_logger', None)
    if result_logger is not None:
        result_logger.log_merge(idx_i, idx_j, len(pairs), tm_search, merge_examples=merge_examples)
    timer.start()

    # build the merged table
    # only two tables are merged at a time, with tuples as nodes; there are at most two connected
    # components, so set operations suffice and union-find is unnecessary

    # prepare the tuple_id -> [tids] mapping
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

    # 1) matched tuple pairs
    for (i, j) in pairs:
        new_tuple = gi[i] + gj[j]
        assert len(new_tuple) > 0
        new_tids.extend(new_tuple)
        new_tuple_ids.extend([new_tuple_cnt] * len(new_tuple))
        new_tuple_cnt += 1
        matched_i.add(i)
        matched_j.add(j)

    # 2) unmatched tuples from table i
    for i in range(size_i):
        if i not in matched_i:
            new_tuple = gi[i]
            assert len(new_tuple) > 0
            new_tids.extend(new_tuple)
            new_tuple_ids.extend([new_tuple_cnt] * len(new_tuple))
            new_tuple_cnt += 1

    # 3) unmatched tuples from table j
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
    """Hierarchically merge all tables, serial version, the old random pairing"""
    cur_tables = [deepcopy(table) for table in tables]
    result_logger = getattr(args, 'result_logger', None)
    hierarchy_level = 1

    while len(cur_tables) > 1:
        # current level number
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
    """Hierarchical merge, joblib parallel version"""
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


# smart pairing versions

def merge_with_smart_pairing(tables: List[Table], all_embeddings: np.array, args: MainArgs) -> Table:
    """Serial merge using SmartTablePairing instead of random shuffle."""
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

        # smart pairing
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

        # merge in pairing order
        merge_monitor = ResourceMonitor()
        merge_monitor.start()
        new_tables = []
        for table_i_idx, table_j_idx in pairs:
            table_i = cur_tables[table_i_idx]
            table_j = cur_tables[table_j_idx]
            new_table = merge_ij(table_i, table_j, all_embeddings, args)
            new_tables.append(new_table)

        # carry over the leftover unpaired table directly
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
    """Smart pairing + parallel merge version"""
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

        # run all pairs in parallel
        def fun(table_i_idx, table_j_idx):
            table_i = cur_tables[table_i_idx]
            table_j = cur_tables[table_j_idx]
            new_table = merge_ij(table_i, table_j, all_embeddings, args)
            return new_table

        # limit concurrency to avoid spawning too many processes
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


