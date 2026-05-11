"""
两表匹配的另一种实现 — 在同一逻辑下少跑点 Python 循环。

老路子：建两次 HNSW、跑两次 search、取交集。
这里语义不变，只是换成更紧的 numpy 向量化收集，实测能快一些。
"""

from typing import List, Tuple, Set
import numpy as np
import hnswlib
from log import log
import time


def efficient_mutual_search(
    embeddings_i: np.ndarray,
    embeddings_j: np.ndarray,
    k: int,
    seed: int,
    min_dis: float,
    metric: str = "cosine"
) -> List[Tuple[int, int]]:
    """
    双向 mutual KNN 的紧凑实现：
    - 用 numpy 向量化代替 for 循环收集
    - 中间 set / list 少建一层

    Args:
        embeddings_i: [N_i, D]
        embeddings_j: [N_j, D]
        k: top-K
        seed: HNSW 随机种子
        min_dis: 距离阈值（越小越严）
        metric: cosine 或 l2

    Returns:
        互匹配对 [(i, j), ...]
    """
    n_i = len(embeddings_i)
    n_j = len(embeddings_j)
    dim = embeddings_i.shape[1]

    t0 = time.time()

    # 两个方向都建一次 HNSW
    index_i = hnswlib.Index(space=metric, dim=dim)
    index_i.init_index(max_elements=n_i, ef_construction=200, M=32, random_seed=seed)
    index_i.add_items(embeddings_i, np.arange(n_i))
    index_i.set_ef(200)

    index_j = hnswlib.Index(space=metric, dim=dim)
    index_j.init_index(max_elements=n_j, ef_construction=200, M=32, random_seed=seed)
    index_j.add_items(embeddings_j, np.arange(n_j))
    index_j.set_ef(200)

    t1 = time.time()
    log(f"  [efficient match] 建索引: {(t1-t0)*1000:.1f}ms")

    # 双向搜索
    I_ij, D_ij = index_j.knn_query(embeddings_i, k=k)
    I_ji, D_ji = index_i.knn_query(embeddings_j, k=k)

    t2 = time.time()
    log(f"  [efficient match] 双向 knn: {(t2-t1)*1000:.1f}ms")

    # 收集候选 (i -> j)
    i_indices = np.repeat(np.arange(n_i), k)
    j_indices_ij = I_ij.flatten()
    d_ij = D_ij.flatten()

    mask_ij = d_ij <= min_dis
    pairs_ij_i = i_indices[mask_ij]
    pairs_ij_j = j_indices_ij[mask_ij]

    pairs_ij = set(zip(pairs_ij_i.tolist(), pairs_ij_j.tolist()))

    # 收集候选 (j -> i)
    j_indices = np.repeat(np.arange(n_j), k)
    i_indices_ji = I_ji.flatten()
    d_ji = D_ji.flatten()

    mask_ji = d_ji <= min_dis
    pairs_ji_j = j_indices[mask_ji]
    pairs_ji_i = i_indices_ji[mask_ji]

    # 注意要把 (j, i) 翻成 (i, j) 再放
    pairs_ji = set(zip(pairs_ji_i.tolist(), pairs_ji_j.tolist()))

    t3 = time.time()
    log(f"  [efficient match] 收集候选: {(t3-t2)*1000:.1f}ms")

    # 取交集就是 mutual KNN
    matches = list(pairs_ij & pairs_ji)

    t4 = time.time()
    log(f"  [efficient match] i->j: {len(pairs_ij)}, j->i: {len(pairs_ji)}, 交集: {len(matches)}")
    log(f"  [efficient match] 总: {(t4-t0)*1000:.1f}ms")

    return matches


def efficient_mutual_search_v2(
    embeddings_i: np.ndarray,
    embeddings_j: np.ndarray,
    k: int,
    seed: int,
    min_dis: float,
    metric: str = "cosine"
) -> List[Tuple[int, int]]:
    """
    单向 + 反向校验版。两表大小差很多时省事。

    流程：
      1) 对 j 建索引，i 全量搜
      2) 对 i 建索引，但只对 i->j 命中的那批 j 做反向搜
      3) 两边交集就是 mutual

    Args / Returns 同 efficient_mutual_search
    """
    n_i = len(embeddings_i)
    n_j = len(embeddings_j)
    dim = embeddings_i.shape[1]

    t0 = time.time()

    # 阶段 1：对 j 建索引，i 搜
    index_j = hnswlib.Index(space=metric, dim=dim)
    index_j.init_index(max_elements=n_j, ef_construction=200, M=32, random_seed=seed)
    index_j.add_items(embeddings_j, np.arange(n_j))
    index_j.set_ef(200)

    I_ij, D_ij = index_j.knn_query(embeddings_i, k=k)

    t1 = time.time()

    i_indices = np.repeat(np.arange(n_i), k)
    j_indices_ij = I_ij.flatten()
    d_ij = D_ij.flatten()
    mask_ij = d_ij <= min_dis

    pairs_ij_i = i_indices[mask_ij]
    pairs_ij_j = j_indices_ij[mask_ij]
    pairs_ij = set(zip(pairs_ij_i.tolist(), pairs_ij_j.tolist()))

    # 阶段 2：对 i 建索引，只搜命中过的那批 j
    index_i = hnswlib.Index(space=metric, dim=dim)
    index_i.init_index(max_elements=n_i, ef_construction=200, M=32, random_seed=seed)
    index_i.add_items(embeddings_i, np.arange(n_i))
    index_i.set_ef(200)

    unique_js = np.unique(pairs_ij_j)
    if len(unique_js) > 0:
        I_ji, D_ji = index_i.knn_query(embeddings_j[unique_js], k=k)

        # 局部 j 序号映射回全局 j 序号
        j_local_indices = np.repeat(np.arange(len(unique_js)), k)
        j_global_indices = unique_js[j_local_indices]
        i_indices_ji = I_ji.flatten()
        d_ji = D_ji.flatten()
        mask_ji = d_ji <= min_dis

        pairs_ji_j = j_global_indices[mask_ji]
        pairs_ji_i = i_indices_ji[mask_ji]
        pairs_ji = set(zip(pairs_ji_i.tolist(), pairs_ji_j.tolist()))
    else:
        pairs_ji = set()

    t2 = time.time()

    matches = list(pairs_ij & pairs_ji)

    log(f"  [efficient match v2] 索引+搜: {(t1-t0)*1000:.1f}ms, 反向校验: {(t2-t1)*1000:.1f}ms")
    log(f"  [efficient match v2] i->j: {len(pairs_ij)}, j->i: {len(pairs_ji)}, 交集: {len(matches)}")

    return matches
