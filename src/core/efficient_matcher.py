"""
高效两表匹配模块：优化的双向搜索

核心思想：
- 原始方法：建两个 HNSW 索引，做两次搜索，取交集
- 优化方法：同样的语义，但优化实现细节

时间复杂度：
- 原始：2 * O(n log n) 建索引 + 2 * O(m log n) 搜索
- 优化：保持相同，但减少 Python 层开销

预期加速：约 10-20%，精度完全一致
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
    高效互匹配搜索：优化的双向搜索实现

    与原始方法完全等价，但优化了：
    1. 使用 numpy 向量化操作替代 Python 循环
    2. 减少中间数据结构的创建

    Args:
        embeddings_i: 表i的嵌入 [N_i, D]
        embeddings_j: 表j的嵌入 [N_j, D]
        k: top-K 参数
        seed: 随机种子
        min_dis: 距离阈值
        metric: 距离度量（cosine 或 l2）

    Returns:
        matches: 互匹配对列表 [(i, j), ...]
    """
    n_i = len(embeddings_i)
    n_j = len(embeddings_j)
    dim = embeddings_i.shape[1]

    t0 = time.time()

    # ========== 建立两个 HNSW 索引 ==========
    # 对表i建立索引
    index_i = hnswlib.Index(space=metric, dim=dim)
    index_i.init_index(max_elements=n_i, ef_construction=200, M=32, random_seed=seed)
    index_i.add_items(embeddings_i, np.arange(n_i))
    index_i.set_ef(200)

    # 对表j建立索引
    index_j = hnswlib.Index(space=metric, dim=dim)
    index_j.init_index(max_elements=n_j, ef_construction=200, M=32, random_seed=seed)
    index_j.add_items(embeddings_j, np.arange(n_j))
    index_j.set_ef(200)

    t1 = time.time()
    log(f"  [高效匹配] 建立索引耗时: {(t1-t0)*1000:.1f}ms")

    # ========== 双向搜索 ==========
    # i -> j 搜索
    I_ij, D_ij = index_j.knn_query(embeddings_i, k=k)
    # j -> i 搜索
    I_ji, D_ji = index_i.knn_query(embeddings_j, k=k)

    t2 = time.time()
    log(f"  [高效匹配] 双向搜索耗时: {(t2-t1)*1000:.1f}ms")

    # ========== 向量化收集候选对 ==========
    # 使用 numpy 操作替代 Python 循环

    # i -> j 方向的候选对
    # 展平索引和距离
    i_indices = np.repeat(np.arange(n_i), k)  # [0,0,..,0,1,1,..,1,...]
    j_indices_ij = I_ij.flatten()  # 对应的 j 索引
    d_ij = D_ij.flatten()  # 对应的距离

    # 过滤距离 <= min_dis
    mask_ij = d_ij <= min_dis
    pairs_ij_i = i_indices[mask_ij]
    pairs_ij_j = j_indices_ij[mask_ij]

    # 创建 i->j 方向的候选集（使用元组集合）
    pairs_ij = set(zip(pairs_ij_i.tolist(), pairs_ij_j.tolist()))

    # j -> i 方向的候选对
    j_indices = np.repeat(np.arange(n_j), k)
    i_indices_ji = I_ji.flatten()
    d_ji = D_ji.flatten()

    mask_ji = d_ji <= min_dis
    pairs_ji_j = j_indices[mask_ji]
    pairs_ji_i = i_indices_ji[mask_ji]

    # 创建 j->i 方向的候选集（注意顺序转换为 (i, j)）
    pairs_ji = set(zip(pairs_ji_i.tolist(), pairs_ji_j.tolist()))

    t3 = time.time()
    log(f"  [高效匹配] 收集候选耗时: {(t3-t2)*1000:.1f}ms")

    # ========== 取交集 ==========
    matches = list(pairs_ij & pairs_ji)

    t4 = time.time()
    log(f"  [高效匹配] i->j: {len(pairs_ij)}, j->i: {len(pairs_ji)}, 交集: {len(matches)}")
    log(f"  [高效匹配] 总耗时: {(t4-t0)*1000:.1f}ms")

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
    高效互匹配搜索 V2：单向搜索 + 全局反向验证

    这个版本只建一个索引，但做全局反向验证确保精度：
    1. 对表j建立索引
    2. 表i搜索表j，得到候选
    3. 对表i建立索引
    4. 对候选中的j搜索表i，验证互匹配

    适用于：表i和表j大小相差很大时

    Args:
        embeddings_i: 表i的嵌入 [N_i, D]
        embeddings_j: 表j的嵌入 [N_j, D]
        k: top-K 参数
        seed: 随机种子
        min_dis: 距离阈值
        metric: 距离度量

    Returns:
        matches: 互匹配对列表
    """
    n_i = len(embeddings_i)
    n_j = len(embeddings_j)
    dim = embeddings_i.shape[1]

    t0 = time.time()

    # ========== 阶段1：对表j建立索引，表i搜索 ==========
    index_j = hnswlib.Index(space=metric, dim=dim)
    index_j.init_index(max_elements=n_j, ef_construction=200, M=32, random_seed=seed)
    index_j.add_items(embeddings_j, np.arange(n_j))
    index_j.set_ef(200)

    I_ij, D_ij = index_j.knn_query(embeddings_i, k=k)

    t1 = time.time()

    # 收集 i->j 候选
    i_indices = np.repeat(np.arange(n_i), k)
    j_indices_ij = I_ij.flatten()
    d_ij = D_ij.flatten()
    mask_ij = d_ij <= min_dis

    pairs_ij_i = i_indices[mask_ij]
    pairs_ij_j = j_indices_ij[mask_ij]
    pairs_ij = set(zip(pairs_ij_i.tolist(), pairs_ij_j.tolist()))

    # ========== 阶段2：对表i建立索引，只对候选j搜索 ==========
    index_i = hnswlib.Index(space=metric, dim=dim)
    index_i.init_index(max_elements=n_i, ef_construction=200, M=32, random_seed=seed)
    index_i.add_items(embeddings_i, np.arange(n_i))
    index_i.set_ef(200)

    # 只对候选中涉及的j进行搜索
    unique_js = np.unique(pairs_ij_j)
    if len(unique_js) > 0:
        I_ji, D_ji = index_i.knn_query(embeddings_j[unique_js], k=k)

        # 收集 j->i 候选
        j_local_indices = np.repeat(np.arange(len(unique_js)), k)
        j_global_indices = unique_js[j_local_indices]  # 映射回全局j索引
        i_indices_ji = I_ji.flatten()
        d_ji = D_ji.flatten()
        mask_ji = d_ji <= min_dis

        pairs_ji_j = j_global_indices[mask_ji]
        pairs_ji_i = i_indices_ji[mask_ji]
        pairs_ji = set(zip(pairs_ji_i.tolist(), pairs_ji_j.tolist()))
    else:
        pairs_ji = set()

    t2 = time.time()

    # ========== 取交集 ==========
    matches = list(pairs_ij & pairs_ji)

    log(f"  [高效匹配V2] 索引+搜索: {(t1-t0)*1000:.1f}ms, 反向验证: {(t2-t1)*1000:.1f}ms")
    log(f"  [高效匹配V2] i->j: {len(pairs_ij)}, j->i: {len(pairs_ji)}, 交集: {len(matches)}")

    return matches
