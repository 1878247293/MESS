import random

import hnswlib
import numpy as np


def knn_search(value: np.array, ids: np.array, query: np.array, k: int, seed: int, metric="cosine", dim=None):
    """
    使用 HNSW 进行 K 近邻搜索

    Args:
        value: 索引向量 [N, D]
        ids: 向量对应的ID
        query: 查询向量 [M, D]
        k: 近邻数量
        seed: 随机种子
        metric: 距离度量（"cosine" 或 "l2"）
        dim: 向量维度（默认None，自动从value中检测）

    Returns:
        I: 近邻索引
        D: 近邻距离
    """
    # 自动检测维度（支持不同模型的嵌入维度）
    if dim is None:
        dim = value.shape[1]

    index = hnswlib.Index(space=metric, dim=dim)
    index.init_index(max_elements=len(value),
                     ef_construction=200, M=32, random_seed=seed)
    index.add_items(value, ids)
    index.set_ef(200)
    I, D = index.knn_query(query, k=k)
    return I, D


def shuffle(x, seed):
    random.Random(seed).shuffle(x)
    return x


def element_wise_cosine_sim(a: np.array, b: np.array):
    return np.sum(a*b, axis=-1)/(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))
