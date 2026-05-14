"""
跨模块复用的小工具。

- `knn_search` —— hnswlib 近邻查询的薄包装（合并阶段用）。
- `shuffle` —— 带 seed 的 list shuffle（保证多次跑的随机配对一致）。
- `element_wise_cosine_sim` —— 两个矩阵逐行余弦相似度，属性选择和评估都用。
"""

import random

import hnswlib
import numpy as np


def knn_search(value: np.array, ids: np.array, query: np.array, k: int, seed: int, metric="cosine", dim=None):
    """
    HNSW 上的 KNN。dim 不传就从 value 推。

    Returns:
        I: 邻居索引
        D: 邻居距离
    """
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
