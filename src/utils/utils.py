"""
Small utilities reused across modules.

- `knn_search` -- thin wrapper around an hnswlib nearest-neighbor query (used in the merge stage).
- `shuffle` -- seeded list shuffle (ensures consistent random pairing across runs).
- `element_wise_cosine_sim` -- row-wise cosine similarity of two matrices, used by attribute selection and evaluation.
"""

import random

import hnswlib
import numpy as np


def knn_search(value: np.array, ids: np.array, query: np.array, k: int, seed: int, metric="cosine", dim=None):
    """
    KNN over HNSW. If dim is not given, infer it from value.

    Returns:
        I: neighbor indices
        D: neighbor distances
    """
    if dim is None:
        dim = value.shape[1]  # value (N, D)

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
