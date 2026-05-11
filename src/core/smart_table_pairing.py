"""
表配对策略：层次合并时，决定哪两张表先合。

提供三种：
- similarity   贪心，按表语义中心相似度从高到低
- optimal      最大权匹配，全局最优
- complementary  大配小，做负载均衡
"""

import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from itertools import combinations

from data import Table
from log import log


class SmartTablePairing:
    def __init__(self, strategy='similarity'):
        """
        Args:
            strategy: 'similarity' / 'optimal' / 'complementary'
        """
        self.strategy = strategy
        self.stats = {
            'total_pairings': 0,
            'avg_similarity': 0.0,
            'strategy_used': strategy
        }

    def compute_table_centers(self, tables: List[Table], embeddings: np.array) -> np.array:
        """
        每张表的语义中心：表里所有实体 embedding 的平均，再 L2 归一化。
        """
        centers = []
        for table in tables:
            table_embeddings = embeddings[table.tids]
            center = table_embeddings.mean(axis=0)
            center_norm = np.linalg.norm(center)
            if center_norm > 0:
                center = center / center_norm
            centers.append(center)

        return np.array(centers)

    def compute_similarity_matrix(self, centers: np.array) -> np.array:
        """对称的余弦相似度矩阵"""
        n = len(centers)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                # 中心已归一化，点积就是 cosine
                sim = np.dot(centers[i], centers[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        np.fill_diagonal(sim_matrix, 1.0)

        return sim_matrix

    def greedy_pairing_by_similarity(self, tables: List[Table],
                                      sim_matrix: np.array) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        贪心：sim 排序，从高到低顺次配对，已配过的跳过。
        快但不保证全局最优。
        """
        n = len(tables)
        paired = set()
        pairs = []

        similarities = []
        for i in range(n):
            for j in range(i+1, n):
                similarities.append((sim_matrix[i, j], i, j))

        similarities.sort(reverse=True, key=lambda x: x[0])

        for sim, i, j in similarities:
            if i not in paired and j not in paired:
                pairs.append((i, j))
                paired.add(i)
                paired.add(j)
                log(f"  Pair tables {i}-{j}, similarity: {sim:.4f}")

        unpaired = [i for i in range(n) if i not in paired]

        if pairs:
            avg_sim = np.mean([sim_matrix[i, j] for i, j in pairs])
            self.stats['avg_similarity'] = avg_sim
            log(f"  Average pairing similarity: {avg_sim:.4f}")

        return pairs, unpaired

    def optimal_pairing_by_assignment(self, tables: List[Table],
                                       sim_matrix: np.array) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        全局最优：networkx 的 max_weight_matching。
        总相似度最大，但 O(n^3)，表多时会贵。
        """
        import networkx as nx

        n = len(tables)

        G = nx.Graph()

        for i in range(n):
            G.add_node(i)

        for i in range(n):
            for j in range(i+1, n):
                weight = sim_matrix[i, j]
                G.add_edge(i, j, weight=weight)

        matching = nx.max_weight_matching(G, maxcardinality=False)

        pairs = []
        paired_indices = set()
        for i, j in matching:
            # 统一 i < j
            if i > j:
                i, j = j, i
            pairs.append((i, j))
            paired_indices.add(i)
            paired_indices.add(j)
            log(f"  Pair tables {i}-{j}, similarity: {sim_matrix[i, j]:.4f}")

        unpaired = [i for i in range(n) if i not in paired_indices]

        if pairs:
            avg_sim = np.mean([sim_matrix[i, j] for i, j in pairs])
            self.stats['avg_similarity'] = avg_sim
            log(f"  Optimal average similarity: {avg_sim:.4f}")

        return pairs, unpaired

    def complementary_pairing(self, tables: List[Table]) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        大表配小表：负载均衡，并行场景下 worker 时间更接近。
        不看语义，所以匹配质量交给后面阶段保。
        """
        sorted_indices = sorted(range(len(tables)),
                                key=lambda i: len(tables[i].tids))

        pairs = []
        n = len(sorted_indices)
        for i in range(n // 2):
            small_idx = sorted_indices[i]
            large_idx = sorted_indices[n - 1 - i]
            pairs.append((small_idx, large_idx))

            small_size = len(tables[small_idx].tids)
            large_size = len(tables[large_idx].tids)
            log(f"  Pair tables {small_idx}-{large_idx}, sizes: {small_size} + {large_size} = {small_size + large_size}")

        # 表数为奇数时中间那张落单
        unpaired = [sorted_indices[n // 2]] if n % 2 == 1 else []

        if unpaired:
            mid_size = len(tables[unpaired[0]].tids)
            log(f"  Unpaired table {unpaired[0]}, size: {mid_size}")

        return pairs, unpaired

    def get_smart_pairing(self, tables: List[Table],
                          embeddings: np.array) -> Tuple[List[Tuple[int, int]], List[int]]:
        """根据 self.strategy 调具体的方法"""
        n = len(tables)
        self.stats['total_pairings'] += 1

        log(f"Smart pairing strategy: {self.strategy}, {n} tables")

        if n == 0:
            return [], []
        if n == 1:
            return [], [0]
        if n == 2:
            return [(0, 1)], []

        if self.strategy == 'complementary':
            # 互补不需要 sim matrix
            pairs, unpaired = self.complementary_pairing(tables)
        else:
            centers = self.compute_table_centers(tables, embeddings)
            sim_matrix = self.compute_similarity_matrix(centers)

            if self.strategy == 'similarity':
                pairs, unpaired = self.greedy_pairing_by_similarity(tables, sim_matrix)
            elif self.strategy == 'optimal':
                pairs, unpaired = self.optimal_pairing_by_assignment(tables, sim_matrix)
            else:
                log(f"Warning: unknown strategy '{self.strategy}', fallback to similarity")
                pairs, unpaired = self.greedy_pairing_by_similarity(tables, sim_matrix)

        log(f"Pairing completed: {len(pairs)} pairs, {len(unpaired)} unpaired")

        return pairs, unpaired

    def get_statistics(self) -> dict:
        return self.stats.copy()


def demo_smart_pairing():
    """跑一遍三种策略，肉眼对比"""
    import numpy as np
    from data import Table

    n_tables = 5
    embedding_dim = 384

    tables = []
    for i in range(n_tables):
        n_entities = np.random.randint(100, 500)
        tids = list(range(i * 1000, i * 1000 + n_entities))
        tuple_ids = list(range(n_entities))
        tables.append(Table(str(i), tids, tuple_ids))

    embeddings = np.random.randn(max(max(t.tids) for t in tables) + 1, embedding_dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    strategies = ['similarity', 'optimal', 'complementary']

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"strategy: {strategy}")
        print('='*60)

        pairing = SmartTablePairing(strategy=strategy)
        pairs, unpaired = pairing.get_smart_pairing(tables, embeddings)

        print(f"\nResults:")
        print(f"  Pairs: {pairs}")
        print(f"  Unpaired: {unpaired}")
        print(f"  Statistics: {pairing.get_statistics()}")


if __name__ == "__main__":
    demo_smart_pairing()
