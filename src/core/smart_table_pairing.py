"""
智能表配对模块 - Smart Table Pairing

基于语义中心的智能表配对策略，优化层次合并效率和效果

核心思想：
1. 计算每个表的语义中心（表示表的整体语义）
2. 根据配对策略选择最优的表配对顺序
3. 支持三种策略：
   - similarity: 优先合并相似的表（贪心）
   - optimal: 使用匈牙利算法找最优配对
   - complementary: 大表配小表（负载均衡）

Author: MultiEM Enhancement
Date: 2025-11-24
"""

import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from itertools import combinations

from data import Table
from log import log


class SmartTablePairing:
    """智能表配对器"""

    def __init__(self, strategy='similarity'):
        """
        Args:
            strategy: 配对策略
                - 'similarity': 贪心相似度配对（快速）
                - 'optimal': 匈牙利算法最优配对（质量最高）
                - 'complementary': 大小互补配对（负载均衡）
        """
        self.strategy = strategy
        self.stats = {
            'total_pairings': 0,
            'avg_similarity': 0.0,
            'strategy_used': strategy
        }

    def compute_table_centers(self, tables: List[Table], embeddings: np.array) -> np.array:
        """
        计算所有表的语义中心

        Args:
            tables: 表列表
            embeddings: 所有实体的嵌入

        Returns:
            centers: [n_tables, embedding_dim] 归一化的表中心向量
        """
        centers = []
        for table in tables:
            # 获取表中所有实体的嵌入
            table_embeddings = embeddings[table.tids]
            # 计算平均作为表中心
            center = table_embeddings.mean(axis=0)
            # 归一化
            center_norm = np.linalg.norm(center)
            if center_norm > 0:
                center = center / center_norm
            centers.append(center)

        return np.array(centers)

    def compute_similarity_matrix(self, centers: np.array) -> np.array:
        """
        计算表间相似度矩阵（余弦相似度）

        Args:
            centers: [n_tables, embedding_dim] 表中心向量

        Returns:
            sim_matrix: [n_tables, n_tables] 对称相似度矩阵
        """
        n = len(centers)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                # 余弦相似度（向量已归一化，直接点积）
                sim = np.dot(centers[i], centers[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        # 对角线设为1（自己和自己完全相似）
        np.fill_diagonal(sim_matrix, 1.0)

        return sim_matrix

    def greedy_pairing_by_similarity(self, tables: List[Table],
                                      sim_matrix: np.array) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        贪心相似度配对：优先合并最相似的表对

        策略：按相似度从高到低排序，贪心选择未配对的最相似表对
        优点：实现简单，速度快
        缺点：可能不是全局最优

        Args:
            tables: 表列表
            sim_matrix: 相似度矩阵

        Returns:
            pairs: [(table_i_idx, table_j_idx), ...] 配对列表
            unpaired: [table_idx, ...] 未配对的表（奇数个表时）
        """
        n = len(tables)
        paired = set()
        pairs = []

        # 收集所有表对及其相似度
        similarities = []
        for i in range(n):
            for j in range(i+1, n):
                similarities.append((sim_matrix[i, j], i, j))

        # 按相似度从高到低排序
        similarities.sort(reverse=True, key=lambda x: x[0])

        # 贪心选择
        for sim, i, j in similarities:
            if i not in paired and j not in paired:
                pairs.append((i, j))
                paired.add(i)
                paired.add(j)
                log(f"  Pair tables {i}-{j}, similarity: {sim:.4f}")

        # 处理未配对的表
        unpaired = [i for i in range(n) if i not in paired]

        # 统计平均相似度
        if pairs:
            avg_sim = np.mean([sim_matrix[i, j] for i, j in pairs])
            self.stats['avg_similarity'] = avg_sim
            log(f"  Average pairing similarity: {avg_sim:.4f}")

        return pairs, unpaired

    def optimal_pairing_by_assignment(self, tables: List[Table],
                                       sim_matrix: np.array) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        最优配对：使用网络流最大权匹配

        策略：找到全局最优的配对方案，使总相似度最大
        优点：全局最优解
        缺点：计算复杂度 O(n^3)

        Args:
            tables: 表列表
            sim_matrix: 相似度矩阵

        Returns:
            pairs: [(table_i_idx, table_j_idx), ...] 配对列表
            unpaired: [table_idx, ...] 未配对的表（奇数个表时）
        """
        import networkx as nx

        n = len(tables)

        # 构造图用于最大权匹配
        G = nx.Graph()

        # 添加所有节点
        for i in range(n):
            G.add_node(i)

        # 添加所有边（权重为相似度）
        for i in range(n):
            for j in range(i+1, n):
                weight = sim_matrix[i, j]
                G.add_edge(i, j, weight=weight)

        # 使用最大权匹配算法
        matching = nx.max_weight_matching(G, maxcardinality=False)

        # 转换为配对列表
        pairs = []
        paired_indices = set()
        for i, j in matching:
            # 确保 i < j
            if i > j:
                i, j = j, i
            pairs.append((i, j))
            paired_indices.add(i)
            paired_indices.add(j)
            log(f"  Pair tables {i}-{j}, similarity: {sim_matrix[i, j]:.4f}")

        # 处理未配对的表
        unpaired = [i for i in range(n) if i not in paired_indices]

        # 统计平均相似度
        if pairs:
            avg_sim = np.mean([sim_matrix[i, j] for i, j in pairs])
            self.stats['avg_similarity'] = avg_sim
            log(f"  Optimal average similarity: {avg_sim:.4f}")

        return pairs, unpaired

    def complementary_pairing(self, tables: List[Table]) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        互补配对：大表配小表（负载均衡）

        策略：最大的表和最小的表配对，次大和次小配对...
        优点：平衡计算负载，提升并行效率
        缺点：不考虑语义相似性

        Args:
            tables: 表列表

        Returns:
            pairs: [(table_i_idx, table_j_idx), ...] 配对列表
            unpaired: [table_idx, ...] 未配对的表（奇数个表时）
        """
        # 按表大小排序（从小到大）
        sorted_indices = sorted(range(len(tables)),
                                key=lambda i: len(tables[i].tids))

        # 大小配对：最小和最大，次小和次大...
        pairs = []
        n = len(sorted_indices)
        for i in range(n // 2):
            small_idx = sorted_indices[i]
            large_idx = sorted_indices[n - 1 - i]
            pairs.append((small_idx, large_idx))

            small_size = len(tables[small_idx].tids)
            large_size = len(tables[large_idx].tids)
            log(f"  Pair tables {small_idx}-{large_idx}, sizes: {small_size} + {large_size} = {small_size + large_size}")

        # 处理未配对的表（奇数情况）
        unpaired = [sorted_indices[n // 2]] if n % 2 == 1 else []

        if unpaired:
            mid_size = len(tables[unpaired[0]].tids)
            log(f"  Unpaired table {unpaired[0]}, size: {mid_size}")

        return pairs, unpaired

    def get_smart_pairing(self, tables: List[Table],
                          embeddings: np.array) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        根据策略获取智能配对方案

        Args:
            tables: 表列表
            embeddings: 所有实体的嵌入

        Returns:
            pairs: [(table_i_idx, table_j_idx), ...] 配对列表
            unpaired: [table_idx, ...] 未配对的表索引
        """
        n = len(tables)
        self.stats['total_pairings'] += 1

        log(f"Smart pairing strategy: {self.strategy}, {n} tables")

        # 特殊情况处理
        if n == 0:
            return [], []
        if n == 1:
            return [], [0]
        if n == 2:
            return [(0, 1)], []

        # 根据策略选择配对方法
        if self.strategy == 'complementary':
            # 互补配对不需要计算相似度
            pairs, unpaired = self.complementary_pairing(tables)
        else:
            # 其他策略需要计算表中心和相似度矩阵
            centers = self.compute_table_centers(tables, embeddings)
            sim_matrix = self.compute_similarity_matrix(centers)

            if self.strategy == 'similarity':
                pairs, unpaired = self.greedy_pairing_by_similarity(tables, sim_matrix)
            elif self.strategy == 'optimal':
                pairs, unpaired = self.optimal_pairing_by_assignment(tables, sim_matrix)
            else:
                log(f"Warning: Unknown strategy '{self.strategy}', fallback to 'similarity'")
                pairs, unpaired = self.greedy_pairing_by_similarity(tables, sim_matrix)

        log(f"Pairing completed: {len(pairs)} pairs, {len(unpaired)} unpaired")

        return pairs, unpaired

    def get_statistics(self) -> dict:
        """获取配对统计信息"""
        return self.stats.copy()


def demo_smart_pairing():
    """演示智能表配对功能"""
    import numpy as np
    from data import Table

    # 创建模拟数据
    n_tables = 5
    embedding_dim = 384

    # 模拟表
    tables = []
    for i in range(n_tables):
        n_entities = np.random.randint(100, 500)
        tids = list(range(i * 1000, i * 1000 + n_entities))
        tuple_ids = list(range(n_entities))
        tables.append(Table(str(i), tids, tuple_ids))

    # 模拟嵌入
    total_entities = sum(len(t.tids) for t in tables)
    embeddings = np.random.randn(max(max(t.tids) for t in tables) + 1, embedding_dim)
    # 归一化
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # 测试三种策略
    strategies = ['similarity', 'optimal', 'complementary']

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy}")
        print('='*60)

        pairing = SmartTablePairing(strategy=strategy)
        pairs, unpaired = pairing.get_smart_pairing(tables, embeddings)

        print(f"\nResults:")
        print(f"  Pairs: {pairs}")
        print(f"  Unpaired: {unpaired}")
        print(f"  Statistics: {pairing.get_statistics()}")


if __name__ == "__main__":
    demo_smart_pairing()
