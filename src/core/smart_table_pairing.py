"""
Table pairing strategy: during hierarchical merging, decide which two tables to merge first.

Greedy pairing by table semantic-center similarity, from highest to lowest.
"""

import numpy as np
from typing import List, Tuple

from data import Table
from log import log


class SmartTablePairing:
    def __init__(self):
        self.stats = {
            'total_pairings': 0,
            'avg_similarity': 0.0,
        }

    def compute_table_centers(self, tables: List[Table], embeddings: np.array) -> np.array:
        """
        Semantic center of each table: the mean of all entity embeddings in the table, then L2-normalized.
        """
        centers = []
        for table in tables:
            table_embeddings = embeddings[table.tids]  # get all encodings
            center = table_embeddings.mean(axis=0)  # take the mean
            center_norm = np.linalg.norm(center)  # compute the vector norm
            if center_norm > 0:
                center = center / center_norm
            centers.append(center)

        return np.array(centers)

    def compute_similarity_matrix(self, centers: np.array) -> np.array:
        """Symmetric cosine similarity matrix"""
        n = len(centers)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                # centers are already normalized, so the dot product is the cosine
                sim = np.dot(centers[i], centers[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        np.fill_diagonal(sim_matrix, 1.0)

        return sim_matrix

    def greedy_pairing_by_similarity(self, tables: List[Table],
                                      sim_matrix: np.array) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Greedy: sort by sim and pair sequentially from highest to lowest, skipping already-paired tables.
        """
        n = len(tables)
        paired = set()  # indices of tables already paired
        pairs = []  # the final pairing result

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

    def get_smart_pairing(self, tables: List[Table],
                          embeddings: np.array) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Greedy pairing by semantic similarity"""
        n = len(tables)
        self.stats['total_pairings'] += 1  # record the total count

        log(f"Smart pairing, {n} tables")

        if n == 0:
            return [], []
        if n == 1:
            return [], [0]
        if n == 2:
            return [(0, 1)], []

        centers = self.compute_table_centers(tables, embeddings)  # compute the "table semantic centers"
        sim_matrix = self.compute_similarity_matrix(centers)  # compute the similarity matrix
        pairs, unpaired = self.greedy_pairing_by_similarity(tables, sim_matrix)  # greedy pairing

        log(f"Pairing completed: {len(pairs)} pairs, {len(unpaired)} unpaired")

        return pairs, unpaired

    def get_statistics(self) -> dict:
        return self.stats.copy()


def demo_smart_pairing():
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

    pairing = SmartTablePairing()
    pairs, unpaired = pairing.get_smart_pairing(tables, embeddings)

    print(f"\nResults:")
    print(f"  Pairs: {pairs}")
    print(f"  Unpaired: {unpaired}")
    print(f"  Statistics: {pairing.get_statistics()}")


if __name__ == "__main__":
    demo_smart_pairing()
