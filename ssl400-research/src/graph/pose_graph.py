"""MediaPipe Pose (33-joint) spatial graph for ST-GCN.

Node order MUST match CSV column order (MediaPipe Pose landmark indices 0..32).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

NUM_JOINTS = 33

# Undirected anatomical edges for MediaPipe Pose BlazePose topology.
POSE_EDGES: list[tuple[int, int]] = [
    # Face / head
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (0, 9), (0, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Approximate root for spatial partitioning (mid-hip between 23 and 24 → use 23).
ROOT_JOINT = 23


def _get_hop_distance(num_node: int, edges: list[tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    adjacency = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    hop = np.full((num_node, num_node), np.inf, dtype=np.float32)
    np.fill_diagonal(hop, 0.0)
    transfer = adjacency.copy()
    for d in range(1, max_hop + 1):
        hop[transfer > 0] = np.minimum(hop[transfer > 0], float(d))
        transfer = transfer @ adjacency
    hop[hop == np.inf] = -1
    return hop


def _normalize_digraph(A: np.ndarray) -> np.ndarray:
    degree = A.sum(axis=0)
    degree[degree == 0] = 1.0
    Dn = np.diag(1.0 / degree)
    return A @ Dn


class Graph:
    """Build partitioned adjacency tensors used by ST-GCN spatial convolution."""

    def __init__(
        self,
        strategy: Literal["uniform", "distance", "spatial"] = "spatial",
        max_hop: int = 1,
        dilation: int = 1,
    ) -> None:
        self.num_node = NUM_JOINTS
        self.edges = list(POSE_EDGES)
        self.strategy = strategy
        self.max_hop = max_hop
        self.dilation = dilation
        self.hop_dis = _get_hop_distance(self.num_node, self.edges, max_hop=max_hop)
        self.A = self._get_adjacency()

    def _get_adjacency(self) -> np.ndarray:
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node), dtype=np.float32)
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1.0

        if self.strategy == "uniform":
            A = _normalize_digraph(adjacency)[None, ...]
        elif self.strategy == "distance":
            A = np.zeros((len(valid_hop), self.num_node, self.num_node), dtype=np.float32)
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = adjacency[self.hop_dis == hop]
                A[i] = _normalize_digraph(A[i])
        elif self.strategy == "spatial":
            A = self._spatial_partition(adjacency)
        else:
            raise ValueError(f"Unknown graph strategy: {self.strategy}")
        return A.astype(np.float32)

    def _spatial_partition(self, adjacency: np.ndarray) -> np.ndarray:
        """Self / centripetal / centrifugal partitions (Yan et al. ST-GCN)."""
        center = ROOT_JOINT
        A = np.zeros((3, self.num_node, self.num_node), dtype=np.float32)
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        for hop in valid_hop:
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if self.hop_dis[j, i] != hop:
                        continue
                    if hop == 0:
                        A[0, j, i] = adjacency[j, i]
                    elif self.hop_dis[j, center] > self.hop_dis[i, center]:
                        A[1, j, i] = adjacency[j, i]
                    else:
                        A[2, j, i] = adjacency[j, i]
        for a in range(A.shape[0]):
            A[a] = _normalize_digraph(A[a])
        return A


def get_adjacency_matrix(strategy: str = "spatial") -> np.ndarray:
    return Graph(strategy=strategy).A
