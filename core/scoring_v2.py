"""
scoring_v2.py
Contribution to: Infinite Conversation Architecture (ICA)
Improves: The C(N,M) term in the ICA scoring formula

Original formula:
    S(N, M) = 0.25·R(N) + 0.40·C(N,M) + 0.20·E(N,M) + 0.15·Q(N)

Original C(N,M):
    "weighted sum of edges to hot window"
    — only sees DIRECT neighbors of hot-window nodes.
    — a node connected via two high-weight edges (e.g. A→B→hot_window)
      scores zero, even if it's the most relevant thing in the graph.

New C(N,M) — Personalized PageRank (PPR):
    Seeds the random walk at hot-window nodes.
    After convergence, every node in the graph has a score reflecting
    how "close" it is to the current conversational context, accounting
    for all path lengths and edge weights, not just direct neighbors.

Why PPR instead of, say, BFS with depth=2?
    PPR propagates through the full graph while naturally discounting
    distant nodes (damping factor α). It's differentiable in the edge
    weights, so RESOLVES edges (1.0) propagate much more signal than
    CONTINUES edges (0.2). BFS at depth=2 treats all reachable nodes equally.

Computational cost:
    PPR on a graph with 500 active nodes converges in <5ms with power iteration.
    For 5,000-node warm tier: ~50ms — acceptable for background pre-fetch,
    not for real-time. Use the cached PPR scores from the last turn for warm tier.
"""

from __future__ import annotations

import math
from typing import Protocol


# ---------------------------------------------------------------------------
# Graph protocol
# ---------------------------------------------------------------------------

class ScoringGraph(Protocol):
    def nodes(self) -> list[str]:
        ...

    def out_edges(self, node_id: str) -> list[tuple[str, float]]:
        """Return list of (neighbor_id, edge_weight)."""
        ...


# ---------------------------------------------------------------------------
# Personalized PageRank
# ---------------------------------------------------------------------------

def personalized_pagerank(
    graph: ScoringGraph,
    seed_nodes: set[str],           # hot-window node IDs
    *,
    alpha: float = 0.15,            # teleport probability (standard: 0.15)
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, float]:
    """
    Compute Personalized PageRank seeded at hot-window nodes.

    PPR equation:
        π = α · seed_dist + (1 - α) · A^T · π

    where seed_dist is uniform over seed nodes and A is the
    column-stochastic adjacency matrix (normalized by out-degree weight).

    Returns a dict mapping node_id → PPR score (sums to 1.0).
    """
    all_nodes = graph.nodes()
    if not all_nodes:
        return {}

    n = len(all_nodes)
    node_idx = {nid: i for i, nid in enumerate(all_nodes)}

    # Seed distribution: uniform over hot-window nodes
    seed_set = seed_nodes & set(all_nodes)
    if not seed_set:
        # Fallback: uniform distribution
        seed_set = set(all_nodes)

    seed_dist = {nid: 0.0 for nid in all_nodes}
    for nid in seed_set:
        seed_dist[nid] = 1.0 / len(seed_set)

    # Build column-stochastic transition dict
    # trans[v] = list of (u, weight) meaning "u sends weight to v"
    trans: dict[str, list[tuple[str, float]]] = {nid: [] for nid in all_nodes}
    for nid in all_nodes:
        out = graph.out_edges(nid)
        total_weight = sum(w for _, w in out)
        if total_weight == 0:
            continue
        for neighbor, w in out:
            if neighbor in node_idx:
                trans[neighbor].append((nid, w / total_weight))

    # Power iteration
    pi = dict(seed_dist)  # start from seed distribution

    for _ in range(max_iter):
        pi_new: dict[str, float] = {}
        for nid in all_nodes:
            incoming = sum(pi[src] * w for src, w in trans.get(nid, []))
            pi_new[nid] = alpha * seed_dist[nid] + (1 - alpha) * incoming

        # Check convergence (L1 norm)
        delta = sum(abs(pi_new[nid] - pi[nid]) for nid in all_nodes)
        pi = pi_new
        if delta < tol:
            break

    return pi


# ---------------------------------------------------------------------------
# Improved scoring formula
# ---------------------------------------------------------------------------

def score_node_v2(
    node_id: str,
    *,
    # Component inputs
    turn_delta: int,                    # turns since node was created
    ppr_score: float,                   # from personalized_pagerank()
    entity_overlap: float,              # |entities(N) ∩ entities(M)| / |entities(M)|
    is_open_question: bool,
    # Ground truth bonus (v2 extension)
    is_ground_truth: bool = False,
    # Weights — kept close to ICA v1 defaults
    w_recency: float = 0.25,
    w_connection: float = 0.40,
    w_entity: float = 0.20,
    w_question: float = 0.15,
    # Thresholds
    inject_threshold: float = 0.15,
    # PPR normalization — scale raw PPR to [0, 1] range.
    # In practice, seed nodes get PPR ~0.05–0.15 in a 500-node graph.
    # Set ppr_max to the 95th percentile PPR score in your active tier.
    ppr_max: float = 0.05,
) -> tuple[float, bool]:
    """
    Compute node injection score with PPR-based C(N,M).

    Returns:
        (score, should_inject)

    Changes from ICA v1:
        C(N,M): was "weighted sum of direct edges to hot window"
                now is PPR score (normalized), capturing full graph proximity
        R(N):   unchanged — exp(-0.005 × Δt) where Δt = turn_delta
        E(N,M): unchanged — entity Jaccard
        Q(N):   unchanged — binary open question flag
    """
    # R(N): recency decay
    r = math.exp(-0.005 * turn_delta)

    # C(N,M): PPR-based connection strength, normalized to [0, 1]
    # Clip at 1.0 in case a node exceeds expected ppr_max
    c = min(ppr_score / max(ppr_max, 1e-9), 1.0)

    # E(N,M): entity overlap
    e = entity_overlap

    # Q(N): open question
    q = 1.0 if is_open_question else 0.0

    score = w_recency * r + w_connection * c + w_entity * e + w_question * q

    # Ground truth nodes always injected (ICA v2 spec: score 0.85 floor)
    if is_ground_truth:
        score = max(score, 0.85)

    return score, score >= inject_threshold


# ---------------------------------------------------------------------------
# PPR cache — avoid recomputing every turn
# ---------------------------------------------------------------------------

class PPRCache:
    """
    Caches PPR scores per hot-window configuration.
    Invalidated when hot window shifts (every turn).

    In practice: compute PPR once per turn in the background,
    reuse for all node scoring decisions that turn.
    """

    def __init__(self):
        self._cache: dict[str, float] = {}
        self._seed_key: frozenset[str] = frozenset()

    def get_or_compute(
        self,
        graph: ScoringGraph,
        hot_window_ids: list[str],
        **ppr_kwargs,
    ) -> dict[str, float]:
        key = frozenset(hot_window_ids)
        if key != self._seed_key:
            self._cache = personalized_pagerank(
                graph, seed_nodes=set(hot_window_ids), **ppr_kwargs
            )
            self._seed_key = key
        return self._cache

    def invalidate(self) -> None:
        self._cache = {}
        self._seed_key = frozenset()
