"""
Infinite Conversation Architecture
Node Scoring Formula & Graph Traversal Algorithm

Author: Eugene Mawuli Attigah
Project: infinite-conversation-architecture
License: CC BY 4.0

─────────────────────────────────────────────────────────────────
THE SCORING FORMULA
─────────────────────────────────────────────────────────────────

Given a candidate node N and the current message M, the relevance
score S(N, M) is computed as:

    S(N, M) = w1 * R(N) + w2 * C(N, M) + w3 * E(N, M) + w4 * Q(N)

Where:

    R(N)    = Recency score       — how recent is this node?
    C(N, M) = Connection score    — how strongly is N connected to current context?
    E(N, M) = Entity overlap score — how many entities does N share with M?
    Q(N)    = Open question score  — does N contain an unresolved question?

Default weights:
    w1 = 0.25  (recency)
    w2 = 0.40  (connection — highest weight, connection is king)
    w3 = 0.20  (entity overlap)
    w4 = 0.15  (open questions)

All component scores are normalised to [0.0, 1.0].
Final score S(N, M) is in [0.0, 1.0].

─────────────────────────────────────────────────────────────────
COMPONENT FORMULAS
─────────────────────────────────────────────────────────────────

RECENCY — R(N):
    R(N) = exp(-λ * Δt)
    Where:
        Δt = current_turn - node.turn_number   (turn distance)
        λ  = 0.005                              (decay rate, tunable)

    This gives R(N) = 1.0 at Δt = 0, decaying toward 0 as distance grows.
    At Δt = 500 turns: R ≈ 0.08
    At Δt = 100 turns: R ≈ 0.61
    At Δt = 20 turns:  R ≈ 0.90

CONNECTION — C(N, M):
    C(N, M) = Σ edge.weight / max_possible_weight
    Summed over all edges connecting N to any node in the current hot window.
    Normalised by dividing by the theoretical maximum connection weight.

    Edge weights by type:
        RESOLVES      → 1.0  (strongest — directly answers something open)
        REFERENCES    → 0.9  (explicit callback)
        CONTRADICTS   → 0.8  (conflict worth surfacing)
        SHARES_ENTITY → 0.6  (same thing being discussed)
        SHARES_TOPIC  → 0.4  (same area, looser connection)
        CONTINUES     → 0.2  (sequential only, lowest weight)

ENTITY OVERLAP — E(N, M):
    E(N, M) = |entities(N) ∩ entities(M)| / |entities(M)|
    Jaccard-style overlap between node's entity tags and current message entities.
    Returns 1.0 if all entities in M appear in N.
    Returns 0.0 if no overlap.

OPEN QUESTION — Q(N):
    Q(N) = 1.0 if node contains unresolved open questions AND
               no RESOLVES edge exists pointing away from N
    Q(N) = 0.0 otherwise

    Binary. A node either has an unresolved question worth surfacing or it doesn't.
─────────────────────────────────────────────────────────────────
"""

import math
from dataclasses import dataclass
from typing import Optional
from schemas import ConversationNode, ConversationEdge, EdgeType


# ─────────────────────────────────────────────
# SCORING WEIGHTS (tunable)
# ─────────────────────────────────────────────

WEIGHT_RECENCY = 0.25
WEIGHT_CONNECTION = 0.40
WEIGHT_ENTITY = 0.20
WEIGHT_OPEN_QUESTION = 0.15

RECENCY_DECAY_RATE = 0.005  # λ in the recency formula

EDGE_WEIGHTS = {
    EdgeType.RESOLVES: 1.0,
    EdgeType.REFERENCES: 0.9,
    EdgeType.CONTRADICTS: 0.8,
    EdgeType.SHARES_ENTITY: 0.6,
    EdgeType.SHARES_TOPIC: 0.4,
    EdgeType.CONTINUES: 0.2,
}


# ─────────────────────────────────────────────
# COMPONENT SCORERS
# ─────────────────────────────────────────────

def recency_score(node: ConversationNode, current_turn: int) -> float:
    """
    R(N) = exp(-λ * Δt)
    Exponential decay based on turn distance from current position.
    """
    delta_t = max(0, current_turn - node.turn_number)
    return math.exp(-RECENCY_DECAY_RATE * delta_t)


def connection_score(
    node: ConversationNode,
    hot_window_node_ids: set[str],
    edges: list[ConversationEdge]
) -> float:
    """
    C(N, M) = sum of edge weights for all edges connecting N to hot window nodes.
    Normalised to [0, 1] by dividing by maximum possible weight.
    """
    max_possible = EDGE_WEIGHTS[EdgeType.RESOLVES] * len(hot_window_node_ids)
    if max_possible == 0:
        return 0.0

    total_weight = 0.0
    for edge in edges:
        is_connected = (
            (edge.source_node_id == node.node_id and edge.target_node_id in hot_window_node_ids) or
            (edge.target_node_id == node.node_id and edge.source_node_id in hot_window_node_ids)
        )
        if is_connected:
            total_weight += EDGE_WEIGHTS.get(edge.edge_type, 0.2) * edge.weight

    return min(1.0, total_weight / max_possible)


def entity_overlap_score(
    node: ConversationNode,
    current_message_entities: list[str]
) -> float:
    """
    E(N, M) = |entities(N) ∩ entities(M)| / |entities(M)|
    Proportion of current message entities that appear in this node.
    """
    if not current_message_entities:
        return 0.0

    node_entities = set(e.lower() for e in node.entity_tags)
    message_entities = set(e.lower() for e in current_message_entities)

    overlap = len(node_entities & message_entities)
    return overlap / len(message_entities)


def open_question_score(
    node: ConversationNode,
    outgoing_edges: list[ConversationEdge]
) -> float:
    """
    Q(N) = 1.0 if node has open questions with no RESOLVES edge pointing away from it.
    Q(N) = 0.0 otherwise.
    """
    if not node.open_questions:
        return 0.0

    # Check if any outgoing edge resolves this node's questions
    for edge in outgoing_edges:
        if edge.source_node_id == node.node_id and edge.edge_type == EdgeType.RESOLVES:
            return 0.0  # Already resolved

    return 1.0


# ─────────────────────────────────────────────
# MASTER SCORING FUNCTION
# ─────────────────────────────────────────────

@dataclass
class NodeScore:
    node: ConversationNode
    total_score: float
    recency: float
    connection: float
    entity: float
    open_question: float


def score_node(
    node: ConversationNode,
    current_turn: int,
    hot_window_node_ids: set[str],
    all_edges: list[ConversationEdge],
    current_message_entities: list[str],
) -> NodeScore:
    """
    S(N, M) = w1*R(N) + w2*C(N,M) + w3*E(N,M) + w4*Q(N)

    Returns a NodeScore with the breakdown of all components.
    """
    r = recency_score(node, current_turn)
    c = connection_score(node, hot_window_node_ids, all_edges)
    e = entity_overlap_score(node, current_message_entities)
    q = open_question_score(node, all_edges)

    total = (
        WEIGHT_RECENCY * r +
        WEIGHT_CONNECTION * c +
        WEIGHT_ENTITY * e +
        WEIGHT_OPEN_QUESTION * q
    )

    return NodeScore(
        node=node,
        total_score=round(total, 4),
        recency=round(r, 4),
        connection=round(c, 4),
        entity=round(e, 4),
        open_question=round(q, 4),
    )


# ─────────────────────────────────────────────
# GRAPH TRAVERSAL ALGORITHM
# ─────────────────────────────────────────────

def traverse_and_rank(
    graph_nodes: dict[str, ConversationNode],
    graph_edges: list[ConversationEdge],
    hot_window: list[ConversationNode],
    current_turn: int,
    current_message_entities: list[str],
    max_depth: int = 4,
    top_k: int = 10,
    min_score_threshold: float = 0.15,
) -> list[NodeScore]:
    """
    Graph traversal with scoring.

    Starting from nodes in the hot window, traverse outward through
    typed edges up to max_depth hops. Score every visited node.
    Return the top_k nodes above the minimum score threshold.

    Parameters:
        graph_nodes         — full node store {node_id: ConversationNode}
        graph_edges         — all edges in the graph
        hot_window          — current hot window nodes (starting points)
        current_turn        — current turn number
        current_message_entities — entities extracted from the new message
        max_depth           — maximum hops from hot window (default 4)
        top_k               — number of nodes to return (default 10)
        min_score_threshold — minimum score to qualify for injection (default 0.15)

    Returns:
        List of NodeScore objects, sorted by total_score descending.
    """

    hot_window_ids = {n.node_id for n in hot_window}

    # Build adjacency index for fast edge lookup
    adjacency: dict[str, list[str]] = {}
    for edge in graph_edges:
        adjacency.setdefault(edge.source_node_id, []).append(edge.target_node_id)
        adjacency.setdefault(edge.target_node_id, []).append(edge.source_node_id)

    # BFS from hot window nodes
    visited: set[str] = set(hot_window_ids)
    frontier: set[str] = set(hot_window_ids)
    candidates: set[str] = set()

    for depth in range(max_depth):
        next_frontier: set[str] = set()
        for node_id in frontier:
            neighbours = adjacency.get(node_id, [])
            for neighbour_id in neighbours:
                if neighbour_id not in visited:
                    visited.add(neighbour_id)
                    next_frontier.add(neighbour_id)
                    if neighbour_id not in hot_window_ids:
                        candidates.add(neighbour_id)
        frontier = next_frontier
        if not frontier:
            break

    # Score all candidates
    scored: list[NodeScore] = []
    for node_id in candidates:
        node = graph_nodes.get(node_id)
        if node is None:
            continue
        ns = score_node(
            node=node,
            current_turn=current_turn,
            hot_window_node_ids=hot_window_ids,
            all_edges=graph_edges,
            current_message_entities=current_message_entities,
        )
        if ns.total_score >= min_score_threshold:
            scored.append(ns)

    # Sort by score descending, return top_k
    scored.sort(key=lambda x: x.total_score, reverse=True)
    return scored[:top_k]
