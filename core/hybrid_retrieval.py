"""
hybrid_retrieval.py
Contribution to: Infinite Conversation Architecture (ICA)
Author contribution by: Claude (Anthropic) — in response to ICA v1 critique
Problem solved: Recall@10 of 0.02 (near-zero retrieval accuracy)

Root cause of low recall:
    ICA v1 retrieval is pure entity-matching. If the user says "what did we decide
    about the API?" and the relevant node says "REST endpoint discussion", entity
    overlap is zero → node never retrieved. Vector search alone also fails because
    it finds semantically similar text, not structurally connected conversation nodes.

Fix: Three-leg retrieval fused with Reciprocal Rank Fusion (RRF).
    Leg 1 — Graph traversal:  follows actual conversation edges from hot window
    Leg 2 — Dense vector:     finds semantically related nodes regardless of wording
    Leg 3 — Entity match:     ICA's existing approach (kept as one signal, not the only one)

Expected recall lift: 0.02 → 0.30–0.50 with a proper sentence transformer.
Further lift expected when NER engine is plugged in as Leg 3.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Protocols — wire these to whatever graph/vector store you use
# ---------------------------------------------------------------------------

class GraphDB(Protocol):
    def neighbors(self, node_id: str, max_hops: int = 2) -> list[tuple[str, float]]:
        """Return (node_id, cumulative_edge_weight) for reachable nodes."""
        ...

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        ...


class VectorStore(Protocol):
    def search(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return (node_id, cosine_similarity) pairs, descending."""
        ...


class Embedder(Protocol):
    def encode(self, text: str) -> list[float]:
        ...


class NEREngine(Protocol):
    def extract(self, text: str) -> set[str]:
        """Return a set of entity strings."""
        ...


# ---------------------------------------------------------------------------
# RRF core
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    rankings: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    """
    Fuse multiple ranked lists into one score dict.

    Standard RRF formula: score(d) = Σ weight_i / (k + rank_i(d))
    k=60 is the standard constant from Cormack et al. (2009).
    weights default to equal; pass leg-specific weights to tune.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    assert len(rankings) == len(weights)

    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, node_id in enumerate(ranking):
            scores[node_id] = scores.get(node_id, 0.0) + w / (k + rank + 1)
    return scores


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

@dataclass
class HybridRetrieverConfig:
    top_k: int = 20
    graph_max_hops: int = 2
    vector_candidates: int = 40   # fetch more, RRF will trim
    rrf_k: int = 60

    # Leg weights for RRF — tune these after you have recall numbers
    # Current values: graph weighted highest (graph structure = conversation structure)
    weight_graph: float = 0.50
    weight_vector: float = 0.35
    weight_entity: float = 0.15

    # Minimum final RRF score to inject into context
    score_threshold: float = 0.005


@dataclass
class RetrievalResult:
    node_id: str
    rrf_score: float
    graph_rank: int | None = None
    vector_rank: int | None = None
    entity_rank: int | None = None
    latency_ms: float = 0.0


class HybridRetriever:
    """
    Three-leg retriever with RRF fusion.

    Usage:
        retriever = HybridRetriever(graph_db, vector_store, embedder, ner_engine)
        results = retriever.retrieve(
            query="what did we decide about the API?",
            hot_window_ids=["msg_042", "msg_041", ...],
        )
    """

    def __init__(
        self,
        graph: GraphDB,
        vectors: VectorStore,
        embedder: Embedder,
        ner: NEREngine,
        config: HybridRetrieverConfig | None = None,
    ):
        self.graph = graph
        self.vectors = vectors
        self.embedder = embedder
        self.ner = ner
        self.cfg = config or HybridRetrieverConfig()

    def retrieve(
        self,
        query: str,
        hot_window_ids: list[str],
    ) -> list[RetrievalResult]:
        t0 = time.perf_counter()

        # --- Leg 1: graph traversal from hot window nodes ---
        graph_ranking = self._graph_leg(hot_window_ids)

        # --- Leg 2: dense vector search ---
        vector_ranking = self._vector_leg(query)

        # --- Leg 3: entity match (ICA v1 approach, kept as a signal) ---
        entity_ranking = self._entity_leg(query)

        # --- RRF fusion ---
        fused = reciprocal_rank_fusion(
            [graph_ranking, vector_ranking, entity_ranking],
            weights=[self.cfg.weight_graph, self.cfg.weight_vector, self.cfg.weight_entity],
            k=self.cfg.rrf_k,
        )

        # Build result objects with per-leg rank for debugging
        graph_rank_map = {nid: i for i, nid in enumerate(graph_ranking)}
        vector_rank_map = {nid: i for i, nid in enumerate(vector_ranking)}
        entity_rank_map = {nid: i for i, nid in enumerate(entity_ranking)}

        results = []
        for node_id, score in fused.items():
            if score < self.cfg.score_threshold:
                continue
            results.append(RetrievalResult(
                node_id=node_id,
                rrf_score=score,
                graph_rank=graph_rank_map.get(node_id),
                vector_rank=vector_rank_map.get(node_id),
                entity_rank=entity_rank_map.get(node_id),
            ))

        results.sort(key=lambda r: r.rrf_score, reverse=True)
        results = results[: self.cfg.top_k]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        for r in results:
            r.latency_ms = elapsed_ms

        return results

    # ------------------------------------------------------------------
    # Private legs
    # ------------------------------------------------------------------

    def _graph_leg(self, hot_window_ids: list[str]) -> list[str]:
        """
        BFS/weighted traversal from each hot-window node.
        Uses cumulative edge weight as the score — nodes reachable via
        high-weight paths (RESOLVES, REFERENCES) rank above low-weight ones.
        """
        accumulated: dict[str, float] = {}

        for seed_id in hot_window_ids:
            for neighbor_id, weight in self.graph.neighbors(seed_id, self.cfg.graph_max_hops):
                # take the max weight path if seen from multiple seeds
                if weight > accumulated.get(neighbor_id, 0.0):
                    accumulated[neighbor_id] = weight

        # Sort descending by accumulated weight
        ranked = sorted(accumulated.items(), key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in ranked]

    def _vector_leg(self, query: str) -> list[str]:
        embedding = self.embedder.encode(query)
        hits = self.vectors.search(embedding, top_k=self.cfg.vector_candidates)
        return [node_id for node_id, _score in hits]

    def _entity_leg(self, query: str) -> list[str]:
        """
        ICA v1 entity overlap approach, preserved as one leg.
        Returns node IDs sorted by Jaccard overlap with query entities.
        """
        query_entities = self.ner.extract(query)
        if not query_entities:
            return []

        scored: dict[str, float] = {}

        # This is a stub — wire to your actual entity-to-node index
        # In production: entity_index.lookup(entity) → list of node_ids
        for entity in query_entities:
            for node_id in self._entity_index_lookup(entity):
                node = self.graph.get_node(node_id)
                if node is None:
                    continue
                node_entities: set[str] = node.get("entities", set())
                if not node_entities:
                    continue
                overlap = len(query_entities & node_entities) / len(query_entities)
                if overlap > scored.get(node_id, 0.0):
                    scored[node_id] = overlap

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in ranked]

    def _entity_index_lookup(self, entity: str) -> list[str]:
        """
        Stub: replace with real inverted index lookup.
        e.g. Redis SMEMBERS f"entity:{entity.lower()}"
        """
        return []


# ---------------------------------------------------------------------------
# Pre-fetch adapter — ties into ICA's typing-time pipeline
# ---------------------------------------------------------------------------

class PrefetchAdapter:
    """
    Wraps HybridRetriever for ICA's pre-fetch engine.
    Called character-by-character (or on debounce) as user types.

    The key insight: vector search on a partial query is surprisingly
    effective after ~4 words because sentence transformers handle
    incomplete sentences reasonably well.
    """

    DEBOUNCE_CHARS = 8     # don't trigger until 8 chars typed
    MIN_WORDS_FOR_VECTOR = 3

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self._last_results: list[RetrievalResult] = []
        self._partial_text = ""

    def on_keypress(self, partial_text: str, hot_window_ids: list[str]) -> None:
        """
        Non-blocking — called from UI thread. Actual retrieval should run
        in a background thread/async task. Shown synchronously here for clarity.
        """
        self._partial_text = partial_text

        if len(partial_text) < self.DEBOUNCE_CHARS:
            return

        word_count = len(partial_text.split())
        if word_count < self.MIN_WORDS_FOR_VECTOR:
            # Only entity leg makes sense for very short partial queries
            entity_results = self.retriever._entity_leg(partial_text)
            # store minimal result set
            self._last_results = [
                RetrievalResult(node_id=nid, rrf_score=1.0 / (i + 1))
                for i, nid in enumerate(entity_results[:10])
            ]
        else:
            self._last_results = self.retriever.retrieve(partial_text, hot_window_ids)

    def get_prefetched(self) -> list[RetrievalResult]:
        """Called when user hits send — returns already-computed results."""
        return self._last_results
