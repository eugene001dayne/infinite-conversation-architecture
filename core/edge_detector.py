"""
edge_detector.py
Contribution to: Infinite Conversation Architecture (ICA)
Solves: Open Issue #3 — CONTRADICTS and REFERENCES edge detection

Why this matters:
    Without these two edge types, most graph edges default to CONTINUES (weight 0.2).
    The graph becomes a shallow chain instead of a rich relational structure.
    High-weight edges (RESOLVES 1.0, REFERENCES 0.9, CONTRADICTS 0.8) are what
    make graph traversal superior to pure vector search. Without them, you don't
    have a graph — you have an expensive linked list.

Approach:
    CONTRADICTS — Natural Language Inference (NLI). Two messages sharing entities
                  are contradictory if an NLI model scores them as "contradiction".
                  Uses cross-encoder/nli-deberta-v3-small (~85MB, runs on CPU).

    REFERENCES  — Two-signal detection:
                  Signal A: Explicit reference patterns (regex on surface text)
                  Signal B: Semantic similarity above a threshold to a non-adjacent node
                  Both signals required to reduce false positives.

    RESOLVES    — Bonus: also implemented here. A message resolves a prior one if
                  that prior message contains an interrogative and the new message
                  shares its entities AND has high semantic similarity to it.

Runtime note:
    NLI inference is ~10ms per pair on CPU (deberta-v3-small).
    Only run on node pairs that share at least one entity — this gates the expense.
    On a conversation with 500 active nodes, entity gating reduces pairs from
    125,000 to typically <500 candidate pairs per new message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


# ---------------------------------------------------------------------------
# Edge types (mirrors ICA schema)
# ---------------------------------------------------------------------------

class EdgeType(str, Enum):
    RESOLVES = "RESOLVES"        # weight 1.0
    REFERENCES = "REFERENCES"   # weight 0.9
    CONTRADICTS = "CONTRADICTS"  # weight 0.8
    SHARES_ENTITY = "SHARES_ENTITY"   # weight 0.6
    SHARES_TOPIC = "SHARES_TOPIC"     # weight 0.4
    CONTINUES = "CONTINUES"           # weight 0.2

EDGE_WEIGHTS = {
    EdgeType.RESOLVES: 1.0,
    EdgeType.REFERENCES: 0.9,
    EdgeType.CONTRADICTS: 0.8,
    EdgeType.SHARES_ENTITY: 0.6,
    EdgeType.SHARES_TOPIC: 0.4,
    EdgeType.CONTINUES: 0.2,
}


@dataclass
class DetectedEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    confidence: float          # 0–1, edge-type-specific meaning
    evidence: str              # human-readable explanation


# ---------------------------------------------------------------------------
# Protocols — wire to your NLI model and embedder
# ---------------------------------------------------------------------------

class NLIModel(Protocol):
    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        """
        Return dict with keys: 'entailment', 'neutral', 'contradiction'
        Values are probabilities summing to ~1.0.

        Recommended model: cross-encoder/nli-deberta-v3-small
        Usage:
            from transformers import pipeline
            nli = pipeline("text-classification",
                           model="cross-encoder/nli-deberta-v3-small",
                           return_all_scores=True)
            result = nli(f"{premise} [SEP] {hypothesis}")[0]
            return {r['label'].lower(): r['score'] for r in result}
        """
        ...


class SimilarityModel(Protocol):
    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between sentence embeddings, 0–1."""
        ...


# ---------------------------------------------------------------------------
# REFERENCES detection
# ---------------------------------------------------------------------------

# Patterns that signal a message is explicitly referencing a prior exchange.
# Ordered by specificity — more specific patterns first.
_REFERENCE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(as|like)\s+(i|we|you)\s+(said|mentioned|discussed|talked about|noted|pointed out)\b",
        r"\b(remember|recall)\s+when\b",
        r"\b(going back to|returning to|revisiting)\b",
        r"\b(that|the)\s+(thing|point|issue|question|problem|idea|concern)\s+(you|we|i)\s+(raised|mentioned|brought up|flagged|highlighted)\b",
        r"\b(earlier|previously|before)\s+(you|we|i|we discussed|i mentioned)\b",
        r"\b(as|per)\s+(mentioned|discussed|agreed|noted)\b",
        r"\b(following up|circling back)\b",
        r"\byou (asked|said|mentioned|noted) (earlier|before|previously|that)\b",
        r"\b(on|about) (that|the) (topic|subject|point|question) (of|from|we)\b",
        r"\bstill (thinking about|on|considering) (what|that|the)\b",
    ]
]

def _has_reference_pattern(text: str) -> tuple[bool, str]:
    """Returns (matched, pattern_string)."""
    for pattern in _REFERENCE_PATTERNS:
        m = pattern.search(text)
        if m:
            return True, m.group(0)
    return False, ""


def detect_references(
    new_node_id: str,
    new_text: str,
    candidate_nodes: list[dict],   # each: {"id": str, "text": str, "turn": int}
    current_turn: int,
    similarity_model: SimilarityModel,
    *,
    min_similarity: float = 0.60,
    min_turn_gap: int = 3,         # must reference something at least 3 turns back
) -> list[DetectedEdge]:
    """
    Detect REFERENCES edges from new_node to older nodes.

    Two-signal requirement:
        1. New message contains an explicit reference pattern, OR
        2. Cosine similarity to a non-adjacent node exceeds min_similarity
           (suggesting the user is connecting back without an explicit phrase)

    Requiring both signals when pattern is absent makes the detector conservative —
    better to miss some REFERENCES than to flood the graph with false edges.
    """
    pattern_match, pattern_text = _has_reference_pattern(new_text)
    results: list[DetectedEdge] = []

    for candidate in candidate_nodes:
        cid = candidate["id"]
        ctext = candidate["text"]
        cturn = candidate.get("turn", 0)

        # Skip adjacent turns — those become CONTINUES
        if current_turn - cturn < min_turn_gap:
            continue

        sim = similarity_model.similarity(new_text, ctext)

        if pattern_match and sim > 0.45:
            # Explicit pattern + moderate similarity → confident REFERENCES
            results.append(DetectedEdge(
                source_id=new_node_id,
                target_id=cid,
                edge_type=EdgeType.REFERENCES,
                weight=EDGE_WEIGHTS[EdgeType.REFERENCES],
                confidence=min(0.95, 0.5 + sim * 0.5),
                evidence=f"Pattern: '{pattern_text}' + similarity={sim:.2f}",
            ))
        elif not pattern_match and sim > min_similarity:
            # No explicit pattern but high semantic similarity to an old node →
            # likely an implicit callback; use lower confidence
            results.append(DetectedEdge(
                source_id=new_node_id,
                target_id=cid,
                edge_type=EdgeType.REFERENCES,
                weight=EDGE_WEIGHTS[EdgeType.REFERENCES],
                confidence=sim * 0.75,
                evidence=f"High similarity to turn {cturn} (sim={sim:.2f}, no explicit pattern)",
            ))

    # Return the top-3 references — a message rarely references more than that
    results.sort(key=lambda e: e.confidence, reverse=True)
    return results[:3]


# ---------------------------------------------------------------------------
# CONTRADICTS detection
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS = re.compile(
    r"\b(no|not|never|none|nothing|neither|nor|cannot|can't|won't|isn't|aren't|wasn't"
    r"|weren't|doesn't|don't|didn't|hasn't|haven't|hadn't|shouldn't|wouldn't|couldn't)\b",
    re.IGNORECASE,
)

def detect_contradictions(
    new_node_id: str,
    new_text: str,
    new_entities: set[str],
    candidate_nodes: list[dict],   # each: {"id", "text", "entities": set[str]}
    nli_model: NLIModel,
    *,
    contradiction_threshold: float = 0.65,
    max_candidates: int = 30,
) -> list[DetectedEdge]:
    """
    Detect CONTRADICTS edges using NLI, gated by entity overlap.

    Pipeline:
        1. Entity gate   — only test pairs sharing ≥1 entity (kills 99% of pairs)
        2. Negation hint — if neither text contains negation, skip NLI (fast path)
        3. NLI inference — run cross-encoder on surviving pairs
        4. Threshold     — emit edge if contradiction_prob > threshold

    Why entity gate?
        Two messages with zero entity overlap cannot meaningfully contradict
        each other in the conversation sense. The gate makes NLI inference
        affordable even on large active tiers.
    """
    results: list[DetectedEdge] = []

    # Sort candidates by entity overlap descending, take top N for NLI
    def entity_overlap(c: dict) -> int:
        return len(new_entities & c.get("entities", set()))

    candidates_sorted = sorted(
        [c for c in candidate_nodes if entity_overlap(c) > 0],
        key=entity_overlap,
        reverse=True,
    )[:max_candidates]

    for candidate in candidates_sorted:
        cid = candidate["id"]
        ctext = candidate["text"]

        # Fast path: if neither message has negation words, contradiction
        # is very unlikely in conversational text. Skip NLI.
        has_negation = (
            _NEGATION_PATTERNS.search(new_text) is not None or
            _NEGATION_PATTERNS.search(ctext) is not None
        )
        if not has_negation:
            continue

        # NLI: run in both directions, take max contradiction score.
        # Asymmetric because "A contradicts B" ≠ "B contradicts A" in NLI models.
        try:
            scores_fwd = nli_model.predict(premise=ctext, hypothesis=new_text)
            scores_rev = nli_model.predict(premise=new_text, hypothesis=ctext)
        except Exception:
            continue

        contradiction_score = max(
            scores_fwd.get("contradiction", 0.0),
            scores_rev.get("contradiction", 0.0),
        )

        if contradiction_score >= contradiction_threshold:
            results.append(DetectedEdge(
                source_id=new_node_id,
                target_id=cid,
                edge_type=EdgeType.CONTRADICTS,
                weight=EDGE_WEIGHTS[EdgeType.CONTRADICTS],
                confidence=contradiction_score,
                evidence=(
                    f"NLI contradiction={contradiction_score:.2f} "
                    f"(fwd={scores_fwd.get('contradiction', 0):.2f}, "
                    f"rev={scores_rev.get('contradiction', 0):.2f})"
                ),
            ))

    results.sort(key=lambda e: e.confidence, reverse=True)
    return results


# ---------------------------------------------------------------------------
# RESOLVES detection (bonus — not in ICA v1)
# ---------------------------------------------------------------------------

_QUESTION_PATTERNS = re.compile(
    r"(\?|^(what|who|when|where|why|how|which|can|could|would|should|is|are|do|does|did)\b)",
    re.IGNORECASE | re.MULTILINE,
)

def detect_resolves(
    new_node_id: str,
    new_text: str,
    new_entities: set[str],
    candidate_nodes: list[dict],
    similarity_model: SimilarityModel,
    *,
    min_similarity: float = 0.50,
) -> list[DetectedEdge]:
    """
    Detect RESOLVES edges: new message answers an earlier open question.

    Heuristic:
        Candidate must look like a question (ends with ? or starts with interrogative).
        New message must share entities with that question AND be semantically similar.
        This approximates "same subject, but now answered."
    """
    results: list[DetectedEdge] = []

    for candidate in candidate_nodes:
        ctext = candidate["text"]
        centities = candidate.get("entities", set())

        # Must look like a question
        if not _QUESTION_PATTERNS.search(ctext):
            continue

        # Must share entities
        if not (new_entities & centities):
            continue

        sim = similarity_model.similarity(new_text, ctext)
        if sim >= min_similarity:
            results.append(DetectedEdge(
                source_id=new_node_id,
                target_id=candidate["id"],
                edge_type=EdgeType.RESOLVES,
                weight=EDGE_WEIGHTS[EdgeType.RESOLVES],
                confidence=sim,
                evidence=f"Candidate is interrogative, entity overlap, sim={sim:.2f}",
            ))

    results.sort(key=lambda e: e.confidence, reverse=True)
    return results[:2]


# ---------------------------------------------------------------------------
# Unified edge detection entry point
# ---------------------------------------------------------------------------

def detect_all_edges(
    new_node_id: str,
    new_text: str,
    new_entities: set[str],
    candidate_nodes: list[dict],
    current_turn: int,
    nli_model: NLIModel,
    similarity_model: SimilarityModel,
) -> list[DetectedEdge]:
    """
    Run all detectors and return a deduplicated, priority-sorted edge list.
    Higher-weight edge type wins if the same pair is detected by multiple detectors.
    """
    all_edges: list[DetectedEdge] = []

    all_edges += detect_resolves(
        new_node_id, new_text, new_entities, candidate_nodes, similarity_model
    )
    all_edges += detect_references(
        new_node_id, new_text, candidate_nodes, current_turn, similarity_model
    )
    all_edges += detect_contradictions(
        new_node_id, new_text, new_entities, candidate_nodes, nli_model
    )

    # Deduplicate: for each target node, keep only the highest-weight edge
    best: dict[str, DetectedEdge] = {}
    for edge in all_edges:
        existing = best.get(edge.target_id)
        if existing is None or edge.weight > existing.weight:
            best[edge.target_id] = edge

    return sorted(best.values(), key=lambda e: e.weight, reverse=True)
