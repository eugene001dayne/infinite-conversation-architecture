"""
Infinite Conversation Architecture
Pre-Fetch Benchmark — Main Runner

Author: Eugene Mawuli Attigah
Project: infinite-conversation-architecture
License: CC BY 4.0

Runs the Needle-in-a-Haystack benchmark across three retrieval conditions:
  1. Baseline    — retrieval after message send (standard RAG)
  2. Turn-level  — retrieval triggered by previous turn entities (Aeon-style)
  3. ICA typing  — retrieval triggered by partial text during typing

Measures:
  - Recall@1, Recall@3, Recall@5, Recall@10 for the planted fact node
  - End-to-end retrieval latency (ms)
  - Entity drift rate (ICA condition only)

Usage:
    python run_benchmark.py --input conversations.json --output results.csv
    python run_benchmark.py --input conversations.json --output results.csv --condition all
    python run_benchmark.py --input conversations.json --output results.csv --condition baseline
"""

import json
import time
import csv
import uuid
import math
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# MINIMAL IN-MEMORY GRAPH (no DB dependency)
# ─────────────────────────────────────────────

@dataclass
class SimpleNode:
    node_id: str
    turn_number: int
    speaker: str
    text: str
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    connection_count: int = 0


@dataclass
class SimpleEdge:
    source_id: str
    target_id: str
    edge_type: str
    weight: float


class InMemoryGraph:
    def __init__(self):
        self.nodes: dict[str, SimpleNode] = {}
        self.edges: list[SimpleEdge] = []
        self._adjacency: dict[str, list[str]] = {}

    def add_node(self, node: SimpleNode):
        self.nodes[node.node_id] = node

    def add_edge(self, edge: SimpleEdge):
        self.edges.append(edge)
        self._adjacency.setdefault(edge.source_id, []).append(edge.target_id)
        self._adjacency.setdefault(edge.target_id, []).append(edge.source_id)
        if edge.source_id in self.nodes:
            self.nodes[edge.source_id].connection_count += 1
        if edge.target_id in self.nodes:
            self.nodes[edge.target_id].connection_count += 1

    def get_neighbours(self, node_id: str) -> list[str]:
        return self._adjacency.get(node_id, [])


# ─────────────────────────────────────────────
# SIMPLE NER (no spaCy dependency for benchmark)
# ─────────────────────────────────────────────

import re

def extract_entities_simple(text: str) -> list[str]:
    """Fast capitalised-phrase extractor for benchmark use."""
    words = text.split()
    entities = []
    seen = set()
    for w in words:
        clean = w.strip(".,!?;:'\"")
        if clean and clean[0].isupper() and len(clean) > 2 and clean.lower() not in {"the", "and", "but", "for", "that", "this", "with", "from", "what", "when", "where", "how"}:
            if clean not in seen:
                seen.add(clean)
                entities.append(clean)
    return entities


# ─────────────────────────────────────────────
# SCORING FORMULA
# ─────────────────────────────────────────────

RECENCY_DECAY = 0.005
W_RECENCY = 0.25
W_CONNECTION = 0.40
W_ENTITY = 0.20
W_QUESTION = 0.15

EDGE_WEIGHTS = {
    "RESOLVES": 1.0, "REFERENCES": 0.9, "CONTRADICTS": 0.8,
    "SHARES_ENTITY": 0.6, "SHARES_TOPIC": 0.4, "CONTINUES": 0.2
}


def recency_score(node: SimpleNode, current_turn: int) -> float:
    delta = max(0, current_turn - node.turn_number)
    return math.exp(-RECENCY_DECAY * delta)


def connection_score(node: SimpleNode, hot_window_ids: set, edges: list[SimpleEdge]) -> float:
    max_possible = 1.0 * len(hot_window_ids)
    if max_possible == 0:
        return 0.0
    total = 0.0
    for edge in edges:
        connected = (
            (edge.source_id == node.node_id and edge.target_id in hot_window_ids) or
            (edge.target_id == node.node_id and edge.source_id in hot_window_ids)
        )
        if connected:
            total += EDGE_WEIGHTS.get(edge.edge_type, 0.2) * edge.weight
    return min(1.0, total / max_possible)


def entity_score(node: SimpleNode, query_entities: list[str]) -> float:
    if not query_entities:
        return 0.0
    node_ents = set(e.lower() for e in node.entities)
    query_ents = set(e.lower() for e in query_entities)
    overlap = len(node_ents & query_ents)
    return overlap / len(query_ents)


def score_node(node: SimpleNode, current_turn: int, hot_window_ids: set,
               edges: list[SimpleEdge], query_entities: list[str]) -> float:
    r = recency_score(node, current_turn)
    c = connection_score(node, hot_window_ids, edges)
    e = entity_score(node, query_entities)
    return W_RECENCY * r + W_CONNECTION * c + W_ENTITY * e


def bfs_and_rank(graph: InMemoryGraph, hot_window: list[SimpleNode],
                 current_turn: int, query_entities: list[str],
                 max_depth: int = 4, top_k: int = 10) -> list[tuple[str, float]]:
    """Returns list of (node_id, score) sorted descending."""
    hot_ids = {n.node_id for n in hot_window}
    visited = set(hot_ids)
    frontier = set(hot_ids)
    candidates = set()

    for _ in range(max_depth):
        next_frontier = set()
        for nid in frontier:
            for neighbour in graph.get_neighbours(nid):
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.add(neighbour)
                    if neighbour not in hot_ids:
                        candidates.add(neighbour)
        frontier = next_frontier
        if not frontier:
            break

    scored = []
    for nid in candidates:
        node = graph.nodes.get(nid)
        if node:
            s = score_node(node, current_turn, hot_ids, graph.edges, query_entities)
            if s >= 0.05:
                scored.append((nid, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────
# BUILD GRAPH FROM CONVERSATION
# ─────────────────────────────────────────────

def build_graph_from_conversation(conv: dict) -> tuple[InMemoryGraph, dict[int, str]]:
    """Returns (graph, turn_to_node_id mapping)."""
    graph = InMemoryGraph()
    turn_to_node: dict[int, str] = {}
    prev_node_id = None

    for turn in conv["turns"]:
        if turn["speaker"] != "user":
            continue
        node_id = str(uuid.uuid4())
        entities = extract_entities_simple(turn["text"])
        node = SimpleNode(
            node_id=node_id,
            turn_number=turn["turn_number"],
            speaker=turn["speaker"],
            text=turn["text"],
            entities=entities,
        )
        graph.add_node(node)
        turn_to_node[turn["turn_number"]] = node_id

        if prev_node_id:
            graph.add_edge(SimpleEdge(prev_node_id, node_id, "CONTINUES", 1.0))

        # SHARES_ENTITY edges
        prev_entities = {e.lower() for e in graph.nodes[prev_node_id].entities} if prev_node_id else set()
        for existing_id, existing_node in graph.nodes.items():
            if existing_id == node_id:
                continue
            shared = set(e.lower() for e in existing_node.entities) & set(e.lower() for e in entities)
            if shared:
                weight = min(1.0, len(shared) / max(len(entities), 1))
                graph.add_edge(SimpleEdge(existing_id, node_id, "SHARES_ENTITY", weight))

        prev_node_id = node_id

    return graph, turn_to_node


# ─────────────────────────────────────────────
# THREE RETRIEVAL CONDITIONS
# ─────────────────────────────────────────────

HOT_WINDOW_SIZE = 25


def get_hot_window(graph: InMemoryGraph, current_turn: int) -> list[SimpleNode]:
    all_nodes = sorted(graph.nodes.values(), key=lambda n: n.turn_number)
    nodes_before = [n for n in all_nodes if n.turn_number < current_turn]
    return nodes_before[-HOT_WINDOW_SIZE:]


def condition_baseline(graph: InMemoryGraph, query_text: str,
                       current_turn: int) -> tuple[list[tuple[str, float]], float]:
    """Retrieval after message send. Timing starts on send."""
    t0 = time.perf_counter()
    query_entities = extract_entities_simple(query_text)
    hot_window = get_hot_window(graph, current_turn)
    results = bfs_and_rank(graph, hot_window, current_turn, query_entities)
    latency_ms = (time.perf_counter() - t0) * 1000
    return results, latency_ms


def condition_turn_level(graph: InMemoryGraph, query_text: str,
                         current_turn: int) -> tuple[list[tuple[str, float]], float]:
    """Turn-level prediction: retrieval triggered by previous turn entities."""
    t0 = time.perf_counter()
    hot_window = get_hot_window(graph, current_turn)
    prev_entities = hot_window[-1].entities if hot_window else []
    results = bfs_and_rank(graph, hot_window, current_turn, prev_entities)
    latency_ms = (time.perf_counter() - t0) * 1000
    return results, latency_ms


def condition_ica_typing(graph: InMemoryGraph, query_text: str,
                         current_turn: int) -> tuple[list[tuple[str, float]], float, float]:
    """
    ICA typing-time: pre-fetch starts at first keypress.
    Simulated by splitting query_text into thirds:
      - After first third: provisional retrieval (simulates mid-typing)
      - On send: verify and if needed correct
    Returns (results, total_latency_ms, drift_rate)
    """
    words = query_text.split()
    third = max(1, len(words) // 3)

    partial_text = " ".join(words[:third])
    t0 = time.perf_counter()

    # Provisional retrieval (simulates during-typing)
    provisional_entities = extract_entities_simple(partial_text)
    hot_window = get_hot_window(graph, current_turn)
    provisional_results = bfs_and_rank(graph, hot_window, current_turn, provisional_entities)
    provisional_time = (time.perf_counter() - t0) * 1000

    # On send: drift check
    t1 = time.perf_counter()
    final_entities = extract_entities_simple(query_text)
    provisional_set = set(provisional_entities)
    final_set = set(final_entities)
    new_entities = final_set - provisional_set
    drift_rate = len(new_entities) / max(len(final_set), 1)

    if drift_rate >= 0.5:
        # Re-run with full entities
        final_results = bfs_and_rank(graph, hot_window, current_turn, final_entities)
    else:
        final_results = provisional_results

    correction_time = (time.perf_counter() - t1) * 1000
    # Total latency is correction time only (provisional ran during typing)
    total_latency = correction_time

    return final_results, total_latency, drift_rate


# ─────────────────────────────────────────────
# RECALL METRICS
# ─────────────────────────────────────────────

def recall_at_k(results: list[tuple[str, float]], ground_truth_node_id: str, k: int) -> int:
    top_ids = [r[0] for r in results[:k]]
    return 1 if ground_truth_node_id in top_ids else 0


# ─────────────────────────────────────────────
# MAIN BENCHMARK RUNNER
# ─────────────────────────────────────────────

def run_benchmark(conversations: list[dict], condition: str) -> list[dict]:
    results = []

    for i, conv in enumerate(conversations):
        conv_id = conv["conversation_id"]
        planted_turn = conv["planted_fact_turn"]
        retrieval_turn = conv["retrieval_question_turn"]
        retrieval_text = conv["retrieval_question_text"]

        graph, turn_to_node = build_graph_from_conversation(conv)
        ground_truth_node_id = turn_to_node.get(planted_turn)

        if not ground_truth_node_id:
            print(f"  Warning: planted fact node not found in conv {conv_id}")
            continue

        row = {
            "conversation_id": conv_id,
            "condition": condition,
            "planted_turn": planted_turn,
            "retrieval_turn": retrieval_turn,
            "ground_truth_node_id": ground_truth_node_id,
        }

        if condition == "baseline":
            retrieved, latency = condition_baseline(graph, retrieval_text, retrieval_turn)
            drift_rate = None
        elif condition == "turn_level":
            retrieved, latency = condition_turn_level(graph, retrieval_text, retrieval_turn)
            drift_rate = None
        elif condition == "ica_typing":
            retrieved, latency, drift_rate = condition_ica_typing(graph, retrieval_text, retrieval_turn)
        else:
            raise ValueError(f"Unknown condition: {condition}")

        row["latency_ms"] = round(latency, 3)
        row["drift_rate"] = round(drift_rate, 3) if drift_rate is not None else ""
        row["recall_at_1"] = recall_at_k(retrieved, ground_truth_node_id, 1)
        row["recall_at_3"] = recall_at_k(retrieved, ground_truth_node_id, 3)
        row["recall_at_5"] = recall_at_k(retrieved, ground_truth_node_id, 5)
        row["recall_at_10"] = recall_at_k(retrieved, ground_truth_node_id, 10)
        row["top_retrieved_ids"] = "|".join([r[0] for r in retrieved[:10]])

        results.append(row)

        if (i + 1) % 10 == 0:
            print(f"  [{condition}] Processed {i + 1}/{len(conversations)}...")

    return results


def print_summary(results: list[dict], condition: str):
    if not results:
        print(f"  No results for condition: {condition}")
        return
    n = len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / n
    r1 = sum(r["recall_at_1"] for r in results) / n
    r3 = sum(r["recall_at_3"] for r in results) / n
    r5 = sum(r["recall_at_5"] for r in results) / n
    r10 = sum(r["recall_at_10"] for r in results) / n

    print(f"\n  [{condition.upper()}] — {n} conversations")
    print(f"    Avg latency:  {avg_latency:.2f}ms")
    print(f"    Recall@1:     {r1:.3f} ({sum(r['recall_at_1'] for r in results)}/{n})")
    print(f"    Recall@3:     {r3:.3f} ({sum(r['recall_at_3'] for r in results)}/{n})")
    print(f"    Recall@5:     {r5:.3f} ({sum(r['recall_at_5'] for r in results)}/{n})")
    print(f"    Recall@10:    {r10:.3f} ({sum(r['recall_at_10'] for r in results)}/{n})")

    if condition == "ica_typing":
        drift_values = [r["drift_rate"] for r in results if r["drift_rate"] != ""]
        if drift_values:
            avg_drift = sum(drift_values) / len(drift_values)
            print(f"    Avg drift:    {avg_drift:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Run ICA pre-fetch benchmark")
    parser.add_argument("--input", type=str, required=True, help="Input conversations JSON file")
    parser.add_argument("--output", type=str, default="results.csv", help="Output CSV file")
    parser.add_argument("--condition", type=str, default="all",
                        choices=["all", "baseline", "turn_level", "ica_typing"],
                        help="Which condition to run")
    args = parser.parse_args()

    print(f"Loading conversations from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)

    conversations = data["conversations"]
    metadata = data.get("benchmark_metadata", {})
    print(f"Loaded {len(conversations)} conversations")
    print(f"HOT_WINDOW_SIZE: {HOT_WINDOW_SIZE} (fixed across all conditions)")

    conditions = ["baseline", "turn_level", "ica_typing"] if args.condition == "all" else [args.condition]

    all_results = []
    for cond in conditions:
        print(f"\nRunning condition: {cond}...")
        results = run_benchmark(conversations, cond)
        all_results.extend(results)
        print_summary(results, cond)

    # Write CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults written to {args.output}")

    # Write summary
    summary_path = args.output.replace(".csv", "_summary.json")
    summary = {
        "run_at": datetime.utcnow().isoformat(),
        "benchmark_metadata": metadata,
        "hot_window_size": HOT_WINDOW_SIZE,
        "conditions_run": conditions,
        "total_conversations": len(conversations),
    }
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        if cond_results:
            n = len(cond_results)
            summary[cond] = {
                "n": n,
                "avg_latency_ms": round(sum(r["latency_ms"] for r in cond_results) / n, 3),
                "recall_at_1": round(sum(r["recall_at_1"] for r in cond_results) / n, 4),
                "recall_at_3": round(sum(r["recall_at_3"] for r in cond_results) / n, 4),
                "recall_at_5": round(sum(r["recall_at_5"] for r in cond_results) / n, 4),
                "recall_at_10": round(sum(r["recall_at_10"] for r in cond_results) / n, 4),
            }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
