# Research Findings
## State of the Art in Graph-Based Memory for LLM Conversation Continuity

**Compiled by:** Eugene Mawuli Attigah  
**Sources:** Perplexity, Gemini Deep Research, Grok  
**Date:** March 2026  
**Purpose:** Literature review to contextualise the Infinite Conversation Architecture against existing work

---

## Summary

The research confirms three things:

1. The problem this architecture addresses is real, well-documented, and actively being worked on by serious researchers and companies.
2. The individual components we designed (typed graph, sliding window, pre-fetch, state document, memory manager) each have partial precedents in existing work — but no system combines all five into a unified architecture.
3. The specific gap this architecture fills — pre-fetch during user typing, typed conversation graph with real-time traversal, and persistent identity combined into one coherent system — does not exist in the current literature.

---

## 1. Existing Graph-Based Memory Implementations

Several open-source projects and research papers have implemented conversation graph stores with typed edges for LLM context management.

**Graphiti / Zep** is the most production-ready example. It is an open-source temporal context graph engine that ingests message episodes, automatically extracts entities via LLM, and constructs a directed graph with typed edges and bi-temporal validity windows. Old facts are invalidated rather than deleted, preserving provenance. Retrieval combines semantic embeddings, BM25, and graph traversal. It supports Neo4j, FalkorDB, Kuzu, and Neptune as backends. Sub-200ms retrieval is claimed in production.

Key gap in Graphiti: it does not implement predictive pre-fetch during typing, and its intelligent sliding-window selection inside the transformer context is limited.

**Mem0 / Mem0g** uses Neo4j for entity-relation triplets extracted from conversations, with typed relationships and conflict resolution. On the LoCoMo benchmark (600-turn conversations), Mem0g reaches 68.4% J-score — the best result among all tested systems. However, it adds latency and has been criticised for noisy extraction, where irrelevant social exchanges are retrieved alongside critical facts.

**Memory Palace** implements a persistent semantic memory layer separating long-term storage from the model's active context window, using typed edges including: `relates_to`, `derived_from`, `contradicts`, `exemplifies`, `refines`, and `supersedes`. Notably, it includes a `contradicts` edge type — consistent with our architecture's CONTRADICTS edge.

**Hindsight** organises memory into four logical networks: World Facts, Agent Experiences, Synthesized Entity Summaries, and Evolving Beliefs. Edges represent causal lineage and hierarchical relationships between concepts. It explicitly distinguishes between objective evidence and subjective beliefs — a level of epistemic structure not present in our current architecture.

**Aeon** introduces a neuro-symbolic episodic graph with CAUSAL, NEXT, and REFERS_TO edge types. It operates on cognitive state tracking and includes a Semantic Lookaside Buffer (SLB) — the closest existing system to our pre-fetch engine (see Section 4).

**SGMem** uses sentence-level graphs within chunks for turn/round/session associations, specifically targeting multi-turn dialogue coherence — the closest academic precedent to our typed conversation graph.

**Research papers of note:**
- "Graph-based Agent Memory: Taxonomy, Techniques, and Applications" (arXiv:2602.05665, 2026) — most comprehensive current survey
- "From Experience to Strategy: Empowering LLM Agents with Trainable Graph Memory" (arXiv:2511.07800)
- "Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects" (arXiv)
- "Aeon: High-Performance Neuro-Symbolic Memory Management for Long-Horizon LLM Agents" (arXiv)

---

## 2. Graph Databases for Real-Time AI Workloads

The workload this architecture requires: write one node per message turn, detect edges in real time, traverse outward from recent nodes at depth 4, at 10K–100K nodes. Here is what the benchmarks show:

**FalkorDB** is the strongest match. On a graph of 820K nodes and 9.8M edges:
- P50 query latency: 36ms
- P95: 74ms
- P99: 83ms
- PageRank: 18.53ms (vs Neo4j's 417ms — 22x faster)
- Memory footprint: 100MB (vs Neo4j's 600MB — 6x lower)

FalkorDB uses matrix-based execution via GraphBLAS and in-memory design, eliminating GC pauses. It is the recommended backend for this architecture's active tier.

**Kuzu** (embedded, lightweight) shows 18x faster ingestion and 100–188x faster multi-hop queries than Neo4j on social graphs. Its vectorised execution and Worst-Case Optimal Joins handle expensive traversals with minimal overhead. Best choice for self-hosted or resource-constrained deployments. Kuzu is also the embedded option — it runs in-process with the application, eliminating network latency entirely.

**Memgraph** (in-memory, streaming-first) reports up to 120x faster than Neo4j on streaming workloads, using only a quarter of the memory. Best suited when write volume dominates over read traversal.

**Neo4j** remains mature and ACID-compliant but shows P99 latency of 46 seconds under peak load — orders of magnitude too slow for real-time conversation use. Not recommended for the active tier.

**Recommendation for this architecture:**
- Active tier → FalkorDB or Kuzu
- Warm tier → Neo4j or PostgreSQL (lower real-time requirement)
- Cold tier → any persistent store

No published benchmark exactly matches the 10K–100K + depth-4 conversational workload. This is itself a gap — and a benchmark this project should produce.

---

## 3. Sliding Context Window Research

Academic work on intelligent context selection has advanced significantly but primarily addresses architectural efficiency rather than conversation-specific selection.

**StreamingLLM** (ICLR 2024) combines attention sinks — the observation that LLMs assign disproportionately high attention to initial tokens regardless of semantic content — with a sliding window of recent tokens. This allows models trained on finite sequences to generalise to infinite-length inputs without fine-tuning. The initial tokens in our State Document serve a structurally similar function to attention sinks.

**H2O — Heavy-Hitter Oracle** (NeurIPS 2023) identifies a small set of tokens that contribute most to attention scores and formulates KV cache eviction as a dynamic submodular problem. Reduces KV cache memory by up to 80% with minimal accuracy loss. 29x throughput improvement reported.

**ARKV** (arXiv 2025) dynamically allocates precision levels to cached tokens based on per-layer attention dynamics. Preserves 97% of baseline accuracy while reducing KV memory by 4x.

**Key gap confirmed:** Dedicated papers on conversation-specific intelligent selection — particularly graph-guided selection where historical nodes are retrieved based on relational structure rather than attention mechanics — remain sparse. Most advances are architectural (how the transformer attends internally) rather than infrastructural (what the system decides to feed the transformer). Our architecture operates at the infrastructural layer, which is largely unexplored.

---

## 4. Pre-Fetch Patterns for LLM Retrieval

This is the area with the least existing published work — and therefore the area where this architecture's contribution is most novel.

**TeleRAG** (arXiv:2502.20969) introduces lookahead prefetching, anticipating retrieval during LLM generation — not during user typing, but during the model's own response generation — via query similarity. The concept of starting retrieval before the query is complete is directly parallel to our pre-fetch engine.

**StreamingRAG** combines sliding eviction with dynamic RAG recall, prefetching evicted tokens. Again, operates during generation rather than during typing.

**Aeon's Semantic Lookaside Buffer** is the closest existing system to our pre-fetch engine. It calculates:

```
S_i = (q · c_i) / (‖q‖ · ‖c_i‖)
```

If the best matching score exceeds a threshold, the pointer is returned immediately at sub-5μs latency. The concept of Semantic Inertia — the observation that the topic at turn i+1 is highly correlated with turn i — directly validates our provisional retrieval approach.

**The confirmed gap:** True streaming retrieval during user input (character-by-character or word-by-word as the user types, before the message is sent) has not been published in peer-reviewed literature. Practitioner implementations reference it but no large-scale benchmarks exist. This is a genuine research contribution this architecture can claim.

---

## 5. Persistent State Management — MemGPT, Zep, Mem0 Compared

**MemGPT / Letta** treats memory as an agent's editable state with explicit paging between RAM, disk, and cold storage layers via tool calls. Strong for session continuity but introduces high latency due to OS-like paging logic. Performance on complex recall tasks: F1 score 0.09–0.26 (poor). Not suitable for real-time conversational continuity.

**Zep (Graphiti-backed)** maintains a single evolving temporal knowledge graph per user with bi-temporal validity tracking. Every fact carries validity windows and provenance. Strong persistent identity and cross-session coherence with automatic invalidation of obsolete information. However, the community self-hosting edition has been deprecated, limiting open deployment.

**Mem0** implements multi-level (user/session/agent) persistent memory with LLM-orchestrated extraction using ADD/UPDATE/DELETE/NOOP operations. On LoCoMo benchmark (600-turn conversations):
- Mem0: 66.9% J-score
- Mem0g (graph variant): 68.4% J-score — best result
- Zep: 66.0%
- OpenAI Memory: 52.9%
- MemGPT: lower

Mem0g P95 total latency: ~1.44 seconds. Token usage: ~7,000–14,000 (vs Zep's 600,000+ — 40x more efficient).

**Confirmed shortcomings across all systems:**
- No system implements "predecessor instance" — a structured protocol for a new agent instance to inherit the cognitive state of its predecessor
- Heavy reliance on semantic retrieval misses situational context that shares no direct embedding similarity with the current query
- No system combines typed conversation graphs with real-time depth-limited traversal AND pre-fetch AND persistent identity in a single unified architecture

---

## 6. The Gap This Architecture Fills

Synthesised from all three research sources, the gap is confirmed and specific:

| Capability | Graphiti/Zep | Mem0g | Aeon | ICA (This Architecture) |
|---|---|---|---|---|
| Typed conversation graph | ✅ | Partial | ✅ | ✅ |
| Real-time edge detection | ✅ | Partial | ✅ | ✅ |
| Depth-limited traversal at scale | ✅ | ❌ | Partial | ✅ |
| Pre-fetch during user typing | ❌ | ❌ | Partial (SLB) | ✅ |
| Persistent state document | Partial | ✅ | ❌ | ✅ |
| Self-correcting state review | ❌ | ❌ | ❌ | ✅ |
| Tiered graph (active/warm/cold) | ❌ | ❌ | ❌ | ✅ |
| Dedicated memory manager process | ❌ | ❌ | ❌ | ✅ |
| Cross-session identity | ✅ | ✅ | ❌ | ✅ |
| Open, model-agnostic | ✅ | ✅ | ❌ | ✅ |

No existing system combines all of these. The Infinite Conversation Architecture is the first to propose a unified design covering all layers from real-time edge detection through tiered storage through pre-fetch through persistent identity — in a single open, model-agnostic framework.

---

## 7. Recommended Next Research Steps

Based on this review, the following workstreams would most strengthen the project's research contribution:

**Benchmark the architecture** — Run conversations to 500, 1,000, and 5,000 turns and measure coherence against a Mem0g and Graphiti baseline. The lack of conversational-specific benchmarks at these scales is a documented gap this project can fill.

**Formalise the pre-fetch contribution** — The typing-time retrieval approach is the most novel component. A paper specifically benchmarking pre-fetch vs post-submit retrieval latency and accuracy would be a direct research contribution to an area with no published benchmarks.

**Evaluate FalkorDB vs Kuzu** — Run the specific workload (one node write + edge detection + depth-4 traversal) at 10K, 50K, and 100K nodes on both databases. No published benchmark covers this exact workload.

**Explore Hindsight's epistemic structure** — The distinction between world facts, experiences, opinions, and observations could strengthen the State Document beyond its current four-section structure.

---

## Sources

- Graphiti: https://github.com/getzep/graphiti
- Mem0: https://github.com/mem0ai/mem0
- Aeon: arXiv — High-Performance Neuro-Symbolic Memory Management for Long-Horizon LLM Agents
- Graph-based Agent Memory Survey: arXiv:2602.05665 (2026)
- Hindsight: arXiv — Building Agent Memory that Retains, Recalls, and Reflects
- TeleRAG: arXiv:2502.20969
- StreamingLLM: ICLR 2024 — mit-han-lab/streaming-llm
- H2O: NeurIPS 2023
- FalkorDB benchmarks: falkordb.com/blog/graph-database-performance-benchmarks
- Kuzu benchmarks: github.com/prrao87/kuzudb-study
- Memgraph benchmarks: memgraph.com/blog/memgraph-vs-neo4j-performance-benchmark-comparison
- LoCoMo benchmark: Mem0 paper arXiv:2504.19413
- Memory Palace: github.com/jeffpierce/memory-palace
- Continuum Memory: arXiv — Continuum Memory Architectures for Long-Horizon LLM Agents
