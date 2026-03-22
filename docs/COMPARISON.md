# ICA vs Existing Systems
## Infinite Conversation Architecture — Comparative Analysis

**Author:** Eugene Mawuli Attigah  
**Sources:** Research findings from Perplexity, Gemini Deep Research, Grok (March 2026)  
**License:** CC BY 4.0

---

## Overview

This document compares ICA against the four most relevant existing systems: Graphiti/Zep, Mem0g, Aeon, and MemGPT. The comparison is evidence-based, drawing from published benchmarks, open-source repositories, and peer-reviewed papers.

The conclusion is not that existing systems are poor. The conclusion is that no existing system combines all the capabilities ICA was designed around, and the combination itself produces properties none of them individually exhibit.

---

## The Systems

**Graphiti / Zep** — Open-source temporal context graph engine. Ingests message episodes, extracts entities via LLM, constructs directed graph with typed edges and bi-temporal validity windows. Sub-200ms retrieval in production. Supports FalkorDB, Kuzu, Neo4j, Neptune. GitHub: https://github.com/getzep/graphiti

**Mem0 / Mem0g** — Multi-level persistent memory with LLM-orchestrated extraction (ADD/UPDATE/DELETE/NOOP). Mem0g adds Neo4j graph layer for typed relationships and conflict resolution. On LoCoMo benchmark (600-turn conversations): 68.4% J-score — best result tested. P95 latency ~1.44s. Token usage ~7,000–14,000. GitHub: https://github.com/mem0ai/mem0

**Aeon** — Neuro-symbolic memory with Semantic Lookaside Buffer and Speculative Fetch Algorithm. Sub-5μs retrieval via cached cluster proximity. Paper: "Aeon: High-Performance Neuro-Symbolic Memory Management for Long-Horizon LLM Agents" (arXiv)

**MemGPT / Letta** — OS-style paging between RAM, disk, cold storage via agent tool calls. Agent manages its own memory explicitly. F1 score 0.09–0.26 on complex recall. High latency. GitHub: https://github.com/cpacker/MemGPT

---

## Feature Comparison

| Capability | Graphiti/Zep | Mem0g | Aeon | MemGPT | ICA |
|---|---|---|---|---|---|
| Typed conversation graph | ✅ | Partial | ✅ | ❌ | ✅ |
| Real-time edge detection | ✅ | Partial | ✅ | ❌ | ✅ |
| Depth-limited traversal at scale | ✅ | ❌ | Partial | ❌ | ✅ |
| Pre-fetch during user typing | ❌ | ❌ | Partial (turn-level) | ❌ | ✅ |
| Persistent state document | Partial | ✅ | ❌ | ✅ | ✅ |
| Self-correcting state review | ❌ | ❌ | ❌ | ❌ | ✅ |
| Tiered graph (active/warm/cold) | ❌ | ❌ | ❌ | ❌ | ✅ |
| Dedicated memory manager process | ❌ | ❌ | ❌ | ❌ | ✅ |
| Cross-session identity | ✅ | ✅ | ❌ | ✅ | ✅ |
| Model-agnostic | ✅ | ✅ | ❌ | ✅ | ✅ |
| Fully open source | ✅ (core) | ✅ | ❌ | ✅ | ✅ |

---

## Key Comparisons

### On typed edges

Graphiti uses custom ontologies with bi-temporal validity — edges carry four timestamps tracking when facts were valid and when they were ingested. Strong for evolving facts.

ICA uses six fixed edge types with explicit retrieval weights. The weight determines how strongly an historical node is surfaced — edge type drives scoring, not just labelling. The RESOLVES edge type, which detects that a new message answers an open question from an older node, is not present in Graphiti or Mem0g. This enables the architecture to actively close conversational loops.

### On retrieval latency

Aeon achieves sub-5μs via its Semantic Lookaside Buffer — the fastest of any system reviewed. It predicts based on the current turn's topic cluster. ICA's pre-fetch retrieves based on what the user is actively typing — more accurate for topic shifts which frequently occur at message boundaries. Zep claims sub-200ms after message send. Mem0g P95 is ~1.44s — too slow for real-time feel.

### On token efficiency

| System | Tokens per context |
|--------|-------------------|
| Mem0g | ~7,000–14,000 |
| ICA | ~8,000–16,000 |
| Zep | 600,000+ (reported cases) |

ICA is comparable to Mem0g and dramatically better than Zep at scale.

### On persistent identity

Zep's bi-temporal graph is the most rigorous approach to tracking evolving facts — every fact carries validity windows and provenance. ICA's State Document is simpler and cheaper to inject but adds what Zep lacks: a self-correcting review loop where a model explicitly reads recent conversation and asks "what is now wrong?" This catches nuance that entity-level invalidation misses.

---

## Where Each System Wins

**Graphiti/Zep** — Temporal fact tracking, bi-temporal validity, broadest backend support, most production-tested.

**Mem0g** — Best benchmark score (68.4% on LoCoMo), best token efficiency, easiest integration.

**Aeon** — Fastest raw retrieval (sub-5μs), hardware-level optimization.

**MemGPT** — Deliberate agent-controlled memory for cases where explicit memory management matters.

**ICA** — The combination. Tiered graph, typing-time pre-fetch, self-correcting state document, dedicated memory manager process — no existing system has all of these.

---

## The Honest Gaps

ICA is a proposal and early implementation — not production-tested.

- **No benchmark results yet.** Mem0g has LoCoMo scores. ICA has design arguments. Closing this gap is the most important next step.
- **CONTRADICTS and REFERENCES edge detection is a placeholder.** LLM-based extraction in Graphiti handles this far better than ICA's current keyword heuristics.
- **Zep's bi-temporal model is more rigorous** for tracking fact evolution over long periods.
- **Aeon's retrieval speed is currently unmatched** by ICA's pre-fetch, which depends on DB traversal latency.

These are engineering problems, not architectural ones. The design is sound. The implementation needs to catch up with the claim.

---

## Summary

No existing system is a direct competitor to ICA because no existing system solves all five problems simultaneously: real-time typed graph, tiered storage, typing-time pre-fetch, self-correcting state document, dedicated memory process. The field has converged on solving these in isolation. ICA's contribution is the unified architecture.

The most practical near-term path: benchmark ICA against Mem0g and Graphiti on LoCoMo, publish results. That converts the design argument into evidence.
