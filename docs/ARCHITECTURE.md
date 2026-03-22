# Architecture Deep Dive
## Infinite Conversation Architecture — Full Technical Specification

**Author:** Eugene Mawuli Attigah  
**Version:** 1.0  
**License:** CC BY 4.0

---

## Overview

The Infinite Conversation Architecture (ICA) is a five-component memory infrastructure that wraps around any existing language model to make AI conversation genuinely continuous — without growing the context window.

The system is grounded in one insight:

> **The context window is not the problem. The assumption that conversation is a flat, growing list is the problem.**

Current AI systems append every message to a single sequence until it overflows. ICA replaces that flat list with a structured, living memory system. The model's working memory stays fixed. What changes is how intelligently the system decides what fills that fixed space on every turn.

---

## Component 1 — The Sliding Hot Window

### What it is

A fixed-size buffer of the most recent message turns, always included in the model's context window verbatim.

### Design

```
Window size: 20–25 turns (configurable)
Content: Full message text for all turns in window
Movement: Slides forward with each new turn
Overflow: Turns exiting the window are handed to the Memory Manager
```

### Why 20–25 turns

This range covers the immediate conversational thread. Empirical observation from existing long-context research suggests the most critical coherence dependencies in conversation exist within a 15–30 turn radius. Beyond that, relational retrieval handles context better than brute-force inclusion.

### What it solves

Real-time conversational coherence. The model always has a complete, uncompressed view of what is happening right now.

---

## Component 2 — The Conversation Graph

### Why a graph — not a vector store

Vector stores retrieve by semantic similarity — they find messages that sound like the current message. Human conversation is non-linear. A question in turn 200 may only make sense in light of a decision made in turn 12. These connections are relational, not textual. A graph traversal follows the connection. A vector search finds similarity. Conversation needs traversal.

### Node schema

Every message turn becomes a `ConversationNode`:

```
node_id, conversation_id, turn_number
full_text         — Complete message (active tier only)
compressed_text   — Short summary (all tiers)
speaker           — USER or ASSISTANT
topic_tags        — Auto-extracted topic keywords
entity_tags       — Named entities: people, projects, places
open_questions    — Unresolved questions in this turn
emotional_register — neutral / curious / frustrated / excited / uncertain / assertive
tier              — ACTIVE / WARM / COLD
connection_count  — Number of edges on this node
```

### Edge types and weights

| Type | Weight | Meaning |
|------|--------|---------|
| `RESOLVES` | 1.0 | Answers an open question from the source node |
| `REFERENCES` | 0.9 | Explicitly refers back to the source node |
| `CONTRADICTS` | 0.8 | Conflicts with a claim in the source node |
| `SHARES_ENTITY` | 0.6 | Both messages mention the same named entity |
| `SHARES_TOPIC` | 0.4 | Same topic, differently worded |
| `CONTINUES` | 0.2 | Sequential — source immediately precedes target |

### Retrieval algorithm

```
1. Start from all nodes in the hot window
2. Traverse edges outward (BFS), depth limit: 4 hops
3. Score each candidate node: S(N,M) = w1·R + w2·C + w3·E + w4·Q
4. Return top 8–12 nodes scoring above 0.15 threshold
```

### Tiered graph

| Tier | Scope | Detail | Traversal Speed |
|------|-------|--------|----------------|
| Active | Last ~500 turns | Full text | Fast |
| Warm | Turns 500–5,000 | Compressed | Medium |
| Cold | Everything older | Archived | On-demand |

Highly connected nodes (3+ connections) stay active regardless of age.

### Recommended database backends

| Tier | Recommended | Why |
|------|-------------|-----|
| Active | FalkorDB or Kuzu | Sub-100ms traversal, in-memory |
| Warm | Neo4j or PostgreSQL | Persistence over speed |
| Cold | Any persistent store | On-demand only |

FalkorDB achieves P99 latency of 83ms on 820K-node graphs. Kuzu's embedded architecture eliminates network latency entirely.

---

## Component 3 — The State Document

### What it is

A living structured document always injected at the top of every context window. Gives the model persistent identity across turns and across sessions.

### Four sections

**User Identity** — name, role, goals, projects, preferences, technical level  
**Conversation State** — active threads, decisions made, open questions, agreements  
**Key Facts** — user-established facts, corrections made  
**Relationship History** — tone, duration, notable moments

### Self-correction loop

Every 20 turns a background process reviews the State Document against recent turns and produces a structured diff:

```
REMOVE: [outdated items]
ADD: [new information]
UPDATE: [old value → new value]
NO_CHANGES: [if nothing changed]
```

The document is versioned with rollback capability.

### Cross-session identity

The State Document persists tied to a user identity. Every new session loads the previous document first. The model never starts from zero.

---

## Component 4 — The Memory Manager

### What it is

A dedicated lightweight process running in parallel with the main model. Its only responsibility is memory operations. It never responds to users. It never reasons about conversation content.

### Why it is separate

Memory management in the main model consumes context tokens and competes with reasoning tasks. A dedicated process is always running, always consistent, regardless of conversation intensity. The main model trusts what it is handed.

### Operations on every turn

```
1. extract_metadata(message) → entities, topics, emotional_register (via NER engine)
2. Create ConversationNode
3. detect_edges() → CONTINUES + SHARES_ENTITY + SHARES_TOPIC + RESOLVES
4. Write node and edges to graph
5. Evaluate tier promotions/demotions
6. Check if state document review is due (every 20 turns)
7. Pre-compute retrieval query for next turn
```

### NER engine

- **spaCy `en_core_web_sm`** — entity recognition (PERSON, ORG, GPE, LOC, PRODUCT, EVENT, LANGUAGE, LAW)
- **YAKE** — topic keywords, deterministic, under 1ms
- **Lexicon + VADER** — emotional register, under 2ms
- Two pipelines: full (for complete turns) and NER-only (for typing path, 3–10ms)

---

## Component 5 — The Pre-Fetch Engine

### What it is

A retrieval process that begins the moment the user starts typing — not when they send.

### How it works

```
User begins typing
    → Extract provisional entities from partial text (NER-only, <10ms)
    → If entities changed: trigger provisional graph traversal (background)

User hits send
    → Verify provisional result against final message entities
    → If drift < 50%: use provisional result directly
    → If drift >= 50%: targeted correction pass only
    → Result ready before model starts processing
```

### Short text threshold

Below 15 characters, regex-only path. NER has insufficient context on 1–2 words.

### Why this matters

This is the most novel component of the architecture. No peer-reviewed paper covers typing-time retrieval. The closest existing work is Aeon's Semantic Lookaside Buffer (turn-level, not character-level). From the user's perspective, the conversation feels as fast as a system with no retrieval — because retrieval is already done.

---

## Full Turn Flow

```
USER STARTS TYPING
    → Pre-Fetch: partial text → provisional entities → provisional traversal (background)

USER HITS SEND
    → Memory Manager:
        extract_metadata → create node → detect edges → write to graph
        evaluate tiers → check review cycle

    → Context Assembler:
        verify pre-fetch → build hot window → retrieve graph nodes
        render state doc → assemble: [State Doc][Retrieved][Hot Window][New Message]

    → Main Model:
        receives assembled context → generates response

    → Memory Manager:
        writes response to graph → detects edges
        if turn 20: state doc review in background

LATENCY ADDED FOR USER: Near zero
```

---

## Known Limitations and Open Problems

**CONTRADICTS and REFERENCES edge detection** — Current implementation uses keyword heuristics. Production requires a lightweight semantic classifier. Highest-priority engineering gap.

**Open question detection** — Currently keyword overlap. Should use embedding similarity between stored questions and new message text.

**State document extraction quality** — Only as good as the review model. The review prompt is a critical system prompt — tune carefully.

**Graph at extreme scale** — Tiering handles thousands of turns well. Multi-year deployments with tens of millions of turns will require TTL or importance-weighted archival beyond tiering.

**Multi-conversation identity** — Current architecture tracks one conversation at a time. Parallel threads or group contexts require a user-level graph aggregating across conversations.

---

## Implementation

**Language:** Python 3.11+

**Startup:**
```python
from core.ner_engine import warm_up
warm_up()  # Pre-load spaCy models once at server startup
```

**Minimum viable call:**
```python
from core.memory_manager import process_turn
from core.schemas import Speaker

result = process_turn(
    message_text="Your message here",
    speaker=Speaker.USER,
    turn_number=current_turn,
    conversation_id="conv_id",
    existing_nodes=list(graph_nodes.values()),
    state_document=state_doc,
    extracted_open_questions=[],
)
```
