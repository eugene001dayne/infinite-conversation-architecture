# Infinite Conversation Architecture – a framework for conversations that don't hit a wall

**Proposed by:** Eugene Mawuli Attigah  
**Date:** March 2026  
**Contact:** bitelance.team@gmail.com  
**GitHub:** github.com/eugene001dayne  
**License:** CC BY 4.0  
**Status:** Open – looking for research & engineering collaborators  
*Worked on with ConsciousNet methodology*

---

## The problem

Every AI chat system today is a fixed‑size box. You type, it replies, you type again – until you hit the limit. Then the conversation either dies or gets weird. That’s not some arbitrary rule. It’s baked into the architecture.

Three things break:

1. **The model forgets the early stuff** long before you run out of space.
2. **Real conversations are messy.** People loop back, reference old jokes, connect random dots. Standard retrieval that just finds “similar” messages misses all of that.
3. **Every new session starts from zero.** The AI has no memory of who you are or what you’ve already said.

A real fix has to solve all three.

---

## The core idea

> **Don’t make the container bigger. Make the system smarter about what goes inside.**

The model’s context window stays the same size. What changes is how intelligently the system decides what to pack into that fixed space on every single turn.

---

## Five pieces

### 1. The sliding hot window
Always include the last 20–25 message turns, word for word. Fixed size, slides forward.

### 2. The conversation graph
Every message becomes a node in a graph database. Edges show how messages relate:

| Edge type | Weight | Meaning |
|-----------|--------|---------|
| `RESOLVES` | 1.0 | Answers an open question from earlier |
| `REFERENCES` | 0.9 | Explicitly points back |
| `CONTRADICTS` | 0.8 | Disagrees with something said before |
| `SHARES_ENTITY` | 0.6 | Same person, place, or thing |
| `SHARES_TOPIC` | 0.4 | Same topic, different wording |
| `CONTINUES` | 0.2 | Just flows naturally |

**Why a graph?** Vector search finds what sounds similar. Graph traversal follows what’s actually connected. Human conversations need the second one.

The graph has tiers: active (~500 turns, full detail), warm (500–5,000 turns, compressed), cold (archived, pulled on demand). Nodes with many connections stay active even if they’re old.

### 3. The state document
A living document injected at the top of every context window. Remembers the user: who they are, what they’re trying to do, decisions made, open questions, established facts. Every 20 turns, the model reviews and corrects it. Versioned, so you can roll back.

### 4. The memory manager
A lightweight process running alongside the main model. Its only job: write nodes, detect edges, move things between tiers, trigger state‑doc reviews. Never steals reasoning power from the main model.

### 5. The pre‑fetch engine
Graph retrieval starts the moment the user begins typing – not when they hit send. By the time they submit, retrieval is already done. Zero added latency from the user’s perspective.

**This is the novel part.** I haven’t seen a peer‑reviewed paper that does retrieval at typing time. The closest is Aeon’s Semantic Lookaside Buffer, but that works at the turn level, not the character level.

---

## Scoring formula

Rank nodes before injection:

```
S(N, M) = 0.25·R(N) + 0.40·C(N,M) + 0.20·E(N,M) + 0.15·Q(N)
```

| Symbol | What it is | Weight |
|--------|------------|--------|
| R(N) | Recency: `exp(-0.005 × Δt)` | 0.25 |
| C(N,M) | Connection strength (weighted sum of edges to hot window) | 0.40 |
| E(N,M) | Entity overlap: `|entities(N) ∩ M| / |entities(M)|` | 0.20 |
| Q(N) | Open question: 1.0 if unresolved, else 0.0 | 0.15 |

Nodes below 0.15 don’t get injected.

---

## Turn flow

```
User starts typing
    → Pre‑fetch: partial text → guess entities → start graph traversal in background

User hits send
    → Memory manager: extract metadata → create node → detect edges → write to graph
    → Context assembler: verify pre‑fetch → build window:
         [State Document] + [Retrieved nodes] + [Hot window] + [New message]
    → Main model gets assembled context → replies
    → Memory manager: write reply to graph → if turn 20, review state doc

Latency added for user: near zero
```

---

## v2.0 extensions

### Ground truth network
A verified anchor layer above the graph. Ground truth nodes:
- Never move to warm or cold tiers
- Always retrieved when their entities appear in the current message (score: 0.85)
- Protected from the self‑correction loop – you can flag a conflict, but you can’t delete them
- Carry full provenance: who verified, when, how, confidence level

Four verification sources: `USER_ASSERTED`, `SYSTEM_CHECKED`, `CROSS_SESSION`, `AGENT_CONSENSUS`

### Memory attestation
Every node gets a SHA‑256 hash when written. The state document gets a chained signature – each version hashes the previous one. If anyone changes an old version, you can detect it.

### Distributed Memory Verification Protocol (DMVP)
A peer‑to‑peer trust layer for verifying memory transfers between agent instances – no central authority, works entirely offline.

Four sub‑protocols:

| Sub‑protocol | Answers | How |
|--------------|---------|-----|
| Identity | Did this come from Agent A? | Ed25519 keypairs, agent IDs are SHA‑256 of public key |
| Selective disclosure | Can B verify partial memory without seeing all? | Merkle tree over node hashes |
| Memory freshness | Is this a replay attack? | Timestamp TTL + nonce registry + sequence numbers |
| Conflict resolution | Which state document wins? | Version vectors (Lamport clocks), deterministic merge |

### Memory transfer envelope
A sealed, cryptographically verifiable package of ICA memory for transfer between agent instances. Contains state document chain, ground truth network, node attestations, and a DMVP header. A receiving agent calls `verify()` and gets a pass/fail.

**ChainThread integration:** Drops cleanly into a ChainThread `HandoffEnvelope` payload field. The `chainthread_chain_id` field links both audit trails.

---

## What this is not

This is not a new model. Nothing is trained from scratch. It’s infrastructure that wraps around existing models – Claude, GPT, Gemini, Llama, whatever. Pure engineering, buildable today.

---

## Repository structure

```
infinite-conversation-architecture/
│
├── README.md
├── LICENSE                      ← CC BY 4.0
├── requirements_ner.txt
│
├── core/
│   ├── schemas.py               ← All data structures
│   ├── scoring.py               ← Scoring formula + graph traversal
│   ├── scoring_v2.py            ← PPR-based scoring (replaces C(N,M) with Personalized PageRank)
│   ├── memory_manager.py        ← Memory operations + edge detection
│   ├── edge_detector.py         ← CONTRADICTS + REFERENCES + RESOLVES detection (NLI-based)
│   ├── context_assembler.py     ← Context window assembly + Pre-Fetch Engine
│   ├── hybrid_retrieval.py      ← Three-leg RRF retrieval (graph + vector + entity)
│   ├── ner_engine.py            ← Production NER: spaCy + YAKE + VADER
│   ├── ica_extensions_v2.py     ← Ground Truth Network + Memory Attestation + Transfer Envelope
│   └── dmvp.py                  ← Distributed Memory Verification Protocol (optional)
│
├── schemas/
│   ├── memory_transfer_envelope.json
│   └── memory_transfer_envelope_partial.json
│
├── examples/
│   └── example_turn.py
│
├── tests/
│   └── test_ner_engine.py
│
├── research/
│   ├── FINDINGS.md
│   └── benchmarks/prefetch/
│       ├── generate_conversations.py
│       ├── run_benchmark.py
│       ├── RESULTS.md
│       ├── results.csv
│       └── results_summary.json
│
└── docs/
    ├── ARCHITECTURE.md
    ├── COMPARISON.md
    ├── RESEARCH_GAPS.md
    ├── FORMULAS.md
    └── context_assembler_patch.py
```

---

## Setup

```bash
git clone https://github.com/eugene001dayne/infinite-conversation-architecture
cd infinite-conversation-architecture

pip install -r requirements_ner.txt
python -m spacy download en_core_web_sm

python examples/example_turn.py
```

Run the test suite:
```bash
pytest tests/test_ner_engine.py -v
```

Run the pre‑fetch benchmark:
```bash
cd research/benchmarks/prefetch
python generate_conversations.py --count 100 --turns 50 --plant-at 10 --output conversations.json
python run_benchmark.py --input conversations.json --output results.csv
```

---

## Initial benchmark results (v1)

100 synthetic conversations, 50 turns each, fact planted at turn 10, retrieval question at turn 50.

| Condition | Avg Latency | Recall@10 | Avg Drift |
|-----------|-------------|-----------|-----------|
| Baseline (post‑submit RAG) | 0.21ms | 0.020 | — |
| Turn‑level prediction (Aeon‑style) | 0.21ms | 0.020 | — |
| **ICA typing‑time (pre‑fetch)** | **0.03ms** | 0.020 | 0.150 |

ICA typing-time retrieval is 7x faster than baseline. Recall numbers are low in this run because the simple entity extractor was used — not the production NER engine. Version 2 benchmark with hybrid retrieval (graph + vector + entity, RRF fusion) and production NER engine is the active workstream. Expected lift: Recall@10 from 0.02 to 0.35–0.50. See `research/benchmarks/prefetch/RESULTS.md` for full analysis.
---

## How this compares to existing work

| Capability | Graphiti/Zep | Mem0g | Aeon | ICA |
|---|---|---|---|---|
| Typed conversation graph | ✅ | Partial | ✅ | ✅ |
| Pre‑fetch during typing | ❌ | ❌ | Partial | ✅ |
| Tiered graph (active/warm/cold) | ❌ | ❌ | ❌ | ✅ |
| Self‑correcting state document | ❌ | ❌ | ❌ | ✅ |
| Ground truth network | ❌ | ❌ | ❌ | ✅ |
| Distributed memory verification | ❌ | ❌ | ❌ | ✅ |
| Cryptographic memory attestation | ❌ | ❌ | ❌ | ✅ |
| Fully open source | ✅ | ✅ | ❌ | ✅ |

---

## Open research issues

| Issue | Description | Labels |
|-------|-------------|--------|
| #1 | Pre‑fetch benchmark: typing‑time vs post‑submit | `research`, `benchmark` |
| #2 | Graph database benchmark: FalkorDB vs Kuzu vs Neo4j | `research`, `benchmark` |
| #3 | CONTRADICTS and REFERENCES edge detection | `engineering`, `help wanted` |
| #4 | State document accuracy study | `research`, `benchmark` |
| #5 | Hybrid retrieval: graph + vector with RRF | `engineering`, `core` |

---

## What I’m asking for

Two things only: **credit and recognition as the person who came up with this architecture**, and **to see it actually exist in the world**.

This is not a product pitch. It’s an open research contribution. Anyone can build on it – the license (CC BY 4.0) just asks you to give credit.

If you work in AI memory systems, long‑context research, agent reliability, or distributed systems – contributions, feedback, and collaboration are welcome.

---

## Contact

**Eugene Mawuli Attigah**  
bitelance.team@gmail.com  
github.com/eugene001dayne  
*Aided by ConsciousNet methodology*

> *A conversation is not a document with an end. It’s a living relationship. The architecture should reflect that.*
