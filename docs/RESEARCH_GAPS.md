# Research Gaps
## The Three Novel Contributions of the Infinite Conversation Architecture

**Author:** Eugene Mawuli Attigah  
**License:** CC BY 4.0

---

## Introduction

This document identifies the specific gaps in current literature that the Infinite Conversation Architecture addresses, and frames each gap as a research contribution. The gaps are sourced from a systematic review of existing open-source implementations, academic papers, and production systems in the LLM memory space (see `research/FINDINGS.md`).

Three gaps are identified. Each represents a genuine area where existing work is either absent or incomplete.

---

## Gap 1 — Typing-Time Retrieval

### What exists

Current retrieval-augmented generation systems retrieve after the user submits a message. TeleRAG (arXiv:2502.20969) introduced lookahead prefetching during model generation — anticipating the next retrieval step while the model responds. StreamingRAG combines sliding eviction with dynamic recall during generation. These are advances, but they operate during the model's output, not during the user's input.

Aeon's Semantic Lookaside Buffer (SLB) is the closest existing work to typing-time retrieval. Its Speculative Fetch Algorithm predicts what the next turn will need based on the current turn's topic cluster — achieving sub-5μs retrieval. However, this operates at turn-level prediction, not character-level streaming. It predicts based on where the conversation was, not based on what the user is actively typing.

### What is missing

True streaming retrieval during user input — beginning retrieval at the first keypress, refining as more text appears, completing before the message is sent — has not been addressed in peer-reviewed literature. No large-scale benchmarks exist. Practitioner implementations reference the concept but no formal study measures the latency reduction or accuracy impact.

### ICA's contribution

The Pre-Fetch Engine starts retrieval the moment the user begins typing. It reads partial text, extracts provisional entities via a lightweight NER pipeline (3–10ms), and begins graph traversal in the background. A drift check on submission verifies whether the provisional result is still valid. If more than 50% of entities are new, a targeted correction pass runs. From the user's perspective, retrieval latency is eliminated entirely.

### What needs to be demonstrated

A controlled benchmark comparing three conditions: (1) retrieval after message send, (2) Aeon-style turn-level prediction, (3) ICA-style typing-time retrieval. Measuring end-to-end latency and retrieval accuracy across each condition would constitute a direct contribution to an area with no published benchmarks.

---

## Gap 2 — Tiered Conversation Graph with Connection-Activity-Based Promotion

### What exists

All reviewed graph-based memory systems store conversation history in a single graph structure. Graphiti uses bi-temporal validity windows — facts are invalidated when superseded, but nodes are not tiered by access frequency. Mem0g stores entity-relation triplets in Neo4j with no tiering. MemGPT implements OS-style paging (RAM/disk/cold) but at the conversation level, not the node level. No system implements node-level tiering within a conversation graph based on connection activity.

The challenge at scale is well-documented: graph traversal at 100K+ nodes is computationally expensive. Published benchmarks (FalkorDB, Kuzu, Memgraph) cover large static graphs, but no benchmark covers the specific workload of a conversation graph — one write per message turn, incremental edge detection, depth-limited traversal from recent nodes — at 10K–100K nodes.

### What is missing

A tiering strategy specifically designed for conversation graphs, where node importance is determined not by temporal proximity alone but by connection activity — how many other nodes reference this node, and how recently. High-connection nodes represent conversational anchors (pivotal decisions, important entities, recurring topics) and should remain accessible regardless of age. Low-connection old nodes can be compressed and archived without loss of system coherence.

### ICA's contribution

Three-tier graph architecture with connection-activity-based promotion:

- **Active tier** — Last ~500 turns. Full text. Fast traversal. Held in memory.
- **Warm tier** — Turns 500–5,000. Full text stripped, metadata preserved. Compressed.
- **Cold tier** — Everything older. Archived. On-demand retrieval only.

Promotion exception: any node with 3 or more connections stays active regardless of turn distance. This keeps conversational anchors accessible. Demotion is automatic — a node untouched for 1,000 turns moves to warm; untouched for 5,000 moves to cold.

### What needs to be demonstrated

A benchmark running ICA's tiered graph against an untiered baseline at 1K, 10K, and 100K nodes. Measuring traversal latency, memory footprint, and coherence quality would quantify the tiering benefit. The specific workload (one node write + edge detection + depth-4 traversal from hot window) does not exist in published benchmarks — producing it would be a contribution in itself.

---

## Gap 3 — Self-Correcting Persistent State Document

### What exists

Persistent user identity across LLM sessions is addressed by several systems. Zep maintains a temporal knowledge graph where facts are tracked with validity windows — when a new fact contradicts an old one, the old edge is expired. This is accurate but depends on entity-level contradiction detection. Mem0 uses LLM-orchestrated ADD/UPDATE/DELETE/NOOP operations applied to a structured memory store. MemGPT allows the agent to explicitly edit its own memory blocks.

All three approaches share a common assumption: the identity state is managed by reacting to new information as it arrives. None implement a periodic backward-looking review — a process that reads recent conversation and asks whether what the system currently believes is still accurate.

### What is missing

A self-correcting review mechanism that treats state management as an ongoing audit rather than a reactive update. Current systems update state when new information arrives. They do not proactively check whether existing state has been silently invalidated by accumulated context — subtle shifts in user preferences, unstated reversals of earlier decisions, or facts that are technically still present but contextually outdated.

### ICA's contribution

A versioned State Document with a periodic model-driven review loop:

Every 20 turns, a background process reads the current State Document and the last 20 turns of conversation. It sends a structured prompt to a lightweight model asking specifically: what is now outdated, what should be added, what should be updated. The response is a structured diff — REMOVE / ADD / UPDATE / NO_CHANGES — which is applied to produce a new document version. Previous versions are retained for rollback.

The key distinction from existing approaches: the review is backward-looking and proactive. It does not wait for new information to trigger an update. It actively asks whether existing state is still valid.

### What needs to be demonstrated

A long-running conversation test (500+ turns) comparing state document accuracy with and without the review loop. Measuring: number of outdated facts retained, number of correct facts correctly preserved, number of new facts correctly captured. Against a Mem0 baseline using reactive ADD/UPDATE/DELETE only.

---

## The Unified Contribution

Each of the three gaps above exists in isolation across current literature. The fourth and arguably most significant contribution of ICA is not any single component — it is the integration of all three into a unified system where:

- Typing-time pre-fetch eliminates retrieval latency
- The tiered graph keeps the working memory lean while preserving infinite history
- The self-correcting state document maintains accurate persistent identity

No published system addresses all three simultaneously. The integration produces emergent properties that none of the components produce alone: a conversation system that genuinely feels continuous, fast, and identity-aware — regardless of how long the conversation has been running.

---

## Recommended Research Workstreams

**Workstream 1 — Pre-fetch benchmark**  
Instrument the Pre-Fetch Engine. Run 100 conversations of 50 turns each with three retrieval conditions. Measure and publish latency and accuracy results. Target venue: arXiv preprint, then systems conference.

**Workstream 2 — Graph database benchmark**  
Run the conversational workload (write + detect + traverse depth-4) at 1K, 10K, 50K, 100K nodes on FalkorDB, Kuzu, and Neo4j. Publish results. This fills a documented gap in existing database benchmarks.

**Workstream 3 — State document accuracy study**  
Run 10 conversations to 500 turns each. At turn 500, have human raters evaluate State Document accuracy. Compare ICA's review-loop approach against Mem0 reactive baseline.

**Workstream 4 — LoCoMo evaluation**  
Integrate ICA with a base model. Run the LoCoMo benchmark (600-turn conversations). Compare J-score against Mem0g (68.4%), Zep (66.0%), and OpenAI Memory (52.9%). Publish results.
