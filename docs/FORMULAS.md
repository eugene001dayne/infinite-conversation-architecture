# Scoring Formulas — Infinite Conversation Architecture

**Author:** Eugene Mawuli Attigah  
**Project:** infinite-conversation-architecture  
**License:** CC BY 4.0

---

## The Master Scoring Formula

Given a candidate node **N** and the current message **M**, the relevance score is:

```
S(N, M) = w1 · R(N) + w2 · C(N, M) + w3 · E(N, M) + w4 · Q(N)
```

| Symbol | Component | Default Weight |
|--------|-----------|---------------|
| R(N) | Recency | w1 = 0.25 |
| C(N, M) | Connection | w2 = 0.40 |
| E(N, M) | Entity Overlap | w3 = 0.20 |
| Q(N) | Open Question | w4 = 0.15 |

All components are normalised to **[0.0, 1.0]**.  
Final score **S(N, M) ∈ [0.0, 1.0]**.  
Connection carries the highest weight — in conversation, *what is connected matters more than what is recent.*

---

## Component 1 — Recency: R(N)

```
R(N) = e^(-λ · Δt)

Where:
  Δt = current_turn - node.turn_number   (turn distance)
  λ  = 0.005                             (decay rate, tunable)
```

Exponential decay. A node scores 1.0 at the current turn and decays toward 0 as distance grows.

| Turn Distance (Δt) | R(N) |
|-------------------|------|
| 0 | 1.00 |
| 20 | 0.90 |
| 100 | 0.61 |
| 200 | 0.37 |
| 500 | 0.08 |
| 1000 | 0.007 |

Tune λ to control how fast older nodes decay. Higher λ = faster decay = system prefers recent content more strongly.

---

## Component 2 — Connection: C(N, M)

```
C(N, M) = Σ (edge.weight × edge_type_weight) / max_possible_weight

Summed over all edges connecting N to any node in the current hot window.
```

**Edge type weights (by strength):**

| Edge Type | Weight |
|-----------|--------|
| RESOLVES | 1.0 |
| REFERENCES | 0.9 |
| CONTRADICTS | 0.8 |
| SHARES_ENTITY | 0.6 |
| SHARES_TOPIC | 0.4 |
| CONTINUES | 0.2 |

RESOLVES edges carry the highest weight because a node that directly answers an open question is the most valuable retrieval possible.

---

## Component 3 — Entity Overlap: E(N, M)

```
E(N, M) = |entities(N) ∩ entities(M)| / |entities(M)|
```

Proportion of current message entities that appear in the candidate node.

- Returns **1.0** if every entity in the current message also appears in node N
- Returns **0.0** if no entity overlap exists
- Partial overlap returns a proportional value

---

## Component 4 — Open Question: Q(N)

```
Q(N) = 1.0  if node contains open_questions
            AND no RESOLVES edge exists pointing away from N

Q(N) = 0.0  otherwise
```

Binary. A node either has an unresolved question worth surfacing or it does not. Nodes with unresolved questions are prioritised for retrieval so the model can close them.

---

## Tier Promotion Formula

A node's tier is determined by:

```
if connection_count >= MIN_CONNECTIONS_TO_STAY_ACTIVE (3):
    if Δt < WARM_TIER_LIMIT (5000): tier = ACTIVE
else:
    if Δt < ACTIVE_TIER_LIMIT (500):  tier = ACTIVE
    if Δt < WARM_TIER_LIMIT (5000):   tier = WARM
    else:                              tier = COLD
```

Highly connected nodes stay active longer regardless of age.  
This prevents important early-conversation nodes from being archived prematurely.

---

## Minimum Score Threshold

Nodes with `S(N, M) < 0.15` are excluded from injection regardless of retrieval.  
This prevents low-signal noise from polluting the context window.

The threshold of **0.15** means a node must score at least 15% of the maximum possible relevance to be considered worth injecting.
