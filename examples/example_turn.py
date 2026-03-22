"""
Infinite Conversation Architecture
Example — Full Turn Flow

Author: Eugene Mawuli Attigah
Project: infinite-conversation-architecture
License: CC BY 4.0

This example walks through a single complete turn to demonstrate
how all five components work together.
"""

from datetime import datetime
from schemas import (
    ConversationNode, ConversationEdge, StateDocument,
    UserIdentity, ConversationState, KeyFacts, RelationshipHistory,
    Speaker, NodeTier, EmotionalRegister
)
from memory_manager import process_turn
from context_assembler import assemble_context, PreFetchEngine
from scoring import traverse_and_rank


# ─────────────────────────────────────────────
# SIMULATE AN EXISTING CONVERSATION
# ─────────────────────────────────────────────

# Pretend we are at turn 42 of an ongoing conversation
# about building an AI tool called Iron-Thread

node_1 = ConversationNode(
    node_id="node_001",
    conversation_id="conv_abc",
    turn_number=1,
    full_text="I am building an output validation tool for AI agents called Iron-Thread.",
    compressed_text="Building Iron-Thread, an AI output validation tool.",
    speaker=Speaker.USER,
    topic_tags=["iron-thread", "validation", "ai-agents"],
    entity_tags=["Iron-Thread", "AI agents"],
    open_questions=[],
    tier=NodeTier.ACTIVE,
    connection_count=4,
)

node_8 = ConversationNode(
    node_id="node_008",
    conversation_id="conv_abc",
    turn_number=8,
    full_text="What database should I use for storing validation logs?",
    compressed_text="Question: which database for validation logs?",
    speaker=Speaker.USER,
    topic_tags=["database", "validation", "iron-thread"],
    entity_tags=["Iron-Thread", "database"],
    open_questions=["What database should I use for storing validation logs?"],
    tier=NodeTier.ACTIVE,
    connection_count=2,
)

node_9 = ConversationNode(
    node_id="node_009",
    conversation_id="conv_abc",
    turn_number=9,
    full_text="For validation logs I recommend PostgreSQL via Supabase. It gives you a hosted database on the free tier with good query performance.",
    compressed_text="Recommendation: PostgreSQL via Supabase for validation logs.",
    speaker=Speaker.ASSISTANT,
    topic_tags=["database", "supabase", "postgresql"],
    entity_tags=["PostgreSQL", "Supabase", "Iron-Thread"],
    open_questions=[],
    tier=NodeTier.ACTIVE,
    connection_count=3,
)

node_40 = ConversationNode(
    node_id="node_040",
    conversation_id="conv_abc",
    turn_number=40,
    full_text="The Supabase setup is done. Iron-Thread backend is deployed on Railway.",
    compressed_text="Supabase done. Iron-Thread deployed on Railway.",
    speaker=Speaker.USER,
    topic_tags=["supabase", "railway", "deployment", "iron-thread"],
    entity_tags=["Supabase", "Railway", "Iron-Thread"],
    open_questions=[],
    tier=NodeTier.ACTIVE,
    connection_count=5,
)

node_41 = ConversationNode(
    node_id="node_041",
    conversation_id="conv_abc",
    turn_number=41,
    full_text="Great. The next step is wiring in the Anthropic API key for the auto-correction loop and publishing SDK v0.2.0.",
    compressed_text="Next: wire Anthropic API key, publish SDK v0.2.0.",
    speaker=Speaker.ASSISTANT,
    topic_tags=["anthropic", "sdk", "api", "iron-thread"],
    entity_tags=["Anthropic", "Iron-Thread", "SDK"],
    open_questions=["Have you set up the Anthropic API key yet?"],
    tier=NodeTier.ACTIVE,
    connection_count=3,
)

# Graph store
all_nodes = {
    "node_001": node_1,
    "node_008": node_8,
    "node_009": node_9,
    "node_040": node_40,
    "node_041": node_41,
}

# Sample edges
all_edges = [
    ConversationEdge(source_node_id="node_001", target_node_id="node_008", edge_type="continues", weight=1.0),
    ConversationEdge(source_node_id="node_008", target_node_id="node_009", edge_type="resolves", weight=1.0),
    ConversationEdge(source_node_id="node_001", target_node_id="node_040", edge_type="shares_entity", weight=0.8, shared_value="Iron-Thread"),
    ConversationEdge(source_node_id="node_009", target_node_id="node_040", edge_type="shares_entity", weight=0.7, shared_value="Supabase"),
    ConversationEdge(source_node_id="node_040", target_node_id="node_041", edge_type="continues", weight=1.0),
]

# State document
state_doc = StateDocument(
    conversation_id="conv_abc",
    version=3,
    last_reviewed_at_turn=20,
    user_identity=UserIdentity(
        name="Eugene",
        role="Independent builder",
        stated_goals=["Build Thread Suite", "Get GitHub traction", "Target acquisition"],
        ongoing_projects=["Iron-Thread", "TestThread"],
        known_preferences=["step-by-step instructions", "direct answers", "no fluff"],
        technical_level="non-technical builder",
    ),
    conversation_state=ConversationState(
        active_threads=["Iron-Thread deployment", "SDK publishing"],
        decisions_made=["Use Railway for backend", "Use Supabase for database", "Python + JS SDKs"],
        open_questions=["Anthropic API key wired in?", "SDK v0.2.0 published?"],
        agreements=["Claude leads architecture decisions", "Eugene executes step by step"],
    ),
    key_facts=KeyFacts(
        user_established_facts=[
            "Iron-Thread is live on Railway",
            "Supabase is set up",
            "GitHub repo: eugene001dayne/iron-thread",
        ],
        corrections_made=[],
    ),
    relationship_history=RelationshipHistory(
        total_turns=41,
        established_tone="technical co-founder dynamic",
    ),
)


# ─────────────────────────────────────────────
# NEW MESSAGE ARRIVES AT TURN 42
# ─────────────────────────────────────────────

new_message = "Okay the Anthropic API key is wired in. What do I do to publish the SDK?"
current_turn = 42
message_entities = ["Anthropic", "Iron-Thread", "SDK"]
message_topics = ["anthropic", "sdk", "publishing", "iron-thread"]

print("=" * 60)
print("INFINITE CONVERSATION ARCHITECTURE — TURN FLOW DEMO")
print("=" * 60)
print(f"\nTurn: {current_turn}")
print(f"New message: \"{new_message}\"")
print(f"Extracted entities: {message_entities}")


# ─────────────────────────────────────────────
# STEP 1 — MEMORY MANAGER PROCESSES THE TURN
# ─────────────────────────────────────────────

print("\n--- MEMORY MANAGER ---")
mm_result = process_turn(
    message_text=new_message,
    speaker=Speaker.USER,
    turn_number=current_turn,
    conversation_id="conv_abc",
    existing_nodes=list(all_nodes.values()),
    state_document=state_doc,
    extracted_entities=message_entities,
    extracted_topics=message_topics,
    extracted_open_questions=[],
    emotional_register=EmotionalRegister.NEUTRAL,
)

print(f"New node created: {mm_result.new_node.node_id}")
print(f"New edges detected: {len(mm_result.new_edges)}")
for edge in mm_result.new_edges:
    print(f"  {edge.source_node_id} --[{edge.edge_type}]--> {edge.target_node_id} (weight: {edge.weight})")
print(f"Tier updates needed: {len(mm_result.tier_updates)}")
print(f"State doc review triggered: {mm_result.review_triggered}")


# ─────────────────────────────────────────────
# STEP 2 — CONTEXT ASSEMBLER BUILDS THE WINDOW
# ─────────────────────────────────────────────

print("\n--- CONTEXT ASSEMBLER ---")

# Add the new node to the graph before assembly
all_nodes[mm_result.new_node.node_id] = mm_result.new_node
all_edges.extend(mm_result.new_edges)

assembled = assemble_context(
    new_message=new_message,
    all_nodes=all_nodes,
    all_edges=all_edges,
    state_document=state_doc,
    current_turn=current_turn,
    current_message_entities=message_entities,
    prefetch_result=None,
)

print(f"Hot window size: {len(assembled.hot_window)} turns")
print(f"Retrieved nodes: {len(assembled.retrieved_nodes)}")
for node in assembled.retrieved_nodes:
    print(f"  Turn {node.turn_number}: \"{node.compressed_text}\"")
print(f"Retrieval latency: {assembled.retrieval_latency_ms}ms")
print(f"Pre-fetch used: {assembled.prefetch_used}")


# ─────────────────────────────────────────────
# STEP 3 — SHOW FINAL CONTEXT WINDOW
# ─────────────────────────────────────────────

print("\n--- FINAL CONTEXT WINDOW (passed to model) ---")
messages = assembled.to_messages()
print(f"Total messages in context: {len(messages)}")
print(f"\nContext structure:")
print(f"  [0] System — State document ({len(assembled.state_document_text)} chars)")
if assembled.retrieved_nodes:
    print(f"  [1] System — Retrieved history ({len(assembled.retrieved_nodes)} nodes)")
print(f"  [...] Hot window ({len(assembled.hot_window)} turns)")
print(f"  [last] New user message")

print("\n--- STATE DOCUMENT PREVIEW ---")
print(assembled.state_document_text)

print("\n" + "=" * 60)
print("Turn complete. Model receives context. Responds. Memory manager")
print("writes response to graph. Cycle continues — infinitely.")
print("=" * 60)
