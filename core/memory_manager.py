"""
Infinite Conversation Architecture
Memory Manager — The Dedicated Parallel Process

Author: Eugene Mawuli Attigah
Project: infinite-conversation-architecture
License: CC BY 4.0

The Memory Manager is deliberately separate from the main model.
Its only job is memory operations. It never responds to users.
It never reasons about conversation content.
It manages: writes, edge detection, tier management,
state document reviews, and pre-fetch preparation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from schemas import (
    ConversationNode, ConversationEdge, StateDocument,
    EdgeType, NodeTier, Speaker, EmotionalRegister
)
from core.ner_engine import extract_metadata


# ─────────────────────────────────────────────
# TIER MANAGEMENT CONSTANTS
# ─────────────────────────────────────────────

ACTIVE_TIER_LIMIT = 500       # Turns before a node is eligible for warm
WARM_TIER_LIMIT = 5000        # Turns before a node is eligible for cold
STATE_REVIEW_INTERVAL = 20    # How often state doc review triggers (turns)
MIN_CONNECTIONS_TO_STAY_ACTIVE = 3  # Highly connected nodes stay active regardless of age


# ─────────────────────────────────────────────
# EDGE DETECTION
# ─────────────────────────────────────────────

def detect_edges(
    new_node: ConversationNode,
    existing_nodes: list[ConversationNode],
    hot_window_size: int = 25,
) -> list[ConversationEdge]:
    """
    Detects edges between the new node and existing nodes.

    Rules:
    - Always creates a CONTINUES edge to the immediately preceding node
    - Scans recent nodes for entity/topic overlap → SHARES_ENTITY or SHARES_TOPIC
    - Scans for open questions that this message resolves → RESOLVES
    - Scans for contradictions → CONTRADICTS (requires semantic check — placeholder here)
    - Scans for explicit references → REFERENCES (requires semantic check — placeholder here)

    In production, CONTRADICTS and REFERENCES detection
    should use a lightweight embedding model or classifier.
    The others are rule-based and fast.
    """
    edges = []

    if not existing_nodes:
        return edges

    # Sort existing nodes by turn number
    sorted_nodes = sorted(existing_nodes, key=lambda n: n.turn_number)
    preceding_node = sorted_nodes[-1] if sorted_nodes else None

    # 1. CONTINUES — always connect to the previous node
    if preceding_node:
        edges.append(ConversationEdge(
            source_node_id=preceding_node.node_id,
            target_node_id=new_node.node_id,
            edge_type=EdgeType.CONTINUES,
            weight=1.0,
        ))

    # 2. SHARES_ENTITY — scan recent nodes for entity overlap
    new_entities = set(e.lower() for e in new_node.entity_tags)
    scan_window = sorted_nodes[-(hot_window_size * 3):]  # Look back 3x the hot window

    for node in scan_window:
        if node.node_id == preceding_node.node_id:
            continue  # Already connected via CONTINUES

        node_entities = set(e.lower() for e in node.entity_tags)
        shared_entities = new_entities & node_entities

        if shared_entities:
            weight = min(1.0, len(shared_entities) / max(len(new_entities), 1))
            edges.append(ConversationEdge(
                source_node_id=node.node_id,
                target_node_id=new_node.node_id,
                edge_type=EdgeType.SHARES_ENTITY,
                weight=weight,
                shared_value=", ".join(shared_entities),
            ))

    # 3. SHARES_TOPIC — scan for topic overlap
    new_topics = set(t.lower() for t in new_node.topic_tags)

    for node in scan_window:
        node_topics = set(t.lower() for t in node.topic_tags)
        shared_topics = new_topics & node_topics

        if shared_topics:
            # Check we haven't already added a SHARES_ENTITY edge for this node
            already_connected = any(
                e.source_node_id == node.node_id and
                e.edge_type == EdgeType.SHARES_ENTITY
                for e in edges
            )
            if not already_connected:
                weight = min(1.0, len(shared_topics) / max(len(new_topics), 1))
                edges.append(ConversationEdge(
                    source_node_id=node.node_id,
                    target_node_id=new_node.node_id,
                    edge_type=EdgeType.SHARES_TOPIC,
                    weight=weight,
                    shared_value=", ".join(shared_topics),
                ))

    # 4. RESOLVES — check if new node answers any open questions from recent nodes
    for node in scan_window:
        if node.open_questions:
            # Placeholder: in production, use embedding similarity between
            # node.open_questions and new_node.full_text
            # For now, check if any question keywords appear in the new message
            for question in node.open_questions:
                question_keywords = set(question.lower().split()) - {"what", "how", "why", "when", "where", "is", "are", "the", "a"}
                new_text_words = set(new_node.full_text.lower().split())
                if len(question_keywords & new_text_words) >= 2:
                    edges.append(ConversationEdge(
                        source_node_id=node.node_id,
                        target_node_id=new_node.node_id,
                        edge_type=EdgeType.RESOLVES,
                        weight=1.0,
                    ))
                    break

    return edges


# ─────────────────────────────────────────────
# TIER MANAGEMENT
# ─────────────────────────────────────────────

def evaluate_tier(
    node: ConversationNode,
    current_turn: int,
) -> NodeTier:
    """
    Determines what tier a node should be in based on:
    - Turn distance from current position
    - Number of connections (highly connected nodes stay active longer)
    - Last referenced time
    """

    turn_distance = current_turn - node.turn_number

    # Highly connected nodes stay active regardless of age
    if node.connection_count >= MIN_CONNECTIONS_TO_STAY_ACTIVE:
        if turn_distance < WARM_TIER_LIMIT:
            return NodeTier.ACTIVE

    # Standard tier assignment by age
    if turn_distance < ACTIVE_TIER_LIMIT:
        return NodeTier.ACTIVE
    elif turn_distance < WARM_TIER_LIMIT:
        return NodeTier.WARM
    else:
        return NodeTier.COLD


def compress_node(node: ConversationNode) -> ConversationNode:
    """
    Compresses a node moving from ACTIVE to WARM tier.
    Strips full_text, keeps compressed_text and all metadata.
    Original full_text should be archived separately before calling this.
    """
    node.full_text = ""  # Full text archived, not held in active memory
    node.tier = NodeTier.WARM
    return node


# ─────────────────────────────────────────────
# STATE DOCUMENT REVIEW
# ─────────────────────────────────────────────

def should_trigger_review(turn_number: int, last_reviewed_at_turn: int) -> bool:
    """Returns True if the state document review cycle is due."""
    return (turn_number - last_reviewed_at_turn) >= STATE_REVIEW_INTERVAL


def build_review_prompt(
    state_document: StateDocument,
    recent_turns: list[ConversationNode],
) -> str:
    """
    Builds the prompt sent to the model for state document review.
    The model reads the current state and recent turns,
    then returns a diff of what should change.
    """
    recent_text = "\n".join([
        f"[Turn {n.turn_number} — {'User' if n.speaker == Speaker.USER else 'Assistant'}]: {n.full_text}"
        for n in recent_turns
    ])

    return f"""You are reviewing a conversation state document for accuracy.

CURRENT STATE DOCUMENT:
{state_document.to_context_string()}

RECENT CONVERSATION (last {len(recent_turns)} turns):
{recent_text}

Review the state document against the recent conversation.
Identify:
1. Anything that is now outdated or incorrect
2. New information that should be added
3. Decisions or facts that have changed
4. New open questions that have emerged
5. Open questions that have been resolved

Respond in this exact format:
REMOVE: [items to remove, one per line]
ADD: [items to add, one per line]
UPDATE: [items to update, format "old value → new value", one per line]
NO_CHANGES: [write this if nothing needs to change]
"""


# ─────────────────────────────────────────────
# MEMORY MANAGER — MAIN INTERFACE
# ─────────────────────────────────────────────

@dataclass
class MemoryManagerResult:
    """Result returned by the memory manager on each turn."""
    new_node: ConversationNode
    new_edges: list[ConversationEdge]
    tier_updates: list[tuple[str, NodeTier]]  # (node_id, new_tier)
    review_triggered: bool
    review_prompt: Optional[str] = None


def process_turn(
    message_text: str,
    speaker: Speaker,
    turn_number: int,
    conversation_id: str,
    existing_nodes: list[ConversationNode],
    state_document: StateDocument,
    extracted_open_questions: list[str],
) -> MemoryManagerResult:
    """
    Main memory manager function. Called on every turn.

    1. Extracts metadata automatically from message text
    2. Creates a new node for the message
    3. Detects and creates edges
    4. Evaluates tier status of existing nodes
    5. Checks if state document review is due
    6. Returns everything needed to update the graph
    """

    # 1. Extract metadata automatically
    metadata = extract_metadata(message_text)

    # 2. Create new node
    new_node = ConversationNode(
        conversation_id=conversation_id,
        turn_number=turn_number,
        full_text=message_text,
        compressed_text=message_text[:200] + "..." if len(message_text) > 200 else message_text,
        speaker=speaker,
        topic_tags=metadata["topics"],
        entity_tags=metadata["entities"],
        open_questions=extracted_open_questions,
        emotional_register=EmotionalRegister(metadata["emotional_register"]),
    )

    # 2. Detect edges
    new_edges = detect_edges(new_node, existing_nodes)

    # Update connection counts
    connected_node_ids = set()
    for edge in new_edges:
        connected_node_ids.add(edge.source_node_id)
        connected_node_ids.add(edge.target_node_id)

    # 3. Evaluate tier updates
    tier_updates = []
    for node in existing_nodes:
        correct_tier = evaluate_tier(node, turn_number)
        if correct_tier != node.tier:
            tier_updates.append((node.node_id, correct_tier))

    # 4. Check state document review
    review_needed = should_trigger_review(turn_number, state_document.last_reviewed_at_turn)
    review_prompt = None

    if review_needed:
        recent_turns = sorted(existing_nodes, key=lambda n: n.turn_number)[-STATE_REVIEW_INTERVAL:]
        review_prompt = build_review_prompt(state_document, recent_turns)

    return MemoryManagerResult(
        new_node=new_node,
        new_edges=new_edges,
        tier_updates=tier_updates,
        review_triggered=review_needed,
        review_prompt=review_prompt,
    )
