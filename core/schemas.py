"""
Infinite Conversation Architecture
Core Schemas — Nodes, Edges, State Document

Author: Eugene Mawuli Attigah
Project: infinite-conversation-architecture
License: CC BY 4.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
import uuid


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class Speaker(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class EmotionalRegister(str, Enum):
    NEUTRAL = "neutral"
    CURIOUS = "curious"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"
    UNCERTAIN = "uncertain"
    ASSERTIVE = "assertive"


class EdgeType(str, Enum):
    CONTINUES = "continues"           # Direct sequential flow
    REFERENCES = "references"         # Explicit callback to earlier message
    CONTRADICTS = "contradicts"       # Conflicts with earlier message
    RESOLVES = "resolves"             # Answers an earlier open question
    SHARES_ENTITY = "shares_entity"   # Same person/project/thing mentioned
    SHARES_TOPIC = "shares_topic"     # Same topic, different wording


class NodeTier(str, Enum):
    ACTIVE = "active"   # Last ~500 turns — full detail, fast traversal
    WARM = "warm"       # Turns 500–5000 — compressed, medium speed
    COLD = "cold"       # Everything older — archived, on-demand only


# ─────────────────────────────────────────────
# GRAPH NODE
# ─────────────────────────────────────────────

@dataclass
class ConversationNode:
    """
    A single message turn stored in the conversation graph.
    Every message — user or assistant — becomes a node.
    """

    # Identity
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    turn_number: int = 0

    # Content
    full_text: str = ""
    compressed_text: str = ""          # Short summary for fast scanning
    speaker: Speaker = Speaker.USER
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Extracted metadata
    topic_tags: list[str] = field(default_factory=list)
    entity_tags: list[str] = field(default_factory=list)   # People, projects, decisions
    open_questions: list[str] = field(default_factory=list) # Unresolved questions in this turn
    emotional_register: EmotionalRegister = EmotionalRegister.NEUTRAL

    # Graph management
    tier: NodeTier = NodeTier.ACTIVE
    connection_count: int = 0          # How many edges this node has
    last_referenced_at: Optional[datetime] = None  # When this node was last retrieved

    # Scoring cache (updated by memory manager)
    base_relevance_score: float = 0.0


# ─────────────────────────────────────────────
# GRAPH EDGE
# ─────────────────────────────────────────────

@dataclass
class ConversationEdge:
    """
    A typed relationship between two nodes in the conversation graph.
    Edges are directional — from source to target.
    """

    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = ""
    target_node_id: str = ""
    edge_type: EdgeType = EdgeType.CONTINUES

    # Edge weight — how strong is this connection?
    # 1.0 = maximum, 0.0 = negligible
    weight: float = 1.0

    # What shared entity or topic triggered this edge (if applicable)
    shared_value: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# STATE DOCUMENT
# ─────────────────────────────────────────────

@dataclass
class UserIdentity:
    name: Optional[str] = None
    role: Optional[str] = None
    context: Optional[str] = None                  # What kind of user, what they do
    stated_goals: list[str] = field(default_factory=list)
    ongoing_projects: list[str] = field(default_factory=list)
    known_preferences: list[str] = field(default_factory=list)
    explicit_dislikes: list[str] = field(default_factory=list)
    technical_level: Optional[str] = None          # beginner / intermediate / expert


@dataclass
class ConversationState:
    active_threads: list[str] = field(default_factory=list)     # Open, unresolved topics
    decisions_made: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)     # Questions not yet answered
    agreements: list[str] = field(default_factory=list)


@dataclass
class KeyFacts:
    user_established_facts: list[str] = field(default_factory=list)
    corrections_made: list[str] = field(default_factory=list)   # Things user corrected


@dataclass
class RelationshipHistory:
    first_conversation_at: Optional[datetime] = None
    total_turns: int = 0
    established_tone: Optional[str] = None
    notable_moments: list[str] = field(default_factory=list)


@dataclass
class StateDocument:
    """
    The living document injected at the top of every context window.
    Always present. Always current. Gives the model persistent identity.
    """

    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    version: int = 1
    last_updated_at: datetime = field(default_factory=datetime.utcnow)
    last_reviewed_at_turn: int = 0     # Which turn triggered last review

    # The four sections
    user_identity: UserIdentity = field(default_factory=UserIdentity)
    conversation_state: ConversationState = field(default_factory=ConversationState)
    key_facts: KeyFacts = field(default_factory=KeyFacts)
    relationship_history: RelationshipHistory = field(default_factory=RelationshipHistory)

    def to_context_string(self) -> str:
        """
        Renders the state document as a string for injection
        at the top of the model's context window.
        """
        lines = ["=== CONVERSATION STATE ==="]

        u = self.user_identity
        if u.name:
            lines.append(f"User: {u.name}" + (f" | {u.role}" if u.role else ""))
        if u.stated_goals:
            lines.append(f"Goals: {'; '.join(u.stated_goals)}")
        if u.ongoing_projects:
            lines.append(f"Projects: {'; '.join(u.ongoing_projects)}")
        if u.known_preferences:
            lines.append(f"Preferences: {'; '.join(u.known_preferences)}")

        s = self.conversation_state
        if s.active_threads:
            lines.append(f"Active threads: {'; '.join(s.active_threads)}")
        if s.open_questions:
            lines.append(f"Open questions: {'; '.join(s.open_questions)}")
        if s.decisions_made:
            lines.append(f"Decisions made: {'; '.join(s.decisions_made)}")

        f = self.key_facts
        if f.user_established_facts:
            lines.append(f"Established facts: {'; '.join(f.user_established_facts)}")

        lines.append("=========================")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# ASSEMBLED CONTEXT WINDOW
# ─────────────────────────────────────────────

@dataclass
class AssembledContext:
    """
    The final context window passed to the model on each turn.
    Order is fixed: state doc → retrieved nodes → hot window → new message.
    """

    state_document_text: str = ""
    retrieved_nodes: list[ConversationNode] = field(default_factory=list)
    hot_window: list[ConversationNode] = field(default_factory=list)
    new_message: str = ""

    # Metadata
    total_tokens_estimate: int = 0
    retrieval_latency_ms: float = 0.0
    prefetch_used: bool = False

    def to_messages(self) -> list[dict]:
        """
        Formats assembled context into the messages array
        expected by any standard LLM API.
        """
        messages = []

        # State document as system context
        if self.state_document_text:
            messages.append({
                "role": "system",
                "content": self.state_document_text
            })

        # Retrieved historical nodes
        if self.retrieved_nodes:
            retrieved_text = "\n\n=== RELEVANT HISTORY ===\n"
            for node in self.retrieved_nodes:
                speaker_label = "User" if node.speaker == Speaker.USER else "Assistant"
                retrieved_text += f"[Turn {node.turn_number} — {speaker_label}]: {node.compressed_text}\n"
            retrieved_text += "========================\n"
            messages.append({
                "role": "system",
                "content": retrieved_text
            })

        # Hot window — recent turns
        for node in self.hot_window:
            messages.append({
                "role": node.speaker.value,
                "content": node.full_text
            })

        # New message
        messages.append({
            "role": "user",
            "content": self.new_message
        })

        return messages
