"""
Infinite Conversation Architecture
Context Assembler — Builds the Window on Every Turn

Author: Eugene Mawuli Attigah
Project: infinite-conversation-architecture
License: CC BY 4.0

The Context Assembler is the piece that brings everything together.
On every turn it takes:
  - The state document
  - Retrieved graph nodes (from scoring.py)
  - The hot window
  - The new message
And assembles them into the final context window passed to the model.
"""

from schemas import (
    ConversationNode, StateDocument, AssembledContext, Speaker
)
from scoring import traverse_and_rank, NodeScore
from schemas import ConversationEdge


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

HOT_WINDOW_SIZE = 25       # Number of recent turns always included
MAX_RETRIEVED_NODES = 10   # Max nodes injected from graph retrieval
MAX_DEPTH = 4              # Graph traversal depth
MIN_SCORE = 0.15           # Minimum score for a node to be injected


# ─────────────────────────────────────────────
# CONTEXT ASSEMBLER
# ─────────────────────────────────────────────

def assemble_context(
    new_message: str,
    all_nodes: dict[str, ConversationNode],
    all_edges: list[ConversationEdge],
    state_document: StateDocument,
    current_turn: int,
    current_message_entities: list[str],
    prefetch_result: list[NodeScore] | None = None,
) -> AssembledContext:
    """
    Assembles the full context window for a single turn.

    Order (fixed, always):
    1. State document text    — top, always present
    2. Retrieved graph nodes  — relevant historical context
    3. Hot window             — last HOT_WINDOW_SIZE turns
    4. New message            — bottom

    If prefetch_result is provided (pre-fetch engine ran during typing),
    it is used directly and skips the traversal step.
    """

    import time
    start = time.time()

    # Step 1 — Hot window (last N turns)
    sorted_nodes = sorted(all_nodes.values(), key=lambda n: n.turn_number)
    hot_window = sorted_nodes[-HOT_WINDOW_SIZE:] if len(sorted_nodes) >= HOT_WINDOW_SIZE else sorted_nodes

    # Step 2 — Retrieved nodes (from pre-fetch or live traversal)
    prefetch_used = False

    if prefetch_result is not None:
        retrieved_scores = prefetch_result
        prefetch_used = True
    else:
        retrieved_scores = traverse_and_rank(
            graph_nodes=all_nodes,
            graph_edges=all_edges,
            hot_window=hot_window,
            current_turn=current_turn,
            current_message_entities=current_message_entities,
            max_depth=MAX_DEPTH,
            top_k=MAX_RETRIEVED_NODES,
            min_score_threshold=MIN_SCORE,
        )

    # Extract node objects from scores, exclude hot window nodes
    hot_window_ids = {n.node_id for n in hot_window}
    retrieved_nodes = [
        ns.node for ns in retrieved_scores
        if ns.node.node_id not in hot_window_ids
    ]

    # Step 3 — State document text
    state_text = state_document.to_context_string()

    latency = (time.time() - start) * 1000  # ms

    return AssembledContext(
        state_document_text=state_text,
        retrieved_nodes=retrieved_nodes,
        hot_window=hot_window,
        new_message=new_message,
        retrieval_latency_ms=round(latency, 2),
        prefetch_used=prefetch_used,
    )


# ─────────────────────────────────────────────
# PRE-FETCH ENGINE
# ─────────────────────────────────────────────

class PreFetchEngine:
    """
    Begins graph retrieval as soon as the user starts typing.
    By the time the message is sent, retrieval is done or nearly done.

    Usage:
        engine = PreFetchEngine(all_nodes, all_edges, current_turn)
        engine.on_keypress(partial_text)   # called on each keystroke
        result = engine.finalise(full_message, message_entities)  # called on send
    """

    def __init__(
        self,
        all_nodes: dict[str, ConversationNode],
        all_edges: list[ConversationEdge],
        current_turn: int,
        hot_window: list[ConversationNode],
    ):
        self.all_nodes = all_nodes
        self.all_edges = all_edges
        self.current_turn = current_turn
        self.hot_window = hot_window
        self._provisional_result: list[NodeScore] | None = None
        self._provisional_entities: list[str] = []

    def on_keypress(self, partial_text: str) -> None:
        """
        Called on each keystroke. Runs a provisional retrieval
        based on partial message. Overwrites previous provisional result.
        In production this runs in a background thread.
        """
        if len(partial_text) < 10:
            return  # Too short to extract meaningful entities yet

        provisional_entities = self._extract_entities_fast(partial_text)

        if provisional_entities == self._provisional_entities:
            return  # Entities haven't changed, skip re-traversal

        self._provisional_entities = provisional_entities
        self._provisional_result = traverse_and_rank(
            graph_nodes=self.all_nodes,
            graph_edges=self.all_edges,
            hot_window=self.hot_window,
            current_turn=self.current_turn,
            current_message_entities=provisional_entities,
            max_depth=MAX_DEPTH,
            top_k=MAX_RETRIEVED_NODES,
            min_score_threshold=MIN_SCORE,
        )

    def finalise(
        self,
        full_message: str,
        full_message_entities: list[str],
    ) -> list[NodeScore]:
        """
        Called when user hits send.
        Verifies provisional result against final message entities.
        If entities changed significantly, runs a correction pass.
        Returns final retrieval result.
        """
        if self._provisional_result is None:
            # Pre-fetch didn't run, do full traversal now
            return traverse_and_rank(
                graph_nodes=self.all_nodes,
                graph_edges=self.all_edges,
                hot_window=self.hot_window,
                current_turn=self.current_turn,
                current_message_entities=full_message_entities,
                max_depth=MAX_DEPTH,
                top_k=MAX_RETRIEVED_NODES,
                min_score_threshold=MIN_SCORE,
            )

        # Check entity drift between provisional and final
        provisional_set = set(self._provisional_entities)
        final_set = set(full_message_entities)
        new_entities = final_set - provisional_set

        if len(new_entities) > len(provisional_set) * 0.5:
            # More than 50% new entities — re-run full traversal
            return traverse_and_rank(
                graph_nodes=self.all_nodes,
                graph_edges=self.all_edges,
                hot_window=self.hot_window,
                current_turn=self.current_turn,
                current_message_entities=full_message_entities,
                max_depth=MAX_DEPTH,
                top_k=MAX_RETRIEVED_NODES,
                min_score_threshold=MIN_SCORE,
            )

        # Provisional result is good enough — return it
        return self._provisional_result

    def _extract_entities_fast(self, text: str) -> list[str]:
        """
        Lightweight entity extraction for use during typing.
        In production: replace with a fast NER model or keyword extractor.
        This placeholder extracts capitalised words as a rough proxy.
        """
        words = text.split()
        entities = [
            w.strip(".,!?") for w in words
            if w and w[0].isupper() and len(w) > 2
        ]
        return list(set(entities))
