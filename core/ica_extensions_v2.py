"""
Infinite Conversation Architecture
ICA Extensions v2.0 — Ground Truth Network + Memory Attestation

Author: Eugene Mawuli Attigah
Project: infinite-conversation-architecture
License: CC BY 4.0

Two extensions that transform ICA from a single-session memory framework
into a verifiable, distributed memory infrastructure:

1. GROUND TRUTH NETWORK
   Verified nodes that the memory manager treats as permanent anchors.
   Never compressed. Never archived. Always retrieved when entities match.
   Protected from self-correction loop overwrite.

2. MEMORY ATTESTATION
   Every node gets a SHA-256 hash at write time.
   The State Document gets a chain signature.
   Memory can be shared between agent instances with full tamper-evidence.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# PART 1 — GROUND TRUTH NETWORK
# ─────────────────────────────────────────────

class VerificationSource(str, Enum):
    """How a Ground Truth node was verified."""
    USER_ASSERTED = "user_asserted"       # User explicitly marked as ground truth
    SYSTEM_CHECKED = "system_checked"     # System verified against external source
    CROSS_SESSION = "cross_session"       # Carried forward from prior verified session
    AGENT_CONSENSUS = "agent_consensus"   # Multiple agents agreed on this fact


@dataclass
class GroundTruthRecord:
    """
    Attached to a ConversationNode to mark it as a verified ground truth anchor.

    Ground truth nodes:
    - Never move to warm or cold tier
    - Always retrieved when their entities appear in the current message
    - Cannot be overwritten by the self-correction loop (only flagged as conflict)
    - Carry full provenance of how they were verified
    """
    verified_at: datetime = field(default_factory=datetime.utcnow)
    verification_source: VerificationSource = VerificationSource.USER_ASSERTED
    verified_by: Optional[str] = None        # Agent ID or user ID that verified
    confidence: float = 1.0                  # 0.0–1.0, user assertions = 1.0
    external_source_url: Optional[str] = None  # If system-checked, the source
    conflict_flagged: bool = False           # True if self-correction loop found a conflict
    conflict_note: Optional[str] = None      # What the conflict is, if any
    attestation_hash: Optional[str] = None  # Set by Memory Attestation on write


class GroundTruthNetwork:
    """
    Manages the Ground Truth Network — the verified anchor layer of the conversation graph.

    The GTN sits above the normal conversation graph. It is a separate index of
    verified nodes that the retrieval system always consults, regardless of graph
    traversal results.
    """

    def __init__(self):
        # node_id → GroundTruthRecord
        self._verified_nodes: dict[str, GroundTruthRecord] = {}
        # entity (lowercase) → list of node_ids that are verified anchors for this entity
        self._entity_index: dict[str, list[str]] = {}

    def register(
        self,
        node_id: str,
        entity_tags: list[str],
        source: VerificationSource = VerificationSource.USER_ASSERTED,
        verified_by: Optional[str] = None,
        confidence: float = 1.0,
        external_source_url: Optional[str] = None,
    ) -> GroundTruthRecord:
        """
        Register a node as a Ground Truth anchor.
        Called by the memory manager when a user marks something as ground truth,
        or when the system verifies a fact.
        """
        record = GroundTruthRecord(
            verification_source=source,
            verified_by=verified_by,
            confidence=confidence,
            external_source_url=external_source_url,
        )
        self._verified_nodes[node_id] = record

        # Index by entity for fast lookup during retrieval
        for entity in entity_tags:
            key = entity.lower()
            if key not in self._entity_index:
                self._entity_index[key] = []
            if node_id not in self._entity_index[key]:
                self._entity_index[key].append(node_id)

        return record

    def is_verified(self, node_id: str) -> bool:
        return node_id in self._verified_nodes

    def get_record(self, node_id: str) -> Optional[GroundTruthRecord]:
        return self._verified_nodes.get(node_id)

    def lookup_by_entities(self, entities: list[str]) -> list[str]:
        """
        Returns all verified node_ids that match any of the given entities.
        These are always injected into context regardless of graph traversal score.
        """
        matched = set()
        for entity in entities:
            key = entity.lower()
            for node_id in self._entity_index.get(key, []):
                matched.add(node_id)
        return list(matched)

    def flag_conflict(self, node_id: str, conflict_note: str) -> None:
        """
        Called by the self-correction loop when recent conversation contradicts
        a ground truth node. Does not remove the node — only flags it for human review.
        The memory manager surfaces the conflict flag in the State Document.
        """
        record = self._verified_nodes.get(node_id)
        if record:
            record.conflict_flagged = True
            record.conflict_note = conflict_note

    def resolve_conflict(self, node_id: str) -> None:
        """Called when a flagged conflict has been reviewed and resolved."""
        record = self._verified_nodes.get(node_id)
        if record:
            record.conflict_flagged = False
            record.conflict_note = None

    def export(self) -> dict:
        """Serialises the full GTN for persistence or transfer."""
        return {
            "verified_nodes": {
                nid: {
                    "verified_at": rec.verified_at.isoformat(),
                    "verification_source": rec.verification_source.value,
                    "verified_by": rec.verified_by,
                    "confidence": rec.confidence,
                    "external_source_url": rec.external_source_url,
                    "conflict_flagged": rec.conflict_flagged,
                    "conflict_note": rec.conflict_note,
                    "attestation_hash": rec.attestation_hash,
                }
                for nid, rec in self._verified_nodes.items()
            },
            "entity_index": self._entity_index,
        }

    @classmethod
    def from_export(cls, data: dict) -> "GroundTruthNetwork":
        """Restores a GTN from a serialised export (cross-session persistence)."""
        gtn = cls()
        for nid, rec_data in data.get("verified_nodes", {}).items():
            rec = GroundTruthRecord(
                verified_at=datetime.fromisoformat(rec_data["verified_at"]),
                verification_source=VerificationSource(rec_data["verification_source"]),
                verified_by=rec_data.get("verified_by"),
                confidence=rec_data.get("confidence", 1.0),
                external_source_url=rec_data.get("external_source_url"),
                conflict_flagged=rec_data.get("conflict_flagged", False),
                conflict_note=rec_data.get("conflict_note"),
                attestation_hash=rec_data.get("attestation_hash"),
            )
            gtn._verified_nodes[nid] = rec
        gtn._entity_index = data.get("entity_index", {})
        return gtn


# ─────────────────────────────────────────────
# RETRIEVAL INTEGRATION — GROUND TRUTH BOOST
# ─────────────────────────────────────────────

GROUND_TRUTH_SCORE_BOOST = 0.85
"""
Score assigned to ground truth nodes when they match current message entities.
Set high (0.85) to ensure they always appear in the top-K, above the 0.15 threshold.
This is not additive — it replaces the normal score for verified nodes
that match the current entities, ensuring they are always surfaced.
"""


def apply_ground_truth_boost(
    scored_nodes: list[tuple[str, float]],
    gtn: GroundTruthNetwork,
    current_message_entities: list[str],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Takes the output of traverse_and_rank() and applies the Ground Truth boost.

    Any verified node matching current message entities gets its score set to
    GROUND_TRUTH_SCORE_BOOST, ensuring it appears in the final injected context.

    Called by context_assembler.assemble_context() after graph traversal.
    """
    # Find all verified nodes matching current entities
    gtn_matched = set(gtn.lookup_by_entities(current_message_entities))

    # Build result: preserve existing scores, boost verified matches
    node_scores = {nid: score for nid, score in scored_nodes}

    for gtn_node_id in gtn_matched:
        node_scores[gtn_node_id] = max(
            node_scores.get(gtn_node_id, 0.0),
            GROUND_TRUTH_SCORE_BOOST
        )

    # Re-sort by score descending
    result = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    return result[:top_k]


# ─────────────────────────────────────────────
# PART 2 — MEMORY ATTESTATION
# ─────────────────────────────────────────────

class MemoryAttestationEngine:
    """
    Provides cryptographic integrity for ICA memory.

    Every node written to the graph gets a SHA-256 hash of its content.
    The State Document gets a chain signature — each version signs the previous.

    When memory is shared with another agent instance, the receiving agent
    can call verify_node() and verify_state_chain() to confirm nothing was tampered with.

    This is the bridge between ICA and ChainThread. An agent receiving a
    ChainThread handoff can verify not just the envelope but the full
    memory state of the sending agent.
    """

    @staticmethod
    def hash_node(
        node_id: str,
        turn_number: int,
        full_text: str,
        speaker: str,
        conversation_id: str,
        timestamp: str,
    ) -> str:
        """
        Generates a SHA-256 hash for a conversation node.

        The canonical form sorts all fields to ensure deterministic output
        regardless of serialisation order.
        """
        canonical = json.dumps({
            "node_id": node_id,
            "turn_number": turn_number,
            "full_text": full_text,
            "speaker": speaker,
            "conversation_id": conversation_id,
            "timestamp": timestamp,
        }, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def verify_node(
        node_id: str,
        turn_number: int,
        full_text: str,
        speaker: str,
        conversation_id: str,
        timestamp: str,
        expected_hash: str,
    ) -> bool:
        """
        Verifies that a received node has not been tampered with.
        Returns True if the node content matches its hash.
        """
        computed = MemoryAttestationEngine.hash_node(
            node_id, turn_number, full_text, speaker, conversation_id, timestamp
        )
        return computed == expected_hash

    @staticmethod
    def hash_state_document(
        document_id: str,
        version: int,
        content_string: str,
        previous_hash: Optional[str],
    ) -> str:
        """
        Generates a chained hash for a State Document version.

        Each version includes the hash of the previous version,
        creating a tamper-evident chain. If previous_hash is None
        (first version), the chain starts fresh.
        """
        canonical = json.dumps({
            "document_id": document_id,
            "version": version,
            "content": content_string,
            "previous_hash": previous_hash or "GENESIS",
        }, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def verify_state_chain(versions: list[dict]) -> tuple[bool, Optional[str]]:
        """
        Verifies the full chain of State Document versions.

        Takes a list of version records in order:
        [{"document_id": ..., "version": N, "content": ..., "hash": ..., "previous_hash": ...}]

        Returns (is_valid, error_message).
        Returns (True, None) if the chain is intact.
        Returns (False, "description of where it broke") if tampered.
        """
        if not versions:
            return True, None

        for i, ver in enumerate(versions):
            expected_prev = versions[i - 1]["hash"] if i > 0 else None
            computed = MemoryAttestationEngine.hash_state_document(
                document_id=ver["document_id"],
                version=ver["version"],
                content_string=ver["content"],
                previous_hash=expected_prev,
            )
            if computed != ver["hash"]:
                return False, f"Chain broken at version {ver['version']}: hash mismatch"

        return True, None


# ─────────────────────────────────────────────
# MEMORY TRANSFER ENVELOPE
# ─────────────────────────────────────────────

@dataclass
class MemoryTransferEnvelope:
    """
    A signed, verifiable package of ICA memory for transfer between agent instances.

    This is the integration point between ICA and ChainThread.
    A ChainThread handoff envelope can carry a MemoryTransferEnvelope as payload,
    giving the receiving agent full verified memory context from the sender.

    Structure:
    - sender_agent_id: identity of the agent sharing memory
    - conversation_id: which conversation this memory belongs to
    - state_document: current State Document (serialised)
    - state_document_hash: hash of current State Document version
    - state_chain_hashes: full chain hashes for verification
    - ground_truth_network: serialised GTN export
    - node_attestations: dict of node_id → hash for all transferred nodes
    - envelope_hash: SHA-256 of the entire envelope (integrity check)
    - created_at: timestamp
    """
    transfer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_agent_id: str = ""
    conversation_id: str = ""
    state_document_json: str = ""
    state_document_hash: str = ""
    state_chain_hashes: list[dict] = field(default_factory=list)
    ground_truth_network: dict = field(default_factory=dict)
    node_attestations: dict[str, str] = field(default_factory=dict)  # node_id → hash
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    envelope_hash: str = ""

    def compute_envelope_hash(self) -> str:
        """Compute integrity hash of the full envelope (excluding envelope_hash itself)."""
        canonical = json.dumps({
            "transfer_id": self.transfer_id,
            "sender_agent_id": self.sender_agent_id,
            "conversation_id": self.conversation_id,
            "state_document_json": self.state_document_json,
            "state_document_hash": self.state_document_hash,
            "state_chain_hashes": self.state_chain_hashes,
            "ground_truth_network": self.ground_truth_network,
            "node_attestations": self.node_attestations,
            "created_at": self.created_at,
        }, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def seal(self) -> "MemoryTransferEnvelope":
        """Finalise the envelope by computing and storing its hash."""
        self.envelope_hash = self.compute_envelope_hash()
        return self

    def verify(self) -> tuple[bool, Optional[str]]:
        """
        Verify the integrity of a received envelope.
        Returns (is_valid, error_message).
        """
        expected = self.compute_envelope_hash()
        if expected != self.envelope_hash:
            return False, "Envelope hash mismatch — content may have been tampered with"

        # Verify state document chain
        chain_valid, chain_error = MemoryAttestationEngine.verify_state_chain(
            self.state_chain_hashes
        )
        if not chain_valid:
            return False, f"State document chain invalid: {chain_error}"

        return True, None

    def to_dict(self) -> dict:
        return {
            "transfer_id": self.transfer_id,
            "sender_agent_id": self.sender_agent_id,
            "conversation_id": self.conversation_id,
            "state_document_json": self.state_document_json,
            "state_document_hash": self.state_document_hash,
            "state_chain_hashes": self.state_chain_hashes,
            "ground_truth_network": self.ground_truth_network,
            "node_attestations": self.node_attestations,
            "created_at": self.created_at,
            "envelope_hash": self.envelope_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryTransferEnvelope":
        return cls(**data)


# ─────────────────────────────────────────────
# DEMO — BOTH EXTENSIONS WORKING TOGETHER
# ─────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("ICA EXTENSIONS v2.0 DEMO")
    print("=" * 60)

    attest = MemoryAttestationEngine()
    gtn = GroundTruthNetwork()

    # ── GROUND TRUTH NETWORK ──
    print("\n--- GROUND TRUTH NETWORK ---")

    # User asserts a ground truth fact
    node_id = str(uuid.uuid4())
    entities = ["Atlas", "dog"]
    gtn.register(
        node_id=node_id,
        entity_tags=entities,
        source=VerificationSource.USER_ASSERTED,
        verified_by="user",
        confidence=1.0,
    )
    print(f"Registered verified node: {node_id[:12]}...")
    print(f"Entities indexed: {entities}")

    # Simulate retrieval — current message mentions "Atlas"
    current_entities = ["Atlas", "puppy"]
    matched = gtn.lookup_by_entities(current_entities)
    print(f"Current message entities: {current_entities}")
    print(f"GTN matched nodes: {len(matched)} (ground truth always surfaced)")

    # Simulate self-correction loop finding a conflict
    gtn.flag_conflict(node_id, "Recent conversation says user's dog is named 'Orion', contradicts 'Atlas'")
    record = gtn.get_record(node_id)
    print(f"Conflict flagged: {record.conflict_flagged}")
    print(f"Conflict note: {record.conflict_note}")

    # ── MEMORY ATTESTATION ──
    print("\n--- MEMORY ATTESTATION ---")

    node_hash = attest.hash_node(
        node_id=node_id,
        turn_number=10,
        full_text="My dog's name is Atlas.",
        speaker="user",
        conversation_id="conv_abc",
        timestamp="2026-03-27T09:00:00",
    )
    print(f"Node hash: {node_hash[:20]}...")

    valid = attest.verify_node(
        node_id=node_id,
        turn_number=10,
        full_text="My dog's name is Atlas.",
        speaker="user",
        conversation_id="conv_abc",
        timestamp="2026-03-27T09:00:00",
        expected_hash=node_hash,
    )
    print(f"Node verification: {'PASS' if valid else 'FAIL'}")

    tampered = attest.verify_node(
        node_id=node_id,
        turn_number=10,
        full_text="My dog's name is Orion.",  # tampered
        speaker="user",
        conversation_id="conv_abc",
        timestamp="2026-03-27T09:00:00",
        expected_hash=node_hash,
    )
    print(f"Tampered node verification: {'PASS' if tampered else 'FAIL (correctly detected tampering)'}")

    # State document chain
    hash_v1 = attest.hash_state_document("doc_001", 1, '{"user": "Eugene"}', None)
    hash_v2 = attest.hash_state_document("doc_001", 2, '{"user": "Eugene", "project": "ICA"}', hash_v1)
    hash_v3 = attest.hash_state_document("doc_001", 3, '{"user": "Eugene", "project": "ICA", "stage": "v2"}', hash_v2)

    chain = [
        {"document_id": "doc_001", "version": 1, "content": '{"user": "Eugene"}', "hash": hash_v1, "previous_hash": None},
        {"document_id": "doc_001", "version": 2, "content": '{"user": "Eugene", "project": "ICA"}', "hash": hash_v2, "previous_hash": hash_v1},
        {"document_id": "doc_001", "version": 3, "content": '{"user": "Eugene", "project": "ICA", "stage": "v2"}', "hash": hash_v3, "previous_hash": hash_v2},
    ]
    chain_valid, chain_err = attest.verify_state_chain(chain)
    print(f"State chain verification: {'PASS' if chain_valid else f'FAIL: {chain_err}'}")

    # ── MEMORY TRANSFER ENVELOPE ──
    print("\n--- MEMORY TRANSFER ENVELOPE ---")

    envelope = MemoryTransferEnvelope(
        sender_agent_id="agent_research",
        conversation_id="conv_abc",
        state_document_json='{"user": "Eugene", "project": "ICA", "stage": "v2"}',
        state_document_hash=hash_v3,
        state_chain_hashes=chain,
        ground_truth_network=gtn.export(),
        node_attestations={node_id: node_hash},
    ).seal()

    print(f"Envelope sealed: {envelope.envelope_hash[:20]}...")

    is_valid, error = envelope.verify()
    print(f"Envelope verification: {'PASS' if is_valid else f'FAIL: {error}'}")

    # Simulate tampering
    envelope.state_document_json = '{"user": "attacker"}'
    is_valid_after_tamper, error = envelope.verify()
    print(f"Tampered envelope: {'PASS' if is_valid_after_tamper else 'FAIL (correctly detected tampering)'}")

    print("\n" + "=" * 60)
    print("Both extensions working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
