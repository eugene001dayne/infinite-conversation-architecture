"""
Infinite Conversation Architecture
DMVP — Distributed Memory Verification Protocol v1.0

Author: Eugene Mawuli Attigah
Project: infinite-conversation-architecture
License: CC BY 4.0

Extends: core/ica_extensions_v2.py (MemoryAttestationEngine, MemoryTransferEnvelope)

The DMVP answers one question: when Agent B receives a MemoryTransferEnvelope
from Agent A, how does it establish trust without a central authority?

Four sub-protocols, all operating entirely offline:

  Sub-Protocol 1 — Identity
      Did this envelope actually come from Agent A?
      Mechanism: Ed25519 keypairs, self-certifying Agent IDs, TOFU or pre-shared trust

  Sub-Protocol 2 — Selective Disclosure
      Can Agent B verify a partial memory transfer?
      Mechanism: Merkle tree over existing SHA-256 node hashes

  Sub-Protocol 3 — Memory Freshness
      Is this envelope current, or a replay of old data?
      Mechanism: Timestamp TTL + nonce registry + monotonic sequence numbers

  Sub-Protocol 4 — Conflict Resolution
      If Agent B has a State Document, which version wins?
      Mechanism: Version vectors (Lamport clocks per agent), deterministic merge

All cryptographic operations use Python's `cryptography` library.
Compatible with the existing SHA-256 infrastructure in MemoryAttestationEngine.

Installation:
    pip install cryptography
"""

from __future__ import annotations

import os
import json
import base64
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# IMPORTS — cryptography library
# ─────────────────────────────────────────────

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey
    )
    from cryptography.exceptions import InvalidSignature
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False
    print("WARNING: 'cryptography' package not installed. Run: pip install cryptography")
    print("DMVP identity and signing features will not be available.")


# ─────────────────────────────────────────────
# ERROR TYPES
# ─────────────────────────────────────────────

class DMVPError(Exception):
    pass

class IdentityError(DMVPError):
    pass    # KEY_ID_MISMATCH, KEY_SUBSTITUTION_ATTACK, UNKNOWN_AGENT, INVALID_SIGNATURE

class FreshnessError(DMVPError):
    pass    # FUTURE_DATED, STALE, REPLAY_DETECTED, SEQUENCE_REPLAY

class IntegrityError(DMVPError):
    pass    # NODE_HASH_INVALID, STATE_CHAIN_BROKEN

class DisclosureError(DMVPError):
    pass    # MERKLE_PROOF_INVALID

class ConflictError(DMVPError):
    pass    # DISJOINT_LINEAGE, MERGE_DEPTH_EXCEEDED


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

FRESHNESS_TTL_SECONDS = 300          # 5 minutes — configurable
MAX_CLOCK_SKEW_SECONDS = 30          # acceptable future-dated window
MAX_MERGE_DEPTH = 3                  # prevents runaway recursive conflict resolution


# ─────────────────────────────────────────────
# SUB-PROTOCOL 1 — AGENT IDENTITY
# ─────────────────────────────────────────────

@dataclass
class AgentIdentity:
    """
    An agent's cryptographic identity.
    Generated once at agent initialization, persisted locally.
    The private key is NEVER transmitted.
    """
    agent_id: str                    # SHA-256(public_key_bytes).hex() — 64-char hex
    public_key_b64: str              # base64-encoded Ed25519 public key — 44 chars
    _private_key: object = field(default=None, repr=False)  # Ed25519PrivateKey object

    @classmethod
    def generate(cls) -> "AgentIdentity":
        """Generate a new Ed25519 keypair and derive the agent identity."""
        if not _CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for identity generation")
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        pub_bytes = public_key.public_bytes_raw()           # 32 bytes
        agent_id = hashlib.sha256(pub_bytes).hexdigest()    # 64-char hex
        pub_b64 = base64.b64encode(pub_bytes).decode()      # 44 chars
        identity = cls(agent_id=agent_id, public_key_b64=pub_b64)
        identity._private_key = private_key
        return identity

    def sign(self, payload_dict: dict) -> str:
        """
        Sign a payload dict canonically.
        Returns base64-encoded signature string.
        """
        if not _CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for signing")
        if self._private_key is None:
            raise IdentityError("Cannot sign without private key — this is a receive-only identity")
        canonical = _canonical_serialize(payload_dict)
        sig_bytes = self._private_key.sign(canonical)
        return base64.b64encode(sig_bytes).decode()

    def to_dict(self) -> dict:
        """Serialise public identity (never includes private key)."""
        return {
            "agent_id": self.agent_id,
            "public_key_b64": self.public_key_b64,
        }


class TrustMode(str, Enum):
    TOFU = "TOFU"               # Trust On First Use — for open/dynamic deployments
    PRE_SHARED = "PRE_SHARED"   # Pre-shared registry — for closed/known agent sets


@dataclass
class TrustStoreEntry:
    agent_id: str
    public_key_b64: str
    first_seen: str
    trust_mode: TrustMode
    label: Optional[str] = None


class TrustStore:
    """
    Local trust store mapping agent_id → public key.
    Persisted to a JSON file. Updated on first contact (TOFU) or pre-loaded.
    """

    def __init__(self, trust_mode: TrustMode = TrustMode.TOFU):
        self._store: dict[str, TrustStoreEntry] = {}
        self.trust_mode = trust_mode

    def register(self, agent_id: str, public_key_b64: str,
                 label: Optional[str] = None) -> None:
        """Register a known agent (for PRE_SHARED mode or manual TOFU acceptance)."""
        self._store[agent_id] = TrustStoreEntry(
            agent_id=agent_id,
            public_key_b64=public_key_b64,
            first_seen=_utc_now(),
            trust_mode=self.trust_mode,
            label=label,
        )

    def check(self, agent_id: str, public_key_b64: str) -> bool:
        """
        Verify agent identity against trust store.
        TOFU: accepts first contact, rejects key substitution.
        PRE_SHARED: rejects unknown agents entirely.
        Returns True if trusted. Raises IdentityError if suspicious.
        """
        if agent_id in self._store:
            stored = self._store[agent_id]
            if stored.public_key_b64 != public_key_b64:
                raise IdentityError("KEY_SUBSTITUTION_ATTACK")
            return True

        if self.trust_mode == TrustMode.PRE_SHARED:
            raise IdentityError("UNKNOWN_AGENT")

        # TOFU: first contact — record and continue
        self.register(agent_id, public_key_b64)
        return True

    def revoke(self, agent_id: str) -> None:
        """Remove a compromised agent from the trust store."""
        self._store.pop(agent_id, None)

    def to_dict(self) -> dict:
        return {
            "trust_mode": self.trust_mode.value,
            "agents": {
                aid: {
                    "public_key_b64": e.public_key_b64,
                    "first_seen": e.first_seen,
                    "trust_mode": e.trust_mode.value,
                    "label": e.label,
                }
                for aid, e in self._store.items()
            }
        }


def verify_identity(envelope_dict: dict, trust_store: TrustStore) -> None:
    """
    Sub-Protocol 1 — verify agent identity.
    Runs three checks in order. Raises IdentityError on failure.
    """
    if not _CRYPTO_AVAILABLE:
        raise ImportError("cryptography package required for identity verification")

    header = envelope_dict.get("dmvp_header", {})
    agent_id = header.get("agent_id", "")
    pub_b64 = header.get("public_key", "")
    sig_b64 = header.get("signature", "")

    if not all([agent_id, pub_b64, sig_b64]):
        raise IdentityError("INVALID_SIGNATURE — missing DMVP header fields")

    # Step 1 — key-to-ID binding check
    pub_bytes = base64.b64decode(pub_b64)
    computed_id = hashlib.sha256(pub_bytes).hexdigest()
    if computed_id != agent_id:
        raise IdentityError("KEY_ID_MISMATCH")

    # Step 2 — trust store lookup
    trust_store.check(agent_id, pub_b64)

    # Step 3 — signature verification
    public_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
    sig_bytes = base64.b64decode(sig_b64)
    payload_for_sig = {k: v for k, v in envelope_dict.items()
                       if k != "dmvp_header"}
    canonical = _canonical_serialize(payload_for_sig)
    try:
        public_key.verify(sig_bytes, canonical)
    except InvalidSignature:
        raise IdentityError("INVALID_SIGNATURE")


# ─────────────────────────────────────────────
# SUB-PROTOCOL 2 — SELECTIVE DISCLOSURE
# ─────────────────────────────────────────────

def _sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def _merkle_leaf(node_hash_hex: str) -> bytes:
    """SHA-256 of the node's existing SHA-256 hex string."""
    return _sha256_bytes(node_hash_hex.encode("utf-8"))

def _merkle_parent(left: bytes, right: bytes) -> bytes:
    return _sha256_bytes(left + right)


def build_merkle_tree(node_hashes: list[str]) -> tuple[str, dict[str, list[str]]]:
    """
    Builds a Merkle tree from a list of node hash hex strings.
    Returns (merkle_root_hex, proof_dict) where proof_dict maps
    each node_hash to its inclusion proof (list of sibling hex strings).

    node_hashes must be sorted deterministically before calling.
    Recommended: sort(node_hashes) for reproducibility.
    """
    if not node_hashes:
        return hashlib.sha256(b"EMPTY").hexdigest(), {}

    leaves = [_merkle_leaf(h) for h in node_hashes]
    n = len(leaves)

    # Build tree layer by layer
    # tree[0] = leaves, tree[-1] = [root]
    tree: list[list[bytes]] = [leaves]
    current_layer = leaves

    while len(current_layer) > 1:
        next_layer = []
        for i in range(0, len(current_layer), 2):
            left = current_layer[i]
            right = current_layer[i + 1] if i + 1 < len(current_layer) else left
            next_layer.append(_merkle_parent(left, right))
        tree.append(next_layer)
        current_layer = next_layer

    root_hex = tree[-1][0].hex()

    # Generate inclusion proofs for each leaf
    proofs: dict[str, list[str]] = {}
    for idx, node_hash in enumerate(node_hashes):
        proof = []
        current_idx = idx
        for layer in tree[:-1]:
            if current_idx % 2 == 0:
                sibling_idx = current_idx + 1
                if sibling_idx < len(layer):
                    proof.append(layer[sibling_idx].hex())
                else:
                    proof.append(layer[current_idx].hex())  # duplicate if odd
            else:
                sibling_idx = current_idx - 1
                proof.append(layer[sibling_idx].hex())
            current_idx //= 2
        proofs[node_hash] = proof

    return root_hex, proofs


def verify_merkle_proof(node_hash_hex: str, proof: list[str],
                         merkle_root_hex: str, leaf_index: int) -> bool:
    """
    Verifies that a node is a member of the Merkle tree with the given root.
    Returns True if proof is valid.
    """
    current = _merkle_leaf(node_hash_hex)
    current_idx = leaf_index

    for sibling_hex in proof:
        sibling = bytes.fromhex(sibling_hex)
        if current_idx % 2 == 0:
            current = _merkle_parent(current, sibling)
        else:
            current = _merkle_parent(sibling, current)
        current_idx //= 2

    return current.hex() == merkle_root_hex


# ─────────────────────────────────────────────
# SUB-PROTOCOL 3 — MEMORY FRESHNESS
# ─────────────────────────────────────────────

class NonceRegistry:
    """
    Tracks seen nonces to detect replay attacks.
    Expires nonces after TTL to prevent unbounded growth.
    """

    def __init__(self, ttl_seconds: int = FRESHNESS_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        # nonce → expiry_timestamp
        self._seen: dict[str, float] = {}
        # agent_id → highest sequence number seen
        self._sequences: dict[str, int] = {}

    def _evict_expired(self) -> None:
        now = time.time()
        self._seen = {k: v for k, v in self._seen.items() if v > now}

    def check_and_record(self, nonce: str, sequence: Optional[int],
                          agent_id: str, timestamp_iso: str) -> None:
        """
        Checks freshness. Raises FreshnessError on failure.
        Records nonce and sequence on success.
        """
        self._evict_expired()
        now_utc = datetime.now(timezone.utc)

        # Parse envelope timestamp
        try:
            envelope_ts = datetime.fromisoformat(
                timestamp_iso.replace("Z", "+00:00")
            )
        except ValueError:
            raise FreshnessError("STALE — unparseable timestamp")

        # Check for future-dated envelopes (clock skew tolerance)
        skew = (envelope_ts - now_utc).total_seconds()
        if skew > MAX_CLOCK_SKEW_SECONDS:
            raise FreshnessError(f"FUTURE_DATED — envelope dated {skew:.0f}s in the future")

        # Check for stale envelopes
        age = (now_utc - envelope_ts).total_seconds()
        if age > self.ttl_seconds:
            raise FreshnessError(f"STALE — envelope is {age:.0f}s old (TTL: {self.ttl_seconds}s)")

        # Check for nonce replay
        if nonce in self._seen:
            raise FreshnessError("REPLAY_DETECTED — nonce has been seen before")

        # Check sequence number
        if sequence is not None:
            last_seq = self._sequences.get(agent_id, -1)
            if sequence <= last_seq:
                raise FreshnessError(
                    f"SEQUENCE_REPLAY — seq {sequence} not greater than last seen {last_seq}"
                )

        # Record nonce and sequence
        expiry = time.time() + self.ttl_seconds
        self._seen[nonce] = expiry
        if sequence is not None:
            self._sequences[agent_id] = sequence


def generate_nonce() -> str:
    """Generate a 256-bit cryptographically secure nonce as 64-char hex."""
    return os.urandom(32).hex()


# ─────────────────────────────────────────────
# SUB-PROTOCOL 4 — CONFLICT RESOLUTION
# ─────────────────────────────────────────────

class VersionVector:
    """
    A version vector (Lamport clock per agent) tracking causal history
    of a State Document across independent agent instances.

    Keys are agent_ids. Values are integer event counters.
    """

    def __init__(self, vector: Optional[dict[str, int]] = None):
        self._v: dict[str, int] = vector.copy() if vector else {}

    def increment(self, agent_id: str) -> None:
        """Increment this agent's counter — call when making a local update."""
        self._v[agent_id] = self._v.get(agent_id, 0) + 1

    def merge(self, other: "VersionVector") -> "VersionVector":
        """Element-wise maximum of two version vectors."""
        all_agents = set(self._v.keys()) | set(other._v.keys())
        merged = {a: max(self._v.get(a, 0), other._v.get(a, 0))
                  for a in all_agents}
        return VersionVector(merged)

    def happens_before(self, other: "VersionVector") -> bool:
        """Returns True if self causally precedes other (self is an ancestor)."""
        all_agents = set(self._v.keys()) | set(other._v.keys())
        return (
            all(self._v.get(a, 0) <= other._v.get(a, 0) for a in all_agents) and
            any(self._v.get(a, 0) < other._v.get(a, 0) for a in all_agents)
        )

    def concurrent_with(self, other: "VersionVector") -> bool:
        """Returns True if neither version vector happens-before the other."""
        return not (
            self.happens_before(other) or
            other.happens_before(self) or
            self == other
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VersionVector):
            return False
        all_agents = set(self._v.keys()) | set(other._v.keys())
        return all(self._v.get(a, 0) == other._v.get(a, 0) for a in all_agents)

    def to_dict(self) -> dict[str, int]:
        return self._v.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "VersionVector":
        return cls(d)


@dataclass
class StateDocumentMergeResult:
    merged_content: dict
    merged_version_vector: VersionVector
    conflict_fields: list[str]        # fields that required tie-breaking
    resolution_method: str            # "fast_forward" | "union" | "last_write_wins" | "manual_required"
    requires_human_review: bool


def resolve_state_conflict(
    local_doc: dict,
    remote_doc: dict,
    local_vv: VersionVector,
    remote_vv: VersionVector,
    resolving_agent_id: str,
    depth: int = 0,
) -> StateDocumentMergeResult:
    """
    Sub-Protocol 4 — deterministic conflict resolution using version vectors.

    Four outcomes:
    1. remote is a descendant of local → fast-forward to remote
    2. local is a descendant of remote → keep local
    3. identical → no-op
    4. concurrent (true conflict) → field-level merge with tie-breaking

    Section merge strategies:
    - user_identity: scalar fields → last-write-wins, lists → union
    - conversation_state: active_threads/agreements → union, decisions → union
    - key_facts: union (append-only, no deletion)
    - relationship_history: append-only
    """
    if depth > MAX_MERGE_DEPTH:
        raise ConflictError("MERGE_DEPTH_EXCEEDED")

    # Case 1: remote supersedes local
    if local_vv.happens_before(remote_vv):
        merged_vv = remote_vv.merge(local_vv)
        merged_vv.increment(resolving_agent_id)
        return StateDocumentMergeResult(
            merged_content=remote_doc,
            merged_version_vector=merged_vv,
            conflict_fields=[],
            resolution_method="fast_forward",
            requires_human_review=False,
        )

    # Case 2: local supersedes remote
    if remote_vv.happens_before(local_vv):
        return StateDocumentMergeResult(
            merged_content=local_doc,
            merged_version_vector=local_vv,
            conflict_fields=[],
            resolution_method="keep_local",
            requires_human_review=False,
        )

    # Case 3: identical
    if local_vv == remote_vv:
        return StateDocumentMergeResult(
            merged_content=local_doc,
            merged_version_vector=local_vv,
            conflict_fields=[],
            resolution_method="identical",
            requires_human_review=False,
        )

    # Case 4: concurrent — field-level merge
    conflict_fields = []
    merged = {}

    for section in ["user_identity", "conversation_state", "key_facts", "relationship_history"]:
        local_sec = local_doc.get(section, {})
        remote_sec = remote_doc.get(section, {})
        merged_sec = {}

        for key in set(list(local_sec.keys()) + list(remote_sec.keys())):
            local_val = local_sec.get(key)
            remote_val = remote_sec.get(key)

            if local_val == remote_val:
                merged_sec[key] = local_val
            elif local_val is None:
                merged_sec[key] = remote_val
            elif remote_val is None:
                merged_sec[key] = local_val
            elif isinstance(local_val, list) and isinstance(remote_val, list):
                # Lists: union (preserve all, deduplicate)
                merged_sec[key] = list(dict.fromkeys(local_val + remote_val))
            else:
                # Scalar conflict: last-write-wins by version vector magnitude
                # Tie-breaker: lexicographically larger value wins (deterministic)
                conflict_fields.append(f"{section}.{key}")
                local_magnitude = sum(local_vv.to_dict().values())
                remote_magnitude = sum(remote_vv.to_dict().values())
                if remote_magnitude > local_magnitude:
                    merged_sec[key] = remote_val
                elif local_magnitude > remote_magnitude:
                    merged_sec[key] = local_val
                else:
                    # Perfect tie: lexicographic comparison for determinism
                    merged_sec[key] = max(str(local_val), str(remote_val))

        merged[section] = merged_sec

    merged_vv = local_vv.merge(remote_vv)
    merged_vv.increment(resolving_agent_id)

    return StateDocumentMergeResult(
        merged_content=merged,
        merged_version_vector=merged_vv,
        conflict_fields=conflict_fields,
        resolution_method="last_write_wins",
        requires_human_review=len(conflict_fields) > 0,
    )


# ─────────────────────────────────────────────
# DMVP HEADER — extension to MemoryTransferEnvelope
# ─────────────────────────────────────────────

@dataclass
class DMVPHeader:
    """
    Added to every MemoryTransferEnvelope to carry DMVP identity
    and freshness fields. The existing envelope fields are unchanged.
    """
    agent_id: str                      # SHA-256(public_key_bytes).hex()
    public_key: str                    # base64-encoded Ed25519 public key
    signature: str                     # base64-encoded Ed25519 signature
    timestamp: str                     # ISO 8601 UTC with milliseconds
    nonce: str                         # 64-char hex, 256-bit random
    sequence: int                      # monotonic counter per sender→receiver channel
    version_vector: dict[str, int]     # Lamport clocks per agent
    dmvp_version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "public_key": self.public_key,
            "signature": self.signature,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "sequence": self.sequence,
            "version_vector": self.version_vector,
            "dmvp_version": self.dmvp_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DMVPHeader":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────
# FULL VERIFICATION PIPELINE
# ─────────────────────────────────────────────

@dataclass
class VerificationResult:
    passed: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
    merged_state: Optional[dict] = None
    merge_result: Optional[StateDocumentMergeResult] = None
    is_first_contact: bool = False


def verify_envelope(
    envelope_dict: dict,
    trust_store: TrustStore,
    nonce_registry: NonceRegistry,
    local_state_doc: Optional[dict] = None,
    local_version_vector: Optional[VersionVector] = None,
    resolving_agent_id: str = "local",
) -> VerificationResult:
    """
    Full DMVP verification pipeline.
    Runs all four sub-protocols in order. Fails fast on error.

    Phase 1 — Identity (Sub-Protocol 1)
    Phase 2 — Freshness (Sub-Protocol 3)
    Phase 3 — Integrity (existing MemoryAttestationEngine — called externally)
    Phase 4 — Selective Disclosure (Sub-Protocol 2) — if is_selective
    Phase 5 — Conflict Resolution (Sub-Protocol 4) — if local_state_doc exists
    """
    header_dict = envelope_dict.get("dmvp_header", {})

    # Phase 1 — Identity
    try:
        verify_identity(envelope_dict, trust_store)
    except IdentityError as e:
        return VerificationResult(passed=False, error=str(e), error_type="IdentityError")

    # Phase 2 — Freshness
    try:
        nonce_registry.check_and_record(
            nonce=header_dict.get("nonce", ""),
            sequence=header_dict.get("sequence"),
            agent_id=header_dict.get("agent_id", ""),
            timestamp_iso=header_dict.get("timestamp", ""),
        )
    except FreshnessError as e:
        return VerificationResult(passed=False, error=str(e), error_type="FreshnessError")

    # Phase 3 — Integrity (caller is responsible for calling verify_node and verify_state_chain)
    # Skipped here — those functions live in MemoryAttestationEngine and are called
    # by the receiving agent after this pipeline returns OK.

    # Phase 4 — Selective Disclosure
    disclosure_manifest = envelope_dict.get("disclosure_manifest")
    if disclosure_manifest:
        merkle_root = disclosure_manifest.get("merkle_root", "")
        disclosed_nodes = envelope_dict.get("disclosed_nodes", {})
        merkle_proofs = envelope_dict.get("merkle_proofs", {})
        sorted_hashes = disclosure_manifest.get("sorted_node_hashes", [])

        for node_hash, proof in merkle_proofs.items():
            try:
                leaf_index = sorted_hashes.index(node_hash)
            except ValueError:
                return VerificationResult(
                    passed=False,
                    error=f"MERKLE_PROOF_INVALID — node {node_hash[:12]}... not in disclosure manifest",
                    error_type="DisclosureError"
                )
            if not verify_merkle_proof(node_hash, proof, merkle_root, leaf_index):
                return VerificationResult(
                    passed=False,
                    error=f"MERKLE_PROOF_INVALID — proof verification failed for {node_hash[:12]}...",
                    error_type="DisclosureError"
                )

    # Phase 5 — Conflict Resolution
    merge_result = None
    if local_state_doc is not None and envelope_dict.get("state_document_json"):
        try:
            remote_state = json.loads(envelope_dict["state_document_json"])
            remote_vv = VersionVector.from_dict(header_dict.get("version_vector", {}))
            local_vv = local_version_vector or VersionVector()

            merge_result = resolve_state_conflict(
                local_doc=local_state_doc,
                remote_doc=remote_state,
                local_vv=local_vv,
                remote_vv=remote_vv,
                resolving_agent_id=resolving_agent_id,
            )
        except ConflictError as e:
            return VerificationResult(passed=False, error=str(e), error_type="ConflictError")

    return VerificationResult(
        passed=True,
        merged_state=merge_result.merged_content if merge_result else None,
        merge_result=merge_result,
    )


# ─────────────────────────────────────────────
# DMVP SIGNING HELPER
# ─────────────────────────────────────────────

def sign_envelope(
    envelope_dict: dict,
    identity: AgentIdentity,
    sequence: int,
    version_vector: VersionVector,
) -> dict:
    """
    Adds a DMVP header to an envelope dict and signs it.
    Returns the envelope dict with dmvp_header populated.

    The signature covers all envelope fields except dmvp_header itself.
    """
    timestamp = _utc_now()
    nonce = generate_nonce()

    # Build the header (signature computed last)
    header = {
        "agent_id": identity.agent_id,
        "public_key": identity.public_key_b64,
        "timestamp": timestamp,
        "nonce": nonce,
        "sequence": sequence,
        "version_vector": version_vector.to_dict(),
        "dmvp_version": "1.0",
        "signature": "",  # placeholder
    }

    # The payload that gets signed = all envelope fields except dmvp_header
    payload_for_sig = {k: v for k, v in envelope_dict.items() if k != "dmvp_header"}
    sig = identity.sign(payload_for_sig)

    header["signature"] = sig

    result = envelope_dict.copy()
    result["dmvp_header"] = header
    return result


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def _canonical_serialize(payload_dict: dict) -> bytes:
    """
    Deterministic JSON serialization for signature operations.
    sort_keys=True is mandatory — JSON key ordering is not guaranteed.
    """
    return json.dumps(
        payload_dict,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _utc_now() -> str:
    """ISO 8601 UTC timestamp with milliseconds, ending in Z."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def demo():
    if not _CRYPTO_AVAILABLE:
        print("cryptography package not installed. Run: pip install cryptography")
        return

    print("=" * 60)
    print("DMVP v1.0 DEMO")
    print("=" * 60)

    # Agent A generates identity
    print("\n--- AGENT IDENTITY ---")
    agent_a = AgentIdentity.generate()
    print(f"Agent A ID: {agent_a.agent_id[:20]}...")
    print(f"Public key: {agent_a.public_key_b64[:20]}...")

    # Agent B sets up trust store
    trust_store = TrustStore(trust_mode=TrustMode.TOFU)
    nonce_registry = NonceRegistry(ttl_seconds=300)

    # Build a sample envelope
    envelope = {
        "transfer_id": "test-001",
        "sender_agent_id": agent_a.agent_id,
        "conversation_id": "conv_abc",
        "state_document_json": '{"user_identity": {"name": "Eugene"}, "conversation_state": {}}',
        "state_document_hash": hashlib.sha256(b'state').hexdigest(),
        "ground_truth_network": {},
        "node_attestations": {},
        "created_at": _utc_now(),
        "envelope_hash": "",
    }

    # Sign the envelope
    vv = VersionVector()
    vv.increment(agent_a.agent_id)
    signed = sign_envelope(envelope, agent_a, sequence=1, version_vector=vv)
    print(f"\nEnvelope signed. Signature: {signed['dmvp_header']['signature'][:20]}...")

    # Verify the envelope
    print("\n--- VERIFICATION PIPELINE ---")
    result = verify_envelope(
        envelope_dict=signed,
        trust_store=trust_store,
        nonce_registry=nonce_registry,
        local_state_doc=None,
    )
    print(f"Verification: {'PASS' if result.passed else f'FAIL: {result.error}'}")

    # Replay detection
    result2 = verify_envelope(
        envelope_dict=signed,
        trust_store=trust_store,
        nonce_registry=nonce_registry,
    )
    print(f"Replay attempt: {'PASS (bad)' if result2.passed else f'CORRECTLY BLOCKED: {result2.error}'}")

    # Merkle tree
    print("\n--- MERKLE TREE SELECTIVE DISCLOSURE ---")
    node_hashes = sorted([
        hashlib.sha256(f"node_{i}".encode()).hexdigest()
        for i in range(8)
    ])
    merkle_root, proofs = build_merkle_tree(node_hashes)
    print(f"Root: {merkle_root[:20]}...")

    # Verify one node
    test_node = node_hashes[3]
    proof = proofs[test_node]
    valid = verify_merkle_proof(test_node, proof, merkle_root, 3)
    print(f"Node 3 membership proof: {'PASS' if valid else 'FAIL'}")

    # Version vector conflict resolution
    print("\n--- CONFLICT RESOLUTION ---")
    vv_a = VersionVector({"agent_a": 3, "agent_b": 1})
    vv_b = VersionVector({"agent_a": 1, "agent_b": 4})

    local_doc = {"user_identity": {"name": "Eugene", "role": "builder"}, "conversation_state": {}, "key_facts": {}, "relationship_history": {}}
    remote_doc = {"user_identity": {"name": "Eugene", "role": "researcher"}, "conversation_state": {}, "key_facts": {}, "relationship_history": {}}

    merge = resolve_state_conflict(local_doc, remote_doc, vv_a, vv_b, "agent_b")
    print(f"Resolution method: {merge.resolution_method}")
    print(f"Conflict fields: {merge.conflict_fields}")
    print(f"Merged role: {merge.merged_content.get('user_identity', {}).get('role')}")
    print(f"Requires human review: {merge.requires_human_review}")

    print("\n" + "=" * 60)
    print("DMVP v1.0 working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
