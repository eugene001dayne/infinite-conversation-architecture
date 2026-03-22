"""
tests/test_ner_engine.py
════════════════════════
Unit tests for core/ner_engine.py.

Run with:
    pytest tests/test_ner_engine.py -v
    pytest tests/test_ner_engine.py -v --benchmark  # includes latency checks

Author: contributed to the Infinite Conversation Architecture
Architecture by: Eugene Mawuli Attigah
"""

import time
import pytest

# Guard: skip entire module if spaCy model is not installed
pytest.importorskip("spacy", reason="spaCy not installed")

from core.ner_engine import (
    extract_metadata,
    _extract_entities_fast_impl,
    _classify_register,
    warm_up,
)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def preload_models():
    """Load spaCy models once for the full test session."""
    warm_up()


# ────────────────────────────────────────────────────────────────────────────
# extract_metadata — entity extraction
# ────────────────────────────────────────────────────────────────────────────

class TestExtractMetadataEntities:

    def test_person_entity(self):
        result = extract_metadata("I was speaking with Sarah about the deployment.")
        assert "Sarah" in result["entities"]

    def test_org_entity(self):
        result = extract_metadata("OpenAI just announced a new model.")
        assert any("OpenAI" in e or "Openai" in e for e in result["entities"])

    def test_product_entity(self):
        result = extract_metadata("The Postgres database is running slowly today.")
        assert any("Postgres" in e for e in result["entities"])

    def test_location_entity(self):
        result = extract_metadata("The team is based in London.")
        assert any("London" in e for e in result["entities"])

    def test_multiple_entities(self):
        result = extract_metadata(
            "Alice from OpenAI and Bob from Google met in London to discuss GPT-5."
        )
        entities = result["entities"]
        # Should contain at least the two people and the city
        assert len(entities) >= 2

    def test_no_entities(self):
        result = extract_metadata("The weather is nice today.")
        # May or may not find entities — just confirm structure is correct
        assert isinstance(result["entities"], list)

    def test_empty_string(self):
        result = extract_metadata("")
        assert result["entities"] == []
        assert result["topics"] == []
        assert result["emotional_register"] == "neutral"

    def test_whitespace_only(self):
        result = extract_metadata("   ")
        assert result["entities"] == []

    def test_no_duplicate_entities(self):
        result = extract_metadata(
            "Alice talked to Alice about the Alice problem."
        )
        alice_count = sum(1 for e in result["entities"] if "alice" in e.lower())
        assert alice_count == 1, "Entities should be deduplicated"


# ────────────────────────────────────────────────────────────────────────────
# extract_metadata — topic extraction
# ────────────────────────────────────────────────────────────────────────────

class TestExtractMetadataTopics:

    def test_returns_list(self):
        result = extract_metadata("I want to understand the caching strategy for our API.")
        assert isinstance(result["topics"], list)

    def test_topic_count_within_range(self):
        result = extract_metadata(
            "We need to improve the database query performance for the user dashboard."
        )
        assert 0 <= len(result["topics"]) <= 6

    def test_topics_non_empty_for_rich_text(self):
        result = extract_metadata(
            "Let's talk about the memory manager architecture and how it handles "
            "graph traversal for the conversation nodes."
        )
        assert len(result["topics"]) >= 1

    def test_short_message_pads_from_entities(self):
        # "Tell me about Sarah's project." — short, but has entities
        result = extract_metadata("Tell me about Sarah's project.")
        # topics should be padded if sparse, not empty
        # (entities like "Sarah" get promoted)
        total = len(result["topics"]) + len(result["entities"])
        assert total >= 1


# ────────────────────────────────────────────────────────────────────────────
# extract_metadata — emotional register
# ────────────────────────────────────────────────────────────────────────────

class TestEmotionalRegister:

    VALID_REGISTERS = {"neutral", "curious", "frustrated", "excited", "uncertain", "assertive"}

    def _check(self, text: str) -> str:
        result = extract_metadata(text)
        r = result["emotional_register"]
        assert r in self.VALID_REGISTERS, f"Invalid register: {r!r}"
        return r

    def test_valid_register_always_returned(self):
        for text in [
            "Why does this keep happening?",
            "I love this feature!",
            "Not sure if this is right.",
            "We must ship this immediately.",
            "The server crashed again!!",
            "The sky is blue.",
        ]:
            self._check(text)

    def test_curious_detected(self):
        r = _classify_register("Why does the pre-fetch engine start before the user sends?")
        assert r == "curious"

    def test_frustrated_detected(self):
        r = _classify_register("It's still not working and I've already tried everything!")
        assert r == "frustrated"

    def test_excited_detected(self):
        r = _classify_register("This is amazing, it finally works!")
        assert r == "excited"

    def test_uncertain_detected(self):
        r = _classify_register("Maybe the graph approach is better, but I'm not sure.")
        assert r == "uncertain"

    def test_assertive_detected(self):
        r = _classify_register("We definitely need to fix the memory manager before launch.")
        assert r == "assertive"

    def test_neutral_default(self):
        r = _classify_register("The file is saved.")
        assert r == "neutral"

    def test_empty_is_neutral(self):
        r = _classify_register("")
        assert r == "neutral"


# ────────────────────────────────────────────────────────────────────────────
# _extract_entities_fast_impl — typing path
# ────────────────────────────────────────────────────────────────────────────

class TestExtractEntitiesFast:

    def test_empty_returns_empty(self):
        assert _extract_entities_fast_impl("") == []
        assert _extract_entities_fast_impl("   ") == []

    def test_short_text_fast_path(self):
        # Under 15 chars — uses regex only
        result = _extract_entities_fast_impl("Alice")
        assert "Alice" in result

    def test_longer_text_nlp_path(self):
        result = _extract_entities_fast_impl(
            "I was talking to Sarah from OpenAI about"
        )
        assert isinstance(result, list)
        # Should find at least "Sarah" or "OpenAI"
        entity_names = [e.lower() for e in result]
        assert any(n in entity_names for n in ["sarah", "openai"])

    def test_no_duplicates(self):
        result = _extract_entities_fast_impl(
            "Alice went to Alice's meeting at Alice Corp."
        )
        alice_hits = sum(1 for e in result if "alice" in e.lower())
        assert alice_hits <= 2  # "Alice" (person) and "Alice Corp" (org) are distinct

    def test_returns_list_of_strings(self):
        result = _extract_entities_fast_impl("Tell me about the London team")
        assert isinstance(result, list)
        assert all(isinstance(e, str) for e in result)

    def test_mid_word_input_doesnt_crash(self):
        # Simulate user mid-word: "Tell me about Ope"
        result = _extract_entities_fast_impl("Tell me about Ope")
        assert isinstance(result, list)

    def test_punctuation_doesnt_crash(self):
        result = _extract_entities_fast_impl("Why?? What about!!!")
        assert isinstance(result, list)


# ────────────────────────────────────────────────────────────────────────────
# Latency — ensure we stay within the documented budgets
# ────────────────────────────────────────────────────────────────────────────

class TestLatency:
    """
    Latency assertions. These are conservative: we target <20 ms for the
    typing path and <50 ms for the full metadata call, but assert at 2× to
    give headroom for slow CI runners.
    """

    SAMPLES = [
        "I was talking to Sarah from the OpenAI team about the GPT-5 release plan.",
        "Can you explain why the Postgres query is slow on the production server?",
        "We need to ship Iron-Thread v2 before the PyCon deadline in London.",
        "Not sure if this approach makes sense for our architecture.",
        "Why does it keep failing?! I've tried everything already.",
    ]

    def _median_ms(self, fn, text: str, n: int = 20) -> float:
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn(text)
            times.append((time.perf_counter() - t0) * 1000)
        return sorted(times)[n // 2]

    def test_extract_entities_fast_under_40ms(self):
        for sample in self.SAMPLES:
            median = self._median_ms(_extract_entities_fast_impl, sample)
            assert median < 40, (
                f"_extract_entities_fast median {median:.1f}ms > 40ms "
                f"for input: {sample[:60]!r}"
            )

    def test_extract_metadata_under_100ms(self):
        for sample in self.SAMPLES:
            median = self._median_ms(extract_metadata, sample)
            assert median < 100, (
                f"extract_metadata median {median:.1f}ms > 100ms "
                f"for input: {sample[:60]!r}"
            )


# ────────────────────────────────────────────────────────────────────────────
# Return schema integrity
# ────────────────────────────────────────────────────────────────────────────

class TestReturnSchema:
    """Verify extract_metadata always returns the exact schema MemoryManager expects."""

    REQUIRED_KEYS = {"entities", "topics", "emotional_register"}
    VALID_REGISTERS = {"neutral", "curious", "frustrated", "excited", "uncertain", "assertive"}

    def test_schema_keys_always_present(self):
        for text in ["Hi", "", "Complex message about Sarah and OpenAI in London."]:
            result = extract_metadata(text)
            assert set(result.keys()) == self.REQUIRED_KEYS

    def test_entities_is_list_of_strings(self):
        result = extract_metadata("Sarah from OpenAI.")
        assert isinstance(result["entities"], list)
        assert all(isinstance(e, str) for e in result["entities"])

    def test_topics_is_list_of_strings(self):
        result = extract_metadata("We discussed the memory architecture.")
        assert isinstance(result["topics"], list)
        assert all(isinstance(t, str) for t in result["topics"])

    def test_topics_max_six(self):
        result = extract_metadata(
            "A long message about spaCy, Postgres, London, Alice, memory, graphs, "
            "and many other things including NER, topics, and registers."
        )
        assert len(result["topics"]) <= 6

    def test_register_is_valid_string(self):
        result = extract_metadata("Why is this happening?")
        assert result["emotional_register"] in self.VALID_REGISTERS
