"""
context_assembler.py — PreFetchEngine patch
════════════════════════════════════════════════════════════════════════════════
This file shows the exact changes to make in core/context_assembler.py.

Changes required:
  1. Add the import block below to the top of context_assembler.py.
  2. Replace the body of PreFetchEngine._extract_entities_fast with the
     method shown below.
  3. Add warm_up() call to your application startup sequence (see bottom).

Everything else in context_assembler.py stays unchanged.
════════════════════════════════════════════════════════════════════════════════
"""

# ── ADD THESE IMPORTS to the top of context_assembler.py ────────────────────

from core.ner_engine import (
    _load_nlp,
    _ENTITY_LABELS,
    _PROPER_NOUN_RE,
    _MIN_CHARS_FOR_NLP,
    _dedupe_entities,
    warm_up,
)

# ────────────────────────────────────────────────────────────────────────────


# ── REPLACE the existing _extract_entities_fast method body ─────────────────
#
# Inside class PreFetchEngine, swap out the placeholder with this:

class PreFetchEngine:
    """
    Partial implementation showing only the changed method.
    All other PreFetchEngine methods remain exactly as they were.
    """

    def _extract_entities_fast(self, partial_text: str) -> list[str]:
        """
        Real-time entity extraction called on every debounce tick as the
        user types. Must complete in <20 ms.

        Replaces the previous naive capitalisation heuristic with production
        NER backed by spaCy (en_core_web_sm). Two execution paths:

          Short text (<_MIN_CHARS_FOR_NLP chars):
            Regex-only fast path. Catches title-cased proper nouns without
            invoking the spaCy pipeline. Cost: <0.2 ms.

          Longer text:
            NER-only spaCy pipeline (parser and lemmatizer disabled for speed).
            Supplemented by regex to recover proper nouns that NER misses in
            sentence fragments (NER degrades slightly on incomplete sentences
            because it can't rely on parse context).
            Cost: 3–10 ms on CPU.

        Both paths return deduplicated, normalised entity strings ready for
        use in the provisional graph traversal query.

        Args:
            partial_text: Current contents of the user's input field.

        Returns:
            Deduplicated list of entity strings found so far.
        """
        if not partial_text or not partial_text.strip():
            return []

        text = partial_text.strip()

        # ── Fast path: too short for reliable NLP ─────────────────────────
        # Below this threshold, NER has insufficient context to be useful,
        # and regex is faster and comparably accurate.
        if len(text) < _MIN_CHARS_FOR_NLP:
            return _dedupe_entities(_PROPER_NOUN_RE.findall(text))

        # ── NLP path: ner-only pipeline ────────────────────────────────────
        _, nlp_ner = _load_nlp()
        doc = nlp_ner(text)

        entities: list[str] = [
            ent.text for ent in doc.ents if ent.label_ in _ENTITY_LABELS
        ]

        # Regex supplement: catches proper nouns NER missed in fragments.
        # We extend (not replace) to keep NER results authoritative.
        entities.extend(_PROPER_NOUN_RE.findall(text))

        return _dedupe_entities(entities)


# ── STARTUP — add to main.py or server entry point ──────────────────────────
#
# The first spacy.load() call takes ~200 ms. Pre-loading at startup means
# no real user request ever pays that cost.
#
#   from core.ner_engine import warm_up
#   warm_up()   # call once during server initialisation
#
# ────────────────────────────────────────────────────────────────────────────
