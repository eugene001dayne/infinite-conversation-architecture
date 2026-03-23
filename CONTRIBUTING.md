# Contributing to the Infinite Conversation Architecture

Thank you for being here. This project was started by one person who got frustrated every time a conversation hit a wall and ended — and decided to think seriously about why that happens and how to fix it. If you are reading this, you probably felt the same thing.

This is an open research project. The architecture is designed, the codebase is started, and the research gaps are documented. What it needs now is people who want to build, test, critique, and improve it.

---

## What kind of contributions matter

### Engineering
The codebase in `core/` is a solid foundation but several components are placeholders that need production implementations:

- **CONTRADICTS and REFERENCES edge detection** — currently keyword heuristics. Needs a lightweight semantic classifier. This is the highest-priority engineering gap.
- **Open question detection** — currently keyword overlap. Should use embedding similarity.
- **Graph database integration** — the architecture recommends FalkorDB or Kuzu for the active tier but no database adapter exists yet. Building one would be a significant contribution.
- **State document persistence** — cross-session identity requires the State Document to be saved and loaded. No persistence layer exists yet.

### Research
The `research/RESEARCH_GAPS.md` document identifies four specific workstreams where published benchmarks do not exist:

- Pre-fetch during typing vs post-submit retrieval (latency and accuracy)
- Graph database benchmark for the conversational workload (write + edge detect + depth-4 traversal at 10K–100K nodes)
- State document accuracy with and without the self-correction loop
- LoCoMo benchmark evaluation against Mem0g and Graphiti baselines

If you are a researcher, any one of these produces a publishable result in an area with no prior work.

### Testing
The test suite in `tests/` covers the NER engine. Nothing else has tests yet. Every module in `core/` needs coverage.

### Documentation
The `docs/` folder has deep writeups but no tutorials, no quickstart guide, and no worked examples beyond `examples/example_turn.py`.

---

## How to get started

### 1. Read first

Before contributing code, read these in order:

- `README.md` — the full architecture proposal
- `docs/ARCHITECTURE.md` — the technical deep dive
- `core/schemas.py` — all data structures, start here before touching anything else
- `research/FINDINGS.md` — the literature review so you know what already exists

### 2. Set up

```bash
git clone https://github.com/eugene001dayne/infinite-conversation-architecture
cd infinite-conversation-architecture
pip install spacy yake vaderSentiment
python -m spacy download en_core_web_sm
```

Run the example to confirm everything works:

```bash
python examples/example_turn.py
```

Run the test suite:

```bash
pytest tests/test_ner_engine.py -v
```

### 3. Pick something to work on

Check the open GitHub Issues — each one corresponds to a specific, scoped task. If you want to work on something not listed, open an issue first and describe what you want to do. This avoids duplicate work and lets us discuss the approach before you invest time in it.

---

## Contribution standards

### Code

- Python 3.11+
- Follow the existing style in `core/` — type hints, docstrings, clear variable names
- Every new function needs a docstring explaining what it does, its parameters, and what it returns
- Every new module needs tests in `tests/`
- Run the existing test suite before submitting — nothing should break

### Research

- Document your methodology clearly
- If you run a benchmark, include the raw data and the code used to produce it in `research/`
- Cite your sources — the project takes intellectual honesty seriously

### Pull requests

- One thing per PR — a focused change is easier to review than a sprawling one
- Write a clear description of what you changed and why
- Reference the Issue number your PR closes
- Be patient — this is a small project and reviews may take time

---

## What this project is not looking for

- Rewrites of existing components without discussion — if you think something should be redesigned, open an issue first
- Changes that add external API dependencies — everything must run locally without API keys
- Contributions that remove attribution — Eugene Mawuli Attigah is the originator of this architecture and that credit stays in the codebase

---

## Credit and attribution

All contributors are credited in the commit history. Significant contributions — a working database adapter, a published benchmark, a production implementation of a placeholder component — will be acknowledged explicitly in the README.

This project is licensed CC BY 4.0. You are free to build on it, fork it, and use it in your own work as long as you give attribution.

---

## Questions

Open a GitHub Issue with the label `question`. There are no stupid questions here — this is research and the whole point is to figure things out together.

---

*This project was originated by Eugene Mawuli Attigah. Aided by ConsciousNet methodology.*
