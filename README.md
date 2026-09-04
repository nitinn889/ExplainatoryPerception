# Episodic Perception

Memory-augmented, perception-driven mobile robot. See
[episodic_perception_build_spec.md](episodic_perception_build_spec.md) for
the full phased build spec, and [PROGRESS.md](PROGRESS.md) for exactly
what's been built so far and what's next — read that before starting new
work, and if you're using Claude Code, ask it to read PROGRESS.md first.

Split across two contributors:
- **Nano / edge (Phases 0-5):** `nano/` — camera, SSD-MobileNet-V2 detection,
  scene graph, perception-action loop, importance scoring, event client.
- **Laptop / fog (Phases 6-10):** `laptop/` — captioning, embeddings, vector
  memory store, compression, contradiction detection, RAG query answering.

`shared/event_schema.py` is the single source of truth for the JSON event
contract between the two sides — never redefine it independently.

## Laptop-side setup

```
python -m venv .venv
./.venv/Scripts/activate        # or source .venv/bin/activate on Nano/Linux
pip install -r requirements-laptop.txt
pytest tests/
```

Ask questions from the CLI once the store has events in it:

```
python -m laptop.api "where's my charger?"
```

Or run the query API: `uvicorn laptop.api:app --reload` then `GET /query?q=...`.

## Status

- [x] Phase 6 — Captioning, embeddings (sentence-transformers), memory store
      (ChromaDB, persisted to `data/memory_store/`)
- [x] Phase 7 — Compression (`laptop/compression.py`) & contradiction
      detection (`laptop/contradiction.py`)
- [x] Phase 8 — RAG query answering (`laptop/rag_query.py`, `laptop/api.py`) —
      default synthesis is extractive/grounded with zero hallucination risk;
      swap in `anthropic_synthesize` (needs `ANTHROPIC_API_KEY` + the
      `anthropic` package) for nicer prose.
- [ ] Phase 9 — Self-collected dataset & evaluation — **metric tooling is
      built and unit-tested** (`laptop/evaluate.py`), but the actual numbers
      in `docs/evaluation_results.md` are still pending: they require 15-20
      real lab recordings from the physical robot, which needs Phases 0-5
      (Nano side) finished and integrated first.
- [ ] Phase 10 — Polish, docs, demo prep — README and repo skeleton are in
      place; the live end-to-end demo run-through still needs the actual
      hardware and the Nano-side pipeline.

All laptop-side (Phase 6-8) tests pass: `pytest tests/` — 11 passed.

## What's still blocking full completion

This is a two-person build. Everything on the laptop side (Phases 6-8) is
implemented, tested, and working standalone. What's still outstanding is
**not** laptop-side code — it's:

1. **Nano-side implementation** (Phases 0-5, camera/detector/scene
   graph/perception-action loop/motor control/event client) — stubbed here
   with docstrings only, owned by the other contributor.
2. **Real dataset collection** (Phase 9) — requires the physical robot and
   lab environment; can't be fabricated without misrepresenting the
   evaluation numbers in the report.
3. **Live integrated demo** (Phase 10) — requires both sides running
   together against real hardware.
