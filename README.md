# Episodic Perception

Memory-augmented, perception-driven mobile robot. See
[episodic_perception_build_spec.md](episodic_perception_build_spec.md) for
the full phased build spec, and [PROGRESS.md](PROGRESS.md) for exactly
what's been built so far and what's next — read that before starting new
work, and if you're using Claude Code, ask it to read PROGRESS.md first.

## Run the live demo

```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements-laptop.txt -r requirements-nano.txt
python -m demo.run_demo
```

Open <http://localhost:8080>. A simulated robot patrols a three-station lab,
recovers partially-visible objects by repositioning, logs episodes, and answers
questions about what it has seen — the whole concept, end to end, in a browser.

For the higher-fidelity Webots version (rendered camera frames, rigid-body
physics, optionally the real SSD-MobileNet-V2 weights in the loop):

```bash
python -m demo.run_demo --no-sim      # dashboard, one terminal
./scripts/run_webots_demo.sh          # Webots, another
```

Both drive the *same* `nano/` and `laptop/` modules the real robot runs.

- **[`docs/DEMO_HOWTO.md`](docs/DEMO_HOWTO.md)** — step-by-step setup, what to
  show and in what order, and the questions you will get asked. Start here if
  you are presenting this.
- [`demo/README.md`](demo/README.md) — how the two demos differ, and an honest
  list of what they do not do.
- [`docs/webots_demo_results.md`](docs/webots_demo_results.md) — measured
  results.

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
- [x] Phase 10 — Polish, docs, demo prep — two runnable live demos
      (`demo/` in a browser, `webots/` in the Webots simulator), both driving
      the real `nano/` and `laptop/` modules end to end. See
      [`demo/README.md`](demo/README.md) for the presentation script and
      [`docs/webots_demo_results.md`](docs/webots_demo_results.md) for measured
      results. **Still outstanding:** the same run-through on the physical
      chassis.

All laptop-side (Phase 6-8) tests pass: `pytest tests/` — 11 passed.

## What's still blocking full completion

This is a two-person build. Everything on the laptop side (Phases 6-8) is
implemented, tested, and working standalone. What's still outstanding is
**not** laptop-side code — it's:

1. **Hardware bring-up of the Nano side** (Phases 0-5) — the code is
   implemented and tested on host, but it has not been run on the physical
   Jetson Nano, CSI camera and L298N driver.
2. **Real dataset collection** (Phase 9) — requires the physical robot and
   lab environment; can't be fabricated without misrepresenting the
   evaluation numbers in the report. The simulated runs in
   `docs/webots_demo_results.md` exercise the same metric code and give
   numbers to compare against, but they are not the self-collected dataset
   the spec asks for.
3. **Live demo on hardware** (Phase 10) — the integrated demo now runs in
   simulation, both in the browser and in Webots. Running it on the real
   chassis, camera and LAN hop is what is left.
