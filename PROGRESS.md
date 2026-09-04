# Episodic Perception — Progress So Far

Read this alongside `episodic_perception_build_spec.md` (the full build
spec) before starting new work. This file exists so a fresh Claude Code
session — or any new contributor — can pick up exactly where things left
off without re-deriving context.

## Who owns what

- **Nano / edge, Phases 0-5** (camera, SSD-MobileNet-V2 detection, scene
  graph, perception-action loop, motor control, importance scoring, event
  client): **not yet implemented.** Files exist only as stubs with
  docstring headers describing responsibility — see `nano/*.py`. This is
  the next work to do.
- **Laptop / fog, Phases 6-10** (captioning, embeddings, memory store,
  compression, contradiction detection, RAG query answering, evaluation
  tooling): **implemented and tested**, described below.

## What's built and passing (laptop side, Phases 6-8)

Run `pytest tests/` from the repo root — 11 tests pass.

- **`shared/event_schema.py`** — the single source of truth JSON event
  contract (pydantic `Event`, `BBox`, `EventType`). Both Nano and laptop
  code must import from here, never redefine the shape independently.
- **`laptop/captioning.py`** — converts scene-graph triples (e.g.
  `["bottle", "ON", "table"]`) into natural sentences. Handles `moved` and
  `partial_timeout` event types with different phrasing.
- **`laptop/embeddings.py`** — sentence-transformers (`all-MiniLM-L6-v2`)
  wrapper: `embed()` / `embed_batch()`.
- **`laptop/memory_store.py`** — ChromaDB-backed vector store
  (persists to `data/memory_store/`). `add_episode()`, `add_episodes()`,
  `search()`, `count()`. Verified with a real semantic-search test: a query
  worded differently than the stored caption still retrieves the right
  episode.
- **`laptop/compression.py`** — `EpisodeCompressor`: merges repeated
  observations of the same object/location/relationship into one
  duration-based episode (extends `end_time`) instead of storing one row
  per frame. Episodes are keyed by `(sorted objects, location_tag, sorted
  relationships)`, with a configurable gap-tolerance (default 5 min)
  between observations to still count as the same episode.
- **`laptop/contradiction.py`** — `LocationTracker`: tracks each object's
  last known location; when a new observation of that object shows a
  different location, retypes the event to `event_type: "moved"` instead
  of letting it look like an unrelated new fact. **Simplifying assumption:
  `event.objects[0]` is treated as the tracked/movable subject**, later
  entries as reference objects (e.g. "table" in "bottle ON table") — this
  mirrors how `scene_graph.py` is expected to order its triples once built.
- **`laptop/rag_query.py`** — `answer_question(question, store, k)`:
  retrieves top-k episodes and synthesizes an answer. Default synthesis
  (`extractive_synthesize`) is template-based and directly quotes retrieved
  captions — zero hallucination risk, no external API needed. An optional
  `anthropic_synthesize` hook exists for nicer prose if `ANTHROPIC_API_KEY`
  is set and the `anthropic` package is installed, but it is **not**
  required for the pipeline to work.
- **`laptop/api.py`** — FastAPI app (`GET /query?q=...`) plus a CLI:
  `python -m laptop.api "where's my charger?"`.
- **`laptop/evaluate.py`** — metric functions for Phase 9's four required
  numbers (compression ratio, retrieval precision/recall, perception-action
  success rate, query latency stats), unit-tested against synthetic
  numbers in `tests/test_evaluate.py`.

## What's explicitly NOT done, and why

- **Phase 9's real numbers are not filled in.** `docs/evaluation_results.md`
  says so explicitly. They require 15-20 real lab recordings from the
  physical robot (`data/raw_sessions/`, `data/ground_truth_events.json`),
  which needs Phases 0-5 finished and integrated first. `laptop/evaluate.py`
  is ready to consume that data the moment it exists — do not fabricate
  placeholder numbers here.
- **Phase 10's live demo has not been run.** `docs/demo_script.md` has the
  walkthrough script, but it needs both sides running against real
  hardware.
- **`nano/*.py` are stubs only** (docstring headers, no logic) — this is
  the next contributor's starting point. Follow
  `episodic_perception_build_spec.md` Section 4, Phases 0-5, in order.
  Read Section 3 (shared event schema) and Section 5 (explicit non-goals —
  no YOLO, no full SLAM, no true depth, short fixed-duration motor pulses
  only) before writing code.

## Environment notes for whoever continues this

- Laptop-side deps: `requirements-laptop.txt` (fastapi, uvicorn, pydantic,
  sentence-transformers, chromadb, requests). A `.venv/` was set up and
  `pip install -r requirements-laptop.txt` completes cleanly.
- Nano-side deps: `requirements-nano.txt` is currently just a placeholder
  comment — fill it in as Phase 0/1 work happens (opencv-python,
  jetson-inference/TensorRT bindings, Jetson.GPIO, requests, pydantic, etc).
- Repo root has `laptop/__init__.py`, `shared/__init__.py`, `nano/__init__.py`
  so `from laptop.x import y` / `from shared.event_schema import Event`
  absolute imports work when running pytest from the repo root.
- `.gitignore` excludes `.venv/`, `data/memory_store/` (ChromaDB's local
  persisted files), and `data/raw_sessions/` contents.
