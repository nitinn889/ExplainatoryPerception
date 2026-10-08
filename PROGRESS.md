# Episodic Perception — Progress So Far

Read this alongside `episodic_perception_build_spec.md` (the full build
spec) before starting new work. This file exists so a fresh Claude Code
session — or any new contributor — can pick up exactly where things left
off without re-deriving context.

## Who owns what

- **Nano / edge, Phases 0-5** (camera, SSD-MobileNet-V2 detection, scene
  graph, perception-action loop, motor control, importance scoring, event
  client, event server, main loop): **implemented, tested, and passing.**
  Designed with dual hardware/host capability so code runs directly on standard
  systems and ports without modification onto the physical Jetson Nano.
- **Laptop / fog, Phases 6-8** (captioning, embeddings, memory store,
  compression, contradiction detection, RAG query answering, evaluation
  tooling): **implemented and tested**, described below.

## What's built and passing (Phases 0-5 Edge & Phases 6-8 Fog)

Run `pytest` from the repo root — 20 tests pass.

### Edge Pipeline (Nano, Phases 0-5)
- **`nano/motor_control.py`** (Phase 0) — L298N motor driver using fixed-duration
  short pulses (default 150ms). Automatically detects `Jetson.GPIO` and falls back
  to a mock driver logging pulses when running on host systems.
- **`nano/camera.py`** (Phase 0/1) — Unified camera interface supporting CSI
  (`nvarguscamerasrc` GStreamer pipeline), USB (`v4l2`), and synthetic frames
  with configurable mock objects for headless testing.
- **`nano/detector.py`** (Phase 1) — SSD-MobileNet-V2 inference wrapper (no YOLO).
  Supports Jetson TensorRT (`jetson.inference.detectNet`), OpenCV DNN, and
  simulation backends. Normalized 0-1 bbox contract, plus FPS benchmarking helper.
- **`nano/scene_graph.py`** (Phase 2) — Converts detections into spatial triples
  (`ON`, `LEFT OF`, `RIGHT OF`, `BEHIND`) using geometric rules and surfaces,
  guaranteeing movable subjects are ordered first.
- **`nano/perception_action.py`** (Phase 3) — Core novelty perception-action state
  machine (`PATROL -> PARTIAL -> ADJUST -> CONFIRM -> LOGGED`). Implements
  left/right/bottom-clip, too-far signals, largest-area prioritization, 3-frame
  consecutive stabilization, and 7-second timeout guardrails.
- **`nano/importance_scoring.py`** (Phase 4) — Filters redundant repetitive frames;
  only triggers upstream events when an object is newly observed, moved, changed
  relationship, significantly displaced, or timed out.
- **`laptop/event_server.py` & `nano/event_client.py`** (Phase 5) — FastAPI receiver
  at `POST /event` and robust HTTP client transmitting `shared.event_schema.Event`
  objects over LAN/Wi-Fi with logging and healthcheck endpoints.
- **`nano/main_loop.py`** — Orchestrator tying camera -> detector -> perception-action
  loop -> scene graph -> importance filter -> fog event client, with `--benchmark`
  and `--simulate` scenarios for one-command verification.

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

## Live demo (added after Phases 0-8)

Two runnable demos, both driving the **same** `nano/` and `laptop/` modules —
nothing is reimplemented:

- **`demo/`** — one command (`python -m demo.run_demo`), browser dashboard at
  `localhost:8080`. A 2D kinematic lab sim with a real pinhole camera model
  feeds the real perception-action loop; the dashboard shows the camera view
  with live bounding boxes and edge margins, the lab map, the state machine, the
  event JSON, the captions, the episodic memory and a RAG query box.
- **`webots/`** — `webots/worlds/episodic_lab.wbt` plus
  `webots/controllers/episodic_robot/`. A 2WD chassis with a mast camera in a
  three-station lab, built from primitive nodes only (no `EXTERNPROTO`, so it
  loads offline). `WebotsMotorAdapter` exposes the same five-method interface as
  `nano/motor_control.py` and uses the same short fixed-duration pulses.
  `./scripts/run_webots_demo.sh` launches it; `--headless` and `--record` work
  for servers and CI.

Read `demo/README.md` first — it has the presentation script and a frank list of
what the demos do not do. `docs/webots_demo_results.md` has the measured
numbers.

### Changes to Phase 0-8 code that the demo required

All additive; existing behaviour and tests unchanged unless noted.

- **`nano/detector.py`** — the OpenCV DNN path now actually works. It loads the
  COCO-pretrained SSD-MobileNet-V2 TensorFlow graph via
  `cv2.dnn.readNetFromTensorflow` with the right preprocessing, auto-discovers
  the weights `scripts/fetch_ssd_model.sh` downloads, and picks the label set
  from the weight format. Previously it applied Caffe-style preprocessing and
  then indexed the result into the COCO list, which mislabels every detection
  when used with the Caffe VOC weights it was set up for. Verified end to end:
  0.99 "person" and 0.98 "sports ball" on a photograph.
- **`nano/perception_action.py`** — new optional `ignore_classes`. A 1.5 m desk
  viewed from 1.3 m is permanently clipped, always has the largest bounding
  box, and can never be made fully visible by repositioning, so the spec's
  "correct toward the largest partial object first" guardrail fixates on
  furniture and times out at every station. Both demos pass
  `SURFACE_CLASSES`; the default is `None`, so Phase 3 behaviour is unchanged.
  **Worth a line in the report's limitations section.**
- **`laptop/embeddings.py`** — falls back to a deterministic lexical embedder
  when `all-MiniLM-L6-v2` cannot be downloaded, instead of raising at import.
  `active_backend()` reports which is live so nothing over-claims semantic
  retrieval. `tests/test_memory_store.py`'s cross-wording test now skips on the
  fallback rather than failing.
- **`laptop/memory_store.py`** — added `update_episode()`, so Phase 7's
  compression (extending an episode's `end_time`) actually reaches storage
  rather than leaving the stored row stuck at its start time. `add_episode()`
  takes an optional `display_caption`, keeping the clean sentence separate from
  the text that gets embedded.
- **`laptop/rag_query.py`** — `extractive_synthesize` now answers the question
  first and lists supporting observations after, instead of dumping a ranked
  list. Still purely extractive, so still zero hallucination risk.
- **`demo/fog_pipeline.py`** — new: the glue that runs Phases 6-8 in order on
  each incoming event (contradiction → caption → compression → store). The
  modules existed and were tested individually; nothing ran them as a pipeline.

## Voice interface (added after the demo)

Asking by speaking and being answered out loud, **added to** the dashboard
rather than replacing any of it. Every existing path — the typed box, the
suggestion chips, the control buttons, the whole display — is unchanged and
still works with no microphone at all.

- **`laptop/voice.py`** — new, and the only new module. Two pure functions:
  - `route_utterance(text)` decides whether an utterance is a control command
    ("pause", "start patrol") or a question for episodic memory. Question
    shape is checked *first*, because plain keyword matching sends "did
    anything move?" to the move-the-bottle command instead of to retrieval.
  - `spoken_answer(result)` re-phrases a `rag_query` result for being read
    aloud: no bulleted "other observations" list, no `06:20:06` timestamps,
    locations as words ("lab desk 3", not `lab_desk_3`), times as "about four
    minutes ago". It re-orders and re-words the retrieved episodes' own text
    and never adds to it, so the spoken answer carries the same
    zero-hallucination guarantee as the displayed one.
- **`demo/server.py`** — new `POST /voice`, which takes a transcript, routes
  it, and returns either a control result or a full query result, each with a
  `spoken` string. The control logic moved out of the `/control/{action}`
  handler into `_apply_control` so a spoken "pause" and the Pause button
  cannot drift apart. `GET /query` and `laptop/api.py` now also return
  `spoken`, so a *typed* question can be read aloud too.
- **`demo/static/dashboard.html`** — a microphone button next to Ask, a "speak
  answers" toggle, and a status line showing what was heard. Recognition and
  synthesis are the browser's Web Speech API; no new Python dependency, and
  nothing to install before a presentation.
- **`tests/test_voice.py`** — 43 tests: routing (including the "did anything
  move?" trap), relative-time phrasing, clock skew between edge and fog,
  unparseable timestamps, and the `/voice` endpoint for questions, commands,
  commands with no sim attached, and silence.

Deliberately **not** done: on-device speech. Chrome's recognition is a cloud
service, so the microphone needs a network even though the rest of the demo
does not. Running Vosk or whisper.cpp on the Nano would fix that and would also
be the right answer for the physical robot, which has no browser. The phrasing
and routing in `laptop/voice.py` are already independent of the browser, so
that work is a new front end for them, not a rewrite.

## What's explicitly NOT done, and why

- **Phase 9's real numbers are not filled in.** `docs/evaluation_results.md`
  says so explicitly. They require 15-20 real lab recordings from the
  physical robot (`data/raw_sessions/`, `data/ground_truth_events.json`),
  which needs Phases 0-5 finished and integrated first. `laptop/evaluate.py`
  is ready to consume that data the moment it exists — do not fabricate
  placeholder numbers here.
- **Phase 10's live demo now runs in simulation, not on hardware.** Both
  `demo/` and `webots/` run the full chain end to end and are reproducible
  (`docs/webots_demo_results.md`). What they are not is a run on the physical
  robot: `docs/demo_script.md`'s hardware walkthrough still needs the real
  chassis, the real camera and the real LAN hop.
- ~~**`nano/*.py` are stubs only**~~ — out of date, the edge side is
  implemented (see the Phases 0-5 section above). For the original plan, follow
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
