# Episodic Perception — Build Specification

**A memory-augmented, perception-driven mobile robot for the Jetson Nano + companion laptop**

This document is written to be handed to an engineering agent (e.g. Claude Code) and executed **phase by phase**. Each phase has a clear goal, concrete deliverables, file/folder expectations, and an acceptance check before moving to the next phase. Do not skip ahead — later phases assume earlier phases are working and tested.

---

## 0. Project Summary

We are building a small 2WD mobile robot that:

1. **Patrols** a lab environment using a live camera feed.
2. **Detects objects** in each frame using SSD-MobileNet-V2 (not YOLO — explicitly excluded by course requirements).
3. **Notices when an object is only partially visible** (its bounding box is clipped by the frame edge) and **repositions itself** (turn / move forward / reverse) until the object is fully and stably visible. This is the **perception–action loop** and is the core novelty the project must demonstrate clearly.
4. Converts confirmed, fully-visible detections into a **scene graph** (simple spatial relationships like "bottle ON table").
5. Filters observations through an **importance-scoring** step so only new/changed/unusual events get logged (not every frame).
6. Converts important events into short **natural-language captions**, embeds them, and stores them in a **vector database** as **episodic memories**, with **compression** (merging repeated observations into duration-based episodes) and **contradiction detection** (inferring object movement instead of storing conflicting facts).
7. Answers natural-language questions about what it has seen using **retrieval-augmented generation (RAG)** over the episodic memory store.

**Compute split (edge–fog architecture):**
- **Jetson Nano (edge, on the robot):** camera capture, object detection, scene-graph construction, perception-action loop / motor control, importance-scoring, event trigger.
- **Companion laptop (fog):** captioning, embeddings, vector database, memory compression/contradiction logic, RAG query answering.
- Communication: **JSON over HTTP** between Nano and laptop over shared LAN/Wi-Fi. The Nano sends only confirmed, meaningful events — not raw video.

---

## 1. Hardware Assumptions

| Component | Spec |
|---|---|
| Edge compute | NVIDIA Jetson Nano, 4 GB RAM, JetPack installed |
| Camera | IMX219 CSI module (or USB webcam fallback) — **the Nano board has no built-in camera** |
| Mobile platform | 2WD/4WD chassis, 2× DC motors, L298N motor driver, separate battery pack for motors (do not share power rail with the Nano) |
| Fog compute | A laptop on the same LAN/Wi-Fi as the Nano |
| Network | Shared LAN or Wi-Fi hotspot connecting Nano and laptop |

**Constraint:** No YOLO (any version) — the detector must be SSD-MobileNet-V2 (or another explicitly non-YOLO lightweight detector if SSD-MobileNet-V2 proves infeasible; confirm with the user before switching).

---

## 2. Repository Structure

Set up the repo like this before writing any pipeline code:

```
episodic-perception/
├── README.md
├── requirements-nano.txt          # deps for the Jetson Nano (edge)
├── requirements-laptop.txt        # deps for the companion laptop (fog)
├── nano/                           # everything that runs ON the Jetson Nano
│   ├── camera.py                  # camera capture (CSI/USB)
│   ├── detector.py                # SSD-MobileNet-V2 inference wrapper
│   ├── scene_graph.py             # bbox relations -> symbolic triples
│   ├── perception_action.py       # bbox-clip logic + motor control state machine
│   ├── motor_control.py           # low-level GPIO/PWM motor driver interface
│   ├── importance_scoring.py      # filter: is this event worth sending?
│   ├── event_client.py            # sends confirmed events to the laptop over HTTP
│   └── main_loop.py               # ties camera -> detector -> scene graph -> perception-action -> importance -> event_client
├── laptop/                         # everything that runs on the companion laptop
│   ├── event_server.py            # FastAPI server receiving events from the Nano
│   ├── captioning.py              # scene graph -> natural language caption
│   ├── embeddings.py              # caption -> vector (sentence-transformers)
│   ├── memory_store.py            # FAISS/ChromaDB + SQLite metadata wrapper
│   ├── compression.py             # merge repeated observations into duration episodes
│   ├── contradiction.py           # detect object-moved vs. new-observation
│   ├── rag_query.py               # retrieve top-k episodes + LLM synthesis
│   └── api.py                     # user-facing query endpoint ("where is my bottle?")
├── shared/
│   └── event_schema.py            # the JSON event schema used by both sides (single source of truth)
├── tests/
│   ├── test_bbox_logic.py
│   ├── test_scene_graph.py
│   ├── test_importance_scoring.py
│   ├── test_memory_store.py
│   └── test_rag_query.py
├── data/
│   ├── raw_sessions/               # self-collected lab recordings
│   └── ground_truth_events.json    # staged event log for evaluation
└── docs/
    └── evaluation_results.md
```

Create this skeleton (empty files with docstring headers describing responsibility) as the very first step of Phase 0.

---

## 3. Shared Event Schema

This schema is the contract between the Nano and the laptop. Define it once in `shared/event_schema.py` and import it on both sides — never redefine it separately.

```json
{
  "event_id": "uuid",
  "timestamp": "2026-09-01T14:32:00Z",
  "objects": ["bottle", "table"],
  "relationships": ["bottle ON table"],
  "confidence": 0.93,
  "event_type": "new_observation | moved | partial_timeout | confirmed",
  "bbox": {"xmin": 0.12, "ymin": 0.30, "xmax": 0.44, "ymax": 0.71},
  "location_tag": "lab_desk_3"
}
```

- `event_type = "partial_timeout"` is used when the perception-action loop could not achieve a full view within the timeout window (see Phase 3) — these are logged with lower confidence rather than discarded.
- `location_tag` is a coarse, manually-configured zone label (e.g. based on patrol waypoint) — full SLAM/localization is explicitly out of scope.

---

## 4. Phased Implementation Plan

### Phase 0 — Environment & Hardware Bring-Up
**Goal:** prove the hardware chain works before any ML code is written.

- Set up JetPack, Python environment, and camera driver on the Nano; verify `libargus`/`nvarguscamerasrc` (CSI) or `v4l2` (USB) captures frames via OpenCV.
- Wire the L298N motor driver to the Nano's GPIO pins; write a minimal `motor_control.py` with functions `forward(duration_ms)`, `reverse(duration_ms)`, `turn_left(duration_ms)`, `turn_right(duration_ms)`, `stop()`. Each should use short, fixed-duration pulses (~150 ms default) — no continuous unbounded movement.
- Set up the repo skeleton from Section 2.
- Set up the laptop-side Python environment.

**Acceptance check:** camera preview works on Nano; each motor_control function reliably moves the chassis a small, predictable amount; repo skeleton exists and is committed.

---

### Phase 1 — Object Detection Pipeline (Nano)
**Goal:** real-time SSD-MobileNet-V2 inference on the Nano.

- Implement `detector.py`: load a pretrained SSD-MobileNet-V2 (COCO-pretrained is fine to start; fine-tune later once the self-collected dataset exists), run TensorRT-optimized inference on frames from `camera.py`.
- Output format: list of `{class, confidence, bbox (xmin,ymin,xmax,ymax in normalized 0-1 coords)}` per frame.
- Benchmark actual FPS on the Nano and record it — this number is needed in Phase 3 to tune motor pulse duration.

**Acceptance check:** live camera feed with detection boxes drawn on-screen (or logged), at a measured, stable FPS.

---

### Phase 2 — Scene Graph Construction (Nano)
**Goal:** convert raw detections into simple symbolic relationships.

- Implement `scene_graph.py`: given a frame's list of detected objects with bboxes, compute pairwise spatial relationships using simple geometric rules:
  - `A ON B` — A's bottom edge is within a small margin of B's top edge, and A's x-range overlaps B's x-range.
  - `A LEFT OF B` / `A RIGHT OF B` — based on relative x-centers.
  - `A BEHIND B` — based on relative bbox size/position (a proxy for depth without a depth sensor — document this as an approximation, not ground truth).
- Output: a list of triples like `["bottle", "ON", "table"]` per frame.

**Acceptance check:** unit tests in `tests/test_scene_graph.py` covering at least ON, LEFT OF, RIGHT OF with synthetic bbox inputs.

---

### Phase 3 — Perception–Action Loop (Nano) — **core novelty, build carefully**
**Goal:** implement the bbox-clip → motor-correction → confirm state machine.

State machine (see the presentation's "Perception-Action Loop" slide for the reference diagram):

```
PATROL → (partial object detected) → ADJUST → (confirmed full view, N consecutive frames) → LOG EPISODE → back to PATROL
                                         │
                                         └── (timeout, no full view) → LOG PARTIAL-CONFIDENCE EPISODE → back to PATROL
```

Bounding-box correction logic (implement exactly, then tune):

| Signal | Action |
|---|---|
| bbox clipped on LEFT edge (`xmin <= edge_margin`) | `turn_left(pulse_ms)` |
| bbox clipped on RIGHT edge (`xmax >= 1 - edge_margin`) | `turn_right(pulse_ms)` |
| bbox clipped at BOTTOM (`ymax >= 1 - edge_margin`, object too close) | `reverse(pulse_ms)` |
| bbox height below expected minimum (object too far) | `forward(pulse_ms)` |
| none of the above | object is fully visible → proceed to CONFIRM |

Guardrails (do not skip these — they prevent real, observed failure modes):
- Use **short, fixed-duration motor pulses** (default ~150 ms, tunable) rather than continuous movement, to avoid overshoot/oscillation between edges.
- If **multiple objects are partial at once**, correct toward the one with the **largest bbox area** first.
- **Tune pulse duration to the Nano's measured detection FPS from Phase 1** — if the detector runs at 5 FPS, don't issue motor pulses faster than the detector can re-observe their effect.
- **CONFIRM** requires the object to remain fully visible for **3–5 consecutive frames** before logging — this avoids logging mid-adjustment.
- **TIMEOUT** after ~6–8 seconds of failed adjustment attempts — log a `partial_timeout` event and resume patrol rather than looping forever.

Implement `perception_action.py` as an explicit state machine (a simple class with states `PATROL`, `PARTIAL`, `ADJUST`, `CONFIRM`, `LOGGED`, plus a timeout counter), not an ad-hoc set of if/else flags — this will make it far easier to debug and to explain in the report/demo.

**Acceptance check:** `tests/test_bbox_logic.py` covers each correction-signal case with synthetic bboxes. Live test: manually move a real object to the frame edge and confirm the chassis physically corrects toward it and stabilizes.

---

### Phase 4 — Importance Scoring (Nano)
**Goal:** filter confirmed observations so only meaningful events get sent to the laptop.

- Implement `importance_scoring.py`: given a new confirmed scene-graph observation and the last known state for that object/location, decide `is_new`, `is_changed`, `is_unusual`.
- Skip sending duplicate/unchanged observations. Only trigger `event_client.py` to send an event when something is new, moved, or otherwise notable.

**Acceptance check:** `tests/test_importance_scoring.py` — feed a sequence of repeated identical observations followed by a changed one; confirm only the changed one triggers a send.

---

### Phase 5 — Event Server & Nano→Laptop Communication
**Goal:** get confirmed events flowing from the Nano to the laptop.

- Implement `laptop/event_server.py` as a small FastAPI app with a `POST /event` endpoint accepting the schema from Section 3.
- Implement `nano/event_client.py` to POST events there.
- Log received events to a local file/DB on the laptop for now (real storage comes in Phase 6).

**Acceptance check:** trigger a real event on the Nano (e.g. via Phase 3's live test) and confirm it arrives and is logged on the laptop side, over the actual LAN/Wi-Fi link (not localhost-only).

---

### Phase 6 — Captioning, Embedding & Memory Store (Laptop)
**Goal:** turn incoming events into stored, semantically searchable episodic memories.

- `captioning.py`: template or small-model-based conversion of scene-graph triples into natural sentences (e.g. `["bottle", "ON", "table"]` → "A bottle is on the table.").
- `embeddings.py`: use a sentence-transformers model to embed captions.
- `memory_store.py`: wrap FAISS or ChromaDB for vector storage + SQLite (or the vector DB's own metadata store) for timestamp/confidence/location metadata. Provide `add_episode(...)` and `search(query, k)` functions.

**Acceptance check:** `tests/test_memory_store.py` — add several synthetic episodes, run a semantic query with different wording than what was stored (e.g. stored "blue flask beside laptop", query "where's my water bottle"), confirm the right episode is retrieved.

---

### Phase 7 — Memory Compression & Contradiction Detection (Laptop)
**Goal:** implement the two "twist" behaviors that make this more than a plain logger.

- `compression.py`: when a new observation matches an existing unclosed episode for the same object/location, extend that episode's duration instead of creating a new record (e.g. "bottle on Desk 3, 10:00–10:30" instead of 30 separate identical rows).
- `contradiction.py`: when an object's new location conflicts with its last known location, infer and log a "moved" event (`event_type: "moved"`) rather than storing both as unrelated facts.

**Acceptance check:** feed a synthetic sequence — same object repeated at location A (should compress into one episode), then object appears at location B (should trigger a "moved" event, not a duplicate).

---

### Phase 8 — RAG Query Answering (Laptop)
**Goal:** answer natural-language questions using the episodic memory.

- `rag_query.py`: given a user question, embed it, retrieve top-k relevant episodes from `memory_store.py`, and pass them + the question to an LLM (local model or API call — confirm with the user which is available) to synthesize a grounded natural-language answer.
- `api.py`: expose a simple endpoint or CLI to ask questions, e.g. "where's my charger?" → retrieves + answers.

**Acceptance check:** `tests/test_rag_query.py` with a small seeded memory store and a few sample questions; manually verify answers are grounded in retrieved episodes (no hallucinated details not present in stored memories).

---

### Phase 9 — Self-Collected Dataset & Evaluation
**Goal:** produce the numbers needed for the report (compression ratio, retrieval accuracy, perception-action success rate, latency).

- Record 15–20 sessions in the actual lab environment (~2,000–3,000 frames, ~10 object categories), including ~50 staged object-movement events, with a manually-written ground-truth event log (`data/ground_truth_events.json`).
- Run the full pipeline end-to-end against these sessions.
- Compute and record in `docs/evaluation_results.md`:
  - **Compression ratio** — raw frames processed vs. episodes actually stored.
  - **Retrieval accuracy** — precision/recall of RAG answers against ground-truth events.
  - **Perception-action success rate** — % of partial-view detections successfully corrected to a full view vs. timed out.
  - **End-to-end query latency** — from question asked to answer returned.

**Acceptance check:** `docs/evaluation_results.md` contains all four numbers with a short method description for how each was measured.

---

### Phase 10 — Polish, Documentation, Demo Prep
**Goal:** package everything for submission and live demo.

- Write/finalize `README.md` with setup instructions for both the Nano and laptop sides.
- Prepare a short demo script: patrol → object partially visible → robot self-corrects → episode logged → ask a natural-language question → get an answer.
- Cross-check the final report/slides against what was actually built (don't let the presentation claim capabilities that were descoped along the way — if something from this spec didn't get built, say so honestly in the report's limitations section).

**Acceptance check:** a full run-through of the demo script works live, start to finish, without manual intervention beyond starting the two processes.

---

## 5. Explicit Non-Goals (do not build these — descope if suggested)

- Full SLAM / precise localization — patrol uses scripted movement or coarse zone tags only.
- True 3D depth reasoning — "BEHIND" relationships are a bbox-size/position heuristic, not real depth.
- Any YOLO-family model — excluded by course requirement.
- Continuous/unbounded motor movement — always short, fixed-duration pulses.
- Multi-robot or multi-camera support.

---

## 6. Notes for the Coding Agent

- Work through phases **in order**. Each phase's acceptance check should genuinely pass — with real hardware where hardware is involved — before moving on.
- When a phase's acceptance check can't be verified without physical hardware in front of you, implement it fully, write the tests that can run without hardware, and clearly flag what still needs a live hardware test from the user.
- Keep `shared/event_schema.py` as the single source of truth; never let the Nano and laptop sides drift into incompatible event formats.
- If SSD-MobileNet-V2 proves difficult to get running well on the Nano, stop and flag it rather than silently substituting YOLO — the course has explicitly excluded YOLO.
- Favor small, testable functions (especially in `perception_action.py` and `importance_scoring.py`) — these are the modules most likely to need tuning after real-world testing.
