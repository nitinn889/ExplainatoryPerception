# Live demo

**Presenting this to someone? Read
[`../docs/DEMO_HOWTO.md`](../docs/DEMO_HOWTO.md) instead — it is the
step-by-step version.**

Two ways to put the whole concept on a screen. They run the **same** edge and
fog code — the Jetson Nano's own modules from `nano/` and the laptop's from
`laptop/`. Only the robot's body differs.

| | `demo/` (browser) | `webots/` (simulator) |
|---|---|---|
| Install | nothing beyond `requirements-laptop.txt` | Webots R2023b+ (~1 GB) |
| Start-up | ~2 s | ~20 s |
| Camera | pinhole projection, drawn in the browser | real rendered frames |
| Detector | geometric ground truth | ground-truth projection, **or** the real SSD-MobileNet-V2 |
| Physics | differential-drive kinematics | full rigid-body (ODE) |
| Shows the fog side | yes — captions, memory, compression, RAG, all live | streams into the same dashboard |

**Use the browser demo to present**, and Webots when someone asks whether it
works on a real robot with a real detector. Webots was the original ask; the
browser demo is the better answer to "show the concept live", because it shows
the half of the system Webots cannot — the memory and the question answering —
and it cannot fail on a projector in a room with no network.

---

## 1. Browser demo

```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements-laptop.txt -r requirements-nano.txt
python -m demo.run_demo
```

Open <http://localhost:8080>. The robot starts patrolling immediately.

Useful flags: `--fps 6` (simulated detector frame rate), `--time-scale 0.2`
(speed the patrol up for a rehearsal), `--laps 3`, `--port 8080`.

### What you are looking at

* **Camera view** — every box is a projection of a real object at real
  coordinates. The red bands are the `edge_margin`: a box reaching one is what
  the Phase 3 clip test fires on. The purple arrow at the bottom is the motor
  pulse the loop just issued.
* **Lab map** — the robot's pose, its field-of-view wedge, the three patrol
  stations and everything in the room.
* **PATROL → PARTIAL → ADJUST → CONFIRM → LOGGED** — the actual state of
  `nano/perception_action.py`, not a re-implementation.
* **Pipeline** — the last logged observation's trip through the fog side:
  scene-graph triples, the event JSON that crossed the wire, the caption, and
  whether it opened a new episode or extended an existing one.
* **Memory** — stored episodes, and the counters the report needs.
* **Ask the robot** — Phase 8 RAG over the episodic memory, by keyboard or by
  voice. The microphone button transcribes what you say in the browser and
  POSTs the text to `/voice`, which decides whether it was a question for
  memory or a control command (`laptop/voice.py`), then speaks the answer
  back. The panel below still shows the full grounded answer and the retrieved
  episodes, so the audience can see what the spoken sentence came from.

### The five-minute script

1. **Patrol.** Point out the frame counter climbing while the episode counter
   does not. That gap is the importance filter.
2. **The core novelty.** At `lab_desk_3` the bottle sits in the right-hand
   margin band. Watch `ADJUST`, the `turn_right` pulses, the box sliding
   inward, then `CONFIRM` holding for four consecutive frames before anything
   is written down. At `lab_desk_1` the laptop is too small instead of
   off-centre, so the correction is `forward`; at `lab_shelf_2` the bottle is
   clipped on the *left*, so it is `turn_left`.
3. **Episodic memory.** An episode appears with a caption and a location tag —
   not a frame, not a bounding box.
4. **Contradiction detection.** The patrol relocates the bottle while the robot
   is at desk 1. Next time it reaches the shelf, the episode is typed `moved`
   rather than stored as an unrelated new fact. (The **Move the bottle** button
   does this on demand.)
5. **Compression.** Revisiting a place the robot has already described extends
   that episode — the count goes to `2×`, `3×`, with a duration, instead of one
   row per sighting.
6. **Ask it something.** "where is the bottle?" → the latest location, with the
   retrieved episodes and the measured latency.
7. **Ask it out loud.** Click the microphone and say the same question. The
   transcript appears in the box, the same retrieval runs, and the answer is
   read back. Then say "pause" — a spoken command runs the same action as the
   Pause button, because `/voice` and `/control` share one code path.

---

## 2. Webots demo

```bash
python -m demo.run_demo --no-sim      # dashboard only, in one terminal
./scripts/run_webots_demo.sh          # Webots, in another
```

The controller POSTs to the same `/event` and `/telemetry` endpoints, so the
dashboard fills in exactly as above, with the real rendered camera frames in the
camera panel.

```bash
./scripts/run_webots_demo.sh --detector ssd    # real SSD-MobileNet-V2 in the loop
./scripts/run_webots_demo.sh --headless        # no X server
./scripts/run_webots_demo.sh --record out.mp4  # headless + video
```

`webots/worlds/episodic_lab.wbt` is built from primitive Webots nodes only — no
`EXTERNPROTO`, so it loads with no network access. The robot is a 2WD chassis
with a camera on a mast, matching the real hardware, and
`WebotsMotorAdapter` exposes the same
`forward/reverse/turn_left/turn_right/stop` interface as the L298N driver in
`nano/motor_control.py`, with the same short fixed-duration pulses. Issuing a
pulse advances simulation time, so the loop's 7-second timeout means the same
thing here as on the chassis.

### Detector choice

`--detector groundtruth` (default) projects tight bounding boxes from the
simulator's own geometry. It is deterministic, which is what you want in front
of an audience, and the dashboard labels it
`simulated detector (tight ground-truth projection)` so nobody mistakes it for
CNN output.

(Webots' `Recognition` node looks like the obvious choice and was the first
thing this used, but it reports the projection of a bounding *sphere*. For a
1.5 m desk at 1.3 m that fills the whole frame, which breaks both the clip test
and the `ON` relation. The controller's docstring has the measured numbers.)

`--detector ssd` runs the real COCO-pretrained SSD-MobileNet-V2 through
`nano/detector.py` over the rendered frames. Fetch the weights first:

```bash
./scripts/fetch_ssd_model.sh
```

**It does not work well on this world, and that is measured, not assumed.** On
the camera frame at `lab_desk_3`, SSD-MobileNet-V2 returns exactly one
detection at any threshold down to 0.15: `bench 0.85` covering the desk. It
misses the bottle and the vase entirely. The same weights score 0.99 "person"
and 0.98 "sports ball" on photographs, so this is the *renderer* — a
COCO-trained network has never seen a bottle that looks like a smooth blue
cylinder with no label and no texture. The frame and the full output are in
[`../docs/webots_demo_results.md`](../docs/webots_demo_results.md).

To show the detector working on real imagery, point it at a webcam:
`python -m nano.main_loop --mode usb --display`.

---

## Honest limitations

Things this demo does not do, so nothing in a report has to over-claim:

* **The 2D sim has no pixels.** It cannot exercise the real detector; it
  projects geometry. Only the Webots mode can run SSD-MobileNet-V2 on images.
* **No occlusion in the 2D sim.** The lab is arranged so nothing meaningfully
  occludes anything, and overlapping objects are drawn nearest-last rather than
  reasoned about.
* **Patrol is scripted, not navigated.** Waypoint following reads the
  simulator's ground-truth pose. This stands in for the spec's "scripted
  movement or coarse zone tags"; no SLAM, no localisation, and nothing in the
  perception path ever sees a pose.
* **Object movement is staged.** The bottle is relocated by the demo, which is
  the simulated form of the spec's staged object-movement events — not the
  robot inferring that someone walked in.
* **Flat objects break the "too far" rule.** A 3 cm-tall book can never satisfy
  a fixed `min_height`, so the loop reads it as too far and drives forward
  indefinitely. The lab deliberately contains no books, and a per-class
  expected height would be the real fix.
* **Surfaces had to be excluded as targets.** A 1.5 m desk viewed from 1.3 m is
  permanently clipped, always has the largest bounding box, and can never be
  made fully visible by repositioning — so the spec's "correct toward the
  largest partial object first" rule fixates on the furniture and times out at
  every station. Both demos pass `ignore_classes=SURFACE_CLASSES` to
  `PerceptionActionStateMachine`; the default behaviour is unchanged. This came
  out of running the loop against faithful projection geometry and is worth a
  line in the report's limitations section.
* **The `ON` test needs a looser margin in Webots than the library default.**
  It compares an object's bottom edge against the surface's bbox *top* edge,
  but perspective puts a deep surface's projected top edge at its **far** edge,
  not at the contact point — so the gap grows the closer the camera is and the
  further forward on the desk the object sits. Measured at `lab_desk_3`: the
  vase (1.46 m away) is 0.017 off, the bottle (0.82 m away) is 0.118 off. The
  Webots controller passes `y_margin=0.15`; a margin derived from the surface's
  apparent depth would be the principled fix. The 2D sim does not hit this
  because its objects have no depth.
* **Embeddings may be the lexical fallback.** `all-MiniLM-L6-v2` has to be
  downloaded from HuggingFace on first use. Where that is unavailable,
  `laptop/embeddings.py` falls back to a deterministic hashed-token embedder.
  Retrieval then matches on words rather than meaning — "blue flask" will not
  match "water bottle". The dashboard shows which backend is live, and
  `tests/test_memory_store.py` skips the cross-wording test rather than
  pretending the fallback passes it.
* **Voice input is Chrome and Edge only, and not local.** Recognition uses the
  browser's Web Speech API. Firefox does not implement it, so the microphone
  button is disabled there with a note — typing is unaffected, and answers are
  still spoken, because speech *synthesis* is supported everywhere. Chrome's
  recognition is also a cloud service: the audio leaves the machine, so the
  "works on a projector with no network" claim covers the dashboard and the
  typed path, not the microphone. On-device recognition on the Nano itself
  (Vosk, whisper.cpp) would be the fix, and is not built.
* **Spoken commands are a fixed phrase table, not language understanding.**
  `laptop/voice.py` matches a handful of phrasings per action and routes
  anything question-shaped to retrieval instead. "Stop what you are doing"
  works; a paraphrase outside the table becomes a question and retrieves
  nothing useful rather than failing loudly.
* **Answer synthesis is extractive by default.** `extractive_synthesize` only
  rearranges the retrieved episodes' own text, so it cannot hallucinate — and
  it cannot write nicely either. Set `ANTHROPIC_API_KEY` to switch
  `laptop/rag_query.py` to LLM synthesis.
