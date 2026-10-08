# How to run the demo for your teacher

Start to finish. Allow 10 minutes the first time, 30 seconds every time after.

---

## 0. Get the code onto your laptop

If the branch is on GitHub:

```bash
git clone https://github.com/nitinn889/ExplainatoryPerception.git
cd ExplainatoryPerception
git checkout claude/wizardly-noether-uz83tj
```

If it is not pushed yet, use the `.bundle` file from this conversation:

```bash
git clone episodic-perception-demo.bundle ExplainatoryPerception
cd ExplainatoryPerception
git checkout claude/wizardly-noether-uz83tj
```

Check you are in the right place — `ls` should show `demo/`, `webots/`, `nano/`,
`laptop/`.

---

## 1. Install (once)

Needs **Python 3.10 or newer**. Check with `python3 --version`
(`python --version` on Windows).

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-laptop.txt -r requirements-nano.txt
```

That pulls in ChromaDB and sentence-transformers — the real Phase 6 vector store
and embedding model, which is what you want to be able to point at. It is a
**large download (1-3 GB) and can take 10+ minutes.** Do it the night before,
not in the classroom.

### If you are short on time or bandwidth

This installs in about 20 seconds and 280 MB, and the demo still runs end to
end:

```bash
pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.6" \
            "requests>=2.31" "numpy>=1.24" "opencv-python-headless>=4.8"
```

The dashboard will then say `store in-memory` and `embeddings lexical` instead
of `chromadb` and `sentence-transformers`. Everything works; retrieval matches
on keywords rather than meaning. The badges in the top-right always tell the
truth about which backend is live, so you can point at them and say so rather
than being caught out.

**Verify the install before the lesson:**

```bash
pytest tests/ -q
```

42 passed, 1 skipped is correct. (The skip is the semantic-retrieval test; it
only runs when sentence-transformers is installed and can reach the internet.)

---

## 2. Run it

```bash
python -m demo.run_demo
```

Open **<http://localhost:8080>**. The robot starts patrolling on its own. That
is the whole demo — one command, one browser tab.

Useful flags:

| Flag | Why |
|---|---|
| `--time-scale 0.3` | speed the patrol up ~3× if you are short on time |
| `--time-scale 2` | slow it down so the state machine is easier to follow |
| `--port 8090` | if something else is already using 8080 |

Press `Ctrl+C` in the terminal to stop.

### Before you present

- **Full-screen the browser** (F11). The dashboard is three columns and wants
  the width; on a projector it is worth checking it fits.
- **Leave it running for ~60 seconds first**, so there are already a few
  episodes in memory when you start talking. An empty memory panel is a weak
  opening.
- Have a terminal visible if you want to show the logs, but you do not need to.

---

## 3. What to show, in order

Roughly five minutes.

**1 — "This is the architecture."** Point at the header:
`camera → detect → scene graph → perception–action loop → importance filter`
on the edge, `caption → embed → vector memory → RAG` on the fog side. Say the
split is real: on hardware the left half runs on the Jetson Nano, the right
half on a laptop, and they talk over HTTP. The demo runs both on one machine.

**2 — "The robot is actually looking at things."** Camera view, left. Every box
is a projection of an object at real coordinates, not an animation. The red
bands down the sides are the `edge_margin`: a box touching one counts as
partially visible.

**3 — The core novelty.** This is the bit to spend time on. Wait for
`lab_desk_3`. The bottle sits in the right-hand red band, the state badge goes
to **ADJUST**, you see `turn_right` pulses at the bottom of the camera view,
and the box slides inward. Then **CONFIRM** — and say the important part: it
holds for four consecutive frames *before* writing anything down, so it never
logs a half-corrected view. Then **LOGGED**.

The other two stations correct differently, which is worth pointing out:
- `lab_desk_1` — the laptop is too small, not off-centre → the correction is
  `forward`
- `lab_shelf_2` — the bottle is clipped on the **left** → `turn_left`

**4 — "It writes down facts, not frames."** Right column. An episode appears
with a sentence and a location tag — "A bottle is on the dining table",
`lab_desk_3` — not a bounding box. Point at **FRAMES → EPISODES**: a number
like 130×. That gap is the importance filter plus memory compression.

**5 — Contradiction detection.** Click **Move the bottle (staged event)** — this
is you playing the person who moves something while the robot is elsewhere.
When the robot next reaches the shelf, the episode comes back typed **moved**,
not stored as an unrelated new fact. Say plainly that the move is staged: it is
the simulated version of the staged object-movement events the build spec asks
for.

**6 — Compression.** On a later lap, an episode shows `2×` or `3×` with a
duration instead of a second identical row. One durable fact, not one row per
sighting.

**7 — Ask it something.** Type, or click a suggestion:
- *"where is the bottle?"* → the latest location, and it says so if it moved
- *"did anything move?"* → the `moved` episode
- *"what is on desk 3?"* → the right desk

Point at the latency (single-digit milliseconds) and at the retrieved episodes
below the answer — the answer is built from those, so it cannot invent
anything.

---

## 4. If your teacher asks for Webots

Two terminals.

```bash
# terminal 1
python -m demo.run_demo --no-sim

# terminal 2
./scripts/run_webots_demo.sh
```

Webots opens, the robot patrols, and the **same dashboard** fills in — with the
real rendered camera frames in the camera panel instead of the drawn ones.

You need Webots R2023b or newer installed from <https://cyberbotics.com/>
(about 1 GB). Install it ahead of time.

Worth saying: the world is built from primitive Webots nodes only, so it loads
with no network access, and the controller imports the same `nano/` modules —
the motor adapter exposes the identical `forward/reverse/turn_left/turn_right/stop`
interface as the L298N driver, with the same short fixed-duration pulses.

---

## 5. Questions your teacher will probably ask

**"Is this using YOLO?"** No. The detector is SSD-MobileNet-V2, which the spec
requires; `scripts/fetch_ssd_model.sh` downloads the COCO-pretrained weights and
`nano/detector.py` runs them through OpenCV's DNN module. There is no
YOLO anywhere in the repo.

**"So the detections are real CNN output?"** In the demo, no — and say so
before being asked. The demo uses ground-truth detection from the simulator, on
purpose, so the demo is deterministic and so what you are showing is the
perception-action loop rather than detector noise. The real detector is wired in
(`--detector ssd`) and works on photographs (0.99 "person", 0.98 "sports ball"),
but on this world it returns one detection: `bench 0.85`, the desk. A
COCO-trained network has never seen a bottle that looks like an untextured blue
cylinder. The measurement and the frame are in
[`webots_demo_results.md`](webots_demo_results.md). To show the detector working
on real images: `python -m nano.main_loop --mode usb --display` with a webcam.

**"Is the robot navigating?"** No, and that is deliberate — full SLAM is an
explicit non-goal in the spec. Patrol is a scripted list of waypoints, each
carrying a coarse location tag. Nothing in the perception path ever sees a pose.

**"Did you test it?"** `pytest tests/ -q` — 42 tests. The measured demo numbers
are in [`webots_demo_results.md`](webots_demo_results.md) with the raw output in
[`demo_results_data.md`](demo_results_data.md).

**"What doesn't work?"** The honest list is in
[`../demo/README.md`](../demo/README.md) under *Honest limitations* — worth
reading it before you present so nothing surprises you. The short version: no
real pixels in the 2D sim, no occlusion, scripted patrol, staged object moves,
and flat objects break the fixed minimum-height rule.

---

## 6. If something goes wrong

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'demo'` | You are in the wrong folder. `cd` to the repo root (where `README.md` is) and use `python -m demo.run_demo`, not `python demo/run_demo.py`. |
| `Address already in use` | Something else has 8080. Use `--port 8090`. |
| Browser shows `link reconnecting…` | The server stopped. Check the terminal for the error. |
| Dashboard loads but nothing moves | The patrol may have finished. Click **Start patrol**, or restart with `--laps 0` to patrol indefinitely (the default). |
| First run hangs for minutes | sentence-transformers downloading its model (~90 MB) on first use. It is cached after that. Run it once before the lesson. |
| Webots window is black / very slow | No GPU drivers. Use the browser demo instead — it needs no graphics hardware. |

**Have a fallback.** Record the demo beforehand and keep the file on your
laptop, so a projector or a network problem does not sink the presentation:

```bash
./scripts/run_webots_demo.sh --record demo.mp4
```
