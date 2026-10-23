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

Questions can be **typed or spoken**. Click the microphone in *Ask the robot*
and say "where is the bottle?" and the robot answers out loud; spoken control
phrases ("pause", "start patrol") drive the same actions as the buttons. The
dashboard is unchanged by this — it still shows the full grounded answer and
the episodes it was built from, and everything works with the keyboard alone.
Voice needs Chrome or Edge; see [`demo/README.md`](demo/README.md) for what it
does not do.

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
- [x] Voice interface — ask by speaking and get a spoken reply, in addition to
      (never instead of) the dashboard. Routing and spoken phrasing live in
      `laptop/voice.py`; speech recognition and synthesis are the browser's.
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




   You are reproducing a working robot demo. Follow these steps IN ORDER. Ask me before
anything that moves the car, and never move it until I type "start".

== My setup (fill in) ==
- Jetson login: <user>@<jetson-ip>   (if Jetson is on USB-C to this PC: 192.168.55.1, this PC = 192.168.55.100;
  if on the same Wi-Fi: find it with `ip -br addr show wlP1p1s0` on the Jetson)
- This PC's IP as seen by the Jetson: <pc-ip>
- Jetson sudo password: I will type it myself if needed — do not ask me to paste it in chat.

== What the demo does ==
A small 4WD car (Arduino UNO + L298N, Elegoo-style shield) carries a camera facing BACKWARD over the
battery box. A Jetson Orin Nano runs detection and drives the UNO over USB serial. The car spins in place
(turn-only, slow pulses) to: (1) face a bottle and confirm it, (2) face a laptop and confirm it,
(3) wait 15 s while I remove the bottle, (4) turn back, see the bottle is gone and record it as "removed".
A dashboard on this PC shows the live camera and answers "where is the bottle?" from memory.

== Code ==
Repo: https://github.com/nitinn889/ExplainatoryPerception  branch: hardware/arduino-bridge-station-test
Clone this branch on BOTH this PC and the Jetson.

Files that matter:
- arduino/motor_bridge/motor_bridge.ino  UNO firmware: text commands "F|B|L|R <ms> <duty>", "S", "?" at 115200
                                         baud; it times each pulse itself (max 2 s). Boot banner: "EPCAR 1".
- nano/motor_control.py                  serial backend for the UNO (serial_port=..., camera_at_rear=...)
- nano/station_test.py                   THE demo driver: --preview, --list-cameras, --find-duty, --turn-test,
                                         and the bottle→laptop→bottle route. Turns are measured from the camera.
- nano/camera.py, detector.py, perception_action.py, scene_graph.py, event_client.py, shared/event_schema.py
- demo/server.py + demo/fog_pipeline.py + demo/static/dashboard.html   dashboard + memory (runs on this PC)
- laptop/rag_query.py, voice.py, captioning.py, compression.py, contradiction.py   answers + "removed" handling
- scripts/fetch_ssd_model.sh             downloads the SSD-MobileNet-V2 COCO weights (~70 MB) into models/
- requirements-laptop.txt (this PC), requirements-nano.txt (Jetson)

== Step 1: this PC — dashboard ==
python3 -m venv .venv && .venv/bin/pip install -r requirements-laptop.txt -r requirements-nano.txt
.venv/bin/python -m pytest tests -q        # expect all to pass (chromadb tests may skip on old sqlite)
.venv/bin/python -m uvicorn demo.server:app --host 0.0.0.0 --port 8080   (run in background)
Open http://localhost:8080. Check: curl -s localhost:8080/health

== Step 2: Jetson — dependencies and model ==
ssh to the Jetson (install this PC's SSH key with ssh-copy-id so later commands need no password).
cd into the repo. Do NOT pip install opencv-python (JetPack's OpenCV must stay).
pip3 install --user pydantic requests pyserial   (if pip refuses, use: pip3 install --target ~/ep_pydeps ...
and run everything with PYTHONPATH=$HOME/ep_pydeps:.)
bash scripts/fetch_ssd_model.sh
Check the Jetson can reach the dashboard: curl -s http://<pc-ip>:8080/health

== Step 3: Arduino ==
UNO must be plugged into the Jetson; it appears as /dev/ttyACM0 (user must be in group dialout).
Check if the firmware is already on it (sends only "?" and "S", nothing moves):
  python3 -c "import serial,time; s=serial.Serial('/dev/ttyACM0',115200,timeout=0.5); time.sleep(2.5); s.write(b'?\n'); print(s.readline())"
If it does not answer "EPCAR 1": flash arduino/motor_bridge/motor_bridge.ino with arduino-cli
(compile --fqbn arduino:avr:uno, upload -p /dev/ttyACM0) or the Arduino IDE. WARNING: the kit's original
sketch drives the motors on power-up — keep the car battery OFF until the bridge firmware is on.

== Step 4: camera ==
python3 -m nano.station_test --list-cameras --snapshots /tmp/camlist
Use the index reported as "colour" (it was 4 for a RealSense D435i/D455). Look at the saved image:
if it is upside down add --rotate 180 everywhere below. Use --hfov 64 for the D435i colour stream
(90 for a D455, ~66 for a Logitech C920).

== Step 5: live preview (motors off, nothing saved to memory) ==
nohup python3 -m nano.station_test --preview --mode usb --camera-index <N> --rotate 180 \
  --laptop-url http://<pc-ip>:8080 > /tmp/preview.log 2>&1 < /dev/null &
The dashboard now shows the camera. Read detections from this PC with: curl -s localhost:8080/state
(telemetry.detections: class, confidence, bbox xmin..xmax in 0-1). I will place a bottle (text facing
the camera) on the left and a laptop on the right. Both must show green boxes; tell me if they don't
(common fixes: object fully in frame, bottle text facing the camera, nothing blocking the view).
Compute each object's angle: angle = degrees(atan((cx - 0.5) * 2 * tan(hfov/2))), cx = box centre x;
negative = left. If the bottle touches the left edge, add a few degrees further left.

== Step 6: run — ONLY after I type "start" ==
1. Stop the preview (use a bracketed pattern so pkill doesn't kill its own shell):
   pkill -f "nano[.]station_test --preview"
2. Restart the dashboard on this PC so memory starts empty (then wait for /health).
3. Run on the Jetson:
python3 -m nano.station_test --motor-port /dev/ttyACM0 --camera-rear --turn-only \
  --mode usb --camera-index <N> --rotate 180 --hfov 64 \
  --duty 70 --pulse-ms 100 --deg-per-sec 32 \
  --min-confidence 0.4 --min-box-height 0.2 --search-deg 30 \
  --station "1=<bottle angle>:bottle|cup" --station "2=<laptop angle>:laptop|tv" \
  --route 1 2 1 --wait 2=15 --location-tag lab_table \
  --laptop-url http://<pc-ip>:8080 --snapshots data/station_test --log-file data/station_test/run.log
During "waiting 15 s at location 2 - change the scene now" I remove the bottle.
Expected final table: 1 confirmed bottle / 2 confirmed laptop / 1 missing bottle.

== Step 7: show the result ==
curl -s "http://localhost:8080/query?q=where%20is%20the%20bottle"
Expected: "The bottle was not found at lab_table location 1; it has been removed. Checked at ... Before
that it was seen at lab_table location 1, ..."  (I can also ask it on the dashboard.)
Copy the snapshots in data/station_test/ back and show me the last one to confirm the bottle is gone.
Then restart the preview (Step 5) so the dashboard stays live — the preview stops sending after
~20 failed posts, so always restart it after restarting the dashboard.

== Known issues — report them, don't hide them ==
- "turn stalled" warnings, mostly turning left: low battery or a cable pulling. Suggest fresh batteries.
- "Motor bridge did not acknowledge" / "Input/output error": the UNO dropped off USB (battery sag at
  70% duty). Stop, ask for fresh batteries / reseat the UNO cable, rerun.
- 60% duty no longer turns the car; 70% is the working value. Never drive forward/back (always --turn-only).
- The detector is SSD-MobileNet-V2 (COCO): laptops often read as "tv" (alias handles it); plain shiny
  bottles and side-on mice aren't detected; far-away people are ignored by --min-box-height.
- The Jetson clock may say 1970; the dashboard stamps events itself, so answers still show real times.

