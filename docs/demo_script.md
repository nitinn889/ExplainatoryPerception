# Live Demo Script

Requires both sides running: Nano (`nano/main_loop.py`) on the robot and
laptop (`laptop/event_server.py` + `laptop/api.py`) on the same LAN/Wi-Fi.
This is a walkthrough script, not something that can be executed without
the physical hardware.

1. **Start the laptop side.**
   `uvicorn laptop.event_server:app --host 0.0.0.0 --port 8000` (event
   receiver), and separately `uvicorn laptop.api:app --port 8001` (query API).
2. **Start the Nano side.** Run `nano/main_loop.py` — robot enters PATROL.
3. **Show partial visibility.** Place an object (e.g. a bottle) so it's only
   partially in frame at the camera's edge.
4. **Self-correction.** Narrate the state transition PATROL -> PARTIAL ->
   ADJUST as the robot issues short motor pulses to bring the object fully
   into frame, then CONFIRM once it's stable for 3-5 consecutive frames.
5. **Episode logged.** Point out the event arriving at the laptop's
   `/event` endpoint, being captioned, embedded, and stored as an episode.
6. **Ask a question.** `python -m laptop.api "where's the bottle?"` (or hit
   `GET /query?q=...`) — show the answer is grounded in the caption just
   logged, worded differently than the question.
7. **(Optional) Show compression/contradiction.** Move the same object to a
   second location, repeat steps 3-5, then ask the question again — the
   answer should reflect the object's new location (a "moved" event), not
   two unrelated facts.

Cross-check against `README.md`'s status section before presenting: don't
claim capabilities in slides/report that weren't actually built.
