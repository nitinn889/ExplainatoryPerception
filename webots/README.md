# Webots simulation

```bash
python -m demo.run_demo --no-sim      # fog node + dashboard, one terminal
./scripts/run_webots_demo.sh          # Webots, another
```

| File | What it is |
|---|---|
| `worlds/episodic_lab.wbt` | the lab: three stations, a 2WD robot with a mast camera |
| `controllers/episodic_robot/episodic_robot.py` | the Nano's job, running in Webots |

The controller does **not** reimplement the pipeline. It imports
`nano.perception_action`, `nano.scene_graph`, `nano.importance_scoring`,
`nano.event_client` and `shared.event_schema`, and adds only what Webots needs:
a motor adapter with the same five-method interface as the L298N driver, a
detection source, and a scripted patrol.

Flags: `--detector ssd` (real SSD-MobileNet-V2 instead of ground truth, run
`scripts/fetch_ssd_model.sh` first), `--headless`, `--record out.mp4`.

The world is built from primitive Webots nodes only — no `EXTERNPROTO` — so it
loads with no network access.

Notes worth knowing before editing the world:

- **Table tops are at z = 0.74 and every object's base is exactly there.** The
  scene graph's `ON` test compares an object's bottom edge against the surface's
  top edge; if those do not line up you get unstable `LEFT OF` / `BEHIND`
  relations instead of `bottle ON dining table`.
- **The controller projects its own bounding boxes** rather than using the
  `Recognition` node, because `Recognition` reports the projection of a bounding
  *sphere*. For a 1.5 m desk at 1.3 m that fills the whole frame. `OBJECT_EXTENTS`
  in the controller holds each Solid's shape; add an entry when you add an object.
- Vertical extents there are offsets from the Solid's **own origin**, which is
  the base plate for the laptop and the top plate for the tables.

See [`../demo/README.md`](../demo/README.md) for the presentation script and the
honest limitations list.
