# Raw demo run output

Verbatim results from the runs summarised in
[`webots_demo_results.md`](webots_demo_results.md). Recorded 2026-10-08 on a
4-core Linux container with software OpenGL (llvmpipe) and no GPU.

Embedding backend for both runs: **lexical fallback** — `all-MiniLM-L6-v2`
could not be downloaded in that environment. See the limitations section of
`demo/README.md`; on a machine that can reach HuggingFace, retrieval is semantic
and should do no worse.

---

## Webots run

```
python -m demo.run_demo --no-sim &
./scripts/run_webots_demo.sh --headless
```

2 laps × 3 stations, ground-truth detection, ~90 s wall clock.
`webots/controllers/episodic_robot/run_summary.json`:

```json
{
  "detector": "simulated detector (tight ground-truth projection)",
  "frames": 840,
  "confirmations": 244,
  "partial_views": 10,
  "recoveries": 8,
  "timeouts": 2,
  "perception_action_success_rate": 0.8,
  "filtered_out": 235,
  "events_sent": 11,
  "bottle_moves": 2
}
```

Fog side: 11 events in, **6 episodes stored**, event-to-episode merge 1.83×,
overall **140 frames per stored episode**. Query latency: mean 3.9 ms, p95
6.4 ms over 5 queries.

Episodes, as stored:

```
[confirmed] A bottle is on the dining table. A vase is on the dining table.   @lab_desk_3   1x
[confirmed] A vase is on the dining table.                                    @lab_desk_3   1x
[confirmed] A laptop is on the dining table.                                  @lab_desk_1   2x over 33.5s
[confirmed] A cup is on the dining table. A laptop is on the dining table.    @lab_desk_1   4x over 36.4s
[confirmed] A potted plant is on the dining table.                            @lab_shelf_2  1x
[moved    ] A potted plant is on the dining table. A bottle is on the ...     @lab_shelf_2  2x
```

## Browser (2D sim) run

```
python -m demo.run_demo --time-scale 0.08 --laps 3
```

```json
{
  "frames": 810,
  "confirmations": 160,
  "partial_views": 12,
  "recoveries": 12,
  "timeouts": 0,
  "perception_action_success_rate": 1.0,
  "filtered_out": 147,
  "events_sent": 13,
  "stored_episodes": 6,
  "event_merge_ratio": 2.17,
  "frames_per_episode": 135.0,
  "latency_mean_ms": 3.5,
  "latency_p95_ms": 8.0
}
```

## Retrieval check

Five fixed questions, same five in both runs, checked by hand against what the
robot had actually observed. **10/10 correct** (5 questions × 2 runs): in every
case the episode stated as the answer was the right one.

| Question | Expected | Webots run | 2D run |
|---|---|---|---|
| where is the bottle? | wherever it was last logged | ✅ `lab_desk_3` (correct — moved back on lap 2) | ✅ `lab_shelf_2` (correct for 3 laps) |
| what is on desk 3? | vase / bottle | ✅ vase | ✅ vase |
| did anything move? | the `moved` episode | ✅ bottle, `moved`, `lab_shelf_2` | ✅ bottle, `moved`, `lab_shelf_2` |
| have you seen a laptop? | laptop at desk 1 | ✅ `lab_desk_1` | ✅ `lab_desk_1` |
| what is on the shelf? | potted plant | ✅ `lab_shelf_2` | ✅ `lab_shelf_2` |

This is a sanity check on five hand-written questions, not the Phase 9
precision/recall figure. That one needs `data/ground_truth_events.json` filled
in from real recordings and `laptop/evaluate.py` run against it.

## Reading the perception-action numbers

- **partial views** — times the loop entered `ADJUST`, i.e. saw a bbox touching
  an edge margin or below the minimum height.
- **recoveries** — of those, how many reached a confirmed full view.
- **timeouts** — how many gave up after 7 s and logged a `partial_timeout`.

The two Webots timeouts are the cup at `lab_desk_1`. At 10 cm tall it can never
satisfy the fixed `min_height` of 0.15 from a sane viewing distance, so the loop
reads it as "too far", drives forward, and eventually gives up — which is the
designed behaviour, and the reason `partial_timeout` exists in the schema. It is
left in deliberately: the demo should show the loop failing gracefully as well
as succeeding. The 2D run shows 1.0 because at `--time-scale 0.08` the 7-second
wall-clock timeout does not elapse within a station's frame budget.

**`confirmations` is much larger than `partial_views`.** The robot dwells at a
station after logging, and the state machine re-confirms the same already-visible
object every few frames. Those are not partial views and are not counted as
recoveries — they are what the importance filter absorbs, which is why
`filtered_out` is large and `events_sent` is small.
