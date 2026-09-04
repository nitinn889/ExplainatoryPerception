# Evaluation Results

**Status: metric tooling built and unit-tested (`laptop/evaluate.py`,
`tests/test_evaluate.py`); the four numbers below are NOT yet filled in
because they require the real self-collected dataset (spec Phase 9: 15-20
lab sessions, ~2,000-3,000 frames, ~10 object categories, ~50 staged
movement events, recorded with the physical robot) plus a completed,
integrated Nano <-> laptop pipeline. That dataset does not exist yet.**

Once `data/raw_sessions/` and `data/ground_truth_events.json` are populated
with real recordings and the full pipeline has been run end-to-end against
them, call `laptop.evaluate.summarize(...)` with the real counts and fill
in this table:

| Metric | Value | Method |
|---|---|---|
| Compression ratio | _pending_ | raw frames processed / episodes stored, via `compression_ratio()` |
| Retrieval precision / recall | _pending_ | RAG answers' retrieved episode ids vs. ground-truth relevant ids, via `retrieval_precision_recall()` |
| Perception-action success rate | _pending_ | confirmed vs. partial_timeout event counts from the Nano's perception-action loop, via `perception_action_success_rate()` |
| End-to-end query latency | _pending_ | wall-clock time from question asked to `laptop.rag_query.answer_question()` returning, via `query_latency_stats()` |

Do not report placeholder or estimated numbers here in place of real
measurements — the build spec is explicit that descoped/unfinished work
should be stated honestly in the report's limitations section rather than
implied to be complete.
