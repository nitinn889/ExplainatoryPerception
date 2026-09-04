"""
Evaluation metrics (Phase 9).

Computes the four numbers the build spec requires for the report, given
real pipeline output (not fabricated): compression ratio, retrieval
precision/recall, perception-action success rate, and end-to-end query
latency. See docs/evaluation_results.md for how to run this against the
self-collected dataset once it exists.
"""

from __future__ import annotations

import statistics
from typing import Any


def compression_ratio(raw_frame_count: int, episodes_stored: int) -> float:
    """How many raw frames each stored episode represents on average."""
    if episodes_stored == 0:
        raise ValueError("episodes_stored must be > 0")
    return raw_frame_count / episodes_stored


def retrieval_precision_recall(
    retrieved_event_ids: set[str], relevant_event_ids: set[str]
) -> dict[str, float]:
    """Standard precision/recall of a RAG answer's retrieved episodes
    against the ground-truth event ids that were actually relevant."""
    if not retrieved_event_ids and not relevant_event_ids:
        return {"precision": 1.0, "recall": 1.0}

    true_positives = retrieved_event_ids & relevant_event_ids
    precision = len(true_positives) / len(retrieved_event_ids) if retrieved_event_ids else 0.0
    recall = len(true_positives) / len(relevant_event_ids) if relevant_event_ids else 0.0
    return {"precision": precision, "recall": recall}


def perception_action_success_rate(confirmed_count: int, timeout_count: int) -> float:
    """% of partial-view detections that were successfully corrected to a
    full view (event_type == confirmed) vs. timed out (partial_timeout)."""
    total = confirmed_count + timeout_count
    if total == 0:
        raise ValueError("no partial-view detections to evaluate")
    return confirmed_count / total


def query_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Mean / p50 / p95 end-to-end query latency in milliseconds."""
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty")
    sorted_latencies = sorted(latencies_ms)
    return {
        "mean_ms": statistics.mean(sorted_latencies),
        "p50_ms": statistics.median(sorted_latencies),
        "p95_ms": sorted_latencies[int(0.95 * (len(sorted_latencies) - 1))],
    }


def summarize(
    raw_frame_count: int,
    episodes_stored: int,
    retrieval_results: list[dict[str, Any]],
    confirmed_count: int,
    timeout_count: int,
    latencies_ms: list[float],
) -> dict[str, Any]:
    """retrieval_results: list of {"retrieved": set[str], "relevant": set[str]}
    one per evaluated question, averaged into overall precision/recall."""
    precisions, recalls = [], []
    for r in retrieval_results:
        pr = retrieval_precision_recall(r["retrieved"], r["relevant"])
        precisions.append(pr["precision"])
        recalls.append(pr["recall"])

    return {
        "compression_ratio": compression_ratio(raw_frame_count, episodes_stored),
        "retrieval_precision": statistics.mean(precisions) if precisions else None,
        "retrieval_recall": statistics.mean(recalls) if recalls else None,
        "perception_action_success_rate": perception_action_success_rate(confirmed_count, timeout_count),
        "query_latency": query_latency_stats(latencies_ms),
    }
