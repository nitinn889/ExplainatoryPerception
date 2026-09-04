"""
Unit tests for laptop/evaluate.py's metric math, using synthetic numbers
(not the real self-collected dataset, which requires the physical robot).
"""

import pytest

from laptop.evaluate import (
    compression_ratio,
    perception_action_success_rate,
    query_latency_stats,
    retrieval_precision_recall,
    summarize,
)


def test_compression_ratio():
    assert compression_ratio(raw_frame_count=3000, episodes_stored=150) == 20.0


def test_retrieval_precision_recall():
    result = retrieval_precision_recall({"a", "b", "c"}, {"a", "b", "z"})
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)


def test_perception_action_success_rate():
    assert perception_action_success_rate(confirmed_count=45, timeout_count=5) == 0.9


def test_query_latency_stats():
    stats = query_latency_stats([100, 200, 300, 400, 500])
    assert stats["mean_ms"] == 300
    assert stats["p50_ms"] == 300


def test_summarize_aggregates_all_four_metrics():
    result = summarize(
        raw_frame_count=2000,
        episodes_stored=100,
        retrieval_results=[
            {"retrieved": {"e1", "e2"}, "relevant": {"e1"}},
            {"retrieved": {"e3"}, "relevant": {"e3"}},
        ],
        confirmed_count=18,
        timeout_count=2,
        latencies_ms=[120, 150, 180],
    )
    assert result["compression_ratio"] == 20.0
    assert result["perception_action_success_rate"] == 0.9
    assert result["retrieval_precision"] == pytest.approx((0.5 + 1.0) / 2)
    assert "p95_ms" in result["query_latency"]
