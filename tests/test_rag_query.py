"""
Tests for laptop/rag_query.py (Phase 8).

Seeds a small memory store and runs a few sample questions; per the build
spec's acceptance check, answers must be grounded in retrieved episodes
(no hallucinated details not present in stored memories).
"""

import shutil
import tempfile

import pytest

from laptop.memory_store import MemoryStore
from laptop.rag_query import answer_question
from shared.event_schema import BBox, Event, EventType

BBOX = BBox(xmin=0.1, ymin=0.1, xmax=0.5, ymax=0.5)


def make_event(objects, relationships, location_tag):
    return Event(
        objects=objects,
        relationships=relationships,
        confidence=0.9,
        event_type=EventType.NEW_OBSERVATION,
        bbox=BBOX,
        location_tag=location_tag,
    )


@pytest.fixture
def seeded_store():
    tmp_dir = tempfile.mkdtemp()
    store = MemoryStore(persist_dir=tmp_dir)
    store.add_episode(
        make_event(["charger", "desk"], ["charger ON desk"], "lab_desk_1"),
        "A phone charger is on the desk.",
    )
    store.add_episode(
        make_event(["mug", "shelf"], ["mug ON shelf"], "lab_shelf_1"),
        "A ceramic mug sitting on the shelf.",
    )
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_answer_is_grounded_in_retrieved_caption(seeded_store):
    result = answer_question("where's my charger?", seeded_store, k=1)

    assert result["episodes"], "expected at least one retrieved episode"
    top_caption = result["episodes"][0]["caption"]
    assert top_caption == "A phone charger is on the desk."
    # extractive synthesis must directly contain the grounding caption text
    assert top_caption in result["answer"]


def test_no_relevant_memory_does_not_hallucinate(seeded_store):
    tmp_dir = tempfile.mkdtemp()
    empty_store = MemoryStore(persist_dir=tmp_dir)
    try:
        result = answer_question("where's my charger?", empty_store, k=3)
        assert result["episodes"] == []
        assert "don't have any recorded observations" in result["answer"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
