"""
Tests for laptop/memory_store.py (Phase 6).

Adds several synthetic episodes and runs a semantic query worded
differently than what was stored, confirming the right episode is
retrieved (per the Phase 6 acceptance check in the build spec).
"""

import shutil
import tempfile

import pytest

from laptop.memory_store import MemoryStore
from shared.event_schema import BBox, Event, EventType

BBOX = BBox(xmin=0.1, ymin=0.1, xmax=0.5, ymax=0.5)


def make_event(objects, relationships, location_tag, event_type=EventType.NEW_OBSERVATION):
    return Event(
        objects=objects,
        relationships=relationships,
        confidence=0.9,
        event_type=event_type,
        bbox=BBOX,
        location_tag=location_tag,
    )


@pytest.fixture
def store():
    tmp_dir = tempfile.mkdtemp()
    yield MemoryStore(persist_dir=tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_add_and_semantic_search(store):
    store.add_episode(
        make_event(["flask", "laptop"], ["flask LEFT OF laptop"], "lab_desk_3"),
        "A blue flask beside the laptop.",
    )
    store.add_episode(
        make_event(["mug", "shelf"], ["mug ON shelf"], "lab_shelf_1"),
        "A ceramic mug sitting on the shelf.",
    )
    store.add_episode(
        make_event(["keys", "table"], ["keys ON table"], "lab_desk_2"),
        "A set of keys on the table.",
    )

    results = store.search("where's my water bottle?", k=1)

    assert len(results) == 1
    assert results[0]["location_tag"] == "lab_desk_3"
    assert "flask" in results[0]["caption"].lower()


def test_count_reflects_added_episodes(store):
    assert store.count() == 0
    store.add_episode(
        make_event(["bottle", "table"], ["bottle ON table"], "lab_desk_1"),
        "A bottle is on the table.",
    )
    assert store.count() == 1
