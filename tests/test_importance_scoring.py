"""
Tests for nano/importance_scoring.py (Phase 4).
Feeds a sequence of repeated identical observations followed by a changed one;
confirms only the changed one triggers a send.
"""

from datetime import datetime, timezone
import pytest
from nano.importance_scoring import ImportanceScorer
from shared.event_schema import BBox, Event, EventType

BBOX_CENTER = BBox(xmin=0.4, ymin=0.4, xmax=0.6, ymax=0.6)
T0 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)


def make_event(objects, relationships, location_tag, bbox=BBOX_CENTER, event_type=EventType.CONFIRMED):
    return Event(
        objects=objects,
        relationships=relationships,
        confidence=0.92,
        event_type=event_type,
        bbox=bbox,
        location_tag=location_tag,
        timestamp=T0,
    )


def test_repeated_observations_filtered_and_changed_triggers_send():
    scorer = ImportanceScorer()

    # 1. First sighting: bottle ON table @ desk_1 -> New observation, MUST send
    e1 = make_event(["bottle", "table"], ["bottle ON table"], "lab_desk_1")
    res1 = scorer.filter_event(e1)
    assert res1 is not None
    assert res1.objects == ["bottle", "table"]

    # 2. Repeated identical observation 1 second later -> Duplicate, MUST NOT send
    e2 = make_event(["bottle", "table"], ["bottle ON table"], "lab_desk_1")
    res2 = scorer.filter_event(e2)
    assert res2 is None

    # 3. Repeated identical observation again -> MUST NOT send
    e3 = make_event(["bottle", "table"], ["bottle ON table"], "lab_desk_1")
    res3 = scorer.filter_event(e3)
    assert res3 is None

    # 4. Changed observation: bottle now observed at shelf_1 -> Moved, MUST send!
    e4 = make_event(["bottle", "shelf"], ["bottle ON shelf"], "lab_shelf_1")
    res4 = scorer.filter_event(e4)
    assert res4 is not None
    assert res4.event_type == EventType.MOVED
    assert res4.location_tag == "lab_shelf_1"


def test_relationship_change_triggers_send():
    scorer = ImportanceScorer()

    # Sighting: cup ON table @ desk_1
    e1 = make_event(["cup", "table"], ["cup ON table"], "lab_desk_1")
    assert scorer.filter_event(e1) is not None

    # Duplicate -> Suppressed
    assert scorer.filter_event(e1) is None

    # Same location, but relationship changed to cup LEFT OF laptop -> MUST send
    e2 = make_event(["cup", "laptop"], ["cup LEFT OF laptop"], "lab_desk_1")
    res2 = scorer.filter_event(e2)
    assert res2 is not None
    assert res2.relationships == ["cup LEFT OF laptop"]


def test_partial_timeout_always_triggers_send():
    scorer = ImportanceScorer()

    # Normal observation
    e1 = make_event(["bottle", "table"], ["bottle ON table"], "lab_desk_1")
    assert scorer.filter_event(e1) is not None

    # Partial timeout observation of same object
    timeout_event = make_event(
        ["bottle"],
        [],
        "lab_desk_1",
        bbox=BBox(xmin=0.01, ymin=0.2, xmax=0.3, ymax=0.5),
        event_type=EventType.PARTIAL_TIMEOUT,
    )
    res = scorer.filter_event(timeout_event)
    assert res is not None
    assert res.event_type == EventType.PARTIAL_TIMEOUT
