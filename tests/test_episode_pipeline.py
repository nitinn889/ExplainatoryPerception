"""
Tests for laptop/compression.py + laptop/contradiction.py (Phase 7).

Per the build spec's Phase 7 acceptance check: feed a synthetic sequence —
same object repeated at location A (should compress into one episode), then
the object appears at location B (should trigger a "moved" event, not a
duplicate).
"""

from datetime import datetime, timedelta, timezone

from laptop.compression import EpisodeCompressor
from laptop.contradiction import LocationTracker
from shared.event_schema import BBox, Event, EventType

BBOX = BBox(xmin=0.1, ymin=0.1, xmax=0.5, ymax=0.5)
T0 = datetime(2026, 9, 4, 10, 0, 0, tzinfo=timezone.utc)


def make_event(objects, relationships, location_tag, timestamp, event_type=EventType.NEW_OBSERVATION):
    return Event(
        objects=objects,
        relationships=relationships,
        confidence=0.9,
        event_type=event_type,
        bbox=BBOX,
        location_tag=location_tag,
        timestamp=timestamp,
    )


def test_repeated_observation_compresses_then_move_triggers_contradiction():
    tracker = LocationTracker()
    compressor = EpisodeCompressor()

    # 1. bottle ON table @ desk_3
    e1 = make_event(["bottle", "table"], ["bottle ON table"], "lab_desk_3", T0)
    e1 = tracker.process(e1)
    action1, episode1 = compressor.ingest(e1)

    # 2. same fact again a minute later -> should compress into episode1
    e2 = make_event(["bottle", "table"], ["bottle ON table"], "lab_desk_3", T0 + timedelta(minutes=1))
    e2 = tracker.process(e2)
    action2, episode2 = compressor.ingest(e2)

    # 3. bottle now seen at a different location -> contradiction -> "moved"
    e3 = make_event(["bottle", "shelf"], ["bottle ON shelf"], "lab_shelf_1", T0 + timedelta(minutes=2))
    e3 = tracker.process(e3)
    action3, episode3 = compressor.ingest(e3)

    assert action1 == "new"
    assert action2 == "extended"
    assert episode2 is episode1
    assert episode1.observation_count == 2
    assert episode1.end_time == T0 + timedelta(minutes=1)

    assert e3.event_type == EventType.MOVED
    assert action3 == "new"
    assert episode3.observation_count == 1
    assert episode3.location_tag == "lab_shelf_1"

    # only two distinct episodes were ever open, despite three observations
    assert len(compressor.open_episodes()) == 2


def test_unrelated_object_is_not_affected_by_another_objects_move():
    tracker = LocationTracker()

    e1 = make_event(["bottle", "table"], ["bottle ON table"], "lab_desk_3", T0)
    tracker.process(e1)

    e2 = make_event(["mug", "shelf"], ["mug ON shelf"], "lab_shelf_1", T0 + timedelta(minutes=1))
    e2 = tracker.process(e2)

    # mug has never been seen before -> first sighting, not a contradiction
    assert e2.event_type == EventType.NEW_OBSERVATION
    assert tracker.last_known_location("bottle") == "lab_desk_3"
    assert tracker.last_known_location("mug") == "lab_shelf_1"
