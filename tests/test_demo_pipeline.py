"""
Tests for the live-demo code: the 2D lab's camera model, the fog pipeline that
chains Phases 6-8, the perception-action loop's `ignore_classes` guard, and the
lexical embedding fallback.
"""

import math

import pytest

from demo.fog_pipeline import FogPipeline
from demo.sim2d import SimCamera, SimObject, SimRobot, build_lab
from laptop import embeddings
from nano.detector import DetectionResult
from nano.perception_action import PerceptionActionStateMachine, State
from nano.scene_graph import SURFACE_CLASSES
from shared.event_schema import BBox, Event, EventType


# --------------------------------------------------------------------------
# 2D camera: the bounding boxes must come from real projection geometry
# --------------------------------------------------------------------------

def _robot_at(x=0.0, y=0.0, yaw_deg=90.0) -> SimRobot:
    return SimRobot(x=x, y=y, yaw=math.radians(yaw_deg))


def _bottle_at(x, y) -> SimObject:
    return SimObject("bottle", "bottle", x, y, 0.865, 0.045, 0.25, "#1b73bf")


def test_object_dead_ahead_projects_to_frame_centre():
    robot = _robot_at()
    camera = SimCamera(robot, [_bottle_at(0.0, 1.2)])

    (projected,) = camera.detect()
    bbox = projected.detection.bbox
    centre_x = (bbox[0] + bbox[2]) / 2

    assert centre_x == pytest.approx(0.5, abs=0.02)
    assert not projected.clipped


def test_object_to_the_right_projects_right_of_centre():
    robot = _robot_at()
    camera = SimCamera(robot, [_bottle_at(0.45, 1.2)])

    (projected,) = camera.detect()
    centre_x = (projected.detection.bbox[0] + projected.detection.bbox[2]) / 2

    assert centre_x > 0.75


def test_turning_right_brings_a_right_clipped_object_back_into_frame():
    """The whole point of the perception-action loop: the box moves because the
    camera moved, not because anything was scripted."""
    robot = _robot_at()
    camera = SimCamera(robot, [_bottle_at(0.62, 1.1)])

    (before,) = camera.detect()
    assert before.clipped

    robot.yaw -= math.radians(12)  # a turn_right pulse
    (after,) = camera.detect()

    assert not after.clipped
    assert after.detection.bbox[2] < before.detection.bbox[2]


def test_objects_behind_the_camera_are_not_detected():
    robot = _robot_at()
    camera = SimCamera(robot, [_bottle_at(0.0, -1.5)])

    assert camera.detect() == []


def test_lab_objects_sit_exactly_on_the_table_surface():
    """`scene_graph.check_on_relationship` compares an object's bottom edge
    against the surface's top edge, so these have to line up to the millimetre."""
    objects, _ = build_lab()
    tables = {o.name: o for o in objects if o.class_name == "dining table"}
    assert tables, "the lab should have surfaces"

    table_top = next(iter(tables.values())).z_top
    for obj in objects:
        if obj.class_name == "dining table":
            continue
        assert obj.z_bottom == pytest.approx(table_top, abs=1e-6), obj.name


# --------------------------------------------------------------------------
# Perception-action: surfaces are context, never targets
# --------------------------------------------------------------------------

class _RecordingMotor:
    def __init__(self):
        self.action_history = []

    def _record(self, name):
        def pulse(duration_ms=150):
            self.action_history.append((name, duration_ms))
        return pulse

    def __getattr__(self, name):
        if name in ("forward", "reverse", "turn_left", "turn_right"):
            return self._record(name)
        if name == "stop":
            return lambda: self.action_history.append(("stop", 0))
        raise AttributeError(name)


def _machine(**kwargs):
    return PerceptionActionStateMachine(motor=_RecordingMotor(), **kwargs)


def test_without_ignore_classes_the_loop_chases_the_unfittable_surface():
    """Documents the behaviour `ignore_classes` exists to avoid: a desk wider
    than the frame always has the largest bbox and can never be made fully
    visible, so the loop fixates on it."""
    detections = [
        DetectionResult("dining table", 0.9, (0.0, 0.45, 1.0, 1.0)),
        DetectionResult("bottle", 0.9, (0.95, 0.2, 1.0, 0.5)),
    ]
    machine = _machine()
    machine.step(detections)

    assert machine.tracked_target == "dining table"


def test_ignore_classes_makes_the_loop_correct_toward_the_bottle_instead():
    detections = [
        DetectionResult("dining table", 0.9, (0.0, 0.45, 1.0, 1.0)),
        DetectionResult("bottle", 0.9, (0.95, 0.2, 1.0, 0.5)),
    ]
    machine = _machine(ignore_classes=SURFACE_CLASSES)
    state, _ = machine.step(detections)

    assert machine.tracked_target == "bottle"
    assert state == State.ADJUST
    assert machine.motor.action_history[-1][0] == "turn_right"


def test_ignoring_every_detected_class_leaves_nothing_to_track():
    machine = _machine(ignore_classes=SURFACE_CLASSES)
    state, emitted = machine.step([DetectionResult("desk", 0.9, (0.0, 0.4, 1.0, 1.0))])

    assert state == State.PATROL
    assert emitted is None


# --------------------------------------------------------------------------
# Fog pipeline: contradiction -> caption -> compression -> store
# --------------------------------------------------------------------------

BBOX = BBox(xmin=0.3, ymin=0.3, xmax=0.6, ymax=0.7)


def _event(objects, relationships, location, event_type=EventType.CONFIRMED):
    return Event(
        objects=objects,
        relationships=relationships,
        confidence=0.9,
        event_type=event_type,
        bbox=BBOX,
        location_tag=location,
    )


@pytest.fixture
def pipeline():
    return FogPipeline(persist=False)


def test_first_observation_opens_a_new_episode(pipeline):
    result = pipeline.ingest(_event(["bottle", "dining table"], ["bottle ON dining table"], "lab_desk_3"))

    assert result.action == "new"
    assert not result.moved
    assert result.caption == "A bottle is on the dining table."
    assert pipeline.snapshot()["stored_episodes"] == 1


def test_repeating_the_same_fact_extends_the_episode_rather_than_adding_one(pipeline):
    fact = (["bottle", "dining table"], ["bottle ON dining table"], "lab_desk_3")
    pipeline.ingest(_event(*fact))
    result = pipeline.ingest(_event(*fact))

    assert result.action == "extended"
    assert result.episode.observation_count == 2
    assert pipeline.snapshot()["stored_episodes"] == 1


def test_seeing_an_object_somewhere_new_is_logged_as_moved(pipeline):
    pipeline.ingest(_event(["bottle", "dining table"], ["bottle ON dining table"], "lab_desk_3"))
    result = pipeline.ingest(_event(["bottle", "dining table"], ["bottle ON dining table"], "lab_shelf_2"))

    assert result.moved
    assert result.event.event_type == EventType.MOVED.value
    assert "moved from its previous location" in result.caption
    assert pipeline.snapshot()["stored_episodes"] == 2


def test_a_partial_timeout_is_not_treated_as_a_move(pipeline):
    pipeline.ingest(_event(["cup", "dining table"], ["cup ON dining table"], "lab_desk_1"))
    result = pipeline.ingest(
        _event(["cup", "dining table"], ["cup ON dining table"], "lab_shelf_2", EventType.PARTIAL_TIMEOUT)
    )

    assert not result.moved
    assert "could not confirm a full view" in result.caption


def test_questions_retrieve_the_episode_they_are_about(pipeline):
    pipeline.ingest(_event(["bottle", "dining table"], ["bottle ON dining table"], "lab_desk_3"))
    pipeline.ingest(_event(["laptop", "dining table"], ["laptop ON dining table"], "lab_desk_1"))

    answer = pipeline.answer("where is the bottle?", k=2)

    assert "bottle" in answer["answer"].lower()
    assert answer["episodes"][0]["location_tag"] == "lab_desk_3"
    assert answer["latency_ms"] >= 0


def test_the_answer_shows_the_clean_caption_not_the_embedded_retrieval_hints(pipeline):
    pipeline.ingest(_event(["bottle", "dining table"], ["bottle ON dining table"], "lab_desk_3"))

    answer = pipeline.answer("where is the bottle?", k=1)

    assert "[objects:" not in answer["answer"]
    assert "A bottle is on the dining table." in answer["answer"]


def test_compression_ratio_tracks_events_against_stored_episodes(pipeline):
    fact = (["bottle", "dining table"], ["bottle ON dining table"], "lab_desk_3")
    for _ in range(4):
        pipeline.ingest(_event(*fact))

    snapshot = pipeline.snapshot()
    assert snapshot["raw_events"] == 4
    assert snapshot["stored_episodes"] == 1
    assert snapshot["compression_ratio"] == pytest.approx(4.0)


# --------------------------------------------------------------------------
# Embeddings: the fallback has to be usable, not just non-crashing
# --------------------------------------------------------------------------

def test_lexical_embeddings_are_deterministic_and_normalised():
    first = embeddings._lexical_embed("A bottle is on the dining table.")
    second = embeddings._lexical_embed("A bottle is on the dining table.")

    assert first == second
    assert len(first) == embeddings.EMBED_DIM
    assert math.sqrt(sum(v * v for v in first)) == pytest.approx(1.0, abs=1e-6)


def test_lexical_embeddings_rank_a_shared_subject_above_an_unrelated_caption():
    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    query = embeddings._lexical_embed("where is the bottle?")
    about_bottle = embeddings._lexical_embed("A bottle is on the dining table.")
    about_laptop = embeddings._lexical_embed("A laptop is on the dining table.")

    assert cosine(query, about_bottle) > cosine(query, about_laptop)


def test_stopwords_keep_shared_function_words_from_dominating():
    """Both captions share "is on the dining table"; only one shares the noun."""
    assert "the" in embeddings._STOPWORDS
    assert embeddings._lexical_embed("the the the the") == [0.0] * embeddings.EMBED_DIM
