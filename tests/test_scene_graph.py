"""
Tests for nano/scene_graph.py (Phase 2). Covers ON, LEFT OF, RIGHT OF, BEHIND
with synthetic bbox inputs and downstream captioning compatibility.
"""

import pytest
from nano.detector import DetectionResult
from nano.scene_graph import (
    RELATION_BEHIND,
    RELATION_LEFT_OF,
    RELATION_ON,
    RELATION_RIGHT_OF,
    SceneGraphBuilder,
    check_behind_relationship,
    check_left_right_relationship,
    check_on_relationship,
)
from laptop.captioning import parse_relationship, caption_for_relationship


def test_on_relationship_detected():
    builder = SceneGraphBuilder()
    # Bottle placed on table:
    # Table bbox: bottom half of screen (0.1, 0.5, 0.9, 0.9)
    # Bottle bbox: standing directly on table top (0.4, 0.25, 0.6, 0.52)
    bottle = DetectionResult("bottle", 0.95, (0.4, 0.25, 0.6, 0.52))
    table = DetectionResult("table", 0.90, (0.1, 0.5, 0.9, 0.9))

    relationships, objects = builder.build_scene_graph([bottle, table])

    assert len(relationships) == 1
    assert relationships[0] == "bottle ON table"
    assert objects[0] == "bottle"
    assert objects[1] == "table"

    # Verify downstream captioning can parse and caption it
    subj, rel, obj = parse_relationship(relationships[0])
    assert (subj, rel, obj) == ("bottle", "ON", "table")
    caption = caption_for_relationship(relationships[0])
    assert caption == "A bottle is on the table."


def test_left_of_and_right_of_relationship():
    builder = SceneGraphBuilder()
    # Cup on the left (0.1, 0.4, 0.3, 0.6), Laptop on the right (0.6, 0.3, 0.9, 0.7)
    cup = DetectionResult("cup", 0.92, (0.1, 0.4, 0.3, 0.6))
    laptop = DetectionResult("laptop", 0.94, (0.6, 0.3, 0.9, 0.7))

    relationships, objects = builder.build_scene_graph([cup, laptop])

    assert len(relationships) == 1
    assert relationships[0] == "cup LEFT OF laptop"
    assert "cup" in objects
    assert "laptop" in objects

    # Check inverse relative direction
    lr = check_left_right_relationship(builder_obj(laptop), builder_obj(cup))
    assert lr == RELATION_RIGHT_OF


def test_behind_relationship():
    builder = SceneGraphBuilder()
    # Monitor behind keyboard:
    # Keyboard closer to camera (lower in frame): (0.3, 0.6, 0.7, 0.85)
    # Monitor further back (higher in frame): (0.25, 0.2, 0.75, 0.5)
    monitor = DetectionResult("tv", 0.89, (0.25, 0.2, 0.75, 0.5))
    keyboard = DetectionResult("keyboard", 0.91, (0.3, 0.6, 0.7, 0.85))

    relationships, objects = builder.build_scene_graph([monitor, keyboard])

    assert len(relationships) == 1
    assert relationships[0] == "tv BEHIND keyboard"


def test_no_spurious_relationship_when_far_apart():
    # Two objects with no overlap and large separation still get relative LEFT/RIGHT
    builder = SceneGraphBuilder()
    obj1 = DetectionResult("chair", 0.8, (0.05, 0.2, 0.25, 0.5))
    obj2 = DetectionResult("clock", 0.8, (0.75, 0.05, 0.95, 0.25))

    relationships, _ = builder.build_scene_graph([obj1, obj2])
    assert len(relationships) == 1
    assert relationships[0] == "chair LEFT OF clock"


def builder_obj(det: DetectionResult):
    from nano.scene_graph import SceneObject
    return SceneObject(det.class_name, det.bbox, det.confidence)
