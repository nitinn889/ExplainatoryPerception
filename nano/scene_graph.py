"""
Scene graph construction (Phase 2).

Given a frame's list of detected objects with bboxes, computes pairwise
spatial relationships (ON, LEFT OF, RIGHT OF, BEHIND) via geometric rules
and outputs triples like ("bottle", "ON", "table") and formatted strings
like "bottle ON table".
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("nano.scene_graph")

# Spatial relationship types supported by downstream captioning
RELATION_ON = "ON"
RELATION_LEFT_OF = "LEFT OF"
RELATION_RIGHT_OF = "RIGHT OF"
RELATION_BEHIND = "BEHIND"

SUPPORTED_RELATIONS = {RELATION_ON, RELATION_LEFT_OF, RELATION_RIGHT_OF, RELATION_BEHIND}

# Surfaces commonly supporting objects
SURFACE_CLASSES = {"table", "dining table", "desk", "shelf", "bench", "counter", "chair"}


class SceneObject:
    """Standardized representation of an object for spatial reasoning."""

    def __init__(self, class_name: str, bbox: Tuple[float, float, float, float], confidence: float = 1.0):
        self.class_name = class_name
        self.xmin, self.ymin, self.xmax, self.ymax = bbox
        self.confidence = confidence

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0


def _horizontal_overlap(a: SceneObject, b: SceneObject) -> float:
    """Calculates horizontal 1D overlap between two bounding boxes."""
    overlap_min = max(a.xmin, b.xmin)
    overlap_max = min(a.xmax, b.xmax)
    return max(0.0, overlap_max - overlap_min)


def _vertical_overlap(a: SceneObject, b: SceneObject) -> float:
    """Calculates vertical 1D overlap between two bounding boxes."""
    overlap_min = max(a.ymin, b.ymin)
    overlap_max = min(a.ymax, b.ymax)
    return max(0.0, overlap_max - overlap_min)


def check_on_relationship(a: SceneObject, b: SceneObject, y_margin: float = 0.08, min_x_overlap_ratio: float = 0.25) -> bool:
    """
    Checks if object A is ON object B.
    A's bottom edge (ymax) is within a small vertical margin of B's top edge (ymin),
    and A's x-range overlaps B's x-range.
    If B is known not to be a surface (e.g. keyboard, mouse, bottle), it should not
    readily be considered a support surface for large items unless explicitly on it.
    """
    # An object cannot be ON another object if it is below it
    if a.ymin >= b.ymax:
        return False

    # Check vertical contact/proximity between A bottom and B top
    vertical_distance = abs(a.ymax - b.ymin)
    if vertical_distance > y_margin and not (b.ymin <= a.ymax <= b.ymin + y_margin):
        return False

    # Check horizontal overlap
    x_overlap = _horizontal_overlap(a, b)
    min_w = min(a.width, b.width)
    if min_w <= 0 or (x_overlap / min_w) < min_x_overlap_ratio:
        return False

    # A TV or large object is rarely on a smaller object like a keyboard
    if b.class_name not in SURFACE_CLASSES and a.area > b.area * 1.2:
        return False

    return True


def check_behind_relationship(a: SceneObject, b: SceneObject, min_x_overlap_ratio: float = 0.25) -> bool:
    """
    Approximation for depth without a depth sensor:
    In a perspective camera looking at a ground plane / scene,
    an object A that is BEHIND B has its bottom edge (ground contact point)
    higher in the image (a.ymax < b.ymax), its vertical center higher (a.center_y < b.center_y),
    and shares significant horizontal field of view.
    """
    # A must be further away (higher ground contact point and center)
    if a.ymax >= b.ymax or a.center_y >= b.center_y:
        return False

    # Must have horizontal overlap
    x_overlap = _horizontal_overlap(a, b)
    min_w = min(a.width, b.width)
    if min_w <= 0 or (x_overlap / min_w) < min_x_overlap_ratio:
        return False

    # Not ON
    if check_on_relationship(a, b):
        return False

    return True


def check_left_right_relationship(a: SceneObject, b: SceneObject, x_margin: float = 0.05) -> Optional[str]:
    """
    Determines if A is LEFT OF or RIGHT OF B based on relative x-centers.
    """
    if a.center_x < b.center_x - x_margin:
        return RELATION_LEFT_OF
    elif a.center_x > b.center_x + x_margin:
        return RELATION_RIGHT_OF
    return None


def infer_spatial_relation(a: SceneObject, b: SceneObject) -> Optional[str]:
    """
    Infers the most salient relationship between A (subject) and B (object).
    Priority order: ON > BEHIND > LEFT OF / RIGHT OF.
    """
    if check_on_relationship(a, b):
        return RELATION_ON

    if check_behind_relationship(a, b):
        return RELATION_BEHIND

    lr = check_left_right_relationship(a, b)
    if lr:
        return lr

    return None


class SceneGraphBuilder:
    """Constructs symbolic spatial triples from raw detections."""

    def __init__(self, y_margin: float = 0.08, x_margin: float = 0.05):
        self.y_margin = y_margin
        self.x_margin = x_margin

    def build_scene_graph(
        self,
        detections: List[Any],
    ) -> Tuple[List[str], List[str]]:
        """
        Takes detections (DetectionResult objects or dicts) and returns:
        1. List of relationship strings e.g. ["bottle ON table"]
        2. Ordered list of object labels (primary movable subject first)
        """
        if not detections:
            return [], []

        # Convert detections to SceneObject instances
        scene_objs: List[SceneObject] = []
        for d in detections:
            if hasattr(d, "class_name") and hasattr(d, "bbox"):
                scene_objs.append(SceneObject(d.class_name, d.bbox, getattr(d, "confidence", 1.0)))
            elif isinstance(d, dict):
                bbox = d.get("bbox")
                if isinstance(bbox, dict):
                    bbox_tuple = (bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])
                elif isinstance(bbox, (list, tuple)):
                    bbox_tuple = tuple(bbox)
                else:
                    continue
                scene_objs.append(SceneObject(d.get("class", "object"), bbox_tuple, d.get("confidence", 1.0)))

        if not scene_objs:
            return [], []

        if len(scene_objs) == 1:
            return [], [scene_objs[0].class_name]

        # First check if there are any ON relationships
        # Prioritize movable subjects (non-surfaces) on surfaces
        relationships: List[str] = []
        objects_seen: List[str] = []

        n = len(scene_objs)
        # Check ON pairs first: movable object on surface
        for i in range(n):
            a = scene_objs[i]
            for j in range(n):
                if i == j:
                    continue
                b = scene_objs[j]
                if check_on_relationship(a, b, y_margin=self.y_margin):
                    rel_str = f"{a.class_name} {RELATION_ON} {b.class_name}"
                    if rel_str not in relationships:
                        relationships.append(rel_str)
                        if a.class_name not in objects_seen:
                            objects_seen.append(a.class_name)
                        if b.class_name not in objects_seen:
                            objects_seen.append(b.class_name)

        # If no ON relationship found, infer BEHIND or LEFT OF / RIGHT OF in original/left-to-right order
        if not relationships:
            # Sort from left to right so subjects naturally flow left-to-right or depth-wise
            sorted_objs = sorted(scene_objs, key=lambda obj: (obj.xmin, obj.ymin))
            for i in range(len(sorted_objs)):
                a = sorted_objs[i]
                for j in range(i + 1, len(sorted_objs)):
                    b = sorted_objs[j]
                    rel = infer_spatial_relation(a, b)
                    if rel:
                        rel_str = f"{a.class_name} {rel} {b.class_name}"
                        relationships.append(rel_str)
                        if a.class_name not in objects_seen:
                            objects_seen.append(a.class_name)
                        if b.class_name not in objects_seen:
                            objects_seen.append(b.class_name)
        # Ensure all detected object names are in the objects list
        for obj in scene_objs:
            if obj.class_name not in objects_seen:
                objects_seen.append(obj.class_name)

        return relationships, objects_seen
