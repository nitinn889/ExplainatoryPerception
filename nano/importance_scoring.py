"""
Importance scoring (Phase 4).

Given a new confirmed scene-graph observation and the last known state for
that object/location, decides is_new / is_changed / is_unusual, so only
meaningful events get forwarded to event_client.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from shared.event_schema import Event, EventType

logger = logging.getLogger("nano.importance_scoring")


@dataclass
class ObjectState:
    object_name: str
    location_tag: str
    relationships: Set[str]
    center_x: float
    center_y: float
    last_event_type: EventType
    last_seen: datetime
    observation_count: int = 1


class ImportanceScorer:
    """
    Evaluates whether an observation carries novel or notable information
    worthy of transmission from edge (Nano) to fog (Laptop).
    """

    def __init__(self, displacement_threshold: float = 0.25):
        self.displacement_threshold = displacement_threshold
        # Key: object_name -> ObjectState
        self._known_objects: Dict[str, ObjectState] = {}

    def reset(self) -> None:
        """Clear memory cache."""
        self._known_objects.clear()

    def evaluate(self, event: Event) -> Tuple[bool, str]:
        """
        Evaluates an event.
        Returns:
            (should_send: bool, reason: str)
            reason is one of: "new_observation", "moved_location", "changed_relationship",
                              "significant_displacement", "partial_timeout", "unchanged_duplicate"
        """
        # Event must have at least one object
        primary_object = event.objects[0] if event.objects else "unknown"
        current_rels = set(event.relationships)
        cx = (event.bbox.xmin + event.bbox.xmax) / 2.0
        cy = (event.bbox.ymin + event.bbox.ymax) / 2.0

        # Partial timeouts are always important to record as they signal partial visibility
        if event.event_type == EventType.PARTIAL_TIMEOUT or event.event_type == "partial_timeout":
            self._update_state(primary_object, event, current_rels, cx, cy)
            return True, "partial_timeout"

        # 1. Check if new observation
        if primary_object not in self._known_objects:
            self._update_state(primary_object, event, current_rels, cx, cy)
            return True, "new_observation"

        prev_state = self._known_objects[primary_object]

        # 2. Check if location changed (e.g. moved between rooms/desks)
        if event.location_tag != prev_state.location_tag:
            self._update_state(primary_object, event, current_rels, cx, cy)
            return True, "moved_location"

        # 3. Check if relationships changed (e.g. bottle ON table -> bottle LEFT OF laptop)
        if current_rels != prev_state.relationships:
            self._update_state(primary_object, event, current_rels, cx, cy)
            return True, "changed_relationship"

        # 4. Check if spatial displacement within same view is significant
        dx = cx - prev_state.center_x
        dy = cy - prev_state.center_y
        displacement = (dx * dx + dy * dy) ** 0.5
        if displacement >= self.displacement_threshold:
            self._update_state(primary_object, event, current_rels, cx, cy)
            return True, "significant_displacement"

        # 5. Otherwise, this is a redundant / unchanged duplicate observation
        prev_state.observation_count += 1
        prev_state.last_seen = event.timestamp
        return False, "unchanged_duplicate"

    def _update_state(
        self,
        primary_object: str,
        event: Event,
        current_rels: Set[str],
        cx: float,
        cy: float,
    ) -> None:
        self._known_objects[primary_object] = ObjectState(
            object_name=primary_object,
            location_tag=event.location_tag,
            relationships=current_rels,
            center_x=cx,
            center_y=cy,
            last_event_type=event.event_type if isinstance(event.event_type, EventType) else EventType(event.event_type),
            last_seen=event.timestamp,
        )

    def filter_event(self, event: Event) -> Optional[Event]:
        """
        Filters event; returns the event if it should be sent to fog, or None if skipped.
        If the object moved location, sets event.event_type = EventType.MOVED.
        """
        should_send, reason = self.evaluate(event)
        if not should_send:
            return None

        if reason == "moved_location":
            event.event_type = EventType.MOVED

        return event
