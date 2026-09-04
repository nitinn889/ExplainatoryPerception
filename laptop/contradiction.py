"""
Contradiction detection (Phase 7).

When an object's new location conflicts with its last known location,
infers and retypes the event as "moved" (event_type: "moved") rather than
letting it get stored as an unrelated new fact.
"""

from __future__ import annotations

from shared.event_schema import Event, EventType


class LocationTracker:
    """Tracks each object's last known location tag.

    Simplifying assumption: the first entry in `event.objects` is the
    tracked subject (the thing that can move); later entries are reference
    objects (e.g. "table" in "bottle ON table"). This mirrors how
    scene_graph.py orders its relationship triples.
    """

    def __init__(self):
        self._last_location: dict[str, str] = {}

    def process(self, event: Event) -> Event:
        """Return `event`, retyped to `moved` if it contradicts the last
        known location for its subject object."""
        if not event.objects:
            return event

        subject = event.objects[0]
        prior_location = self._last_location.get(subject)
        self._last_location[subject] = event.location_tag

        contradicts = (
            prior_location is not None
            and prior_location != event.location_tag
            and event.event_type != EventType.PARTIAL_TIMEOUT
        )
        if contradicts:
            return event.model_copy(update={"event_type": EventType.MOVED.value})
        return event

    def last_known_location(self, subject: str) -> str | None:
        return self._last_location.get(subject)
