"""
Scene graph -> natural language caption (Phase 6).

Template-based conversion of scene-graph relationship triples (as produced
by nano/scene_graph.py and carried in shared.event_schema.Event) into short
natural-language sentences, e.g. ["bottle", "ON", "table"] ->
"A bottle is on the table."
"""

from __future__ import annotations

import re

from shared.event_schema import Event, EventType

_RELATION_PHRASES = {
    "ON": "on",
    "LEFT OF": "to the left of",
    "RIGHT OF": "to the right of",
    "BEHIND": "behind",
}

_RELATIONSHIP_RE = re.compile(r"^(.*?)\s+(ON|LEFT OF|RIGHT OF|BEHIND)\s+(.*)$")


def parse_relationship(relationship: str) -> tuple[str, str, str]:
    """Split "bottle ON table" -> ("bottle", "ON", "table")."""
    match = _RELATIONSHIP_RE.match(relationship.strip())
    if not match:
        raise ValueError(f"Unrecognized relationship format: {relationship!r}")
    subject, relation, obj = match.groups()
    return subject.strip(), relation.strip(), obj.strip()


def caption_for_relationship(relationship: str) -> str:
    subject, relation, obj = parse_relationship(relationship)
    phrase = _RELATION_PHRASES.get(relation, relation.lower())
    return f"A {subject} is {phrase} the {obj}."


def caption_from_event(event: Event) -> str:
    """Build a caption for a full event, honoring event_type."""
    if not event.relationships:
        objects = ", ".join(event.objects) or "an object"
        base = f"Observed {objects}."
    else:
        base = " ".join(caption_for_relationship(r) for r in event.relationships)

    event_type = event.event_type
    if event_type == EventType.MOVED or event_type == "moved":
        return f"{base} (moved from its previous location)"
    if event_type == EventType.PARTIAL_TIMEOUT or event_type == "partial_timeout":
        return f"{base} (only partially visible — could not confirm a full view)"
    return base
