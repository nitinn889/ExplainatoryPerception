"""
Single source of truth for the JSON event contract between the Nano (edge)
and the laptop (fog). Both sides import from here — never redefine this
shape independently.

Matches Section 3 of episodic_perception_build_spec.md.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    NEW_OBSERVATION = "new_observation"
    MOVED = "moved"
    PARTIAL_TIMEOUT = "partial_timeout"
    CONFIRMED = "confirmed"


class BBox(BaseModel):
    xmin: float = Field(ge=0.0, le=1.0)
    ymin: float = Field(ge=0.0, le=1.0)
    xmax: float = Field(ge=0.0, le=1.0)
    ymax: float = Field(ge=0.0, le=1.0)


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    objects: list[str]
    relationships: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    event_type: EventType
    bbox: BBox
    location_tag: str

    model_config = ConfigDict(use_enum_values=True)


def event_to_json(event: Event) -> str:
    return event.model_dump_json()


def event_from_json(payload: str) -> Event:
    return Event.model_validate_json(payload)
