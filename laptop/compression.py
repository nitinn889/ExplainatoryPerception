"""
Memory compression (Phase 7).

When a new observation matches an existing unclosed episode for the same
object/location/relationship, extends that episode's duration instead of
creating a new record (e.g. "bottle on Desk 3, 10:00-10:30" instead of many
duplicate rows).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from shared.event_schema import Event

DEFAULT_GAP_TOLERANCE = timedelta(minutes=5)


@dataclass
class Episode:
    key: tuple
    objects: list[str]
    relationships: list[str]
    location_tag: str
    event_type: str
    confidence: float
    start_time: datetime
    end_time: datetime
    event_ids: list[str] = field(default_factory=list)
    observation_count: int = 0


def _episode_key(event: Event) -> tuple:
    """Episodes are keyed by the (unordered) object set, location, and
    relationship set — a re-observation of the same fact in the same place."""
    return (tuple(sorted(event.objects)), event.location_tag, tuple(sorted(event.relationships)))


class EpisodeCompressor:
    """Merges repeated observations into duration-based episodes instead of
    one stored row per observed frame."""

    def __init__(self, gap_tolerance: timedelta = DEFAULT_GAP_TOLERANCE):
        self._gap_tolerance = gap_tolerance
        self._open_episodes: dict[tuple, Episode] = {}

    def ingest(self, event: Event) -> tuple[str, Episode]:
        """Extend a matching open episode, or start a new one.

        Returns ("extended", episode) or ("new", episode).
        """
        key = _episode_key(event)
        existing = self._open_episodes.get(key)

        if existing is not None and (event.timestamp - existing.end_time) <= self._gap_tolerance:
            existing.end_time = event.timestamp
            existing.event_ids.append(event.event_id)
            existing.observation_count += 1
            existing.confidence = max(existing.confidence, event.confidence)
            return "extended", existing

        episode = Episode(
            key=key,
            objects=list(event.objects),
            relationships=list(event.relationships),
            location_tag=event.location_tag,
            event_type=event.event_type,
            confidence=event.confidence,
            start_time=event.timestamp,
            end_time=event.timestamp,
            event_ids=[event.event_id],
            observation_count=1,
        )
        self._open_episodes[key] = episode
        return "new", episode

    def open_episodes(self) -> list[Episode]:
        return list(self._open_episodes.values())

    def close(self, episode: Episode) -> None:
        """Remove an episode from the open set (e.g. after a contradiction
        moves the object elsewhere, or it's been flushed to memory_store)."""
        self._open_episodes.pop(episode.key, None)
