"""
The fog (laptop) side of the demo, wired end to end.

Phases 6-8 exist as independent, unit-tested modules in `laptop/`; this is the
glue that runs them in order on each incoming event, which is what the live
demo needs and what the evaluation numbers are measured against:

    Event (from the Nano / Webots / 2D sim)
      -> contradiction.LocationTracker   retype to `moved` on a location conflict
      -> captioning.caption_from_event   scene-graph triples -> a sentence
      -> compression.EpisodeCompressor   extend an open episode, or open a new one
      -> memory_store.MemoryStore        embed + store (or update in place)
      -> rag_query.answer_question       on demand, for user questions

Storage degrades gracefully: if ChromaDB is unavailable, an in-process cosine
store with the same three-method surface is used instead, so the demo runs on a
machine with nothing but numpy installed.
"""

from __future__ import annotations

import atexit
import logging
import math
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from laptop import embeddings
from laptop.captioning import caption_from_event
from laptop.compression import Episode, EpisodeCompressor
from laptop.contradiction import LocationTracker
from laptop.voice import attach_spoken
from shared.event_schema import Event, EventType

logger = logging.getLogger("demo.fog_pipeline")


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

class InMemoryStore:
    """Fallback vector store with MemoryStore's surface.

    Brute-force cosine similarity over a Python list. Fine for a demo-sized
    store (tens to thousands of episodes) and it removes ChromaDB from the list
    of things that can fail in front of an audience.
    """

    backend = "in-memory (cosine)"

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def add_episode(self, event: Event, caption: str, display_caption: str | None = None) -> str:
        self._rows[event.event_id] = {
            "event_id": event.event_id,
            "caption": caption,
            "display_caption": display_caption or caption,
            "vector": embeddings.embed(caption),
            "objects": ", ".join(event.objects),
            "relationships": ", ".join(event.relationships),
            "confidence": event.confidence,
            "event_type": str(event.event_type),
            "location_tag": event.location_tag,
            "timestamp": str(event.timestamp),
        }
        return event.event_id

    def update_episode(self, event_id: str, caption: str, metadata: dict[str, Any]) -> None:
        row = self._rows.get(event_id)
        if row is None:
            return
        row.update(metadata)
        row["caption"] = caption
        row["vector"] = embeddings.embed(caption)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        query_vector = embeddings.embed(query)

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0

        scored = []
        for row in self._rows.values():
            similarity = cosine(query_vector, row["vector"])
            item = {kk: vv for kk, vv in row.items() if kk != "vector"}
            item["distance"] = round(1.0 - similarity, 6)
            scored.append((similarity, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:k]]

    def count(self) -> int:
        return len(self._rows)


def build_store(persist: bool = True) -> Any:
    """The real Phase 6 ChromaDB store if it works, the fallback otherwise.

    `persist=False` still uses ChromaDB - the demo should exercise the actual
    deliverable - but points it at a throwaway directory, so each run starts
    from an empty memory without touching anything already in
    `data/memory_store/`.
    """
    try:
        from laptop.memory_store import MemoryStore

        if persist:
            store = MemoryStore()
            store.backend = "chromadb (persisted to data/memory_store/)"
            return store

        temp_dir = tempfile.mkdtemp(prefix="episodic-demo-store-")
        atexit.register(shutil.rmtree, temp_dir, True)
        store = MemoryStore(persist_dir=temp_dir)
        store.backend = "chromadb (fresh, discarded on exit)"
        return store
    except Exception as exc:
        logger.warning(
            "ChromaDB unavailable (%s); falling back to the in-process cosine store.", exc
        )
        return InMemoryStore()


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

@dataclass
class StoredEpisode:
    """What the UI and the evaluation read back."""

    episode_id: str
    caption: str
    objects: list[str]
    relationships: list[str]
    location_tag: str
    event_type: str
    confidence: float
    start_time: datetime
    end_time: datetime
    observation_count: int

    @property
    def duration_sec(self) -> float:
        return max(0.0, (self.end_time - self.start_time).total_seconds())

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "caption": self.caption,
            "objects": self.objects,
            "relationships": self.relationships,
            "location_tag": self.location_tag,
            "event_type": self.event_type,
            "confidence": round(self.confidence, 3),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_sec": round(self.duration_sec, 1),
            "observation_count": self.observation_count,
        }


@dataclass
class IngestResult:
    """One event's trip through the fog side, as the dashboard displays it."""

    event: Event
    caption: str
    action: str  # "new" | "extended"
    moved: bool
    episode: StoredEpisode
    raw_events: int
    stored_episodes: int

    @property
    def compression_ratio(self) -> float:
        return self.raw_events / self.stored_episodes if self.stored_episodes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.model_dump(mode="json"),
            "caption": self.caption,
            "action": self.action,
            "moved": self.moved,
            "episode": self.episode.to_dict(),
            "raw_events": self.raw_events,
            "stored_episodes": self.stored_episodes,
            "compression_ratio": round(self.compression_ratio, 2),
        }


@dataclass
class FogPipeline:
    """Thread-safe front door to the fog side."""

    persist: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.store = build_store(self.persist)
        self.location_tracker = LocationTracker()
        self.compressor = EpisodeCompressor()
        self.raw_events = 0
        # episode key -> the event_id its row is stored under
        self._episode_rows: dict[tuple, str] = {}
        self._episodes: dict[str, StoredEpisode] = {}
        self._history: list[IngestResult] = []
        self.query_latencies_ms: list[float] = []

    # -- ingest ------------------------------------------------------------
    def ingest(self, event: Event) -> IngestResult:
        with self._lock:
            self.raw_events += 1

            # 1. Contradiction detection, before captioning, so a relocation is
            #    described as a move rather than as an unrelated new fact.
            resolved = self.location_tracker.process(event)
            moved = str(resolved.event_type) == EventType.MOVED.value and str(
                event.event_type
            ) != EventType.MOVED.value

            # 2. Caption.
            caption = caption_from_event(resolved)

            # 3. Compression: extend an open episode or open a new one.
            action, episode = self.compressor.ingest(resolved)

            if action == "new":
                episode_id = resolved.event_id
                self._episode_rows[episode.key] = episode_id
                stored = self._as_stored(episode_id, caption, episode)
                self._episodes[episode_id] = stored
                self.store.add_episode(
                    resolved, self._storage_caption(stored), display_caption=caption
                )
            else:
                episode_id = self._episode_rows.get(episode.key, resolved.event_id)
                stored = self._as_stored(episode_id, caption, episode)
                self._episodes[episode_id] = stored
                try:
                    self.store.update_episode(
                        episode_id,
                        self._storage_caption(stored),
                        self._metadata(stored),
                    )
                except Exception as exc:
                    logger.debug("Episode update skipped (%s)", exc)

            result = IngestResult(
                event=resolved,
                caption=caption,
                action=action,
                moved=moved,
                episode=stored,
                raw_events=self.raw_events,
                stored_episodes=len(self._episodes),
            )
            self._history.append(result)
            logger.info(
                "[fog] %s episode %s | %s | raw=%d stored=%d ratio=%.2fx",
                action,
                episode_id[:8],
                caption,
                self.raw_events,
                len(self._episodes),
                result.compression_ratio,
            )
            return result

    def _as_stored(self, episode_id: str, caption: str, episode: Episode) -> StoredEpisode:
        return StoredEpisode(
            episode_id=episode_id,
            caption=caption,
            objects=list(episode.objects),
            relationships=list(episode.relationships),
            location_tag=episode.location_tag,
            event_type=str(episode.event_type),
            confidence=episode.confidence,
            start_time=episode.start_time,
            end_time=episode.end_time,
            observation_count=episode.observation_count,
        )

    @staticmethod
    def _storage_caption(stored: StoredEpisode) -> str:
        """The text that actually gets embedded.

        Three things are appended to the human-readable caption, because all
        three are things people ask about and none of them survive in the
        sentence alone:

          * the object names, so "where is my bottle?" matches an episode whose
            caption happens to lead with the surface it sits on
          * the location tag
          * the event type, so "did anything move?" finds the `moved` episodes
            instead of whichever caption happens to share the most words
          * the duration, once a compressed episode spans more than a moment,
            so it retrieves as the one durable fact it represents rather than as
            an instant
        """
        parts = [stored.caption, f"[objects: {', '.join(stored.objects)}"]
        parts.append(f"| location: {stored.location_tag}")
        parts.append(f"| event: {stored.event_type}")
        if stored.observation_count > 1:
            parts.append(
                f"| seen {stored.observation_count} times over {stored.duration_sec:.0f}s"
            )
        return " ".join(parts) + "]"

    @staticmethod
    def _metadata(stored: StoredEpisode) -> dict[str, Any]:
        return {
            "objects": ", ".join(stored.objects),
            "relationships": ", ".join(stored.relationships),
            "confidence": stored.confidence,
            "event_type": stored.event_type,
            "location_tag": stored.location_tag,
            "timestamp": str(stored.end_time),
            "observation_count": stored.observation_count,
            # The embedded document carries retrieval hints (object names,
            # location, duration); this is the clean sentence to show a user.
            "display_caption": stored.caption,
        }

    # -- query -------------------------------------------------------------
    def answer(self, question: str, k: int = 4) -> dict[str, Any]:
        from laptop.rag_query import answer_question

        started = time.perf_counter()
        result = answer_question(question, self.store, k=k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.query_latencies_ms.append(latency_ms)
        result["latency_ms"] = round(latency_ms, 1)
        result["embedding_backend"] = embeddings.active_backend()
        # Both the typed and the spoken path come through here, so the
        # read-aloud phrasing is always available to whoever asked.
        return attach_spoken(result)

    # -- introspection -----------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            episodes = sorted(
                (e.to_dict() for e in self._episodes.values()),
                key=lambda e: e["end_time"],
                reverse=True,
            )
            latencies = sorted(self.query_latencies_ms)
            return {
                "raw_events": self.raw_events,
                "stored_episodes": len(self._episodes),
                "compression_ratio": round(
                    self.raw_events / len(self._episodes), 2
                )
                if self._episodes
                else 0.0,
                "episodes": episodes,
                "store_backend": getattr(self.store, "backend", type(self.store).__name__),
                "embedding_backend": embeddings.active_backend(),
                "query_count": len(latencies),
                "query_latency_ms_mean": round(sum(latencies) / len(latencies), 1)
                if latencies
                else None,
                "query_latency_ms_p95": round(
                    latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 1
                )
                if latencies
                else None,
            }

    def history(self, limit: int = 40) -> list[dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self._history[-limit:]]
