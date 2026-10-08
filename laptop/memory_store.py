"""
Vector memory store (Phase 6) — wraps ChromaDB (vectors + metadata in one)
for episodic memory. Captions come from captioning.py, vectors from
embeddings.py.

Provides add_episode(...) and search(query, k) per the build spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from laptop.embeddings import embed, embed_batch
from shared.event_schema import Event

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "memory_store"
COLLECTION_NAME = "episodic_memory"


class MemoryStore:
    def __init__(self, persist_dir: str | Path = DEFAULT_PERSIST_DIR):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    def add_episode(self, event: Event, caption: str, display_caption: str | None = None) -> str:
        """Embed `caption` and store it alongside the event's metadata.

        `caption` is what gets embedded, so callers may append retrieval hints
        to it. Pass `display_caption` to keep the clean sentence for showing to
        a user.
        """
        vector = embed(caption)
        metadata: dict[str, Any] = {
            "display_caption": display_caption or caption,
            "objects": ", ".join(event.objects),
            "relationships": ", ".join(event.relationships),
            "confidence": event.confidence,
            "event_type": event.event_type,
            "location_tag": event.location_tag,
            "timestamp": str(event.timestamp),
        }
        self._collection.add(
            ids=[event.event_id],
            embeddings=[vector],
            documents=[caption],
            metadatas=[metadata],
        )
        return event.event_id

    def add_episodes(self, events: list[Event], captions: list[str]) -> list[str]:
        if len(events) != len(captions):
            raise ValueError("events and captions must be the same length")
        vectors = embed_batch(captions)
        metadatas = [
            {
                "objects": ", ".join(e.objects),
                "relationships": ", ".join(e.relationships),
                "confidence": e.confidence,
                "event_type": e.event_type,
                "location_tag": e.location_tag,
                "timestamp": str(e.timestamp),
            }
            for e in events
        ]
        ids = [e.event_id for e in events]
        self._collection.add(
            ids=ids, embeddings=vectors, documents=captions, metadatas=metadatas
        )
        return ids

    def update_episode(self, event_id: str, caption: str, metadata: dict[str, Any]) -> None:
        """Re-write a stored episode in place.

        Phase 7's compression extends an open episode's end_time instead of
        adding a row per observation; this is how that extension reaches
        storage, so the stored episode keeps saying "10:00-10:30" rather than
        going stale at its start time.
        """
        self._collection.update(
            ids=[event_id],
            embeddings=[embed(caption)],
            documents=[caption],
            metadatas=[metadata],
        )

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Semantic search: returns up to k episodes ranked by similarity."""
        vector = embed(query)
        result = self._collection.query(query_embeddings=[vector], n_results=k)

        episodes: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for event_id, caption, metadata, distance in zip(ids, documents, metadatas, distances):
            episodes.append(
                {
                    "event_id": event_id,
                    "caption": caption,
                    "distance": distance,
                    **metadata,
                }
            )
        return episodes

    def count(self) -> int:
        return self._collection.count()
