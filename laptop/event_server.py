"""
FastAPI event receiver (Phase 5).

Exposes POST /event accepting shared.event_schema.Event, logging received
events to a local file/log (data/received_events.jsonl).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
import uvicorn

from shared.event_schema import Event, event_to_json

logger = logging.getLogger("laptop.event_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Episodic Perception - Event Receiver",
    description="Fog-side server receiving confirmed episodic perception events from the Jetson Nano.",
    version="1.0.0",
)

# Persistence file
LOG_DIR = Path("data")
LOG_FILE = LOG_DIR / "received_events.jsonl"

# In-memory buffer of recent events
received_events: List[Event] = []


def _append_to_log(event: Event) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(event_to_json(event) + "\n")


@app.post("/event", status_code=status.HTTP_201_CREATED)
async def receive_event(event: Event):
    """Receive an episodic event from the Nano edge device."""
    try:
        received_events.append(event)
        _append_to_log(event)
        logger.info(
            f"Received event [{event.event_id[:8]}] type={event.event_type} "
            f"objects={event.objects} relationships={event.relationships} location={event.location_tag}"
        )
        return {
            "status": "ok",
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": str(event.event_type),
        }
    except Exception as e:
        logger.error(f"Error processing event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events", response_model=List[Event])
async def list_events(limit: Optional[int] = 50):
    """Inspect recently received events (most recent first)."""
    return list(reversed(received_events[-limit:]))


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "received_count": len(received_events),
        "log_path": str(LOG_FILE),
    }


def clear_events() -> None:
    """Helper for test cleanup."""
    received_events.clear()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
