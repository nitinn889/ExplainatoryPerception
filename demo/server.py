"""
Demo server: the fog node plus the live dashboard.

One process that:
  * serves the dashboard at  GET  /
  * receives edge events at  POST /event          (shared.event_schema.Event)
  * receives frame state at  POST /telemetry
  * answers questions at     GET  /query?q=...    (Phase 8 RAG)
  * handles spoken input at  POST /voice          (transcript -> command or question)
  * pushes everything to the browser over WS /ws

`POST /event` and `GET /health` are the same contract `laptop/event_server.py`
exposes, so `nano.event_client.EventClient` talks to this without modification -
the Webots controller, the 2D sim and a real Jetson Nano on the LAN are all
interchangeable clients.

Run it with `python -m demo.run_demo` (starts the 2D sim too) or
`uvicorn demo.server:app --port 8080` for the Webots / real-hardware case.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, JSONResponse

from demo.fog_pipeline import FogPipeline
from laptop.voice import route_utterance
from shared.event_schema import Event

logger = logging.getLogger("demo.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"
DASHBOARD = STATIC_DIR / "dashboard.html"

# "1" keeps episodes in ChromaDB under data/memory_store/ across runs;
# "0" (the default for the demo) starts from an empty store every time, which is
# what you want when demonstrating live.
PERSIST = os.environ.get("EPISODIC_PERSIST", "0") == "1"


class Hub:
    """Fan-out of pipeline updates to every connected browser."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = asyncio.Lock()
        self.last_telemetry: dict[str, Any] = {}
        self.recent_events: list[dict[str, Any]] = []

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def register(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)

    async def unregister(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def _broadcast(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message)
        async with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                await client.send_text(payload)
            except Exception:
                await self.unregister(client)

    def publish(self, message: dict[str, Any]) -> None:
        """Thread-safe: callable from the edge-agent thread or a request handler."""
        if message.get("kind") == "telemetry":
            self.last_telemetry = message["data"]
        elif message.get("kind") == "episode":
            self.recent_events.append(message["data"])
            del self.recent_events[:-60]

        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)


hub = Hub()
pipeline = FogPipeline(persist=PERSIST)
agent = None  # set by run_demo when the 2D sim is in use


def attach_agent(edge_agent: Any) -> None:
    global agent
    agent = edge_agent


@asynccontextmanager
async def lifespan(_: FastAPI):
    hub.bind_loop(asyncio.get_running_loop())
    logger.info(
        "Fog node ready | store=%s | embeddings=%s",
        getattr(pipeline.store, "backend", "?"),
        pipeline.snapshot()["embedding_backend"],
    )
    yield
    if agent is not None:
        agent.stop()


app = FastAPI(title="Episodic Perception - Live Demo", lifespan=lifespan)


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD)


@app.get("/health")
def health() -> dict[str, Any]:
    """Same shape laptop/event_server.py returns, so EventClient's connection
    check passes against either server."""
    snapshot = pipeline.snapshot()
    return {
        "status": "healthy",
        "received_count": snapshot["raw_events"],
        "stored_episodes": snapshot["stored_episodes"],
        "store_backend": snapshot["store_backend"],
        "embedding_backend": snapshot["embedding_backend"],
    }


# --------------------------------------------------------------------------
# Edge -> fog
# --------------------------------------------------------------------------

@app.post("/event", status_code=status.HTTP_201_CREATED)
def receive_event(event: Event) -> dict[str, Any]:
    result = pipeline.ingest(event)
    hub.publish({"kind": "episode", "data": result.to_dict()})
    return {
        "status": "ok",
        "event_id": event.event_id,
        "action": result.action,
        "moved": result.moved,
        "stored_episodes": result.stored_episodes,
    }


@app.post("/telemetry")
def receive_telemetry(payload: dict[str, Any]) -> dict[str, str]:
    hub.publish({"kind": "telemetry", "data": payload})
    return {"status": "ok"}


# --------------------------------------------------------------------------
# User -> fog
# --------------------------------------------------------------------------

@app.get("/query")
def query(q: str, k: int = 4) -> dict[str, Any]:
    result = pipeline.answer(q, k=k)
    hub.publish({"kind": "answer", "data": result})
    return result


@app.get("/state")
def state() -> dict[str, Any]:
    """Everything a freshly-opened dashboard needs to render immediately."""
    return {
        "snapshot": pipeline.snapshot(),
        "telemetry": hub.last_telemetry,
        "recent": hub.recent_events[-20:],
        "agent": {
            "available": agent is not None,
            "running": bool(agent and agent.running),
            "paused": bool(agent and agent.paused),
        },
    }


def _apply_control(action: str) -> tuple[int, dict[str, Any]]:
    """Run a control action. Shared by the dashboard buttons and /voice so a
    spoken "pause" and the Pause button cannot drift apart."""
    if agent is None:
        return 409, {"status": "unavailable", "detail": "No 2D sim attached; the edge is external."}

    if action == "start":
        agent.start()
    elif action == "pause":
        agent.pause()
    elif action == "resume":
        agent.resume()
    elif action == "move_bottle":
        if not agent.move_bottle():
            return 400, {"status": "error", "detail": "no bottle in the world"}
    else:
        return 400, {"status": "error", "detail": f"unknown action {action!r}"}

    return 200, {"status": "ok", "running": agent.running, "paused": agent.paused}


@app.post("/control/{action}")
def control(action: str) -> JSONResponse:
    """Dashboard buttons for the 2D sim (no-ops when driving from Webots)."""
    code, body = _apply_control(action)
    return JSONResponse(body, status_code=code)


@app.post("/voice")
def voice(payload: dict[str, Any]) -> JSONResponse:
    """One spoken utterance, already transcribed by the browser.

    Recognition and synthesis stay in the browser (Web Speech API); the
    decision of what an utterance *means* lives here, in `laptop.voice`, where
    it is testable without audio hardware. Every response carries a `spoken`
    string for the browser to read back.
    """
    transcript = str(payload.get("text") or "").strip()
    routed = route_utterance(transcript)
    kind = routed["kind"]

    if kind == "empty":
        return JSONResponse({"kind": kind, "transcript": transcript, "spoken": routed["say"]})

    if kind == "command":
        code, body = _apply_control(routed["action"])
        spoken = (
            routed["say"]
            if code == 200
            else "I can't do that from here — the edge is Webots or real hardware."
        )
        return JSONResponse(
            {
                "kind": kind,
                "transcript": transcript,
                "action": routed["action"],
                "spoken": spoken,
                "control": body,
            },
            status_code=200 if code in (200, 409) else code,
        )

    result = pipeline.answer(routed["question"], k=4)
    # Same broadcast /query makes, so every open dashboard shows the answer
    # to a question that was only ever spoken aloud at one of them.
    hub.publish({"kind": "answer", "data": result})
    return JSONResponse({"kind": kind, "transcript": transcript, **result})


# --------------------------------------------------------------------------
# Live push
# --------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await hub.register(websocket)
    try:
        await websocket.send_text(json.dumps({"kind": "hello", "data": state()}))
        while True:
            # The dashboard does not send anything; this just detects a close.
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await hub.unregister(websocket)
