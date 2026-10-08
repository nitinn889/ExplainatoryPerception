"""
Tests for Phase 5 (laptop/event_server.py and nano/event_client.py)
and end-to-end simulation from nano/main_loop.py.
"""

from datetime import datetime, timezone

import pytest

# Starlette's TestClient is built on httpx, which is a test-only dependency
# (see requirements-dev.txt) rather than something the demo needs at runtime.
pytest.importorskip(
    "httpx", reason="needs the test extras: pip install -r requirements-dev.txt"
)

from fastapi.testclient import TestClient

from laptop.event_server import app, clear_events, received_events
from nano.event_client import EventClient
from nano.main_loop import NanoPatrolAgent
from shared.event_schema import BBox, Event, EventType


@pytest.fixture(autouse=True)
def clean_event_store():
    clear_events()
    yield
    clear_events()


def test_event_server_post_and_get():
    client = TestClient(app)

    event = Event(
        objects=["bottle", "table"],
        relationships=["bottle ON table"],
        confidence=0.94,
        event_type=EventType.CONFIRMED,
        bbox=BBox(xmin=0.2, ymin=0.2, xmax=0.5, ymax=0.7),
        location_tag="lab_desk_3",
    )

    # POST /event
    response = client.post("/event", json=event.model_dump(mode="json"))
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ok"
    assert data["event_id"] == event.event_id

    # GET /events
    get_resp = client.get("/events")
    assert get_resp.status_code == 200
    events = get_resp.json()
    assert len(events) == 1
    assert events[0]["objects"] == ["bottle", "table"]
    assert events[0]["relationships"] == ["bottle ON table"]
    assert events[0]["location_tag"] == "lab_desk_3"

    # GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["received_count"] == 1


def test_nano_agent_simulation_end_to_end(monkeypatch):
    """
    Test the full agent simulation scenario:
    Camera captures synthetic partial bottle -> Repositioning pulses -> Stabilized ->
    Emitted -> Scene graph built -> Importance filter passes -> Event client sends.
    """
    test_client = TestClient(app)
    agent = NanoPatrolAgent(
        camera_mode="synthetic",
        laptop_url="http://testserver",
        location_tag="lab_test_zone",
        force_mock_motor=True,
    )

    # Monkeypatch agent.event_client session to use TestClient
    def mock_post(url, data, headers, timeout):
        class MockResponse:
            status_code = 201
            text = '{"status":"ok"}'
        # Route to TestClient
        resp = test_client.post("/event", content=data, headers=headers)
        mock_resp = MockResponse()
        mock_resp.status_code = resp.status_code
        mock_resp.text = resp.text
        return mock_resp

    monkeypatch.setattr(agent.event_client.session, "post", mock_post)

    events_sent = agent.run_simulation_scenario(steps=10)
    assert events_sent >= 1

    # Verify event stored in fog server
    assert len(received_events) >= 1
    bottle_events = [e for e in received_events if "bottle" in e.objects]
    assert len(bottle_events) >= 1
    stored = bottle_events[0]
    assert stored.objects[0] == "bottle"
    assert stored.location_tag == "lab_test_zone"
    assert stored.event_type in (EventType.CONFIRMED, EventType.NEW_OBSERVATION)
