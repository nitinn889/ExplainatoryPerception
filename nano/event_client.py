"""
Nano -> laptop event sender (Phase 5).

POSTs confirmed events (shared.event_schema.Event) to the laptop's
event_server.py over HTTP, on the shared LAN/Wi-Fi.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from shared.event_schema import Event, event_to_json

logger = logging.getLogger("nano.event_client")


class EventClient:
    """HTTP client on the Jetson Nano to transmit events to the companion laptop."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: float = 3.0,
        max_retries: int = 2,
    ):
        # Normalize url
        self.server_url = server_url.rstrip("/")
        self.event_endpoint = f"{self.server_url}/event"
        self.health_endpoint = f"{self.server_url}/health"
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def check_connection(self) -> bool:
        """Verifies connectivity to the fog server health endpoint."""
        try:
            resp = self.session.get(self.health_endpoint, timeout=self.timeout)
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"Connection check failed to {self.health_endpoint}: {e}")
            return False

    def send_event(self, event: Event) -> bool:
        """
        Sends an Event to the laptop event server.
        Retries up to max_retries on transient network errors.
        """
        payload = event_to_json(event)
        headers = {"Content-Type": "application/json"}

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(
                    self.event_endpoint,
                    data=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code in (200, 201):
                    logger.info(f"Successfully transmitted event {event.event_id[:8]} to fog server.")
                    return True
                else:
                    logger.warning(
                        f"Fog server rejected event with HTTP {resp.status_code}: {resp.text}"
                    )
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed sending event: {e}")
                if attempt < self.max_retries:
                    time.sleep(0.5)

        logger.error(f"Failed to transmit event {event.event_id[:8]} after {self.max_retries} attempts.")
        return False
