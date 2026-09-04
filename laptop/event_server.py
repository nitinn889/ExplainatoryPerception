"""
FastAPI event receiver (Phase 5).

Exposes POST /event accepting shared.event_schema.Event, logging received
events to a local file/DB for now (real storage wired up in Phase 6's
memory_store.py).
"""
