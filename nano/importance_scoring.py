"""
Importance scoring (Phase 4).

Given a new confirmed scene-graph observation and the last known state for
that object/location, decides is_new / is_changed / is_unusual, so only
meaningful events get forwarded to event_client.py.
"""
