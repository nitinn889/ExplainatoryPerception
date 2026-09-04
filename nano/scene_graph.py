"""
Scene graph construction (Phase 2).

Given a frame's list of detected objects with bboxes, computes pairwise
spatial relationships (ON, LEFT OF, RIGHT OF, BEHIND) via geometric rules
and outputs triples like ["bottle", "ON", "table"].
"""
