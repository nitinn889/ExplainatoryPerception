"""
Perception-action loop (Phase 3) — core novelty. Explicit state machine:
PATROL -> PARTIAL -> ADJUST -> CONFIRM -> LOGGED, with a timeout counter.

Implements the bbox-clip -> motor-correction -> confirm logic described in
Section 4, Phase 3 of episodic_perception_build_spec.md.
"""
