"""
Low-level GPIO/PWM motor driver interface for the L298N (Phase 0).

Exposes forward(duration_ms), reverse(duration_ms), turn_left(duration_ms),
turn_right(duration_ms), stop() — short, fixed-duration pulses only
(~150 ms default), never continuous/unbounded movement.
"""
