"""
Tests for nano/perception_action.py bbox-clip correction logic (Phase 3).
Covers each correction-signal case (left/right/bottom-clip, too-far,
fully-visible) with synthetic bboxes, multi-object prioritization,
consecutive-frame confirmation, and timeout behavior.
"""

import pytest
from nano.detector import DetectionResult
from nano.motor_control import MotorController
from nano.perception_action import (
    CorrectionSignal,
    PerceptionActionStateMachine,
    State,
    evaluate_bbox_signal,
)
from shared.event_schema import EventType


def test_evaluate_bbox_signals():
    # 1. Left edge clipped: xmin <= 0.05
    assert evaluate_bbox_signal((0.02, 0.2, 0.4, 0.6)) == CorrectionSignal.TURN_LEFT

    # 2. Right edge clipped: xmax >= 0.95
    assert evaluate_bbox_signal((0.6, 0.2, 0.98, 0.6)) == CorrectionSignal.TURN_RIGHT

    # 3. Bottom edge clipped: ymax >= 0.95 (too close)
    assert evaluate_bbox_signal((0.3, 0.5, 0.7, 0.97)) == CorrectionSignal.REVERSE

    # 4. Too far: height < min_height (default 0.15)
    assert evaluate_bbox_signal((0.4, 0.4, 0.5, 0.48)) == CorrectionSignal.FORWARD

    # 5. Fully visible
    assert evaluate_bbox_signal((0.3, 0.3, 0.6, 0.7)) == CorrectionSignal.FULLY_VISIBLE


def test_state_machine_patrol_to_adjust_to_confirm():
    mock_motor = MotorController(force_mock=True)
    sm = PerceptionActionStateMachine(confirm_frames=3, motor=mock_motor)

    # 1. Initially in PATROL
    assert sm.state == State.PATROL

    # 2. Frame with left-clipped object triggers ADJUST and turn_left pulse
    partial_det = DetectionResult("bottle", 0.9, (0.01, 0.2, 0.35, 0.6))
    state, emitted = sm.step([partial_det], current_time=100.0)
    assert state == State.ADJUST
    assert emitted is None
    assert mock_motor.action_history[-1] == ("turn_left", 150)

    # 3. Object moves inward but still slightly clipped on bottom -> reverse
    bottom_clipped = DetectionResult("bottle", 0.9, (0.2, 0.5, 0.5, 0.98))
    state, emitted = sm.step([bottom_clipped], current_time=100.5)
    assert state == State.ADJUST
    assert emitted is None
    assert mock_motor.action_history[-1] == ("reverse", 150)

    # 4. Object now fully centered -> enters CONFIRM (frame 1 of 3)
    full_det = DetectionResult("bottle", 0.92, (0.3, 0.3, 0.6, 0.7))
    state, emitted = sm.step([full_det], current_time=101.0)
    assert state == State.CONFIRM
    assert sm.consecutive_confirm_count == 1
    assert emitted is None

    # 5. Frame 2 of 3 in CONFIRM
    state, emitted = sm.step([full_det], current_time=101.2)
    assert state == State.CONFIRM
    assert sm.consecutive_confirm_count == 2
    assert emitted is None

    # 6. Frame 3 of 3 in CONFIRM -> Reaches threshold, logs confirmed event!
    state, emitted = sm.step([full_det], current_time=101.4)
    assert state == State.LOGGED
    assert emitted is not None
    assert emitted.target_object == "bottle"
    assert emitted.event_type == EventType.CONFIRMED
    assert emitted.confidence == 0.92

    # 7. Next step returns to PATROL
    state, emitted = sm.step([], current_time=102.0)
    assert state == State.PATROL


def test_state_machine_timeout_guardrail():
    mock_motor = MotorController(force_mock=True)
    sm = PerceptionActionStateMachine(timeout_sec=5.0, motor=mock_motor)

    # Object remains left-clipped persistently
    det = DetectionResult("cup", 0.85, (0.01, 0.3, 0.2, 0.6))

    # T = 0.0: Enters ADJUST
    sm.step([det], current_time=0.0)
    assert sm.state == State.ADJUST

    # T = 3.0: Still adjusting
    state, emitted = sm.step([det], current_time=3.0)
    assert state == State.ADJUST
    assert emitted is None

    # T = 6.0: Timeout exceeded (> 5.0s) -> logs partial_timeout with discounted confidence
    state, emitted = sm.step([det], current_time=6.0)
    assert state == State.LOGGED
    assert emitted is not None
    assert emitted.event_type == EventType.PARTIAL_TIMEOUT
    assert emitted.confidence < 0.85  # Discounted confidence per spec


def test_multi_object_prioritizes_largest_area():
    mock_motor = MotorController(force_mock=True)
    sm = PerceptionActionStateMachine(motor=mock_motor)

    # Small clipped object (area = 0.05 * 0.2 = 0.01)
    small_clip = DetectionResult("mouse", 0.8, (0.01, 0.1, 0.06, 0.3))
    # Large clipped object (area = 0.2 * 0.5 = 0.10)
    large_clip = DetectionResult("laptop", 0.9, (0.85, 0.2, 0.99, 0.7))

    # Should prioritize large_clip (right edge -> turn_right)
    sm.step([small_clip, large_clip], current_time=10.0)
    assert sm.state == State.ADJUST
    assert sm.tracked_target == "laptop"
    assert mock_motor.action_history[-1] == ("turn_right", 150)
