"""
Perception-action loop (Phase 3) — core novelty.
Explicit state machine: PATROL -> PARTIAL -> ADJUST -> CONFIRM -> LOGGED, with a timeout counter.

Implements the bbox-clip -> motor-correction -> confirm logic described in
Section 4, Phase 3 of episodic_perception_build_spec.md.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from nano import motor_control
from shared.event_schema import BBox, EventType

logger = logging.getLogger("nano.perception_action")


class State(str, Enum):
    PATROL = "PATROL"
    PARTIAL = "PARTIAL"
    ADJUST = "ADJUST"
    CONFIRM = "CONFIRM"
    LOGGED = "LOGGED"


class CorrectionSignal(str, Enum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    REVERSE = "reverse"
    FORWARD = "forward"
    FULLY_VISIBLE = "fully_visible"
    NO_DETECTION = "no_detection"


@dataclass
class EmittedObservation:
    target_object: str
    bbox: BBox
    confidence: float
    event_type: EventType


def evaluate_bbox_signal(
    bbox: Tuple[float, float, float, float],
    edge_margin: float = 0.05,
    min_height: float = 0.15,
) -> CorrectionSignal:
    """
    Evaluates bounding box against frame edges and size constraints.
    bbox: (xmin, ymin, xmax, ymax) in normalized 0.0 to 1.0 coords.

    Signal rules (per spec):
    1. xmin <= edge_margin: clipped on LEFT edge -> turn_left
    2. xmax >= 1.0 - edge_margin: clipped on RIGHT edge -> turn_right
    3. ymax >= 1.0 - edge_margin: clipped at BOTTOM (too close) -> reverse
    4. height < min_height: object too far -> forward
    5. none of the above: fully visible -> proceed to CONFIRM
    """
    xmin, ymin, xmax, ymax = bbox
    height = max(0.0, ymax - ymin)

    if xmin <= edge_margin:
        return CorrectionSignal.TURN_LEFT
    if xmax >= (1.0 - edge_margin):
        return CorrectionSignal.TURN_RIGHT
    if ymax >= (1.0 - edge_margin):
        return CorrectionSignal.REVERSE
    if height < min_height:
        return CorrectionSignal.FORWARD

    return CorrectionSignal.FULLY_VISIBLE


class PerceptionActionStateMachine:
    """
    State machine driving active repositioning of the mobile chassis
    to stabilize partially clipped object views.
    """

    def __init__(
        self,
        edge_margin: float = 0.05,
        min_height: float = 0.15,
        pulse_ms: int = 150,
        confirm_frames: int = 3,
        timeout_sec: float = 7.0,
        motor: Optional[Any] = None,
        ignore_classes: Optional[Iterable[str]] = None,
    ):
        self.edge_margin = edge_margin
        self.min_height = min_height
        self.pulse_ms = pulse_ms
        self.confirm_frames = confirm_frames
        self.timeout_sec = timeout_sec
        self.motor = motor if motor is not None else motor_control.get_controller()
        # Classes that are never chased, only used as scene-graph context.
        #
        # Without this the largest-bbox-area guardrail backfires on furniture:
        # a 1.5 m desk viewed from 1.3 m is permanently clipped, always has the
        # largest area, and can never be made fully visible by repositioning -
        # so the loop fixates on it and times out on every station instead of
        # correcting toward the bottle sitting on it. Repositioning can only fix
        # a target that *could* fit in frame. Default stays None so the spec's
        # Phase 3 behaviour is unchanged; the demo drivers pass
        # scene_graph.SURFACE_CLASSES.
        self.ignore_classes = {c.lower() for c in ignore_classes} if ignore_classes else set()

        self.state = State.PATROL
        self.consecutive_confirm_count = 0
        self.adjust_start_time: Optional[float] = None
        self.tracked_target: Optional[str] = None
        self.last_signal: CorrectionSignal = CorrectionSignal.NO_DETECTION
        self.last_emitted: Optional[EmittedObservation] = None

    def _select_target(self, detections: List[Any]) -> Optional[Any]:
        """
        Guardrail: If multiple objects are partial at once, correct toward the one
        with the largest bbox area first.
        If already adjusting or confirming a target, continue tracking that target.
        """
        if not detections:
            return None

        def get_name(d: Any) -> str:
            return getattr(d, "class_name", None) or (d.get("class") if isinstance(d, dict) else "object")

        if self.ignore_classes:
            detections = [d for d in detections if get_name(d).lower() not in self.ignore_classes]
            if not detections:
                return None

        # If actively adjusting or confirming a tracked target, prioritize it
        if self.tracked_target is not None and self.state in (State.ADJUST, State.CONFIRM):
            for d in detections:
                if get_name(d) == self.tracked_target:
                    return d

        # Extract normalized bbox from each detection
        def get_bbox(d: Any) -> Tuple[float, float, float, float]:
            if hasattr(d, "bbox"):
                return d.bbox
            if isinstance(d, dict) and "bbox" in d:
                b = d["bbox"]
                if isinstance(b, dict):
                    return (b["xmin"], b["ymin"], b["xmax"], b["ymax"])
                return tuple(b)
            return (0.0, 0.0, 0.0, 0.0)

        def get_area(d: Any) -> float:
            if hasattr(d, "area"):
                return d.area
            xmin, ymin, xmax, ymax = get_bbox(d)
            return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)

        # Check signals for each detection
        partials = []
        for d in detections:
            bbox = get_bbox(d)
            sig = evaluate_bbox_signal(bbox, self.edge_margin, self.min_height)
            if sig != CorrectionSignal.FULLY_VISIBLE:
                partials.append((d, get_area(d)))

        if partials:
            # Sort by area descending
            partials.sort(key=lambda x: x[1], reverse=True)
            return partials[0][0]

        # If none are partial, pick the largest fully-visible object
        return max(detections, key=get_area)

    def _apply_motor_action(self, signal: CorrectionSignal) -> None:
        """Executes a short fixed-duration pulse according to the signal."""
        if signal == CorrectionSignal.TURN_LEFT:
            self.motor.turn_left(self.pulse_ms)
        elif signal == CorrectionSignal.TURN_RIGHT:
            self.motor.turn_right(self.pulse_ms)
        elif signal == CorrectionSignal.REVERSE:
            self.motor.reverse(self.pulse_ms)
        elif signal == CorrectionSignal.FORWARD:
            self.motor.forward(self.pulse_ms)
        elif signal == CorrectionSignal.FULLY_VISIBLE:
            self.motor.stop()

    def reset_to_patrol(self) -> None:
        self.state = State.PATROL
        self.consecutive_confirm_count = 0
        self.adjust_start_time = None
        self.tracked_target = None
        self.last_signal = CorrectionSignal.NO_DETECTION

    def step(
        self,
        detections: List[Any],
        current_time: Optional[float] = None,
    ) -> Tuple[State, Optional[EmittedObservation]]:
        """
        Executes one tick of the perception-action loop.
        Returns (current_state, optional_emitted_event).
        """
        now = current_time if current_time is not None else time.time()
        self.last_emitted = None

        target = self._select_target(detections)

        if target is None:
            # No object detected in frame
            if self.state in (State.ADJUST, State.CONFIRM):
                # Check timeout
                if self.adjust_start_time is not None and (now - self.adjust_start_time >= self.timeout_sec):
                    emitted = self._emit_timeout(now)
                    self.reset_to_patrol()
                    return State.LOGGED, emitted
            self.consecutive_confirm_count = 0
            return self.state, None

        # Extract target attributes
        target_name = getattr(target, "class_name", None) or (target.get("class") if isinstance(target, dict) else "object")
        confidence = float(getattr(target, "confidence", 1.0) if hasattr(target, "confidence") else target.get("confidence", 1.0))
        if hasattr(target, "bbox"):
            bbox_raw = target.bbox
        elif isinstance(target, dict) and "bbox" in target:
            b = target["bbox"]
            bbox_raw = (b["xmin"], b["ymin"], b["xmax"], b["ymax"]) if isinstance(b, dict) else tuple(b)
        else:
            bbox_raw = (0.2, 0.2, 0.8, 0.8)

        signal = evaluate_bbox_signal(bbox_raw, self.edge_margin, self.min_height)
        self.last_signal = signal
        bbox_obj = BBox(xmin=bbox_raw[0], ymin=bbox_raw[1], xmax=bbox_raw[2], ymax=bbox_raw[3])

        # State transitions
        if self.state == State.PATROL:
            self.tracked_target = target_name
            if signal == CorrectionSignal.FULLY_VISIBLE:
                self.state = State.CONFIRM
                self.consecutive_confirm_count = 1
                if self.consecutive_confirm_count >= self.confirm_frames:
                    return self._emit_confirmed(target_name, bbox_obj, confidence)
            else:
                self.state = State.PARTIAL
                self.adjust_start_time = now
                # Trigger initial adjustment
                self._apply_motor_action(signal)
                self.state = State.ADJUST

        elif self.state == State.PARTIAL:
            self.state = State.ADJUST
            self.adjust_start_time = now
            self._apply_motor_action(signal)

        elif self.state == State.ADJUST:
            # Check timeout guardrail (~6-8s of failed adjustment)
            if self.adjust_start_time is not None and (now - self.adjust_start_time >= self.timeout_sec):
                emitted = EmittedObservation(
                    target_object=self.tracked_target or target_name,
                    bbox=bbox_obj,
                    confidence=round(confidence * 0.65, 3),  # partial timeout logged with lower confidence
                    event_type=EventType.PARTIAL_TIMEOUT,
                )
                self.reset_to_patrol()
                return State.LOGGED, emitted

            if signal == CorrectionSignal.FULLY_VISIBLE:
                # Target has become fully visible! Move to CONFIRM
                self.state = State.CONFIRM
                self.consecutive_confirm_count = 1
                if self.consecutive_confirm_count >= self.confirm_frames:
                    return self._emit_confirmed(target_name, bbox_obj, confidence)
            else:
                # Still partial: issue pulse
                self._apply_motor_action(signal)

        elif self.state == State.CONFIRM:
            if signal == CorrectionSignal.FULLY_VISIBLE:
                self.consecutive_confirm_count += 1
                if self.consecutive_confirm_count >= self.confirm_frames:
                    return self._emit_confirmed(target_name, bbox_obj, confidence)
            else:
                # Lost full view during confirm: reset confirm counter, back to ADJUST
                logger.info("Lost full visibility during CONFIRM — transitioning back to ADJUST.")
                self.consecutive_confirm_count = 0
                self.state = State.ADJUST
                self._apply_motor_action(signal)

        elif self.state == State.LOGGED:
            self.reset_to_patrol()

        return self.state, None

    def _emit_confirmed(self, target_name: str, bbox: BBox, confidence: float) -> Tuple[State, EmittedObservation]:
        emitted = EmittedObservation(
            target_object=target_name,
            bbox=bbox,
            confidence=round(confidence, 3),
            event_type=EventType.CONFIRMED,
        )
        self.reset_to_patrol()
        return State.LOGGED, emitted

    def _emit_timeout(self, now: float) -> EmittedObservation:
        return EmittedObservation(
            target_object=self.tracked_target or "unknown_object",
            bbox=BBox(xmin=0.0, ymin=0.0, xmax=0.5, ymax=0.5),
            confidence=0.5,
            event_type=EventType.PARTIAL_TIMEOUT,
        )
