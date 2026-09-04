"""
Low-level GPIO/PWM motor driver interface for the L298N (Phase 0).

Exposes forward(duration_ms), reverse(duration_ms), turn_left(duration_ms),
turn_right(duration_ms), stop() — short, fixed-duration pulses only
(~150 ms default), never continuous/unbounded movement.

Provides automatic fallback to a mock motor driver when Jetson.GPIO is not
available (e.g., when developing or testing on a laptop/desktop).
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

logger = logging.getLogger("nano.motor_control")

# Check if Jetson.GPIO is available
try:
    import Jetson.GPIO as GPIO
    HAVE_JETSON_GPIO = True
except (ImportError, RuntimeError):
    HAVE_JETSON_GPIO = False


class MotorController:
    """
    Controls 2WD differential drive chassis via an L298N motor driver.
    Uses short, fixed-duration pulses for all movements to prevent overshoot.
    """

    def __init__(
        self,
        in1_pin: int = 35,
        in2_pin: int = 37,
        in3_pin: int = 36,
        in4_pin: int = 38,
        ena_pin: Optional[int] = 33,
        enb_pin: Optional[int] = 32,
        default_pulse_ms: int = 150,
        default_speed: int = 70,  # PWM duty cycle (0-100)
        force_mock: bool = False,
    ):
        self.in1 = in1_pin
        self.in2 = in2_pin
        self.in3 = in3_pin
        self.in4 = in4_pin
        self.ena = ena_pin
        self.enb = enb_pin
        self.default_pulse_ms = default_pulse_ms
        self.default_speed = default_speed

        self.is_mock = force_mock or not HAVE_JETSON_GPIO
        self.action_history: List[Tuple[str, int]] = []
        self._pwm_a = None
        self._pwm_b = None

        if self.is_mock:
            logger.info("Running in MOCK motor mode (no physical GPIO used).")
        else:
            self._setup_gpio()

    def _setup_gpio(self) -> None:
        try:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup([self.in1, self.in2, self.in3, self.in4], GPIO.OUT, initial=GPIO.LOW)
            if self.ena is not None and self.enb is not None:
                GPIO.setup([self.ena, self.enb], GPIO.OUT, initial=GPIO.LOW)
                self._pwm_a = GPIO.PWM(self.ena, 1000)  # 1kHz
                self._pwm_b = GPIO.PWM(self.enb, 1000)
                self._pwm_a.start(self.default_speed)
                self._pwm_b.start(self.default_speed)
            logger.info("Jetson.GPIO initialized successfully for L298N driver.")
        except Exception as e:
            logger.warning(f"Failed to initialize Jetson.GPIO ({e}). Switching to mock driver.")
            self.is_mock = True

    def _pulse(self, action: str, duration_ms: int, left_state: Tuple[int, int], right_state: Tuple[int, int]) -> None:
        duration_sec = max(duration_ms, 0) / 1000.0
        self.action_history.append((action, duration_ms))

        if self.is_mock:
            logger.debug(f"[MOCK MOTOR] {action.upper()} for {duration_ms}ms")
            time.sleep(duration_sec)
            return

        try:
            # Set motor direction
            GPIO.output(self.in1, left_state[0])
            GPIO.output(self.in2, left_state[1])
            GPIO.output(self.in3, right_state[0])
            GPIO.output(self.in4, right_state[1])

            time.sleep(duration_sec)
        finally:
            self.stop()

    def forward(self, duration_ms: Optional[int] = None) -> None:
        ms = duration_ms if duration_ms is not None else self.default_pulse_ms
        # Left forward (HIGH, LOW), Right forward (HIGH, LOW)
        self._pulse("forward", ms, (GPIO.HIGH if not self.is_mock else 1, GPIO.LOW if not self.is_mock else 0),
                                   (GPIO.HIGH if not self.is_mock else 1, GPIO.LOW if not self.is_mock else 0))

    def reverse(self, duration_ms: Optional[int] = None) -> None:
        ms = duration_ms if duration_ms is not None else self.default_pulse_ms
        # Left reverse (LOW, HIGH), Right reverse (LOW, HIGH)
        self._pulse("reverse", ms, (GPIO.LOW if not self.is_mock else 0, GPIO.HIGH if not self.is_mock else 1),
                                   (GPIO.LOW if not self.is_mock else 0, GPIO.HIGH if not self.is_mock else 1))

    def turn_left(self, duration_ms: Optional[int] = None) -> None:
        ms = duration_ms if duration_ms is not None else self.default_pulse_ms
        # Turn left in-place: Left reverse (LOW, HIGH), Right forward (HIGH, LOW)
        self._pulse("turn_left", ms, (GPIO.LOW if not self.is_mock else 0, GPIO.HIGH if not self.is_mock else 1),
                                     (GPIO.HIGH if not self.is_mock else 1, GPIO.LOW if not self.is_mock else 0))

    def turn_right(self, duration_ms: Optional[int] = None) -> None:
        ms = duration_ms if duration_ms is not None else self.default_pulse_ms
        # Turn right in-place: Left forward (HIGH, LOW), Right reverse (LOW, HIGH)
        self._pulse("turn_right", ms, (GPIO.HIGH if not self.is_mock else 1, GPIO.LOW if not self.is_mock else 0),
                                      (GPIO.LOW if not self.is_mock else 0, GPIO.HIGH if not self.is_mock else 1))

    def stop(self) -> None:
        if not self.is_mock:
            try:
                GPIO.output([self.in1, self.in2, self.in3, self.in4], GPIO.LOW)
            except Exception as e:
                logger.error(f"Error stopping motors: {e}")

    def cleanup(self) -> None:
        self.stop()
        if not self.is_mock:
            try:
                if self._pwm_a:
                    self._pwm_a.stop()
                if self._pwm_b:
                    self._pwm_b.stop()
                GPIO.cleanup()
            except Exception as e:
                logger.error(f"Error during GPIO cleanup: {e}")


# Singleton / module-level convenience functions
_controller: Optional[MotorController] = None


def get_controller() -> MotorController:
    global _controller
    if _controller is None:
        _controller = MotorController()
    return _controller


def forward(duration_ms: Optional[int] = None) -> None:
    get_controller().forward(duration_ms)


def reverse(duration_ms: Optional[int] = None) -> None:
    get_controller().reverse(duration_ms)


def turn_left(duration_ms: Optional[int] = None) -> None:
    get_controller().turn_left(duration_ms)


def turn_right(duration_ms: Optional[int] = None) -> None:
    get_controller().turn_right(duration_ms)


def stop() -> None:
    get_controller().stop()
