"""
Low-level motor driver interface for the L298N (Phase 0).

Exposes forward(duration_ms), reverse(duration_ms), turn_left(duration_ms),
turn_right(duration_ms), stop() — short, fixed-duration pulses only
(~150 ms default), never continuous/unbounded movement.

Three backends share that interface:
- serial: an Arduino running arduino/motor_bridge/motor_bridge.ino drives the
  L298N and takes commands over USB. Selected by passing serial_port.
- gpio: the L298N is wired straight to the Jetson header (Jetson.GPIO).
- mock: no hardware, used when developing or testing on a laptop/desktop.
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

# pyserial is only needed for the Arduino bridge backend
try:
    import serial
except ImportError:
    serial = None

# First line the motor_bridge sketch prints on boot and in reply to "?"
SERIAL_BANNER = "EPCAR"
# One-letter commands understood by the motor_bridge sketch
SERIAL_COMMANDS = {"forward": "F", "reverse": "B", "turn_left": "L", "turn_right": "R"}

HIGH, LOW = 1, 0  # same values as GPIO.HIGH / GPIO.LOW


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
        serial_port: Optional[str] = None,  # e.g. /dev/ttyUSB0 -> use the Arduino bridge
        serial_baud: int = 115200,
        camera_at_rear: bool = False,
    ):
        self.in1 = in1_pin
        self.in2 = in2_pin
        self.in3 = in3_pin
        self.in4 = in4_pin
        self.ena = ena_pin
        self.enb = enb_pin
        self.default_pulse_ms = default_pulse_ms
        self.default_speed = default_speed
        # Camera mounted facing backward: "forward" means toward what the camera
        # sees, so the wheels run in reverse. Turns need no change - spinning
        # in place rotates the view the same way whichever end the camera is on.
        self.camera_at_rear = camera_at_rear

        self.is_serial = serial_port is not None and not force_mock
        self.is_mock = force_mock or not (self.is_serial or HAVE_JETSON_GPIO)
        self.action_history: List[Tuple[str, int]] = []
        self._pwm_a = None
        self._pwm_b = None
        self._serial = None

        if self.is_mock:
            logger.info("Running in MOCK motor mode (no physical GPIO used).")
        elif self.is_serial:
            self._setup_serial(serial_port, serial_baud)
        else:
            self._setup_gpio()

    def _setup_serial(self, port: str, baud: int) -> None:
        # An explicitly requested port that does not answer is an error, not a
        # reason to fall back to the mock: the robot would silently not move.
        if serial is None:
            raise RuntimeError("pyserial is not installed (pip install pyserial); needed for serial_port.")
        self._serial = serial.Serial(port, baud, timeout=0.5)
        # Opening the port resets an UNO, so wait for its boot banner. A board
        # that does not auto-reset stays quiet; probe it once the bootloader
        # window has passed.
        start = time.time()
        while time.time() - start < 6.0:
            line = self._serial.readline().decode("ascii", "replace").strip()
            if line.startswith(SERIAL_BANNER):
                logger.info(f"Arduino motor bridge ready on {port} ({line}).")
                return
            if not line and time.time() - start > 2.5:
                self._serial.write(b"?\n")
        self._serial.close()
        self._serial = None
        raise RuntimeError(f"No motor bridge answered on {port}. Is motor_bridge.ino flashed and the baud {baud}?")

    def _serial_command(self, line: str) -> bool:
        """Sends one command line and checks the bridge acknowledged it."""
        try:
            self._serial.reset_input_buffer()
            self._serial.write((line + "\n").encode("ascii"))
            self._serial.flush()
            reply = self._serial.readline().decode("ascii", "replace").strip()
        except Exception as e:
            logger.error(f"Motor bridge serial error on '{line}': {e}")
            return False
        if not reply.startswith("OK"):
            logger.warning(f"Motor bridge did not acknowledge '{line}' (got '{reply}').")
            return False
        return True

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

        if self.is_serial:
            # The Arduino times the pulse and stops the motors itself, so the
            # chassis still stops if this process dies mid-pulse.
            sent_at = time.time()
            self._serial_command(f"{SERIAL_COMMANDS[action]} {max(duration_ms, 0)} {self.default_speed}")
            time.sleep(max(0.0, duration_sec - (time.time() - sent_at)))
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
        if self.camera_at_rear:
            return self._wheels_reverse(duration_ms)
        self._wheels_forward(duration_ms)

    def reverse(self, duration_ms: Optional[int] = None) -> None:
        if self.camera_at_rear:
            return self._wheels_forward(duration_ms)
        self._wheels_reverse(duration_ms)

    def _wheels_forward(self, duration_ms: Optional[int] = None) -> None:
        ms = duration_ms if duration_ms is not None else self.default_pulse_ms
        # Left forward (HIGH, LOW), Right forward (HIGH, LOW)
        self._pulse("forward", ms, (HIGH, LOW),
                                   (HIGH, LOW))

    def _wheels_reverse(self, duration_ms: Optional[int] = None) -> None:
        ms = duration_ms if duration_ms is not None else self.default_pulse_ms
        # Left reverse (LOW, HIGH), Right reverse (LOW, HIGH)
        self._pulse("reverse", ms, (LOW, HIGH),
                                   (LOW, HIGH))

    def turn_left(self, duration_ms: Optional[int] = None) -> None:
        ms = duration_ms if duration_ms is not None else self.default_pulse_ms
        # Turn left in-place: Left reverse (LOW, HIGH), Right forward (HIGH, LOW)
        self._pulse("turn_left", ms, (LOW, HIGH),
                                     (HIGH, LOW))

    def turn_right(self, duration_ms: Optional[int] = None) -> None:
        ms = duration_ms if duration_ms is not None else self.default_pulse_ms
        # Turn right in-place: Left forward (HIGH, LOW), Right reverse (LOW, HIGH)
        self._pulse("turn_right", ms, (HIGH, LOW),
                                      (LOW, HIGH))

    def stop(self) -> None:
        if self.is_serial:
            if self._serial is not None:
                self._serial_command("S")
        elif not self.is_mock:
            try:
                GPIO.output([self.in1, self.in2, self.in3, self.in4], GPIO.LOW)
            except Exception as e:
                logger.error(f"Error stopping motors: {e}")

    def cleanup(self) -> None:
        self.stop()
        if self.is_serial:
            if self._serial is not None:
                self._serial.close()
                self._serial = None
        elif not self.is_mock:
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
