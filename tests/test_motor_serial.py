"""
Tests for the serial (Arduino bridge) backend of nano/motor_control.py.

A fake bridge speaking the motor_bridge.ino protocol sits on the far end of a
pseudo-terminal, so the real pyserial code path is exercised without hardware.
"""

import os
import threading
import time
import tty

import pytest

pytest.importorskip("serial")
pty = pytest.importorskip("pty")

from nano.motor_control import MotorController
from nano.perception_action import PerceptionActionStateMachine, State


class FakeBridge:
    """Answers like motor_bridge.ino and records every line it receives."""

    def __init__(self, banner_on_boot=True, answer=True):
        self.master, slave = pty.openpty()
        self.port = os.ttyname(slave)
        self._slave = slave  # keep open so the pty survives pyserial reopening it
        tty.setraw(slave)  # no terminal echo, like a real USB serial link
        self.answer = answer
        self.received = []
        self._running = True
        if banner_on_boot:
            os.write(self.master, b"EPCAR 1\r\n")
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        buf = b""
        while self._running:
            try:
                data = os.read(self.master, 64)
            except OSError:
                return
            buf += data
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode().strip()
                if not line:
                    continue
                self.received.append(line)
                if not self.answer:
                    continue
                if line == "?":
                    reply = "EPCAR 1"
                elif line[0] in "FBLRS":
                    reply = "OK " + line
                else:
                    reply = "ERR unknown command"
                os.write(self.master, reply.encode() + b"\r\n")

    def commands(self):
        return [line for line in self.received if line != "?"]

    def close(self):
        self._running = False
        os.close(self._slave)
        os.close(self.master)


@pytest.fixture
def bridge():
    b = FakeBridge()
    yield b
    b.close()


def test_serial_backend_sends_one_command_per_pulse(bridge):
    motor = MotorController(serial_port=bridge.port, default_speed=60)
    assert motor.is_serial and not motor.is_mock

    motor.forward(40)
    motor.reverse(40)
    motor.turn_left(40)
    motor.turn_right(40)
    motor.stop()
    time.sleep(0.05)

    assert bridge.commands() == ["F 40 60", "B 40 60", "L 40 60", "R 40 60", "S"]
    assert motor.action_history == [("forward", 40), ("reverse", 40), ("turn_left", 40), ("turn_right", 40)]
    motor.cleanup()


def test_serial_pulse_blocks_for_its_duration(bridge):
    motor = MotorController(serial_port=bridge.port)
    start = time.time()
    motor.forward(200)
    elapsed = time.time() - start
    assert 0.19 <= elapsed < 0.5
    motor.cleanup()


def test_serial_default_pulse_and_cleanup_stops(bridge):
    motor = MotorController(serial_port=bridge.port, default_pulse_ms=30)
    motor.forward()
    motor.cleanup()
    time.sleep(0.05)
    assert bridge.commands() == ["F 30 70", "S"]
    motor.cleanup()  # second cleanup on a closed port must not raise


def test_serial_probes_a_board_that_does_not_reset_on_open():
    b = FakeBridge(banner_on_boot=False)
    try:
        motor = MotorController(serial_port=b.port)
        assert "?" in b.received
        motor.forward(20)
        time.sleep(0.05)
        assert b.commands() == ["F 20 70"]
        motor.cleanup()
    finally:
        b.close()


def test_serial_port_with_no_bridge_raises():
    b = FakeBridge(banner_on_boot=False, answer=False)
    try:
        with pytest.raises(RuntimeError, match="No motor bridge"):
            MotorController(serial_port=b.port)
    finally:
        b.close()


def test_force_mock_wins_over_serial_port():
    motor = MotorController(force_mock=True, serial_port="/dev/does-not-exist")
    assert motor.is_mock and not motor.is_serial
    motor.forward(1)
    motor.cleanup()


def test_state_machine_drives_the_bridge(bridge):
    motor = MotorController(serial_port=bridge.port)
    sm = PerceptionActionStateMachine(pulse_ms=20, motor=motor)
    # Clipped on the left edge -> one turn_left pulse goes out over serial.
    state, _ = sm.step([{"class": "bottle", "confidence": 0.9, "bbox": (0.02, 0.2, 0.4, 0.6)}])
    time.sleep(0.05)
    assert state == State.ADJUST
    assert bridge.commands() == ["L 20 70"]
    motor.cleanup()


def test_camera_at_rear_swaps_forward_and_reverse_only(bridge):
    motor = MotorController(serial_port=bridge.port, camera_at_rear=True)
    motor.forward(20)
    motor.reverse(20)
    motor.turn_left(20)
    motor.turn_right(20)
    time.sleep(0.05)
    assert bridge.commands() == ["B 20 70", "F 20 70", "L 20 70", "R 20 70"]
    motor.cleanup()
