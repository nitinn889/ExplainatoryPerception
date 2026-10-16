"""
Tests for nano/station_test.py against a simulated turntable world: objects
sit at fixed bearings, every turn pulse rotates the camera's view, and the
real turn rate differs from the calibrated one so search and centering have
real work to do.
"""

import cv2
import numpy as np
import pytest

from nano.detector import DetectionResult
from shared.event_schema import BBox
from nano.motor_control import MotorController
from nano.station_test import Station, StationRunner, TurnOnlyMotor, measure_yaw, parse_station

FOV_DEG = 60.0


class World:
    """Camera + detector double. True heading comes from the motor's pulses."""

    def __init__(self, motor, objects, true_deg_per_sec, width_deg=12.0):
        self.motor = motor
        self.objects = objects  # {class_name: bearing_deg}
        self.true_deg_per_sec = true_deg_per_sec
        self.width_deg = width_deg
        self.is_synthetic = False
        self.camera_type = "fake"

    def heading(self):
        h = 0.0
        for action, ms in self.motor.action_history:
            sign = {"turn_right": 1, "turn_left": -1}.get(action, 0)
            h += sign * self.true_deg_per_sec * ms / 1000.0
        return h

    # camera
    def read(self):
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        pass

    # detector
    def detect(self, frame, synthetic_objects=None):
        out = []
        for name, bearing in self.objects.items():
            rel = bearing - self.heading()
            center = 0.5 + rel / FOV_DEG
            half = self.width_deg / FOV_DEG / 2
            xmin, xmax = center - half, center + half
            if xmax <= 0 or xmin >= 1:
                continue
            out.append(DetectionResult(name, 0.9, (max(0.0, xmin), 0.3, min(1.0, xmax), 0.7)))
        return out


def make_runner(objects, true_rate, assumed_rate=100.0):
    motor = MotorController(force_mock=True)
    world = World(motor, objects, true_rate)
    runner = StationRunner(world, world, motor, assumed_rate, settle_sec=0.0, timeout_sec=3.0)
    return runner, world


def test_parse_station():
    assert parse_station("1=-60:cup") == Station("1", -60.0, "cup")
    assert parse_station("2=45") == Station("2", 45.0, None)
    assert parse_station("3=10:wine_glass").target == "wine glass"


def test_route_1_2_1_faces_each_object():
    # Real car turns 25% faster than calibrated: dead reckoning alone would overshoot.
    runner, world = make_runner({"cup": -60.0, "bottle": 50.0}, true_rate=125.0)
    s1, s2 = Station("1", -60.0, "cup"), Station("2", 50.0, "bottle")
    results = [runner.visit(s) for s in (s1, s2, s1)]

    assert [r.outcome for r in results] == ["confirmed"] * 3
    for r in results:
        assert abs(r.center_x - 0.5) <= runner.center_tolerance
    assert abs(world.heading() - (-60.0)) < 8.0


def test_search_finds_object_far_from_the_given_heading():
    # Station heading is off by 50 degrees; the sweep has to find it.
    runner, world = make_runner({"cup": -90.0}, true_rate=100.0)
    result = runner.visit(Station("1", -40.0, "cup"))
    assert result.outcome == "confirmed"
    assert abs(world.heading() - (-90.0)) < 8.0


def test_missing_object_reports_not_found():
    runner, _ = make_runner({"cup": 170.0}, true_rate=100.0)
    runner.search_max_deg = 40.0
    result = runner.visit(Station("1", 0.0, "bottle"))
    assert result.outcome == "not_found"


def test_turn_only_never_drives():
    motor = MotorController(force_mock=True)
    # Object too far away (short bbox) -> the state machine asks for "forward".
    world = World(motor, {"cup": 0.0}, 100.0, width_deg=4.0)
    world.detect = lambda frame, synthetic_objects=None: [DetectionResult("cup", 0.9, (0.47, 0.45, 0.53, 0.5))]
    runner = StationRunner(world, world, TurnOnlyMotor(motor), 100.0, settle_sec=0.0, timeout_sec=0.5)
    result = runner.visit(Station("1", 0.0, "cup"))
    assert result.outcome == "partial_timeout"
    assert all(action not in ("forward", "reverse") for action, _ in motor.action_history)


def test_centring_survives_a_flickering_detection():
    runner, world = make_runner({"bottle": -20.0}, true_rate=100.0)
    real_detect = world.detect
    calls = {"n": 0}

    def flicker(frame, synthetic_objects=None):
        calls["n"] += 1
        return real_detect(frame) if calls["n"] % 3 == 0 else []  # seen 1 frame in 3

    world.detect = flicker
    runner.rough_turn(-5.0)  # bottle in view but well off centre
    cx = runner.center(Station("1", -20.0, "bottle"))
    assert cx is not None and abs(cx - 0.5) <= runner.center_tolerance


def test_measure_yaw_from_a_shifted_image():
    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur(rng.integers(0, 255, (360, 900), dtype=np.uint8), (5, 5), 0)
    before = scene[:, 100:740]
    # Camera turned right -> the scene slides left in the image.
    after = scene[:, 130:770]
    yaw = measure_yaw(before, after, hfov_deg=90.0)
    assert yaw is not None and 2.0 < yaw < 8.0
    assert measure_yaw(after, before, hfov_deg=90.0) < 0
    assert measure_yaw(np.zeros((360, 640), np.uint8), np.zeros((360, 640), np.uint8), 90.0) is None


def test_search_sweep_is_capped_even_if_turns_are_never_measured():
    # Motor "turns" but heading never changes (e.g. mis-measured turns): the
    # sweep must still stop instead of spinning forever.
    runner, world = make_runner({"cup": 170.0}, true_rate=100.0)
    runner._account_turns = lambda: (True, 0.0)
    found = runner.search(Station("1", 0.0, "bottle"), first_direction=1)
    assert found is False
    assert len(runner.motor.action_history) < 200


def test_rough_turn_stops_when_the_camera_sees_no_movement():
    runner, world = make_runner({"cup": 0.0}, true_rate=100.0)
    calls = {"n": 0}
    real_frame = runner._frame
    import nano.station_test as st
    original = st.measure_yaw
    st.measure_yaw = lambda a, b, h: 0.0  # car buzzing in place
    try:
        runner.rough_turn(40.0)
    finally:
        st.measure_yaw = original
    assert runner.stalled
    assert len(runner.motor.action_history) == 3


class RecordingClient:
    def __init__(self):
        self.events = []

    def send_event(self, event):
        self.events.append(event)
        return True


def test_removed_object_is_reported_missing_on_the_way_back():
    runner, world = make_runner({"bottle": -30.0, "laptop": 25.0}, true_rate=100.0)
    client = RecordingClient()
    runner.event_client = client
    s1, s2 = Station("1", -30.0, "bottle"), Station("2", 25.0, "laptop")

    first = runner.visit(s1)
    second = runner.visit(s2)
    del world.objects["bottle"]  # someone takes the bottle away
    back = runner.visit(s1)

    assert (first.outcome, second.outcome, back.outcome) == ("confirmed", "confirmed", "missing")
    assert [(e.event_type, e.objects[0], e.location_tag) for e in client.events] == [
        ("confirmed", "bottle", "lab_table location 1"),
        ("confirmed", "laptop", "lab_table location 2"),
        ("missing", "bottle", "lab_table location 1"),
    ]


def test_never_seen_object_is_not_found_rather_than_missing():
    runner, _ = make_runner({"laptop": 25.0}, true_rate=100.0)
    client = RecordingClient()
    runner.event_client = client
    result = runner.visit(Station("1", -30.0, "bottle"))
    assert result.outcome == "not_found"
    assert client.events == []


def test_small_far_away_detections_are_ignored():
    runner, world = make_runner({"laptop": 0.0}, true_rate=100.0)
    runner.min_box_height = 0.2
    world.detect = lambda frame, synthetic_objects=None: [
        DetectionResult("laptop", 0.9, (0.3, 0.3, 0.7, 0.65)),   # near: 35% tall
        DetectionResult("person", 0.9, (0.8, 0.4, 0.85, 0.52)),  # far: 12% tall
    ]
    runner._detect(None)
    assert [d.class_name for d in runner._last_detections] == ["laptop"]


def test_memory_event_only_describes_the_target():
    runner, world = make_runner({}, true_rate=100.0)
    client = RecordingClient()
    runner.event_client = client
    runner._last_detections = [
        DetectionResult("laptop", 0.95, (0.3, 0.3, 0.7, 0.65)),
        DetectionResult("person", 0.9, (0.75, 0.2, 0.95, 0.9)),
        DetectionResult("dining table", 0.8, (0.0, 0.6, 1.0, 1.0)),
    ]

    class Emitted:
        target_object = "laptop"
        confidence = 0.95
        bbox = BBox(xmin=0.3, ymin=0.3, xmax=0.7, ymax=0.65)

    runner._report_confirmed(Station("2", 0.0, "laptop"), Emitted())
    event = client.events[0]
    assert event.objects[0] == "laptop"
    assert all("laptop" in r for r in event.relationships)
    assert not any(r.startswith("person ON") for r in event.relationships)


def test_alias_target_accepts_either_name_and_reports_the_first():
    runner, world = make_runner({"tv": 0.0}, true_rate=100.0)
    client = RecordingClient()
    runner.event_client = client
    result = runner.visit(Station("2", 0.0, "laptop|tv"))
    assert result.outcome == "confirmed"
    assert client.events[0].objects[0] == "laptop"
