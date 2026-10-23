"""
First on-car test: turn to face a station, adjust onto its object, move on.

The robot stays in place and visits stations by heading only, e.g. route 1 -> 2 -> 1:

  1. Rough turn: open-loop turn pulses from the current heading estimate to
     the station's heading (degrees, positive = clockwise/right, 0 = the way
     the car faces at start).
  2. Search: if the station's object is not in view, sweep in small pulses
     until it is.
  3. Adjust: the real PerceptionActionStateMachine runs until it logs the
     object (CONFIRMED) or gives up (PARTIAL_TIMEOUT).
  4. Center: small turn pulses until the object is near the middle of the frame.

There are no wheel encoders, so the heading is dead reckoning from turn time
(--deg-per-sec, measured with --calibrate). Every turn pulse, including the
state machine's own corrections, is added to it. Search and centering absorb
the error, which is why returning to station 1 still works.

  # which camera index is the colour stream (saves one frame per index)
  python -m nano.station_test --list-cameras

  # lowest duty that still moves the car (keeps it slow)
  python -m nano.station_test --motor-port /dev/ttyUSB0 --find-duty

  # measure how far the car turns: 10 pulses, then measure the angle
  python -m nano.station_test --motor-port /dev/ttyUSB0 --duty 50 --calibrate

  python -m nano.station_test --motor-port /dev/ttyUSB0 --mode oak \\
      --deg-per-sec 120 --station 1=-60:cup --station 2=50:bottle --route 1 2 1
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from nano.camera import Camera
from nano.detector import SSDMobileNetV2Detector, draw_detections
from nano.event_client import EventClient
from nano.motor_control import MotorController
from nano.perception_action import PerceptionActionStateMachine, State
from nano.scene_graph import SURFACE_CLASSES, SceneGraphBuilder
from shared.event_schema import BBox, Event, EventType

logger = logging.getLogger("nano.station_test")


def measure_yaw(before, after, hfov_deg: float, min_matches: int = 40) -> Optional[float]:
    """Camera rotation between two frames in degrees, positive = turned right.

    ORB features are matched across the two frames; for each match the bearing
    change is atan(x/f) after minus before, and the median is robust to people
    walking through the view. None when there is too little texture to tell.
    """
    to_gray = lambda f: f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    a, b = to_gray(before), to_gray(after)
    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < min_matches:
        return None
    w = a.shape[1]
    f = (w / 2) / math.tan(math.radians(hfov_deg / 2))
    cx = w / 2
    shifts = [
        math.atan((kb[m.trainIdx].pt[0] - cx) / f) - math.atan((ka[m.queryIdx].pt[0] - cx) / f)
        for m in matches
    ]
    # Turning right moves the scene left in the image.
    return -math.degrees(float(np.median(shifts)))


@dataclass
class Station:
    name: str
    heading_deg: float
    target: Optional[str]  # COCO class name; None = any non-surface object


@dataclass
class StationResult:
    station: str
    outcome: str  # confirmed | partial_timeout | not_found | missing
    target: Optional[str]
    heading_deg: float
    center_x: Optional[float]
    seconds: float


class TurnOnlyMotor:
    """Passes turns and stops through, drops forward/reverse. For tight spaces:
    the car can only spin on the spot."""

    def __init__(self, motor: MotorController):
        self._motor = motor

    def forward(self, duration_ms=None) -> None:
        logger.info("  (turn-only) skipped forward pulse")

    def reverse(self, duration_ms=None) -> None:
        logger.info("  (turn-only) skipped reverse pulse")

    def __getattr__(self, name):
        return getattr(self._motor, name)


_RELATIONS = (" LEFT OF ", " RIGHT OF ", " BEHIND ", " ON ")


def _split_relationship(rel: str) -> Optional[Tuple[str, str, str]]:
    """'dining table LEFT OF cell phone' -> ('dining table', 'LEFT OF', 'cell phone')."""
    for word in _RELATIONS:
        if word in rel:
            a, b = rel.split(word, 1)
            return a.strip(), word.strip(), b.strip()
    return None


def parse_station(text: str) -> Station:
    """'1=-60:cup' -> Station('1', -60.0, 'cup'); the ':class' part is optional."""
    name, _, rest = text.partition("=")
    heading, _, target = rest.partition(":")
    if not name or not heading:
        raise argparse.ArgumentTypeError(f"station must look like NAME=DEGREES[:class], got '{text}'")
    return Station(name, float(heading), target.replace("_", " ") or None)


class StationRunner:
    def __init__(
        self,
        camera: Camera,
        detector: SSDMobileNetV2Detector,
        motor: MotorController,
        deg_per_sec: float,
        turn_chunk_ms: int = 300,
        settle_sec: float = 0.25,
        search_step_ms: int = 150,
        search_max_deg: float = 120.0,
        center_tolerance: float = 0.12,
        max_center_pulses: int = 8,
        pulse_ms: int = 150,
        timeout_sec: float = 10.0,
        snapshot_dir: Optional[str] = None,
        telemetry_url: Optional[str] = None,
        hfov_deg: Optional[float] = 90.0,
        event_client=None,
        location_tag: str = "lab_table",
        min_box_height: float = 0.0,
        confirm_frames: int = 3,
        rotate_deg: int = 0,
        turn_only: bool = False,
    ):
        self.camera = camera
        self.detector = detector
        self.motor = motor
        self.deg_per_sec = deg_per_sec
        self.turn_chunk_ms = turn_chunk_ms
        self.settle_sec = settle_sec
        self.search_step_ms = search_step_ms
        self.search_max_deg = search_max_deg
        self.center_tolerance = center_tolerance
        self.max_center_pulses = max_center_pulses
        self.pulse_ms = pulse_ms
        self.timeout_sec = timeout_sec
        self.snapshot_dir = snapshot_dir

        self.heading_deg = 0.0
        self._history_seen = 0
        # Horizontal field of view for measuring turns from the camera; None
        # falls back to dead reckoning from --deg-per-sec alone.
        self.hfov_deg = hfov_deg
        # Memory: confirmed sightings and removals go to the fog node as events.
        self.event_client = event_client
        self.location_tag = location_tag
        # Range limit: no depth from a colour image, so "far away" is judged by
        # apparent size. Boxes shorter than this fraction of the frame (distant
        # people, things on other benches) are ignored everywhere.
        self.min_box_height = min_box_height
        self.confirm_frames = confirm_frames
        self.turn_only = turn_only
        self.rotate = {
            0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[rotate_deg % 360]
        self.scene_builder = SceneGraphBuilder()
        self.confirmed_at: Dict[str, str] = {}  # station name -> time it was last confirmed
        self._last_detections: List = []
        self._last_measured: Optional[float] = None
        self._stalled_pulses = 0

        # Live view on the demo dashboard (POST /telemetry), best-effort only.
        self.telemetry_url = telemetry_url
        self._telemetry_failures = 0
        self._frames = 0
        self.phase_state = State.PATROL.value
        self.phase_signal = "starting"
        self.phase_tag = "-"

    def _publish(self, frame, detections: List) -> None:
        self._frames += 1
        if not self.telemetry_url or self._telemetry_failures > 20:
            return
        import requests

        payload = {
            "source": f"jetson station test ({self.camera.camera_type} camera)",
            "state": self.phase_state,
            "signal": self.phase_signal,
            "location_tag": self.phase_tag,
            "detector": f"ssd-mobilenet-v2 ({self.detector.backend})",
            "detections": [d.to_dict() for d in detections],
            "edge_margin": 0.05,
            "stats": {"frames": self._frames, "heading_deg": round(self.heading_deg, 1)},
            "last_motor_action": list(self.motor.action_history[-1]) if self.motor.action_history else None,
        }
        ok, buffer = cv2.imencode(".jpg", draw_detections(frame, detections), [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok:
            payload["frame_jpeg_b64"] = base64.b64encode(buffer.tobytes()).decode("ascii")
        try:
            requests.post(self.telemetry_url, json=payload, timeout=0.6)
            self._telemetry_failures = 0
        except requests.RequestException:
            self._telemetry_failures += 1

    # -- heading bookkeeping ---------------------------------------------

    def _account_turns(self) -> Tuple[bool, float]:
        """Adds any new turn pulses to the heading estimate from turn time.
        Returns (motor moved, degrees added)."""
        new = self.motor.action_history[self._history_seen:]
        self._history_seen = len(self.motor.action_history)
        added = 0.0
        for action, ms in new:
            if action == "turn_right":
                added += self.deg_per_sec * ms / 1000.0
            elif action == "turn_left":
                added -= self.deg_per_sec * ms / 1000.0
        self.heading_deg += added
        return bool(new), added

    def _correct_with_camera(self, before, estimated: float):
        """Replaces the time-based estimate of the last turn with what the camera
        saw. Returns the new frame (taken after the settle pause)."""
        after = self._frame()
        if self.hfov_deg is None or before is None:
            return after
        measured = measure_yaw(before, after, self.hfov_deg)
        self._last_measured = measured
        if measured is not None:
            self.heading_deg += measured - estimated
            self._stalled_pulses = self._stalled_pulses + 1 if abs(measured) < 0.3 else 0
        logger.info(
            "    pulse: estimated %+.1f, camera %s -> heading %+.1f",
            estimated, "n/a" if measured is None else f"{measured:+.1f}", self.heading_deg,
        )
        return after

    @property
    def stalled(self) -> bool:
        """Several turn pulses in a row that the camera says did nothing."""
        return self._stalled_pulses >= 3

    def _turn(self, direction: int, ms: int) -> None:
        before = self._frame()
        if direction > 0:
            self.motor.turn_right(ms)
        else:
            self.motor.turn_left(ms)
        _, estimated = self._account_turns()
        time.sleep(self.settle_sec)
        self._correct_with_camera(before, estimated)

    # -- perception --------------------------------------------------------

    def _frame(self):
        # A USB camera queues a few frames and read() returns the oldest, which
        # after a pause is from before or during the last turn. Empty the queue
        # so decisions are made on what the camera sees now. (The OAK queue
        # already keeps only the newest frame.)
        cap = getattr(self.camera, "cap", None)
        if cap is not None:
            for _ in range(5):
                cap.grab()
        ok, frame = self.camera.read()
        if not ok or frame is None:
            raise RuntimeError("camera returned no frame")
        if self.rotate:
            # A camera mounted upside down (or sideways): turn the picture
            # upright before detection. 180 also restores image left/right.
            frame = cv2.rotate(frame, self.rotate)
        return frame

    def _detect(self, target: Optional[str]) -> Tuple[object, List]:
        frame = self._frame()
        detections = [
            d for d in self.detector.detect(frame)
            if (d.bbox[3] - d.bbox[1]) >= self.min_box_height
        ]
        self._last_detections = detections
        self._publish(frame, detections)
        if target:
            # "laptop|tv": accept any of the names, but call it by the first, so
            # the state machine, scene graph and memory all say "laptop".
            names = target.split("|")
            detections = [
                dataclasses.replace(d, class_name=names[0]) if d.class_name in names else d
                for d in detections
            ]
            self._last_detections = detections
            wanted = [d for d in detections if d.class_name == names[0]]
        else:
            wanted = [d for d in detections if d.class_name not in SURFACE_CLASSES]
        return frame, wanted

    @staticmethod
    def _best(detections: List):
        return max(detections, key=lambda d: d.area) if detections else None

    def _snapshot(self, frame, detections, label: str) -> None:
        if not self.snapshot_dir:
            return
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(self.snapshot_dir, f"{time.strftime('%H%M%S')}_{label}.jpg")
        cv2.imwrite(path, draw_detections(frame, detections))
        logger.info("  snapshot %s", path)

    # -- the four phases ---------------------------------------------------

    def rough_turn(self, heading: float) -> None:
        delta = heading - self.heading_deg
        logger.info("  rough turn %+.0f deg from heading %+.0f", delta, self.heading_deg)
        self.phase_state, self.phase_signal = State.PATROL.value, f"turning {delta:+.0f} deg"
        direction = 1 if delta > 0 else -1
        self._stalled_pulses = 0
        # Turn pulse by pulse until the (camera-corrected) heading arrives, so a
        # car that turns faster or slower than calibrated still stops in place.
        chunk_deg = self.deg_per_sec * self.turn_chunk_ms / 1000.0
        max_pulses = int(abs(delta) / max(chunk_deg, 0.1) * 3) + 3
        for _ in range(max_pulses):
            remaining = (heading - self.heading_deg) * direction
            if remaining <= chunk_deg / 2:
                break
            ms = min(self.turn_chunk_ms, max(30, int(remaining / self.deg_per_sec * 1000.0)))
            self._turn(direction, ms)
            if self.stalled:
                logger.warning("  turn stalled: %d pulses without movement (cable pulling? battery low?)", self._stalled_pulses)
                return

    def search(self, station: Station, first_direction: int) -> bool:
        """Sweeps out to +-search_max_deg around the station heading until the target shows."""
        _, found = self._detect(station.target)
        if found:
            return True
        logger.info("  %s not in view, searching", station.target or "object")
        self.phase_state, self.phase_signal = State.PATROL.value, f"searching for {station.target or 'object'}"
        # Never trust the heading alone to end the sweep: if turns are
        # mis-measured it would spin indefinitely. Cap the pulses per side.
        step_deg = max(self.deg_per_sec * self.search_step_ms / 1000.0, 0.5)
        max_pulses = int(self.search_max_deg / step_deg * 1.5) + 2
        for direction in (first_direction, -first_direction):
            limit = station.heading_deg + direction * self.search_max_deg
            pulses = 0
            self._stalled_pulses = 0
            while (limit - self.heading_deg) * direction > 0 and pulses < max_pulses:
                pulses += 1
                self._turn(direction, self.search_step_ms)
                if self.stalled:
                    logger.warning("  search stalled turning %s", "right" if direction > 0 else "left")
                    break
                _, found = self._detect(station.target)
                if found:
                    return True
            self.rough_turn(station.heading_deg)
        return False

    @staticmethod
    def _framing_only(detections: List) -> List:
        """Turn-only: the car cannot get closer or back away, so being cut off
        at the bottom or looking small cannot be fixed - judge left/right only."""
        return [dataclasses.replace(d, bbox=(d.bbox[0], 0.3, d.bbox[2], 0.7)) for d in detections]

    def adjust(self, station: Station):
        sm = PerceptionActionStateMachine(
            pulse_ms=self.pulse_ms,
            timeout_sec=self.timeout_sec,
            confirm_frames=self.confirm_frames,
            motor=self.motor,
            ignore_classes=None if station.target else SURFACE_CLASSES,
        )
        hard_stop = time.time() + self.timeout_sec + 5.0
        frame, wanted = None, []
        while time.time() < hard_stop:
            frame, wanted = self._detect(station.target)
            prev = sm.state
            state, emitted = sm.step(self._framing_only(wanted) if self.turn_only else wanted)
            self.phase_state, self.phase_signal = state.value, f"adjust: {sm.last_signal.value}"
            if state != prev:
                logger.info("  %s -> %s (%s)", prev.value, state.value, sm.last_signal.value)
            moved, estimated = self._account_turns()
            if moved:
                time.sleep(self.settle_sec)
                self._correct_with_camera(frame, estimated)
            if emitted is not None:
                return emitted, frame, wanted
        return None, frame, wanted

    def center(self, station: Station) -> Optional[float]:
        cx = None
        self.phase_state, self.phase_signal = State.CONFIRM.value, "centring"
        for _ in range(self.max_center_pulses):
            # A weak detection flickers in and out; look a few times before giving up.
            best = None
            for _ in range(6):
                _, wanted = self._detect(station.target)
                best = self._best(wanted)
                if best is not None:
                    break
            if best is None:
                return None
            cx = (best.xmin + best.xmax) / 2.0
            if abs(cx - 0.5) <= self.center_tolerance:
                break
            self._turn(1 if cx > 0.5 else -1, self.search_step_ms)
        return cx

    def station_tag(self, station: Station) -> str:
        return f"{self.location_tag} location {station.name}"

    def _send(self, event: Event) -> None:
        if self.event_client is None:
            return
        ok = self.event_client.send_event(event)
        logger.info("  memory: %s event for %s %s", event.event_type, event.objects[0], "sent" if ok else "NOT sent")

    def _report_confirmed(self, station: Station, emitted) -> None:
        relationships, _ = self.scene_builder.build_scene_graph(self._last_detections)
        subject = emitted.target_object
        # Only facts about the target itself: a person walking past must not
        # become the caption of "where is the laptop?".
        about, others = [], []
        for r in relationships:
            parts = _split_relationship(r)
            if parts and subject in (parts[0], parts[2]):
                about.append(r)
                other = parts[2] if parts[0] == subject else parts[0]
                if other != subject and other not in others:
                    others.append(other)
        self._send(Event(
            objects=[subject] + others,
            relationships=about,
            confidence=emitted.confidence,
            event_type=getattr(emitted, "event_type", EventType.CONFIRMED),
            bbox=emitted.bbox,
            location_tag=self.station_tag(station),
        ))

    def _report_missing(self, station: Station) -> None:
        self._send(Event(
            objects=[(station.target or "object").split("|")[0]],
            relationships=[],
            confidence=0.8,
            event_type=EventType.MISSING,
            bbox=BBox(xmin=0.0, ymin=0.0, xmax=0.0, ymax=0.0),
            location_tag=self.station_tag(station),
        ))

    def visit(self, station: Station) -> StationResult:
        start = time.time()
        logger.info("=== Station %s (heading %+.0f, target %s) ===", station.name, station.heading_deg, station.target or "any")
        self.phase_tag = f"station {station.name} ({station.target or 'any'})"
        turn_dir = 1 if station.heading_deg >= self.heading_deg else -1
        self.rough_turn(station.heading_deg)

        if not self.search(station, turn_dir):
            frame, wanted = self._detect(station.target)
            outcome = "not_found"
            if station.name in self.confirmed_at:
                # It was confirmed here earlier in this run and is gone now.
                outcome = "missing"
                logger.info(
                    "  %s is gone from location %s (confirmed there at %s) -> recording it as removed",
                    station.target or "object", station.name, self.confirmed_at.pop(station.name),
                )
                self._report_missing(station)
            self.phase_state, self.phase_signal = State.LOGGED.value, outcome
            self._snapshot(frame, wanted, f"station{station.name}_{outcome}")
            return StationResult(station.name, outcome, station.target, self.heading_deg, None, time.time() - start)

        emitted, frame, wanted = self.adjust(station)
        outcome = "partial_timeout"
        if emitted is not None and emitted.event_type == EventType.CONFIRMED:
            outcome = "confirmed"
        target = emitted.target_object if emitted is not None else station.target

        if emitted is not None:
            # A timed-out adjust still saw the object here; either way it is
            # remembered, so a later disappearance is caught as "removed".
            self.confirmed_at[station.name] = time.strftime("%H:%M:%S")
            self._report_confirmed(station, emitted)
        cx = self.center(station) if outcome == "confirmed" else None
        if cx is not None and abs(cx - 0.5) <= self.center_tolerance:
            # Facing the object now: trust the camera over dead reckoning.
            station.heading_deg = self.heading_deg
        self.phase_state, self.phase_signal = State.LOGGED.value, outcome
        frame, wanted = self._detect(station.target)
        self._snapshot(frame, wanted, f"station{station.name}_{outcome}")
        return StationResult(station.name, outcome, target, self.heading_deg, cx, time.time() - start)


def calibrate(motor: MotorController, pulses: int, pulse_ms: int, pause_sec: float) -> None:
    print(f"Mark the car's heading, then it will turn RIGHT {pulses} x {pulse_ms} ms.")
    input("Press Enter to start (Ctrl+C to abort)... ")
    for _ in range(pulses):
        motor.turn_right(pulse_ms)
        time.sleep(pause_sec)
    total_sec = pulses * pulse_ms / 1000.0
    print(f"Done: {total_sec:.2f} s of turning in total.")
    try:
        degrees = float(input("How many degrees did it turn? "))
    except ValueError:
        print("No number entered; deg-per-sec = degrees / %.2f" % total_sec)
        return
    print(f"Use --deg-per-sec {degrees / total_sec:.0f}")


def find_duty(motor: MotorController, pulse_ms: int, start: int = 30, stop: int = 90, step: int = 5) -> None:
    """Short right-turn pulses at rising duty until the user says the car moved."""
    print(f"The car will try a {pulse_ms} ms right turn at {start}%, {start + step}%, ... up to {stop}%.")
    print("Wheels buzzing but the car not turning counts as 'no'.")
    for duty in range(start, stop + 1, step):
        motor.default_speed = duty
        input(f"Press Enter for {duty}% (Ctrl+C to abort)... ")
        motor.turn_right(pulse_ms)
        if input("Did the car actually turn? [y/N] ").strip().lower().startswith("y"):
            print(f"Lowest moving duty: {duty}%. Use --duty {duty} (or {duty + 5} if it sometimes stalls).")
            return
    print(f"Nothing moved up to {stop}%: check the motor battery and the L298N wiring.")


def list_cameras(max_index: int, out_dir: str) -> None:
    """Saves one frame from every V4L2 index that opens, to see which is colour."""
    os.makedirs(out_dir, exist_ok=True)
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if not ok or frame is None:
            continue
        path = os.path.join(out_dir, f"camera_index_{index}.jpg")
        cv2.imwrite(path, frame)
        colour = frame.ndim == 3 and frame.shape[2] == 3 and float(abs(frame[..., 0].astype(int) - frame[..., 2]).mean()) > 2.0
        print(f"index {index}: {frame.shape[1]}x{frame.shape[0]} {'colour' if colour else 'grey/IR/depth'} -> {path}")
    print("Open the saved images and use the colour one with --mode usb --camera-index N.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn between stations and adjust onto the object at each one.")
    parser.add_argument("--motor-port", default=None, help="Arduino motor bridge port, e.g. /dev/ttyUSB0")
    parser.add_argument("--mock-motors", action="store_true", help="Do not move; log motor pulses only")
    parser.add_argument("--camera-rear", action="store_true",
                        help="Camera faces backward over the battery box (swaps forward/reverse)")
    parser.add_argument("--mode", choices=["auto", "csi", "usb", "oak"], default="auto", help="Camera mode")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="Rotate every frame clockwise by this much (180 for an upside-down camera)")
    parser.add_argument("--camera-index", type=int, default=0, help="V4L2 index for --mode usb (see --list-cameras)")
    parser.add_argument("--station", action="append", type=parse_station, default=[],
                        help="NAME=DEGREES[:class], e.g. 1=-60:cup (degrees clockwise from the start heading)")
    parser.add_argument("--route", nargs="+", default=["1", "2", "1"], help="Station names to visit in order")
    parser.add_argument("--deg-per-sec", type=float, default=None, help="Turn rate at --duty, from --calibrate")
    parser.add_argument("--duty", type=int, default=50, help="Motor PWM duty %% (find the lowest that moves with --find-duty)")
    parser.add_argument("--pulse-ms", type=int, default=100, help="Length of every adjust/search/centre pulse")
    parser.add_argument("--turn-chunk-ms", type=int, default=100, help="Big turns are split into pulses this long")
    parser.add_argument("--settle", type=float, default=0.4, help="Pause after every pulse, seconds (bigger = slower)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds of adjusting before partial_timeout")
    parser.add_argument("--snapshots", default="data/station_test", help="Folder for annotated frames ('' to disable)")
    parser.add_argument("--calibrate", action="store_true", help="Measure --deg-per-sec and exit")
    parser.add_argument("--calibrate-pulses", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.4,
                        help="Detector cut-off (too low lets far-away look-alikes count)")
    parser.add_argument("--hfov", type=float, default=90.0,
                        help="Camera horizontal field of view, for measuring turns from the image (0 = off)")
    parser.add_argument("--search-deg", type=float, default=45.0,
                        help="How far to sweep each way when the object is not in view")
    parser.add_argument("--laptop-url", default=None,
                        help="Dashboard to stream the live view to, e.g. http://192.168.55.100:8080")
    parser.add_argument("--wait", action="append", default=[],
                        help="NAME=SECONDS: pause after visiting that station, e.g. 2=15 to give time to move things")
    parser.add_argument("--min-box-height", type=float, default=0.2,
                        help="Ignore detections shorter than this fraction of the frame (= far away)")
    parser.add_argument("--confirm-frames", type=int, default=3,
                        help="Consecutive fully-visible frames needed to confirm (lower for weak detections)")
    parser.add_argument("--location-tag", default="lab_table",
                        help="Place name for memory; each station is logged as '<tag> location <name>'")
    parser.add_argument("--turn-only", action="store_true",
                        help="Never drive forward/reverse, only spin in place (tight spaces)")
    parser.add_argument("--turn-test", type=float, default=None,
                        help="Turn this many degrees (camera-measured, + = right) and stop; needs a camera")
    parser.add_argument("--log-file", default=None,
                        help="Also write the log here (default: <snapshots>/run_<time>.log)")
    parser.add_argument("--preview", action="store_true",
                        help="Live view only: stream what the route would detect to --laptop-url; no motors, no memory")
    parser.add_argument("--find-duty", action="store_true", help="Find the lowest duty that turns the car and exit")
    parser.add_argument("--list-cameras", action="store_true", help="Save a frame from each camera index and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    log_dir = args.snapshots or "data/station_test"
    os.makedirs(log_dir, exist_ok=True)
    log_path = args.log_file or os.path.join(log_dir, f"run_{time.strftime('%H%M%S')}.log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(handler)

    if args.list_cameras:
        list_cameras(10, args.snapshots or "data/station_test")
        return

    if args.preview:
        if not args.laptop_url:
            parser.error("--preview needs --laptop-url")
        camera = Camera(camera_type=args.mode, usb_device_index=args.camera_index)
        detector = SSDMobileNetV2Detector(backend="auto", confidence_threshold=args.min_confidence)
        runner = StationRunner(
            camera, detector, MotorController(force_mock=True), 1.0,
            telemetry_url=f"{args.laptop_url.rstrip('/')}/telemetry",
            min_box_height=args.min_box_height,
            rotate_deg=args.rotate,
        )
        runner.phase_signal, runner.phase_tag = "preview (motors off)", "preview"
        logger.info("Preview running; Ctrl+C to stop.")
        try:
            while True:
                runner._detect(None)
        except KeyboardInterrupt:
            pass
        finally:
            camera.release()
        return

    if not args.motor_port and not args.mock_motors:
        parser.error("give --motor-port (or --mock-motors for a dry run)")
    motor = MotorController(
        force_mock=args.mock_motors, serial_port=args.motor_port, default_speed=args.duty,
        camera_at_rear=args.camera_rear,
    )

    try:
        if args.calibrate:
            calibrate(motor, args.calibrate_pulses, args.pulse_ms, pause_sec=args.settle)
            return
        if args.find_duty:
            find_duty(motor, args.pulse_ms)
            return

        if args.deg_per_sec is None:
            parser.error("give --deg-per-sec (measure it with --calibrate first)")

        if args.turn_test is not None:
            camera = Camera(camera_type=args.mode, usb_device_index=args.camera_index)
            if camera.is_synthetic:
                parser.error("no camera found; use --mode oak/usb/csi")
            runner = StationRunner(
                camera, None, TurnOnlyMotor(motor), args.deg_per_sec,
                turn_chunk_ms=args.turn_chunk_ms, settle_sec=args.settle, hfov_deg=args.hfov or None,
                rotate_deg=args.rotate,
            )
            try:
                runner.rough_turn(args.turn_test)
            finally:
                camera.release()
            print(f"Asked for {args.turn_test:+.0f} deg; camera-measured heading now {runner.heading_deg:+.1f} deg.")
            return

        stations: Dict[str, Station] = {s.name: s for s in args.station}
        missing = [name for name in args.route if name not in stations]
        if missing:
            parser.error(f"route uses undefined station(s): {', '.join(missing)}")

        camera = Camera(camera_type=args.mode, usb_device_index=args.camera_index)
        if camera.is_synthetic:
            parser.error("no camera found; use --mode oak/usb/csi")
        detector = SSDMobileNetV2Detector(backend="auto", confidence_threshold=args.min_confidence)
        if detector.backend == "mock":
            parser.error("detector fell back to mock; run scripts/fetch_ssd_model.sh first")
        logger.info("camera=%s detector=%s duty=%d%% deg/s=%.0f", camera.camera_type, detector.backend, args.duty, args.deg_per_sec)

        runner = StationRunner(
            camera, detector, TurnOnlyMotor(motor) if args.turn_only else motor, args.deg_per_sec,
            turn_chunk_ms=args.turn_chunk_ms, settle_sec=args.settle, search_step_ms=args.pulse_ms,
            search_max_deg=args.search_deg,
            telemetry_url=f"{args.laptop_url.rstrip('/')}/telemetry" if args.laptop_url else None,
            hfov_deg=args.hfov or None,
            pulse_ms=args.pulse_ms, timeout_sec=args.timeout, snapshot_dir=args.snapshots or None,
            event_client=EventClient(server_url=args.laptop_url) if args.laptop_url else None,
            location_tag=args.location_tag,
            min_box_height=args.min_box_height,
            confirm_frames=args.confirm_frames,
            rotate_deg=args.rotate,
            turn_only=args.turn_only,
        )
        waits = {k: float(v) for k, v in (w.split("=", 1) for w in args.wait)}
        results = []
        try:
            for i, name in enumerate(args.route):
                results.append(runner.visit(stations[name]))
                pause = waits.get(name, 0.0) if i < len(args.route) - 1 else 0.0
                if pause > 0:
                    logger.info("  waiting %.0f s at location %s - change the scene now", pause, name)
                    runner.phase_signal = f"waiting {pause:.0f} s - change the scene now"
                    end = time.time() + pause
                    while time.time() < end:
                        runner._detect(None)  # keep the dashboard view live
                        time.sleep(0.2)
        finally:
            camera.release()

        print("\nstation  outcome          target      heading  center_x  seconds")
        for r in results:
            cx = f"{r.center_x:.2f}" if r.center_x is not None else "-"
            print(f"{r.station:<8} {r.outcome:<16} {str(r.target):<11} {r.heading_deg:+7.0f}  {cx:>8}  {r.seconds:7.1f}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        motor.cleanup()


if __name__ == "__main__":
    main()
