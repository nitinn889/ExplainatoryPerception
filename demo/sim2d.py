"""
Lightweight lab simulator for the browser demo.

Webots is the high-fidelity option; this is the one that runs anywhere, in a
couple of seconds, with nothing but numpy installed. It is not a canned
animation: the robot has a pose, the lab has objects at real coordinates, and
every bounding box is a **pinhole projection** of an object into the camera
frame. So when the perception-action loop decides a bottle is clipped on the
right edge and pulses `turn_right`, the box moves because the camera actually
rotated - the same way it does on hardware.

What is simulated:
  * differential-drive kinematics driven by short fixed-duration pulses, with a
    little seeded noise so runs are reproducible but not mechanical
  * a pinhole camera (horizontal FOV, aspect ratio, near plane, max range)
  * per-object apparent size and confidence falling off with distance

What is not (and is documented as such in the demo README):
  * occlusion - the lab is laid out so nothing meaningfully occludes anything
  * image formation - there are no pixels, so this mode cannot exercise the real
    SSD-MobileNet-V2 weights. Use the Webots mode with `--detector ssd` for that.

Everything downstream of `detect()` is the real pipeline: the same
PerceptionActionStateMachine, SceneGraphBuilder, ImportanceScorer and
EventClient the Jetson Nano runs.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any

from nano.detector import DetectionResult

logger = logging.getLogger("demo.sim2d")


# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------

@dataclass
class SimObject:
    """A box-shaped thing in the lab, in metres, world frame (x east, y north, z up)."""

    name: str
    class_name: str
    x: float
    y: float
    z_center: float
    half_width: float
    height: float
    color: str
    station: str = ""

    @property
    def z_bottom(self) -> float:
        return self.z_center - self.height / 2.0

    @property
    def z_top(self) -> float:
        return self.z_center + self.height / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class": self.class_name,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "half_width": self.half_width,
            "color": self.color,
            "station": self.station,
        }


@dataclass
class Station:
    """A patrol waypoint and the coarse location_tag it maps to (no SLAM).

    The viewing heading is given as `aim_offset_deg` **relative to the bearing
    to `look_at`**, not as an absolute compass heading. That matters: the
    perception-action loop is only triggered when a box reaches the edge-margin
    band, which is a window a couple of degrees wide at these distances. An
    absolute heading plus a few centimetres of drive-up drift misses it
    entirely, and the robot just logs whatever happens to be centred. Aiming
    relative to the object makes the demo land the same way every run, however
    the approach went.
    """

    tag: str
    x: float
    y: float
    look_at: tuple[float, float]
    aim_offset_deg: float

    def heading_from(self, x: float, y: float) -> float:
        """Heading, in radians, that puts `look_at` at `aim_offset_deg` off-centre."""
        bearing = math.atan2(self.look_at[1] - y, self.look_at[0] - x)
        return _wrap(bearing + math.radians(self.aim_offset_deg))


# The two places the bottle lives. Relocating it between them is what gives the
# fog side a real location contradiction to resolve (Phase 7).
BOTTLE_HOME = (-1.55, 2.00)
BOTTLE_SHELF = (0.35, -2.25)


def build_lab() -> tuple[list[SimObject], list[Station]]:
    """The same three-station lab as webots/worlds/episodic_lab.wbt.

    Table tops sit at z = 0.74, and every object on a table has its bottom face
    exactly there, so `scene_graph.check_on_relationship` sees a real
    bottom-edge/top-edge contact and emits "<thing> ON dining table".

    Note what is *not* here: no books, phones or papers lying flat. A 3 cm-tall
    object can never satisfy the Phase 3 `min_height` rule from any sane
    distance, so the loop reads it as "too far" and drives forward forever. That
    is a genuine limitation of a single fixed minimum-height threshold, and it
    is written up in demo/README.md rather than hidden by picking kind objects.
    """
    top = 0.74
    objects = [
        # --- lab_desk_3 (west) -------------------------------------------
        SimObject("desk 3", "dining table", -2.00, 2.10, top / 2, 0.75, top, "#8a5a2b", "lab_desk_3"),
        SimObject("bottle", "bottle", -1.55, 2.00, top + 0.125, 0.045, 0.25, "#1b73bf", "lab_desk_3"),
        SimObject("vase 1", "vase", -2.35, 2.05, top + 0.150, 0.055, 0.30, "#c9a227", "lab_desk_3"),
        # --- lab_desk_1 (east) -------------------------------------------
        SimObject("desk 1", "dining table", 2.00, 2.10, top / 2, 0.75, top, "#8a5a2b", "lab_desk_1"),
        SimObject("laptop 1", "laptop", 2.25, 2.05, top + 0.12, 0.17, 0.24, "#26272b", "lab_desk_1"),
        SimObject("cup 1", "cup", 1.60, 2.00, top + 0.05, 0.045, 0.10, "#ededea", "lab_desk_1"),
        # --- lab_shelf_2 (south) -----------------------------------------
        SimObject("shelf 2", "dining table", 0.00, -2.30, top / 2, 0.70, top, "#8a5a2b", "lab_shelf_2"),
        # 0.36 tall (pot 0.14 + foliage reaching 1.10), centred so its base is
        # exactly on the 0.74 surface - matching the Webots world's plant.
        SimObject("plant 1", "potted plant", -0.45, -2.30, top + 0.18, 0.13, 0.36, "#35873a", "lab_shelf_2"),
    ]
    # Each station exercises a different branch of the Phase 3 correction table
    # against real projection geometry:
    #   lab_desk_3   bottle sits in the RIGHT edge-margin band -> turn_right
    #   lab_desk_1   laptop is below the minimum bbox height   -> forward
    #   lab_shelf_2  bottle sits in the LEFT edge-margin band  -> turn_left
    stations = [
        Station("lab_desk_3", -1.15, 1.10, BOTTLE_HOME, +25.8),
        Station("lab_desk_1", 2.10, -0.20, (2.25, 2.05), 0.0),
        Station("lab_shelf_2", 0.95, -1.25, BOTTLE_SHELF, -25.8),
    ]
    return objects, stations


# --------------------------------------------------------------------------
# Robot + camera
# --------------------------------------------------------------------------

@dataclass
class SimRobot:
    x: float = 0.0
    y: float = 0.0
    yaw: float = math.radians(90.0)
    camera_height: float = 0.80
    camera_pitch: float = -0.06  # radians, slight downward tilt
    hfov: float = 1.0
    width: int = 320
    height: int = 240
    max_range: float = 5.0
    near: float = 0.18

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "yaw": round(self.yaw, 4),
            "hfov": self.hfov,
            "camera_height": self.camera_height,
        }


class Sim2DMotor:
    """nano.motor_control-compatible driver for the 2D kinematic robot.

    Short fixed-duration pulses only - the same guardrail the spec puts on the
    L298N driver, and the reason the loop converges instead of oscillating
    between frame edges.
    """

    def __init__(
        self,
        robot: SimRobot,
        drive_speed: float = 0.28,   # m/s
        turn_rate: float = 0.85,     # rad/s
        time_scale: float = 1.0,
        seed: int = 7,
    ):
        self.robot = robot
        self.drive_speed = drive_speed
        self.turn_rate = turn_rate
        self.time_scale = time_scale
        self.action_history: list[tuple[str, int]] = []
        self._rng = random.Random(seed)

    def _noise(self, magnitude: float) -> float:
        return self._rng.gauss(0.0, magnitude)

    def _sleep(self, duration_ms: int) -> None:
        if self.time_scale > 0:
            time.sleep((duration_ms / 1000.0) * self.time_scale)

    def _translate(self, metres: float, duration_ms: int, label: str) -> None:
        drift = self._noise(0.012)
        self.robot.x += metres * math.cos(self.robot.yaw + drift)
        self.robot.y += metres * math.sin(self.robot.yaw + drift)
        self.robot.yaw = _wrap(self.robot.yaw + drift * 0.4)
        self.action_history.append((label, duration_ms))
        self._sleep(duration_ms)

    def _rotate(self, radians: float, duration_ms: int, label: str) -> None:
        self.robot.yaw = _wrap(self.robot.yaw + radians * (1.0 + self._noise(0.04)))
        self.action_history.append((label, duration_ms))
        self._sleep(duration_ms)

    def forward(self, duration_ms: int = 150) -> None:
        self._translate(self.drive_speed * duration_ms / 1000.0, duration_ms, "forward")

    def reverse(self, duration_ms: int = 150) -> None:
        self._translate(-self.drive_speed * duration_ms / 1000.0, duration_ms, "reverse")

    def turn_left(self, duration_ms: int = 150) -> None:
        self._rotate(self.turn_rate * duration_ms / 1000.0, duration_ms, "turn_left")

    def turn_right(self, duration_ms: int = 150) -> None:
        self._rotate(-self.turn_rate * duration_ms / 1000.0, duration_ms, "turn_right")

    def stop(self) -> None:
        self.action_history.append(("stop", 0))

    def cleanup(self) -> None:
        self.stop()


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


# --------------------------------------------------------------------------
# The camera: pinhole projection -> normalised bounding boxes
# --------------------------------------------------------------------------

@dataclass
class ProjectedDetection:
    detection: DetectionResult
    color: str
    distance: float
    clipped: bool


class SimCamera:
    """Projects world objects into normalised 0-1 image coordinates.

    Boxes are reported clamped to the frame (that is what a detector outputs),
    but the clip flag is computed from the *unclamped* projection, so an object
    hanging off the edge is genuinely clipped rather than merely near the edge.
    """

    def __init__(self, robot: SimRobot, world: list[SimObject], seed: int = 11):
        self.robot = robot
        self.world = world
        self._rng = random.Random(seed)

    def detect(self) -> list[ProjectedDetection]:
        robot = self.robot
        tan_h = math.tan(robot.hfov / 2.0)
        tan_v = math.tan(math.atan(tan_h / robot.aspect))

        results: list[ProjectedDetection] = []
        for obj in self.world:
            dx, dy = obj.x - robot.x, obj.y - robot.y
            forward = dx * math.cos(robot.yaw) + dy * math.sin(robot.yaw)
            lateral = -dx * math.sin(robot.yaw) + dy * math.cos(robot.yaw)

            if forward <= robot.near or forward > robot.max_range:
                continue

            # Horizontal: positive lateral is to the robot's left -> smaller x.
            cx = 0.5 - (lateral / forward) / (2.0 * tan_h)
            half_w = (obj.half_width / forward) / (2.0 * tan_h)

            # Vertical: project top and bottom faces separately, including the
            # camera's downward tilt, so tall nearby objects clip at the bottom.
            pitch_shift = math.tan(robot.camera_pitch) / (2.0 * tan_v)
            y_top = 0.5 - ((obj.z_top - robot.camera_height) / forward) / (2.0 * tan_v) + pitch_shift
            y_bottom = 0.5 - ((obj.z_bottom - robot.camera_height) / forward) / (2.0 * tan_v) + pitch_shift

            xmin_raw, xmax_raw = cx - half_w, cx + half_w
            ymin_raw, ymax_raw = min(y_top, y_bottom), max(y_top, y_bottom)

            # Entirely outside the frame -> not detected at all.
            if xmax_raw <= 0.0 or xmin_raw >= 1.0 or ymax_raw <= 0.0 or ymin_raw >= 1.0:
                continue
            # Mostly outside -> too little evidence for a detector to fire.
            visible_x = min(xmax_raw, 1.0) - max(xmin_raw, 0.0)
            if visible_x <= 0.0 or visible_x / max(xmax_raw - xmin_raw, 1e-6) < 0.25:
                continue

            distance = math.hypot(dx, dy)
            confidence = 0.96 - 0.055 * distance + self._rng.uniform(-0.015, 0.015)
            confidence = min(0.99, max(0.42, confidence))

            detection = DetectionResult(
                class_name=obj.class_name,
                confidence=round(confidence, 3),
                bbox=(
                    max(0.0, min(1.0, xmin_raw)),
                    max(0.0, min(1.0, ymin_raw)),
                    max(0.0, min(1.0, xmax_raw)),
                    max(0.0, min(1.0, ymax_raw)),
                ),
            )
            results.append(
                ProjectedDetection(
                    detection=detection,
                    color=obj.color,
                    distance=round(distance, 2),
                    clipped=(
                        xmin_raw < 0.0 or xmax_raw > 1.0 or ymin_raw < 0.0 or ymax_raw > 1.0
                    ),
                )
            )

        # Nearest last, so the UI draws close objects on top.
        results.sort(key=lambda r: r.distance, reverse=True)
        return results
