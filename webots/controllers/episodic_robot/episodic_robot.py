"""
Webots controller for the Episodic Perception demo.

This is the Jetson Nano's job, running inside Webots instead of on the real
chassis. It does NOT reimplement the pipeline - it imports the same modules the
physical robot runs:

    nano.perception_action.PerceptionActionStateMachine   (Phase 3)
    nano.scene_graph.SceneGraphBuilder                    (Phase 2)
    nano.importance_scoring.ImportanceScorer              (Phase 4)
    nano.event_client.EventClient                         (Phase 5)
    nano.detector.SSDMobileNetV2Detector                  (Phase 1, --detector ssd)
    shared.event_schema.Event

The only new code here is the glue Webots needs:

  * WebotsMotorAdapter - the same forward/reverse/turn_left/turn_right/stop
    interface as nano.motor_control, implemented with wheel motors. Pulses are
    short and fixed-duration, exactly as on the L298N driver, and advancing a
    pulse advances simulation time, so the state machine's timeout behaves the
    way it does on hardware.
  * a detection source: either tight bounding boxes projected from the
    simulator's own geometry (--detector groundtruth, the reliable default for a
    live demo) or the real COCO-pretrained SSD-MobileNet-V2 run over the
    rendered camera frames (--detector ssd).
  * a scripted patrol between fixed waypoints, each carrying a coarse
    location_tag. Waypoint following reads the robot's pose straight from the
    simulator - that stands in for the spec's "scripted movement or coarse zone
    tags" patrol, and nothing in the perception path ever sees it. No SLAM.

Arguments are passed through the world's controllerArgs field or on the webots
command line, e.g.

    --detector ssd --fog-url http://localhost:8080 --laps 2

Run it through scripts/run_webots_demo.sh rather than by hand.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

# Make the repository importable: .../webots/controllers/episodic_robot -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from controller import Supervisor  # noqa: E402  (provided by Webots)

from nano.detector import DetectionResult  # noqa: E402
from nano.importance_scoring import ImportanceScorer  # noqa: E402
from nano.perception_action import PerceptionActionStateMachine, State  # noqa: E402
from nano.scene_graph import SURFACE_CLASSES, SceneGraphBuilder  # noqa: E402
from shared.event_schema import Event, EventType  # noqa: E402

LOG_PATH = Path(__file__).resolve().parent / "episodic_robot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("webots.episodic_robot")


# --------------------------------------------------------------------------
# Patrol plan: each waypoint is (x, y, heading_to_face, location_tag)
# --------------------------------------------------------------------------
# The bottle's two homes. Relocating it between them is what gives the fog side
# a real location contradiction to resolve into a `moved` episode (Phase 7).
BOTTLE_HOME = (-1.55, 2.00)
BOTTLE_SHELF = (0.35, -2.25)

# Patrol plan, matching demo/sim2d.py's lab so both demos tell the same story.
# The viewing heading is expressed as an offset from the bearing to `look_at`,
# not as an absolute heading: the Phase 3 clip test only fires inside a narrow
# edge-margin band, and a fixed heading plus a few centimetres of drive-up drift
# misses it. Aiming relative to the object lands the same way every run.
#
#   lab_desk_3   bottle in the RIGHT edge-margin band  -> turn_right
#   lab_desk_1   laptop below the minimum bbox height  -> forward
#   lab_shelf_2  bottle in the LEFT edge-margin band   -> turn_left
WAYPOINTS: list[tuple[float, float, tuple[float, float], float, str]] = [
    (-1.15, 1.10, BOTTLE_HOME, math.radians(25.8), "lab_desk_3"),
    (2.10, -0.20, (2.25, 2.05), 0.0, "lab_desk_1"),
    (0.95, -1.25, BOTTLE_SHELF, math.radians(-25.8), "lab_shelf_2"),
]


class WebotsMotorAdapter:
    """nano.motor_control-compatible driver for the Webots wheel motors.

    Same five-method interface as the L298N controller, same short fixed-duration
    pulse semantics. Issuing a pulse steps the simulation for its duration, so
    simulated time and motor time stay consistent - which is what lets the
    perception-action loop's 7-second timeout mean the same thing here as on the
    real chassis.
    """

    def __init__(
        self,
        robot: Supervisor,
        timestep: int,
        drive_speed: float = 4.0,
        turn_speed: float = 2.6,
    ):
        self._robot = robot
        self._timestep = timestep
        self._drive_speed = drive_speed
        self._turn_speed = turn_speed
        self.action_history: list[tuple[str, int]] = []
        self.quit_requested = False

        self._left = robot.getDevice("left wheel motor")
        self._right = robot.getDevice("right wheel motor")
        for motor in (self._left, self._right):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

    # -- primitive ---------------------------------------------------------
    def _pulse(self, left: float, right: float, duration_ms: int, label: str) -> None:
        self._left.setVelocity(left)
        self._right.setVelocity(right)

        remaining = max(0, int(duration_ms))
        while remaining > 0:
            if self._robot.step(self._timestep) == -1:
                self.quit_requested = True
                break
            remaining -= self._timestep

        self._left.setVelocity(0.0)
        self._right.setVelocity(0.0)
        # One settle step so the chassis is not read mid-wobble.
        if not self.quit_requested and self._robot.step(self._timestep) == -1:
            self.quit_requested = True

        self.action_history.append((label, int(duration_ms)))

    # -- nano.motor_control interface --------------------------------------
    def forward(self, duration_ms: int = 150) -> None:
        self._pulse(self._drive_speed, self._drive_speed, duration_ms, "forward")

    def reverse(self, duration_ms: int = 150) -> None:
        self._pulse(-self._drive_speed, -self._drive_speed, duration_ms, "reverse")

    def turn_left(self, duration_ms: int = 150) -> None:
        self._pulse(-self._turn_speed, self._turn_speed, duration_ms, "turn_left")

    def turn_right(self, duration_ms: int = 150) -> None:
        self._pulse(self._turn_speed, -self._turn_speed, duration_ms, "turn_right")

    def stop(self) -> None:
        self._left.setVelocity(0.0)
        self._right.setVelocity(0.0)

    def cleanup(self) -> None:
        self.stop()


# Known extents of every observable Solid in episodic_lab.wbt. Positions are
# read live from the simulator, so a staged relocation is picked up
# automatically; only the shapes are fixed, and they are fixed here rather than
# derived from the world file because Webots exposes a Solid's bounding geometry
# only indirectly.
#
# The vertical pair is (z_lo, z_hi) **as offsets from the Solid's own origin**,
# not half-heights. That distinction is load-bearing: a laptop's origin is its
# base plate, a table's is its top plate, a bottle's is the middle of its body.
# Treating them all as vertically centred puts the laptop's bottom edge 11 cm
# below the desk surface, and then "laptop ON dining table" never fires.
OBJECT_EXTENTS: dict[str, tuple[str, float, float, float, float]] = {
    # solid name:   (class,          half_x, half_y,  z_lo,   z_hi)
    "desk 1":       ("dining table", 0.75,  0.35,   -0.72,   0.02),
    "desk 3":       ("dining table", 0.75,  0.35,   -0.72,   0.02),
    "shelf 2":      ("dining table", 0.70,  0.30,   -0.72,   0.02),
    "bottle":       ("bottle",       0.038, 0.038,  -0.095,  0.171),
    "vase 1":       ("vase",         0.055, 0.055,  -0.15,   0.15),
    "laptop 1":     ("laptop",       0.17,  0.12,   -0.01,   0.244),
    "cup 1":        ("cup",          0.045, 0.045,  -0.05,   0.05),
    "plant 1":      ("potted plant", 0.13,  0.13,   -0.07,   0.29),
}


class GroundTruthDetector:
    """Tight bounding boxes projected from the simulator's own geometry.

    Webots' `Recognition` node looks like the obvious choice here, and it is
    what the first version of this controller used - but what it reports is the
    projection of a *bounding sphere*, not a tight box. For a 1.5 m desk viewed
    from 1.3 m that inflates the box to fill the entire frame: the clip test
    fires permanently, and the desk's reported top edge ends up nowhere near its
    actual surface, so `scene_graph.check_on_relationship` can never see the
    bottom-edge/top-edge contact that makes "bottle ON dining table". Measured
    at the lab_desk_3 viewing pose, the desk came back as
    xmin=0.00 xmax=1.00 ymin=0.06 ymax=1.00 with a reported 3D size of
    1.91 x 1.91 m - the diagonal of the desk, not the desk.

    So this projects the eight corners of each object's axis-aligned box through
    the camera's real pose and intrinsics and takes the extremes: the same thing
    a detector outputs, and the same geometry demo/sim2d.py uses. It is still
    ground truth, and it is labelled as such in telemetry; the real CNN path is
    SSDDetector below.
    """

    name = "simulated detector (tight ground-truth projection)"

    def __init__(self, robot: Supervisor, camera, min_pixels: int = 5):
        self._robot = robot
        self._camera = camera
        self._min_pixels = min_pixels
        # getFromDevice() wants the C device tag, not the Python wrapper.
        self._camera_node = robot.getFromDevice(camera._tag)
        self._nodes: dict[str, Any] = {}
        for name in OBJECT_EXTENTS:
            node = _find_solid_by_name(robot, name)
            if node is None:
                logger.warning("World has no Solid named %r; it will not be detected.", name)
            else:
                self._nodes[name] = node

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        width, height = self._camera.getWidth(), self._camera.getHeight()
        tan_h = math.tan(self._camera.getFov() / 2.0)
        tan_v = tan_h * height / width

        # World -> camera. getPose() gives the camera's 4x4 row-major transform
        # in world coordinates; invert it with R^T and -R^T t.
        pose = self._camera_node.getPose()
        rotation = [pose[0:3], pose[4:7], pose[8:11]]
        translation = (pose[3], pose[7], pose[11])

        results: list[DetectionResult] = []
        for name, node in self._nodes.items():
            class_name, half_x, half_y, z_lo, z_hi = OBJECT_EXTENTS[name]
            ox, oy, oz = node.getPosition()
            z_bottom, z_top = oz + z_lo, oz + z_hi

            xs: list[float] = []
            ys: list[float] = []
            depths: list[float] = []
            for corner_x in (ox - half_x, ox + half_x):
                for corner_y in (oy - half_y, oy + half_y):
                    for corner_z in (z_bottom, z_top):
                        world = (corner_x - translation[0], corner_y - translation[1], corner_z - translation[2])
                        # Camera frame: +x forward (the direction the lens
                        # faces), +y left, +z up.
                        cam_x = sum(rotation[i][0] * world[i] for i in range(3))
                        cam_y = sum(rotation[i][1] * world[i] for i in range(3))
                        cam_z = sum(rotation[i][2] * world[i] for i in range(3))
                        if cam_x <= 1e-3:
                            continue
                        xs.append(0.5 - (cam_y / cam_x) / (2.0 * tan_h))
                        ys.append(0.5 - (cam_z / cam_x) / (2.0 * tan_v))
                        depths.append(cam_x)

            if not xs:
                continue  # entirely behind the camera

            xmin_raw, xmax_raw = min(xs), max(xs)
            ymin_raw, ymax_raw = min(ys), max(ys)
            if xmax_raw <= 0.0 or xmin_raw >= 1.0 or ymax_raw <= 0.0 or ymin_raw >= 1.0:
                continue
            # Too little of it in frame for a detector to fire on.
            visible = min(xmax_raw, 1.0) - max(xmin_raw, 0.0)
            if visible <= 0.0 or visible / max(xmax_raw - xmin_raw, 1e-6) < 0.25:
                continue
            if (xmax_raw - xmin_raw) * width < self._min_pixels:
                continue

            distance = min(depths)
            confidence = float(np.clip(0.96 - 0.055 * distance, 0.45, 0.99))
            results.append(
                DetectionResult(
                    class_name=class_name,
                    confidence=round(confidence, 3),
                    bbox=(
                        float(np.clip(xmin_raw, 0.0, 1.0)),
                        float(np.clip(ymin_raw, 0.0, 1.0)),
                        float(np.clip(xmax_raw, 0.0, 1.0)),
                        float(np.clip(ymax_raw, 0.0, 1.0)),
                    ),
                )
            )
        return results

def _find_solid_by_name(robot: Supervisor, name: str) -> Optional[Any]:
    """Depth-first search of the scene tree for a Solid with this `name`."""
    stack = [robot.getRoot().getField("children")]
    while stack:
        children = stack.pop()
        for index in range(children.getCount()):
            node = children.getMFNode(index)
            name_field = node.getField("name")
            if name_field is not None and name_field.getSFString() == name:
                return node
            nested = node.getField("children")
            if nested is not None:
                stack.append(nested)
    return None


class SSDDetector:
    """The real COCO-pretrained SSD-MobileNet-V2, run over rendered frames."""

    name = "ssd-mobilenet-v2 (opencv dnn)"

    def __init__(self, confidence_threshold: float = 0.35):
        from nano.detector import SSDMobileNetV2Detector

        self._detector = SSDMobileNetV2Detector(
            backend="opencv", confidence_threshold=confidence_threshold
        )

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        return self._detector.detect(frame)


class TelemetryPublisher:
    """Best-effort POST of per-frame state to the demo dashboard.

    Never raises and never blocks the control loop for long: if the dashboard
    is not running, the robot just keeps patrolling.
    """

    def __init__(self, url: Optional[str], timeout: float = 0.6):
        self._url = f"{url.rstrip('/')}/telemetry" if url else None
        self._timeout = timeout
        self._failures = 0

    def publish(self, payload: dict[str, Any]) -> None:
        if not self._url or self._failures > 20:
            return
        try:
            request = urllib.request.Request(
                self._url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self._timeout):
                pass
            self._failures = 0
        except (urllib.error.URLError, OSError, TimeoutError):
            self._failures += 1
            if self._failures == 1:
                logger.info("Dashboard not reachable at %s - continuing headless.", self._url)


class EpisodicRobot:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)


        self.motor = WebotsMotorAdapter(self.robot, self.timestep)
        self.perception_action = PerceptionActionStateMachine(
            edge_margin=args.edge_margin,
            min_height=args.min_height,
            pulse_ms=args.pulse_ms,
            confirm_frames=args.confirm_frames,
            timeout_sec=args.timeout_sec,
            motor=self.motor,
            # Desks and shelves are scene-graph context, never targets to chase
            # - see PerceptionActionStateMachine.ignore_classes.
            ignore_classes=SURFACE_CLASSES,
        )
        # y_margin is loosened from the library default of 0.08.
        #
        # The ON test compares an object's bottom edge against the surface's
        # bbox top edge, but on a 0.7 m-deep desk the projected top edge comes
        # from the *far* edge, not from the contact point - so the gap grows
        # with how far forward on the desk a thing sits and how close the camera
        # is. Measured at lab_desk_3: the vase (1.46 m away) is 0.017 off, the
        # bottle (0.82 m away) is 0.118 off. 0.08 catches the first and misses
        # the second, which is why the first version of this demo produced
        # unstable "dining table LEFT OF bottle" relations instead of
        # "bottle ON dining table". A per-surface margin derived from the
        # surface's apparent depth would be the principled fix; see the
        # limitations section of demo/README.md.
        self.scene_builder = SceneGraphBuilder(y_margin=0.15)
        self.importance_scorer = ImportanceScorer()

        if args.detector == "ssd":
            self.detector = SSDDetector(confidence_threshold=args.confidence)
        else:
            self.detector = GroundTruthDetector(self.robot, self.camera)
        logger.info("Detection source: %s", self.detector.name)

        from nano.event_client import EventClient

        self.event_client = EventClient(server_url=args.fog_url)
        self.telemetry = TelemetryPublisher(args.fog_url if args.telemetry else None)

        self.location_tag = WAYPOINTS[0][4]
        self.frames = 0
        self.events_sent = 0
        self.confirmations = 0
        self.timeouts = 0
        self.partial_views = 0
        self.recoveries = 0
        self.filtered_out = 0
        # True once the loop has entered ADJUST for the target it is tracking,
        # so a confirmation can be told apart from one that needed no
        # correction at all. Without this, every re-confirmation during a dwell
        # counts as a recovery and the success rate becomes meaningless.
        self._corrected_this_target = False
        self.bottle_moves = 0

        if self.event_client.check_connection():
            logger.info("Fog server reachable at %s", args.fog_url)
        else:
            logger.warning(
                "Fog server not reachable at %s - events will still be attempted.",
                args.fog_url,
            )

    # -- sensing -----------------------------------------------------------
    def grab_frame(self) -> Optional[np.ndarray]:
        raw = self.camera.getImage()
        if raw is None:
            return None
        height, width = self.camera.getHeight(), self.camera.getWidth()
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
        return np.ascontiguousarray(bgra[:, :, :3])  # BGR

    def pose(self) -> tuple[float, float, float]:
        node = self.robot.getSelf()
        x, y, _ = node.getPosition()
        orientation = node.getOrientation()
        yaw = math.atan2(orientation[3], orientation[0])
        return x, y, yaw

    # -- patrol ------------------------------------------------------------
    def _step_once(self) -> bool:
        if self.robot.step(self.timestep) == -1:
            return False
        return True

    def drive_to(
        self, target_x: float, target_y: float, look_at: tuple[float, float], aim_offset: float
    ) -> bool:
        """Scripted waypoint approach built only from short motor pulses.

        Pulse length is scaled to the remaining error (floored at 30 ms) so the
        chassis can settle inside a tolerance tighter than one full pulse
        instead of ping-ponging across it.
        """
        for _ in range(self.args.max_patrol_pulses):
            if self.motor.quit_requested:
                return False
            x, y, yaw = self.pose()
            dx, dy = target_x - x, target_y - y
            distance = math.hypot(dx, dy)
            if distance < 0.06:
                break
            bearing_error = self._wrap(math.atan2(dy, dx) - yaw)
            if abs(bearing_error) > math.radians(7):
                self._turn(bearing_error)
            else:
                self.motor.forward(140)
            self.publish_telemetry(mode="PATROL", detections=[], signal="patrol_drive")

        # Final aim is taken relative to the station's look-at point, so
        # arriving a few centimetres off does not move the target in frame.
        for _ in range(self.args.max_patrol_pulses):
            if self.motor.quit_requested:
                return False
            x, y, yaw = self.pose()
            desired = math.atan2(look_at[1] - y, look_at[0] - x) + aim_offset
            heading_error = self._wrap(desired - yaw)
            if abs(heading_error) < math.radians(1.5):
                break
            self._turn(heading_error)
            self.publish_telemetry(mode="PATROL", detections=[], signal="patrol_align")
        return True

    def _turn(self, error_rad: float) -> None:
        """One turn pulse, no longer than the error needs."""
        wheel_radius, half_track = 0.052, 0.115
        rate = self.motor._turn_speed * wheel_radius / half_track  # rad/s of yaw
        pulse = int(max(30.0, min(140.0, abs(error_rad) / rate * 1000.0)))
        (self.motor.turn_left if error_rad > 0 else self.motor.turn_right)(pulse)

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # -- perception-action at a station -------------------------------------
    def observe_station(self, max_frames: int) -> None:
        """Run the perception-action loop until it logs an episode or runs out
        of frames. This is the core-novelty section of the demo."""
        self.perception_action.reset_to_patrol()
        self.importance_scorer  # retained across stations on purpose

        for _ in range(max_frames):
            if self.motor.quit_requested or not self._step_once():
                return

            frame = self.grab_frame()
            if frame is None:
                continue
            self.frames += 1

            detections = self.detector.detect(frame)
            entry_state = self.perception_action.state
            state, emitted = self.perception_action.step(
                detections, current_time=self.robot.getTime()
            )
            # reset_to_patrol() clears last_signal on the frame that logs, so
            # report what actually triggered the emit, not the reset value.
            signal = (
                "fully_visible"
                if emitted is not None and emitted.event_type == EventType.CONFIRMED
                else self.perception_action.last_signal.value
            )

            if state == State.ADJUST:
                if not self._corrected_this_target:
                    self.partial_views += 1
                self._corrected_this_target = True

            if entry_state != state:
                logger.info(
                    "[t=%.2f] %s -> %s  signal=%s  dets=%d",
                    self.robot.getTime(),
                    entry_state.value,
                    state.value,
                    signal,
                    len(detections),
                )

            self.publish_telemetry(
                mode=state.value, detections=detections, signal=signal, frame=frame
            )

            if emitted is not None:
                if emitted.event_type == EventType.PARTIAL_TIMEOUT:
                    self.timeouts += 1
                else:
                    self.confirmations += 1
                    if self._corrected_this_target:
                        self.recoveries += 1
                self._corrected_this_target = False
                self.dispatch(emitted, detections)
                # Keep observing rather than leaving the moment something is
                # logged: a patrolling robot dwells, and the repeat
                # observations are what the Phase 4 importance filter is for.

    def dispatch(self, emitted, detections) -> None:
        """Scene graph -> importance filter -> event POST. Mirrors
        nano/main_loop.py::_process_emitted_observation."""
        relationships, objects = self.scene_builder.build_scene_graph(detections)

        if emitted.target_object in objects:
            objects.remove(emitted.target_object)
        objects.insert(0, emitted.target_object)

        # The confirmed target is what the episode is about, so put its
        # relationship first rather than burying the subject.
        relationships.sort(key=lambda r: not r.startswith(f"{emitted.target_object} "))

        event = Event(
            objects=objects,
            relationships=relationships,
            confidence=emitted.confidence,
            event_type=emitted.event_type,
            bbox=emitted.bbox,
            location_tag=self.location_tag,
        )
        filtered = self.importance_scorer.filter_event(event)
        if filtered is None:
            self.filtered_out += 1
            # Per-frame, this fires a couple of times a second for the whole
            # dwell and buries everything else. The dashboard shows the running
            # count live; the console gets the state transitions that matter.
            logger.debug(
                "Observation dropped as redundant: %s %s @%s",
                event.event_type,
                event.objects,
                event.location_tag,
            )
            return

        logger.info(
            "Observation: type=%s objects=%s rels=%s @%s",
            filtered.event_type,
            filtered.objects,
            filtered.relationships,
            filtered.location_tag,
        )
        if self.event_client.send_event(filtered):
            self.events_sent += 1

    # -- staged object movement (for contradiction detection) ---------------
    def stage_bottle_move(self) -> None:
        """Relocate the bottle between lab_desk_3 and lab_shelf_2.

        The simulated stand-in for the spec's staged object-movement events: the
        operator moves something while the robot is looking elsewhere, so the
        fog side gets a real location contradiction to resolve. Called twice by
        the patrol (out, then back), because the return trip is what makes
        memory *compression* observable - revisiting a place the robot has
        already described extends that episode instead of storing a duplicate.
        """
        bottle = _find_solid_by_name(self.robot, "bottle")
        if bottle is None:
            logger.warning("Could not find the bottle node - skipping the staged move.")
            return

        field = bottle.getField("translation")
        current = field.getSFVec3f()
        at_shelf = math.hypot(current[0] - BOTTLE_SHELF[0], current[1] - BOTTLE_SHELF[1]) < 0.25
        destination = BOTTLE_HOME if at_shelf else BOTTLE_SHELF
        # 0.835 puts the bottle's base on a 0.74 table top, same as the world
        # file's starting value - not on the floor.
        field.setSFVec3f([destination[0], destination[1], 0.835])
        bottle.resetPhysics()
        self.bottle_moves += 1
        logger.info(
            "STAGED EVENT: bottle relocated to %s",
            "lab_desk_3" if destination == BOTTLE_HOME else "lab_shelf_2",
        )

    # -- telemetry ---------------------------------------------------------
    def publish_telemetry(
        self,
        mode: str,
        detections: list[DetectionResult],
        signal: str,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        if not self.args.telemetry:
            return
        x, y, yaw = self.pose()
        payload: dict[str, Any] = {
            "source": "webots",
            "sim_time": round(self.robot.getTime(), 3),
            "state": mode,
            "signal": signal,
            "location_tag": self.location_tag,
            "pose": {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 4)},
            "detector": self.detector.name,
            "detections": [d.to_dict() for d in detections],
            "edge_margin": self.args.edge_margin,
            "stats": {
                "frames": self.frames,
                "confirmations": self.confirmations,
                "partial_views": self.partial_views,
                "recoveries": self.recoveries,
                "timeouts": self.timeouts,
                "events_sent": self.events_sent,
                "filtered_out": self.filtered_out,
            },
            "last_motor_action": (
                list(self.motor.action_history[-1]) if self.motor.action_history else None
            ),
        }

        if frame is not None and self.frames % self.args.frame_stride == 0:
            try:
                import cv2

                ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                if ok:
                    payload["frame_jpeg_b64"] = base64.b64encode(buffer.tobytes()).decode("ascii")
            except Exception:
                pass

        self.telemetry.publish(payload)

    # -- main --------------------------------------------------------------
    def run(self) -> None:
        logger.info("=== Episodic Perception patrol starting (Webots) ===")
        for lap in range(1, self.args.laps + 1):
            for index, (x, y, look_at, aim_offset, tag) in enumerate(WAYPOINTS):
                if self.motor.quit_requested:
                    break
                self.location_tag = tag
                logger.info("--- lap %d, waypoint %d (%s) ---", lap, index + 1, tag)
                if not self.drive_to(x, y, look_at, aim_offset):
                    break
                self.observe_station(self.args.station_frames)

                # Stage the two relocations between desk 1 and the shelf, i.e.
                # while the robot is not looking at either endpoint.
                if self.args.stage_move and index == 1 and self.bottle_moves < 2 and lap in (1, 2):
                    self.stage_bottle_move()

        self.motor.cleanup()
        logger.info(
            "=== Patrol finished: frames=%d confirmations=%d partial_views=%d "
            "recoveries=%d timeouts=%d filtered=%d events_sent=%d ===",
            self.frames,
            self.confirmations,
            self.partial_views,
            self.recoveries,
            self.timeouts,
            self.filtered_out,
            self.events_sent,
        )
        summary = {
            "detector": self.detector.name,
            "frames": self.frames,
            "confirmations": self.confirmations,
            "partial_views": self.partial_views,
            "recoveries": self.recoveries,
            "timeouts": self.timeouts,
            "perception_action_success_rate": (
                round(self.recoveries / self.partial_views, 3) if self.partial_views else None
            ),
            "filtered_out": self.filtered_out,
            "events_sent": self.events_sent,
            "bottle_moves": self.bottle_moves,
        }
        (Path(__file__).resolve().parent / "run_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        if self.args.quit_when_done:
            self.robot.simulationQuit(0)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Episodic Perception - Webots edge controller")
    parser.add_argument(
        "--detector",
        choices=["groundtruth", "ssd"],
        default=os.environ.get("EPISODIC_DETECTOR", "groundtruth"),
        help="tight ground-truth projection (default) or the real SSD-MobileNet-V2",
    )
    parser.add_argument(
        "--fog-url", default=os.environ.get("EPISODIC_FOG_URL", "http://localhost:8080")
    )
    parser.add_argument("--laps", type=int, default=2)
    parser.add_argument("--station-frames", type=int, default=140)
    parser.add_argument("--max-patrol-pulses", type=int, default=260)
    parser.add_argument("--edge-margin", type=float, default=0.05)
    parser.add_argument("--min-height", type=float, default=0.15)
    parser.add_argument("--pulse-ms", type=int, default=150)
    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument("--timeout-sec", type=float, default=7.0)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--no-telemetry", dest="telemetry", action="store_false")
    parser.add_argument("--no-stage-move", dest="stage_move", action="store_false")
    parser.add_argument("--keep-running", dest="quit_when_done", action="store_false")
    parser.set_defaults(telemetry=True, stage_move=True, quit_when_done=True)
    # Webots may append its own arguments; ignore anything unrecognised.
    namespace, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.debug("Ignoring unrecognised controller arguments: %s", unknown)
    return namespace


if __name__ == "__main__":
    try:
        EpisodicRobot(parse_args(sys.argv[1:])).run()
    except Exception:
        logger.exception("Controller crashed")
        raise
