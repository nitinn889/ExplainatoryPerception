"""
The edge (Nano) side of the browser demo, driving the 2D lab simulator.

This is the counterpart of webots/controllers/episodic_robot/episodic_robot.py:
same pipeline, different body. Both of them import the Nano's own modules
rather than reimplementing anything -

    nano.perception_action.PerceptionActionStateMachine   (Phase 3, core novelty)
    nano.scene_graph.SceneGraphBuilder                    (Phase 2)
    nano.importance_scoring.ImportanceScorer              (Phase 4)
    nano.event_client.EventClient                         (Phase 5)

- and the only thing that differs between the two is where frames and motor
pulses come from.

Patrol is scripted: a fixed list of waypoints, each carrying a coarse
location_tag, reached with short motor pulses. Waypoint following reads the
simulator's pose, which stands in for the spec's "scripted movement or coarse
zone tags". Nothing in the perception path ever sees it - no SLAM, no
localisation.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable, Optional

from demo.sim2d import (
    BOTTLE_HOME,
    BOTTLE_SHELF,
    ProjectedDetection,
    SimCamera,
    Sim2DMotor,
    SimRobot,
    Station,
    build_lab,
)
from nano.importance_scoring import ImportanceScorer
from nano.perception_action import PerceptionActionStateMachine, State
from nano.scene_graph import SURFACE_CLASSES, SceneGraphBuilder
from shared.event_schema import Event, EventType

logger = logging.getLogger("demo.edge_agent")

TelemetrySink = Callable[[dict[str, Any]], None]
EventSink = Callable[[Event], bool]


class EdgeAgent:
    """Runs the full edge loop over the 2D lab, in its own thread."""

    def __init__(
        self,
        on_telemetry: TelemetrySink,
        on_event: EventSink,
        fps: float = 6.0,
        time_scale: float = 1.0,
        edge_margin: float = 0.05,
        min_height: float = 0.15,
        pulse_ms: int = 120,
        confirm_frames: int = 4,
        timeout_sec: float = 7.0,
        station_frames: int = 90,
        laps: int = 0,  # 0 = patrol forever
    ):
        self.world, self.stations = build_lab()
        self.robot = SimRobot()
        self.camera = SimCamera(self.robot, self.world)
        self.motor = Sim2DMotor(self.robot, time_scale=time_scale)
        self.perception_action = PerceptionActionStateMachine(
            edge_margin=edge_margin,
            min_height=min_height,
            pulse_ms=pulse_ms,
            confirm_frames=confirm_frames,
            timeout_sec=timeout_sec,
            motor=self.motor,
            # Desks and shelves are context for the scene graph, never targets
            # to chase - see PerceptionActionStateMachine.ignore_classes.
            ignore_classes=SURFACE_CLASSES,
        )
        self.scene_builder = SceneGraphBuilder()
        self.importance_scorer = ImportanceScorer()

        self.on_telemetry = on_telemetry
        self.on_event = on_event
        self.frame_interval = 1.0 / max(fps, 0.5)
        self.station_frames = station_frames
        self.laps = laps
        self.edge_margin = edge_margin

        self.location_tag = self.stations[0].tag
        self.frames = 0
        self.confirmations = 0
        self.timeouts = 0
        self.partial_views = 0
        self.recoveries = 0
        self.events_sent = 0
        self.filtered_out = 0
        # True once the loop has entered ADJUST for the target it is tracking,
        # so a confirmation can be told apart from one that needed no
        # correction at all. Without this, every re-confirmation during a dwell
        # counts as a recovery and the success rate becomes meaningless.
        self._corrected_this_target = False
        self.lap = 0
        self.bottle_moves = 0
        self.last_detections: list[ProjectedDetection] = []
        self._dropped_here = 0

        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._resume.set()
        self._thread = threading.Thread(target=self._run, name="edge-agent", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    def _wait(self) -> bool:
        """Block while paused; return False once the agent should exit."""
        while not self._resume.wait(timeout=0.2):
            if self._stop.is_set():
                return False
        return not self._stop.is_set()

    # -- staged object movement -------------------------------------------
    def move_bottle(self, to: Optional[tuple[float, float]] = None) -> bool:
        """Relocate the bottle between lab_desk_3 and lab_shelf_2.

        The simulated stand-in for the spec's staged object-movement events: the
        operator moves something while the robot is looking elsewhere, so the
        fog side gets a real location contradiction to resolve into a `moved`
        episode. Called twice by the patrol (out, then back) because the return
        trip is what makes memory *compression* observable - the second visit to
        a place the robot has already described extends that episode instead of
        storing a duplicate. Also wired to a dashboard button.
        """
        for obj in self.world:
            if obj.class_name != "bottle":
                continue
            if to is None:
                at_shelf = math.hypot(obj.x - BOTTLE_SHELF[0], obj.y - BOTTLE_SHELF[1]) < 0.2
                to = BOTTLE_HOME if at_shelf else BOTTLE_SHELF
            obj.x, obj.y = to
            obj.station = "lab_shelf_2" if to == BOTTLE_SHELF else "lab_desk_3"
            self.bottle_moves += 1
            logger.info("STAGED EVENT: bottle relocated to %s (%.2f, %.2f)", obj.station, *to)
            return True
        return False

    # -- patrol ------------------------------------------------------------
    def _drive_to(self, station: Station) -> bool:
        """Scripted waypoint approach built only from short motor pulses.

        Pulse length is scaled to the remaining error (floored at 30 ms) so the
        robot can settle inside a tolerance tighter than one full pulse, instead
        of ping-ponging across it. The final aim is taken relative to the
        station's look-at point, so arriving a few centimetres off does not
        change where the target lands in frame.
        """
        for _ in range(500):
            if not self._wait():
                return False
            dx, dy = station.x - self.robot.x, station.y - self.robot.y
            distance = math.hypot(dx, dy)
            if distance < 0.04:
                break
            bearing_error = _wrap(math.atan2(dy, dx) - self.robot.yaw)
            if abs(bearing_error) > math.radians(6):
                self._turn(bearing_error)
            else:
                self.motor.forward(self._drive_pulse_ms(distance))
            self._emit_telemetry("PATROL", "patrol_drive")

        for _ in range(200):
            if not self._wait():
                return False
            heading_error = _wrap(station.heading_from(self.robot.x, self.robot.y) - self.robot.yaw)
            if abs(heading_error) < math.radians(1.2):
                break
            self._turn(heading_error)
            self._emit_telemetry("PATROL", "patrol_align")
        return True

    def _turn(self, error_rad: float) -> None:
        """One turn pulse, no longer than the error needs."""
        needed_ms = abs(error_rad) / self.motor.turn_rate * 1000.0
        pulse = int(max(30.0, min(140.0, needed_ms)))
        (self.motor.turn_left if error_rad > 0 else self.motor.turn_right)(pulse)

    def _drive_pulse_ms(self, distance_m: float) -> int:
        needed_ms = distance_m / self.motor.drive_speed * 1000.0
        return int(max(40.0, min(170.0, needed_ms)))

    # -- perception-action -------------------------------------------------
    def _observe_station(self) -> None:
        self.perception_action.reset_to_patrol()
        self._dropped_here = 0
        logged_here = 0

        for _ in range(self.station_frames):
            if not self._wait():
                return
            frame_started = time.perf_counter()

            projected = self.camera.detect()
            self.last_detections = projected
            detections = [p.detection for p in projected]
            self.frames += 1

            entry_state = self.perception_action.state
            state, emitted = self.perception_action.step(detections)
            # reset_to_patrol() clears last_signal on the frame that logs, so
            # report what actually triggered the emit rather than the reset value.
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
                    "%s -> %s  signal=%s  dets=%d", entry_state.value, state.value, signal, len(detections)
                )

            self._emit_telemetry(state.value, signal)

            if emitted is not None:
                if emitted.event_type == EventType.PARTIAL_TIMEOUT:
                    self.timeouts += 1
                else:
                    self.confirmations += 1
                    if self._corrected_this_target:
                        self.recoveries += 1
                self._corrected_this_target = False
                before = self.events_sent
                self._dispatch(emitted, detections)
                logged_here += self.events_sent - before
                # Keep observing rather than leaving the moment something is
                # logged: a patrolling robot dwells, and the repeat
                # observations are exactly what the Phase 4 importance filter
                # is for. Watch the `filtered` counter climb while `episodes`
                # does not.

            elapsed = time.perf_counter() - frame_started
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)

        logger.info(
            "Leaving %s: logged %d new episode(s), importance filter dropped %d repeat(s).",
            self.location_tag,
            logged_here,
            self._dropped_here,
        )

    def _dispatch(self, emitted, detections) -> None:
        """Scene graph -> importance filter -> event send.
        Mirrors nano/main_loop.py::_process_emitted_observation."""
        relationships, objects = self.scene_builder.build_scene_graph(detections)

        if emitted.target_object in objects:
            objects.remove(emitted.target_object)
        objects.insert(0, emitted.target_object)

        # The confirmed target is what the episode is about, so put its
        # relationship first: the caption then reads "A bottle is on the table.
        # A vase is on the table." instead of burying the subject.
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
            self._dropped_here += 1
            # Per-frame, this fires a couple of times a second for the whole
            # dwell and buries everything else. The count is reported once when
            # the robot leaves the station instead; the dashboard shows it live.
            logger.debug(
                "Observation dropped as redundant: %s %s @%s",
                event.event_type, event.objects, event.location_tag,
            )
            self._emit_telemetry(
                State.LOGGED.value, "filtered_redundant", extra={"filtered_event": event.model_dump(mode="json")}
            )
            return

        logger.info(
            "Observation: type=%s objects=%s rels=%s @%s",
            filtered.event_type, filtered.objects, filtered.relationships, filtered.location_tag,
        )
        if self.on_event(filtered):
            self.events_sent += 1

    # -- telemetry ---------------------------------------------------------
    def _emit_telemetry(self, state: str, signal: str, extra: Optional[dict[str, Any]] = None) -> None:
        payload: dict[str, Any] = {
            "source": "sim2d",
            "state": state,
            "signal": signal,
            "location_tag": self.location_tag,
            "lap": self.lap,
            "pose": self.robot.to_dict(),
            "detector": "sim2d pinhole projection (geometric ground truth)",
            "edge_margin": self.edge_margin,
            "detections": [
                {
                    **p.detection.to_dict(),
                    "color": p.color,
                    "distance": p.distance,
                    "clipped": p.clipped,
                }
                for p in self.last_detections
            ],
            "world": [o.to_dict() for o in self.world],
            "stations": [
                {"tag": s.tag, "x": s.x, "y": s.y} for s in self.stations
            ],
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
            "bottle_moves": self.bottle_moves,
        }
        if extra:
            payload.update(extra)
        self.on_telemetry(payload)

    # -- main loop ---------------------------------------------------------
    def _run(self) -> None:
        logger.info("Edge agent patrolling (2D lab).")
        try:
            while not self._stop.is_set():
                self.lap += 1
                if self.laps and self.lap > self.laps:
                    break
                for index, station in enumerate(self.stations):
                    if self._stop.is_set():
                        break
                    self.location_tag = station.tag
                    if not self._drive_to(station):
                        break
                    self._observe_station()

                    # Stage the two relocations between desk 1 and the shelf,
                    # i.e. while the robot is not looking at either endpoint.
                    if index == 1 and self.bottle_moves < 2 and self.lap in (1, 2):
                        self.move_bottle()
        except Exception:
            logger.exception("Edge agent crashed")
        finally:
            self.motor.cleanup()
            logger.info(
                "Edge agent stopped: frames=%d confirmations=%d partial_views=%d "
                "recoveries=%d timeouts=%d filtered=%d sent=%d",
                self.frames,
                self.confirmations,
                self.partial_views,
                self.recoveries,
                self.timeouts,
                self.filtered_out,
                self.events_sent,
            )


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
