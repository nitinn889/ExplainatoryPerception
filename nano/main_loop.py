"""
Nano main loop.

Ties camera.py -> detector.py -> scene_graph.py -> perception_action.py ->
importance_scoring.py -> event_client.py together into the patrol loop.
"""

from __future__ import annotations

import argparse
import base64
import logging
import sys
import time
from typing import Optional

import cv2
import requests

from nano.camera import Camera
from nano.detector import SSDMobileNetV2Detector, benchmark_fps, draw_detections
from nano.event_client import EventClient
from nano.importance_scoring import ImportanceScorer
from nano.motor_control import MotorController, get_controller
from nano.perception_action import CorrectionSignal, PerceptionActionStateMachine, State
from nano.scene_graph import SceneGraphBuilder
from shared.event_schema import BBox, Event, EventType

logger = logging.getLogger("nano.main_loop")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class NanoPatrolAgent:
    """
    Main orchestrator running on the Jetson Nano (edge).
    Coordinates camera capture, detection, perception-action repositioning,
    scene-graph construction, importance filtering, and upstream event dispatch.
    """

    def __init__(
        self,
        camera_mode: str = "auto",
        camera_index: int = 0,
        laptop_url: str = "http://localhost:8000",
        location_tag: str = "lab_desk_3",
        edge_margin: float = 0.05,
        min_height: float = 0.15,
        pulse_ms: int = 150,
        confirm_frames: int = 3,
        timeout_sec: float = 7.0,
        force_mock_motor: bool = False,
        motor_port: Optional[str] = None,
        camera_at_rear: bool = False,
        display: bool = False,
        telemetry: bool = True,
    ):
        self.location_tag = location_tag
        self.display = display
        self.edge_margin = edge_margin

        # Per-frame state for the demo dashboard's camera view (POST /telemetry),
        # the same payload the Webots controller sends. Best-effort only.
        self._telemetry_url = f"{laptop_url.rstrip('/')}/telemetry" if telemetry else None
        self._telemetry_session = requests.Session()
        self._telemetry_failures = 0
        self.stats = {
            "frames": 0, "confirmations": 0, "partial_views": 0, "recoveries": 0,
            "timeouts": 0, "events_sent": 0, "filtered_out": 0,
        }
        self._corrected_this_target = False

        logger.info("Initializing Nano Patrol Agent components...")
        self.camera = Camera(camera_type=camera_mode, usb_device_index=camera_index)
        self.detector = SSDMobileNetV2Detector(backend="auto")
        self.motor = MotorController(force_mock=force_mock_motor, serial_port=motor_port, camera_at_rear=camera_at_rear)
        self.perception_action = PerceptionActionStateMachine(
            edge_margin=edge_margin,
            min_height=min_height,
            pulse_ms=pulse_ms,
            confirm_frames=confirm_frames,
            timeout_sec=timeout_sec,
            motor=self.motor,
        )
        self.scene_builder = SceneGraphBuilder()
        self.importance_scorer = ImportanceScorer()
        self.event_client = EventClient(server_url=laptop_url)

        # Connection check to companion laptop
        if self.event_client.check_connection():
            logger.info(f"Connected to fog event server at {laptop_url}.")
        else:
            logger.warning(
                f"Fog event server at {laptop_url} not currently reachable. "
                "Events will be queued/retried when generated."
            )

    def run_simulation_scenario(self, steps: int = 15) -> int:
        """
        Runs a synthetic simulation scenario to demonstrate:
        1. Object detected clipped on left edge (PARTIAL).
        2. Robot issues turn_left pulses (ADJUST).
        3. Object stabilizes in frame (CONFIRM for 3 frames).
        4. Episode emitted -> Scene graph generated -> Importance scored -> Sent to Laptop!
        """
        logger.info("--- Starting Synthetic Perception-Action Demonstration ---")
        self.camera.camera_type = "synthetic"
        self.camera.is_synthetic = True

        # Simulate object starting at xmin = 0.02 (clipped on left) and shifting right on each turn_left
        events_sent = 0
        obj_x = 0.02
        obj_w = 0.25
        obj_h = 0.35

        for i in range(1, steps + 1):
            # Configure synthetic camera frame
            self.camera.set_synthetic_objects([
                {"name": "bottle", "bbox": (obj_x, 0.25, obj_x + obj_w, 0.25 + obj_h), "color": (0, 200, 255)},
                {"name": "table", "bbox": (0.10, 0.55, 0.90, 0.90), "color": (120, 80, 50)},
            ])

            ret, frame = self.camera.read()
            if not ret:
                break

            # Detection
            synth = self.camera.get_synthetic_objects() if self.camera.is_synthetic else None
            detections = self.detector.detect(frame, synthetic_objects=synth)

            # Perception-action step
            prev_state = self.perception_action.state
            state, emitted = self.perception_action.step(detections)

            action_note = ""
            if self.motor.action_history:
                last_act = self.motor.action_history[-1]
                action_note = f"-> Motor Action: {last_act[0]} ({last_act[1]}ms)"

            logger.info(
                f"[Tick {i:02d}] State: {prev_state.value} -> {state.value} "
                f"| Target xmin={obj_x:.2f} {action_note}"
            )

            # If motor turned left, the object moves relatively towards the center
            if self.motor.action_history and self.motor.action_history[-1][0] == "turn_left":
                obj_x = min(0.35, obj_x + 0.08)

            # Process emitted event when CONFIRM succeeds or timeout occurs
            if emitted is not None:
                events_sent += self._process_emitted_observation(emitted, detections)

        logger.info(f"--- Demonstration complete. Total events transmitted: {events_sent} ---")
        return events_sent

    def run(self, max_iterations: Optional[int] = None) -> None:
        """Main real-time loop."""
        logger.info("Entering Nano patrol perception-action loop...")
        iteration = 0

        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame from camera.")
                    time.sleep(0.1)
                    continue

                # 1. Object Detection (SSD-MobileNet-V2)
                synth = self.camera.get_synthetic_objects() if self.camera.is_synthetic else None
                detections = self.detector.detect(frame, synthetic_objects=synth)

                # 2. Perception-Action State Machine Tick
                state, emitted = self.perception_action.step(detections)
                self.stats["frames"] += 1
                # step() resets last_signal on the frame that logs, so report
                # what actually triggered the emit.
                signal = (
                    CorrectionSignal.FULLY_VISIBLE.value
                    if emitted is not None and emitted.event_type == EventType.CONFIRMED
                    else self.perception_action.last_signal.value
                )
                if state == State.ADJUST:
                    if not self._corrected_this_target:
                        self.stats["partial_views"] += 1
                    self._corrected_this_target = True

                # 3. If an observation was logged (confirmed or partial_timeout)
                if emitted is not None:
                    if emitted.event_type == EventType.PARTIAL_TIMEOUT:
                        self.stats["timeouts"] += 1
                    else:
                        self.stats["confirmations"] += 1
                        if self._corrected_this_target:
                            self.stats["recoveries"] += 1
                    self._corrected_this_target = False
                    self._process_emitted_observation(emitted, detections)

                self._publish_telemetry(state, signal, detections, frame)

                # 4. Optional local preview display
                if self.display:
                    annotated = draw_detections(frame, detections)
                    cv2.imshow("Nano Edge Perception Preview", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            logger.info("Patrol loop stopped by user interrupt.")
        finally:
            self.cleanup()

    def _process_emitted_observation(self, emitted, detections) -> int:
        """Builds scene graph, applies importance scoring, and dispatches event."""
        # 1. Scene graph construction
        relationships, objects = self.scene_builder.build_scene_graph(detections)

        # Ensure primary subject is first
        if emitted.target_object not in objects:
            objects.insert(0, emitted.target_object)
        else:
            objects.remove(emitted.target_object)
            objects.insert(0, emitted.target_object)

        # 2. Package shared event
        event = Event(
            objects=objects,
            relationships=relationships,
            confidence=emitted.confidence,
            event_type=emitted.event_type,
            bbox=emitted.bbox,
            location_tag=self.location_tag,
        )

        logger.info(
            f"Observation generated: type={event.event_type} "
            f"objects={event.objects} rels={event.relationships}"
        )

        # 3. Importance scoring filter (Phase 4)
        filtered_event = self.importance_scorer.filter_event(event)
        if filtered_event is None:
            self.stats["filtered_out"] += 1
            logger.info("Event filtered out by importance scoring (redundant observation).")
            return 0

        # 4. Transmit event to laptop fog server (Phase 5)
        logger.info(f"Dispatching event [{filtered_event.event_id[:8]}] to laptop...")
        success = self.event_client.send_event(filtered_event)
        if success:
            self.stats["events_sent"] += 1
        return 1 if success else 0

    def _publish_telemetry(self, state: State, signal: str, detections, frame) -> None:
        """Best-effort POST of this frame's state and JPEG to the dashboard.
        Never raises: if the dashboard is not running, the robot keeps going."""
        if not self._telemetry_url or self._telemetry_failures > 20:
            return
        payload = {
            "source": f"jetson ({self.camera.camera_type} camera)",
            "state": state.value,
            "signal": signal,
            "location_tag": self.location_tag,
            "detector": f"ssd-mobilenet-v2 ({self.detector.backend})",
            "detections": [d.to_dict() for d in detections],
            "edge_margin": self.edge_margin,
            "stats": dict(self.stats),
            "last_motor_action": (
                list(self.motor.action_history[-1]) if self.motor.action_history else None
            ),
        }
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok:
            payload["frame_jpeg_b64"] = base64.b64encode(buffer.tobytes()).decode("ascii")
        try:
            self._telemetry_session.post(self._telemetry_url, json=payload, timeout=0.6)
            self._telemetry_failures = 0
        except requests.RequestException:
            self._telemetry_failures += 1
            if self._telemetry_failures == 1:
                logger.info("Dashboard telemetry not reachable at %s.", self._telemetry_url)

    def cleanup(self) -> None:
        logger.info("Cleaning up resources...")
        self.camera.release()
        self.motor.cleanup()
        if self.display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Episodic Perception - Nano Edge Patrol")
    parser.add_argument("--mode", choices=["auto", "csi", "usb", "oak", "synthetic"], default="auto", help="Camera mode")
    parser.add_argument("--camera-index", type=int, default=0, help="V4L2 index for --mode usb (python -m nano.station_test --list-cameras)")
    parser.add_argument("--laptop-url", default="http://localhost:8000", help="Fog server URL")
    parser.add_argument("--location-tag", default="lab_desk_3", help="Current robot waypoint/location tag")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--mock-motors", action="store_true", help="Force software mock for motors")
    parser.add_argument("--motor-port", default=None,
                        help="Serial port of the Arduino motor bridge, e.g. /dev/ttyUSB0 (default: Jetson GPIO)")
    parser.add_argument("--camera-rear", action="store_true",
                        help="Camera faces backward over the battery box (swaps forward/reverse)")
    parser.add_argument("--display", action="store_true", help="Show live OpenCV preview window")
    parser.add_argument("--no-telemetry", dest="telemetry", action="store_false",
                        help="Do not stream frames/state to the dashboard's /telemetry")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark SSD-MobileNet-V2 FPS and exit")
    parser.add_argument("--simulate", action="store_true", help="Run the automated perception-action demo scenario")
    args = parser.parse_args()

    if args.benchmark:
        detector = SSDMobileNetV2Detector(backend="auto")
        fps = benchmark_fps(detector, num_frames=50)
        print(f"Measured detector inference speed: {fps:.2f} FPS")
        return

    agent = NanoPatrolAgent(
        camera_mode=args.mode,
        camera_index=args.camera_index,
        laptop_url=args.laptop_url,
        location_tag=args.location_tag,
        force_mock_motor=args.mock_motors,
        motor_port=args.motor_port,
        camera_at_rear=args.camera_rear,
        display=args.display,
        telemetry=args.telemetry,
    )

    if args.simulate or args.mode == "synthetic":
        agent.run_simulation_scenario()
    else:
        agent.run(max_iterations=args.max_frames)


if __name__ == "__main__":
    main()
