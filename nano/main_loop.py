"""
Nano main loop.

Ties camera.py -> detector.py -> scene_graph.py -> perception_action.py ->
importance_scoring.py -> event_client.py together into the patrol loop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

import cv2

from nano.camera import Camera
from nano.detector import SSDMobileNetV2Detector, benchmark_fps, draw_detections
from nano.event_client import EventClient
from nano.importance_scoring import ImportanceScorer
from nano.motor_control import MotorController, get_controller
from nano.perception_action import PerceptionActionStateMachine, State
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
        laptop_url: str = "http://localhost:8000",
        location_tag: str = "lab_desk_3",
        edge_margin: float = 0.05,
        min_height: float = 0.15,
        pulse_ms: int = 150,
        confirm_frames: int = 3,
        timeout_sec: float = 7.0,
        force_mock_motor: bool = False,
        display: bool = False,
    ):
        self.location_tag = location_tag
        self.display = display

        logger.info("Initializing Nano Patrol Agent components...")
        self.camera = Camera(camera_type=camera_mode)
        self.detector = SSDMobileNetV2Detector(backend="auto")
        self.motor = MotorController(force_mock=force_mock_motor)
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

                # 3. If an observation was logged (confirmed or partial_timeout)
                if emitted is not None:
                    self._process_emitted_observation(emitted, detections)

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
            logger.info("Event filtered out by importance scoring (redundant observation).")
            return 0

        # 4. Transmit event to laptop fog server (Phase 5)
        logger.info(f"Dispatching event [{filtered_event.event_id[:8]}] to laptop...")
        success = self.event_client.send_event(filtered_event)
        return 1 if success else 0

    def cleanup(self) -> None:
        logger.info("Cleaning up resources...")
        self.camera.release()
        self.motor.cleanup()
        if self.display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Episodic Perception - Nano Edge Patrol")
    parser.add_argument("--mode", choices=["auto", "csi", "usb", "synthetic"], default="auto", help="Camera mode")
    parser.add_argument("--laptop-url", default="http://localhost:8000", help="Fog server URL")
    parser.add_argument("--location-tag", default="lab_desk_3", help="Current robot waypoint/location tag")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--mock-motors", action="store_true", help="Force software mock for motors")
    parser.add_argument("--display", action="store_true", help="Show live OpenCV preview window")
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
        laptop_url=args.laptop_url,
        location_tag=args.location_tag,
        force_mock_motor=args.mock_motors,
        display=args.display,
    )

    if args.simulate or args.mode == "synthetic":
        agent.run_simulation_scenario()
    else:
        agent.run(max_iterations=args.max_frames)


if __name__ == "__main__":
    main()
