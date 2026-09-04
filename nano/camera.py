"""
Camera capture for the Jetson Nano (Phase 0/1).

Responsible for opening the CSI (IMX219 via nvarguscamerasrc) or USB (v4l2)
camera through OpenCV and yielding frames to detector.py.

Includes synthetic/simulation mode for running tests and pipeline validation
on host machines without Jetson CSI hardware or webcams.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("nano.camera")


def gstreamer_pipeline(
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 640,
    display_height: int = 360,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """Returns a GStreamer pipeline string for CSI camera capture via libargus."""
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


class Camera:
    """
    Unified Camera capture supporting CSI (IMX219), USB (V4L2), and Synthetic frames.
    """

    def __init__(
        self,
        camera_type: str = "auto",  # 'csi', 'usb', 'synthetic', or 'auto'
        usb_device_index: int = 0,
        width: int = 640,
        height: int = 480,
        framerate: int = 30,
        flip_method: int = 0,
    ):
        self.camera_type = camera_type.lower()
        self.usb_device_index = usb_device_index
        self.width = width
        self.height = height
        self.framerate = framerate
        self.flip_method = flip_method

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_synthetic = False
        self._frame_count = 0
        self._synthetic_objects = []

        self._init_camera()

    def _init_camera(self) -> None:
        if self.camera_type == "synthetic":
            self._setup_synthetic()
            return

        if self.camera_type in ("csi", "auto"):
            # Try CSI first if requested or auto
            pipeline = gstreamer_pipeline(
                display_width=self.width,
                display_height=self.height,
                framerate=self.framerate,
                flip_method=self.flip_method,
            )
            try:
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    self.cap = cap
                    self.camera_type = "csi"
                    logger.info("Initialized CSI camera via GStreamer nvarguscamerasrc.")
                    return
            except Exception as e:
                logger.debug(f"CSI init failed: {e}")

        if self.camera_type in ("usb", "auto"):
            # Try USB webcam
            try:
                cap = cv2.VideoCapture(self.usb_device_index)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.framerate)
                    self.cap = cap
                    self.camera_type = "usb"
                    logger.info(f"Initialized USB camera on index {self.usb_device_index}.")
                    return
            except Exception as e:
                logger.debug(f"USB init failed: {e}")

        # Fallback to synthetic
        logger.info("No physical CSI/USB camera detected. Falling back to synthetic frame generator.")
        self._setup_synthetic()

    def _setup_synthetic(self) -> None:
        self.camera_type = "synthetic"
        self.is_synthetic = True
        self.cap = None

    def set_synthetic_objects(self, objects: list[dict]) -> None:
        """
        Configure synthetic objects to render.
        Each dict: {'name': str, 'bbox': (xmin, ymin, xmax, ymax), 'color': (B, G, R)}
        """
        self._synthetic_objects = objects

    def get_synthetic_objects(self) -> list[dict]:
        """Returns currently active synthetic objects."""
        return self._synthetic_objects

    def read(self) -> Tuple[bool, np.ndarray]:
        """Reads a frame. Returns (ret, frame) where frame is a BGR uint8 numpy array."""
        if self.is_synthetic or self.cap is None:
            self._frame_count += 1
            frame = np.full((self.height, self.width, 3), 40, dtype=np.uint8)

            # Draw a subtle grid background
            for x in range(0, self.width, 40):
                cv2.line(frame, (x, 0), (x, self.height), (60, 60, 60), 1)
            for y in range(0, self.height, 40):
                cv2.line(frame, (0, y), (self.width, y), (60, 60, 60), 1)

            # Draw configured synthetic objects
            for obj in self._synthetic_objects:
                xmin, ymin, xmax, ymax = obj.get("bbox", (0.2, 0.2, 0.4, 0.4))
                x1 = int(xmin * self.width)
                y1 = int(ymin * self.height)
                x2 = int(xmax * self.width)
                y2 = int(ymax * self.height)
                color = obj.get("color", (0, 255, 128))
                name = obj.get("name", "object")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, name, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            time.sleep(1.0 / max(self.framerate, 1))
            return True, frame

        ret, frame = self.cap.read()
        return ret, frame

    def is_opened(self) -> bool:
        if self.is_synthetic:
            return True
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
