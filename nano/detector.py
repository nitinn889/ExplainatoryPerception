"""
SSD-MobileNet-V2 inference wrapper (Phase 1).
No YOLO — explicitly excluded by course requirement.

Loads a pretrained SSD-MobileNet-V2 detector (via Jetson TensorRT detectNet if available,
or OpenCV DNN / synthetic fallback), runs inference on frames from camera.py.
Outputs a list of {class, confidence, bbox (xmin, ymin, xmax, ymax normalized 0-1)} per frame.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("nano.detector")

# COCO 90 classes often used with SSD-MobileNet
COCO_CLASSES = [
    "unlabeled", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
    "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror",
    "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


@dataclass
class DetectionResult:
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) in 0-1 normalized coords

    @property
    def xmin(self) -> float:
        return self.bbox[0]

    @property
    def ymin(self) -> float:
        return self.bbox[1]

    @property
    def xmax(self) -> float:
        return self.bbox[2]

    @property
    def ymax(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": {
                "xmin": round(float(self.xmin), 4),
                "ymin": round(float(self.ymin), 4),
                "xmax": round(float(self.xmax), 4),
                "ymax": round(float(self.ymax), 4),
            },
        }


class SSDMobileNetV2Detector:
    """
    SSD-MobileNet-V2 detector supporting:
    1. Jetson TensorRT via jetson.inference.detectNet (on physical Jetson Nano)
    2. OpenCV DNN (if Caffe / TF / ONNX weights are supplied)
    3. Heuristic / Synthetic fallback (for laptop development / tests without Nano hardware)
    """

    def __init__(
        self,
        backend: str = "auto",  # 'jetson', 'opencv', 'mock', or 'auto'
        confidence_threshold: float = 0.5,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.backend = backend.lower()
        self.net = None
        self._jetson_net = None
        self._synthetic_detections: Optional[List[DetectionResult]] = None

        self._init_backend(model_path, config_path)

    def _init_backend(self, model_path: Optional[str], config_path: Optional[str]) -> None:
        if self.backend in ("jetson", "auto"):
            try:
                import jetson.inference
                import jetson.utils
                self._jetson_net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=self.confidence_threshold)
                self.backend = "jetson"
                logger.info("Initialized Jetson TensorRT detectNet (ssd-mobilenet-v2).")
                return
            except (ImportError, Exception) as e:
                logger.debug(f"jetson.inference not available: {e}")

        if self.backend in ("opencv", "auto") and model_path:
            try:
                if config_path:
                    self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
                else:
                    self.net = cv2.dnn.readNet(model_path)
                self.backend = "opencv"
                logger.info(f"Initialized OpenCV DNN SSD-MobileNet with {model_path}.")
                return
            except Exception as e:
                logger.debug(f"OpenCV DNN init failed: {e}")

        # Fallback to mock / synthetic detector
        self.backend = "mock"
        logger.info("Running in MOCK/SYNTHETIC detector mode (SSD-MobileNet-V2 interface).")

    def set_mock_detections(self, detections: List[DetectionResult]) -> None:
        """Inject specific detections (for unit testing and controlled validation)."""
        self._synthetic_detections = detections

    def detect(
        self,
        frame: np.ndarray,
        synthetic_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DetectionResult]:
        """
        Run inference on an OpenCV BGR image frame.
        Returns a list of DetectionResult objects with normalized 0-1 bboxes.
        """
        if self._synthetic_detections is not None:
            return self._synthetic_detections

        if synthetic_objects is not None and self.backend == "mock":
            return [
                DetectionResult(
                    class_name=obj.get("name", "bottle"),
                    confidence=float(obj.get("confidence", 0.92)),
                    bbox=tuple(obj.get("bbox", (0.2, 0.2, 0.5, 0.5))),
                )
                for obj in synthetic_objects
            ]

        h, w = frame.shape[:2]

        if self.backend == "jetson" and self._jetson_net is not None:
            import jetson.utils
            # Convert OpenCV BGR to CUDA RGBA
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            cuda_mem = jetson.utils.cudaFromNumpy(rgba)
            jetson_detections = self._jetson_net.Detect(cuda_mem)

            results: List[DetectionResult] = []
            for d in jetson_detections:
                results.append(
                    DetectionResult(
                        class_name=self._jetson_net.GetClassDesc(d.ClassID),
                        confidence=float(d.Confidence),
                        bbox=(
                            float(np.clip(d.Left / w, 0.0, 1.0)),
                            float(np.clip(d.Top / h, 0.0, 1.0)),
                            float(np.clip(d.Right / w, 0.0, 1.0)),
                            float(np.clip(d.Bottom / h, 0.0, 1.0)),
                        ),
                    )
                )
            return results

        elif self.backend == "opencv" and self.net is not None:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                0.007843,
                (300, 300),
                127.5,
            )
            self.net.setInput(blob)
            output = self.net.forward()

            results = []
            for i in range(output.shape[2]):
                confidence = float(output[0, 0, i, 2])
                if confidence > self.confidence_threshold:
                    idx = int(output[0, 0, i, 1])
                    class_name = COCO_CLASSES[idx] if idx < len(COCO_CLASSES) else f"class_{idx}"
                    xmin = float(np.clip(output[0, 0, i, 3], 0.0, 1.0))
                    ymin = float(np.clip(output[0, 0, i, 4], 0.0, 1.0))
                    xmax = float(np.clip(output[0, 0, i, 5], 0.0, 1.0))
                    ymax = float(np.clip(output[0, 0, i, 6], 0.0, 1.0))
                    results.append(DetectionResult(class_name, confidence, (xmin, ymin, xmax, ymax)))
            return results

        # In mock mode: detect brightly colored synthetic rectangles in the frame if any exist
        return self._detect_mock_heuristics(frame)

    def _detect_mock_heuristics(self, frame: np.ndarray) -> List[DetectionResult]:
        """Simple color/contour detector for synthetic frames generated in Camera."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Threshold out dark background
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results: List[DetectionResult] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Map to normalized
                xmin = float(np.clip(x / w, 0.0, 1.0))
                ymin = float(np.clip(y / h, 0.0, 1.0))
                xmax = float(np.clip((x + cw) / w, 0.0, 1.0))
                ymax = float(np.clip((y + ch) / h, 0.0, 1.0))
                results.append(
                    DetectionResult(
                        class_name="bottle",
                        confidence=0.88,
                        bbox=(xmin, ymin, xmax, ymax),
                    )
                )

        return results


def draw_detections(frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    """Utility to draw bounding boxes and labels on an OpenCV image."""
    annotated = frame.copy()
    h, w = frame.shape[:2]
    for d in detections:
        x1 = int(d.xmin * w)
        y1 = int(d.ymin * h)
        x2 = int(d.xmax * w)
        y2 = int(d.ymax * h)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.class_name}: {d.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return annotated


def benchmark_fps(detector: SSDMobileNetV2Detector, num_frames: int = 50, frame_size: Tuple[int, int] = (640, 480)) -> float:
    """Benchmark detector inference speed in Frames Per Second (FPS)."""
    dummy_frame = np.random.randint(0, 255, (frame_size[1], frame_size[0], 3), dtype=np.uint8)
    # Warmup
    for _ in range(5):
        detector.detect(dummy_frame)

    t0 = time.perf_counter()
    for _ in range(num_frames):
        detector.detect(dummy_frame)
    elapsed = time.perf_counter() - t0

    fps = num_frames / max(elapsed, 1e-6)
    logger.info(f"Detector benchmark: {fps:.2f} FPS across {num_frames} frames.")
    return fps
