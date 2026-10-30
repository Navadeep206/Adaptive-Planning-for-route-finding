"""Local YOLO object detection module for MacBook camera perception.

Provides structured, standalone object detection using lightweight pretrained YOLO models
(e.g., yolo11n.pt / yolov8n.pt). Runs 100% locally with zero paid APIs and safe fallback handling.
Decoupled from A* path planning and grid simulation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Structured representation of a single detected object in an image frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates

    @property
    def x1(self) -> int:
        """Left bounding coordinate."""
        return self.bbox[0]

    @property
    def y1(self) -> int:
        """Top bounding coordinate."""
        return self.bbox[1]

    @property
    def x2(self) -> int:
        """Right bounding coordinate."""
        return self.bbox[2]

    @property
    def y2(self) -> int:
        """Bottom bounding coordinate."""
        return self.bbox[3]

    @property
    def width(self) -> int:
        """Bounding box pixel width."""
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        """Bounding box pixel height."""
        return max(0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[int, int]:
        """Center coordinate (cx, cy) of bounding box."""
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to serializable dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
        }


class YOLOObjectDetector:
    """Local object detector powered by Ultralytics YOLO.

    Accepts raw video frames, runs local neural network inference,
    and returns strongly-typed Detection records without coupling to path planning.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        device: Optional[str] = None,
        auto_load: bool = True
    ):
        """Initialize the object detector.

        Args:
            model_name: Pretrained YOLO model file or identifier (default: yolo11n.pt).
            conf_threshold: Minimum detection confidence threshold (0.0 to 1.0).
            imgsz: Inference input image size (default: 640).
            device: Optional inference device (e.g., 'cpu', 'mps', 'cuda'). None selects best default.
            auto_load: Automatically load the model during instantiation if True.
        """
        self.model_name: str = model_name
        self.conf_threshold: float = conf_threshold
        self.imgsz: int = imgsz
        self.device: Optional[str] = device
        self._model: Optional[Any] = None
        self._is_loaded: bool = False
        self.class_names: Dict[int, str] = {}
        self.last_inference_time_ms: float = 0.0

        if auto_load:
            self.load_model()

    def load_model(self) -> bool:
        """Load the YOLO model locally.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        if self._is_loaded and self._model is not None:
            return True

        try:
            logger.info("Initializing Ultralytics YOLO detector with model '%s'...", self.model_name)
            from ultralytics import YOLO  # Lazy import to avoid importing torch in unneeded contexts

            self._model = YOLO(self.model_name)
            if hasattr(self._model, "names") and isinstance(self._model.names, dict):
                self.class_names = dict(self._model.names)
            elif hasattr(self._model, "names") and isinstance(self._model.names, list):
                self.class_names = {i: name for i, name in enumerate(self._model.names)}
            else:
                self.class_names = {}

            self._is_loaded = True
            logger.info(
                "YOLO model '%s' loaded successfully (%d classes recognized).",
                self.model_name,
                len(self.class_names)
            )
            return True

        except ImportError as exc:
            self._is_loaded = False
            self._model = None
            logger.warning(
                "Ultralytics is not installed in the environment (%s). Please install with 'pip install ultralytics'.",
                exc
            )
            return False

        except Exception as exc:
            self._is_loaded = False
            self._model = None
            logger.error("Failed to load YOLO model '%s': %s", self.model_name, exc, exc_info=True)
            return False

    def is_loaded(self) -> bool:
        """Check whether the model is loaded and ready for inference."""
        return self._is_loaded and self._model is not None

    @property
    def status_text(self) -> str:
        """Return human-readable status of the model."""
        if self.is_loaded():
            return f"Ready ({self.model_name})"
        return "Not Loaded"

    def detect(self, frame: Optional[np.ndarray]) -> Tuple[List[Detection], float]:
        """Perform object detection on an image frame.

        Args:
            frame: Input image as a NumPy BGR or RGB array (H, W, C).

        Returns:
            Tuple[List[Detection], float]:
                - List of Detection objects found in the frame.
                - Inference execution time in milliseconds.
        """
        # 1. Validation checks
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return [], 0.0

        if not self.is_loaded():
            logger.debug("Object detector called while model is not loaded.")
            return [], 0.0

        # 2. Run inference locally
        start_time = time.perf_counter()
        try:
            kwargs = {
                "conf": self.conf_threshold,
                "imgsz": self.imgsz,
                "verbose": False
            }
            if self.device is not None:
                kwargs["device"] = self.device

            results = self._model(frame, **kwargs)
            self.last_inference_time_ms = (time.perf_counter() - start_time) * 1000.0

            if not results:
                return [], self.last_inference_time_ms

            detections: List[Detection] = []
            boxes = getattr(results[0], "boxes", None)
            if boxes is None or len(boxes) == 0:
                return [], self.last_inference_time_ms

            # 3. Parse boxes and build structured Detection objects
            for box in boxes:
                # Extract coordinates
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                # Extract confidence and class ID
                conf = float(box.conf[0].detach().cpu().numpy())
                cls_id = int(box.cls[0].detach().cpu().numpy())
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2)
                    )
                )

            if len(detections) == 0:
                logger.debug(
                    "YOLO inference found 0 detections meeting conf_threshold=%.2f in %.1f ms.",
                    self.conf_threshold,
                    self.last_inference_time_ms
                )
            else:
                logger.debug(
                    "YOLO inference found %d detections in %.1f ms.",
                    len(detections),
                    self.last_inference_time_ms
                )

            return detections, self.last_inference_time_ms

        except Exception as exc:
            logger.error("Exception during YOLO inference: %s", exc, exc_info=True)
            self.last_inference_time_ms = (time.perf_counter() - start_time) * 1000.0
            return [], self.last_inference_time_ms

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        inference_time_ms: Optional[float] = None,
        mapping_enabled: Optional[bool] = None
    ) -> np.ndarray:
        """Render detection bounding boxes, labels, and telemetry onto a frame copy.

        Args:
            frame: Input image array.
            detections: List of Detection instances to render.
            inference_time_ms: Optional inference latency in ms to display.
            mapping_enabled: Optional flag indicating whether perception grid mapping is ON or OFF.

        Returns:
            np.ndarray: Annotated frame copy.
        """
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Resolution-adaptive scaling for text and bounding boxes
        min_dim = min(w, h)
        font_scale = max(0.45, min(0.85, (min_dim / 720.0) * 0.55))
        thickness = max(1, int(round(font_scale * 2)))
        box_thickness = max(2, int(round(font_scale * 3)))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Clamp coordinates to frame boundaries and cast to int
            x1 = int(round(max(0, min(x1, w - 1))))
            y1 = int(round(max(0, min(y1, h - 1))))
            x2 = int(round(max(0, min(x2, w - 1))))
            y2 = int(round(max(0, min(y2, h - 1))))

            # Double-contour bounding box for maximum contrast against any lighting
            # Dark outer contour (1px wider) + Vibrant inner Cyan/Green box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), box_thickness + 2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 240, 160), box_thickness)

            # Label text with percentage
            conf_pct = int(round(det.confidence * 100))
            label = f"{det.class_name} {conf_pct}%"

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Ceiling-safe badge positioning: avoid clipping off-screen when box touches top
            if y1 - text_h - 8 >= 0:
                badge_y1 = y1 - text_h - 8
                badge_y2 = y1
                text_y = y1 - 4
            else:
                badge_y1 = y1
                badge_y2 = y1 + text_h + 8
                text_y = y1 + text_h + 4

            # Dark badge with vibrant border and clean white text
            cv2.rectangle(canvas, (x1, badge_y1), (x1 + text_w + 8, badge_y2), (15, 15, 15), -1)
            cv2.rectangle(canvas, (x1, badge_y1), (x1 + text_w + 8, badge_y2), (0, 240, 160), 1)
            cv2.putText(
                canvas,
                label,
                (x1 + 4, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        # Draw HUD summary overlay at top-left
        hud_w = max(330, int(w * 0.32))
        hud_h = 76
        cv2.rectangle(canvas, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (10, 10), (10 + hud_w, 10 + hud_h), (0, 200, 255), 1)

        cv2.putText(
            canvas,
            f"YOLO Detector: {self.model_name}",
            (18, 30),
            font,
            0.5,
            (0, 200, 255),
            1,
            cv2.LINE_AA
        )

        inf_str = f"{inference_time_ms:.1f} ms" if inference_time_ms is not None else "N/A"
        if len(detections) > 0:
            det_text = f"Detections: {len(detections)}  |  Latency: {inf_str}"
            det_color = (0, 240, 160)
        else:
            det_text = f"Detections: 0 (No objects)  |  Latency: {inf_str}"
            det_color = (180, 180, 180)

        cv2.putText(
            canvas,
            det_text,
            (18, 52),
            font,
            0.45,
            det_color,
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"Confidence Thresh: {self.conf_threshold:.2f}",
            (18, 70),
            font,
            0.4,
            (160, 160, 160),
            1,
            cv2.LINE_AA
        )

        if mapping_enabled is not None:
            map_status = "ON" if mapping_enabled else "OFF"
            map_color = (0, 255, 120) if mapping_enabled else (140, 140, 140)
            cv2.putText(
                canvas,
                f"Perception Mapping: {map_status} (Toggle: 'M')",
                (18, 88),
                font,
                0.4,
                map_color,
                1,
                cv2.LINE_AA
            )

        return canvas
