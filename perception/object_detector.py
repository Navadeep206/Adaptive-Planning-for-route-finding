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
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        max_det: int = 50,
        device: Optional[str] = None,
        preprocess_mode: str = "none",
        gamma_value: float = 1.5,
        clahe_clip_limit: float = 2.0,
        auto_load: bool = True
    ):
        """Initialize the object detector.

        Args:
            model_name: Pretrained YOLO model file or identifier (default: yolo11n.pt).
            conf_threshold: Minimum detection confidence threshold (default: 0.25).
            iou_threshold: Non-Maximum Suppression (NMS) IoU overlap threshold (default: 0.45).
            imgsz: Inference input image size: 640 or 800 (default: 640).
            max_det: Maximum detections per frame for real-time robotics scenes (default: 50).
            device: Optional inference device (e.g., 'cpu', 'mps', 'cuda'). None selects best default.
            preprocess_mode: Preprocessing enhancement: 'none', 'clahe', or 'gamma' (default: 'none').
            gamma_value: Exponent for gamma correction brightening (default: 1.5).
            clahe_clip_limit: Contrast clip limit for CLAHE equalization (default: 2.0).
            auto_load: Automatically load the model during instantiation if True.
        """
        self.model_name: str = model_name
        self.conf_threshold: float = conf_threshold
        self.iou_threshold: float = iou_threshold
        self.imgsz: int = imgsz
        self.max_det: int = max_det
        self.device: Optional[str] = device
        self.preprocess_mode: str = preprocess_mode.lower()
        self.gamma_value: float = gamma_value
        self.clahe_clip_limit: float = clahe_clip_limit
        self._model: Optional[Any] = None
        self._is_loaded: bool = False
        self.class_names: Dict[int, str] = {}
        self.last_inference_time_ms: float = 0.0
        self.last_preprocess_time_ms: float = 0.0

        # Precompute gamma lookup table for zero-allocation real-time correction
        self._gamma_lut = self._build_gamma_lut(self.gamma_value)

        if auto_load:
            self.load_model()

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        """Precompute 256-element integer lookup table for fast gamma correction."""
        if gamma <= 0.0:
            gamma = 1.0
        inv_gamma = 1.0 / gamma
        return np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)]).astype(np.uint8)

    def set_preprocessing(self, mode: str, gamma: Optional[float] = None, clip_limit: Optional[float] = None) -> None:
        """Update preprocessing mode and parameters dynamically at runtime."""
        self.preprocess_mode = mode.lower()
        if gamma is not None:
            self.gamma_value = gamma
            self._gamma_lut = self._build_gamma_lut(self.gamma_value)
        if clip_limit is not None:
            self.clahe_clip_limit = clip_limit
        logger.info("Detector preprocessing set to: '%s' (gamma=%.2f, clahe_clip=%.2f)",
                    self.preprocess_mode, self.gamma_value, self.clahe_clip_limit)

    def set_thresholds(
        self,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_det: Optional[int] = None
    ) -> None:
        """Update inference hyperparameters dynamically without reloading weights."""
        if conf is not None:
            self.conf_threshold = max(0.01, min(0.99, conf))
        if iou is not None:
            self.iou_threshold = max(0.05, min(0.95, iou))
        if imgsz is not None:
            self.imgsz = imgsz
        if max_det is not None:
            self.max_det = max(1, max_det)

    def apply_preprocessing(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply configured preprocessing (CLAHE or Gamma Correction) to frame.

        Returns:
            Tuple[np.ndarray, float]: (processed_frame, execution_time_ms)
        """
        if frame is None or frame.size == 0 or self.preprocess_mode in ("none", "", "original"):
            return frame, 0.0

        t0 = time.perf_counter()
        if self.preprocess_mode == "clahe":
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            out = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        elif self.preprocess_mode == "gamma":
            out = cv2.LUT(frame, self._gamma_lut)
        else:
            out = frame

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.last_preprocess_time_ms = elapsed_ms
        return out, elapsed_ms

    def load_model(self, fallback_to_pretrained: bool = True) -> bool:
        """Load the YOLO model locally with fallback protection.

        Args:
            fallback_to_pretrained: If True and a custom model fails to load,
                                   automatically falls back to 'yolo11n.pt'.

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
            model_tag = "CUSTOM OBSTACLE" if self.is_custom_model else "PRETRAINED COCO"
            logger.info(
                "YOLO model '%s' [%s] loaded successfully (%d classes recognized).",
                self.model_name,
                model_tag,
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
            if fallback_to_pretrained and self.model_name != "yolo11n.pt":
                logger.warning(
                    "Failed to load custom model '%s': %s. Falling back to default pretrained 'yolo11n.pt'.",
                    self.model_name, exc
                )
                try:
                    from ultralytics import YOLO
                    self.model_name = "yolo11n.pt"
                    self._model = YOLO("yolo11n.pt")
                    if hasattr(self._model, "names") and isinstance(self._model.names, dict):
                        self.class_names = dict(self._model.names)
                    elif hasattr(self._model, "names") and isinstance(self._model.names, list):
                        self.class_names = {i: name for i, name in enumerate(self._model.names)}
                    self._is_loaded = True
                    logger.info("Successfully loaded fallback pretrained model 'yolo11n.pt' (%d classes).", len(self.class_names))
                    return True
                except Exception as fb_exc:
                    logger.error("Fallback to 'yolo11n.pt' also failed: %s", fb_exc)

            self._is_loaded = False
            self._model = None
            logger.error("Failed to load YOLO model '%s': %s", self.model_name, exc, exc_info=True)
            return False

    def switch_model(self, new_model_path: str, fallback_to_pretrained: bool = True) -> bool:
        """Switch active YOLO model dynamically at runtime.

        Args:
            new_model_path: Path or identifier of new model to load.
            fallback_to_pretrained: If True, reverts to previous or default model on failure.

        Returns:
            bool: True if switch succeeded, False otherwise.
        """
        logger.info("Switching YOLO model from '%s' to '%s'...", self.model_name, new_model_path)
        old_model = self._model
        old_name = self.model_name
        old_classes = self.class_names

        try:
            from ultralytics import YOLO
            new_model = YOLO(new_model_path)
            new_names = {}
            if hasattr(new_model, "names") and isinstance(new_model.names, dict):
                new_names = dict(new_model.names)
            elif hasattr(new_model, "names") and isinstance(new_model.names, list):
                new_names = {i: name for i, name in enumerate(new_model.names)}

            self._model = new_model
            self.model_name = new_model_path
            self.class_names = new_names
            self._is_loaded = True
            tag = "CUSTOM OBSTACLE" if self.is_custom_model else "PRETRAINED COCO"
            logger.info("Successfully switched model to '%s' [%s] (%d classes).", new_model_path, tag, len(self.class_names))
            return True
        except Exception as exc:
            logger.warning("Failed to switch to model '%s': %s. Restoring '%s'.", new_model_path, exc, old_name)
            self._model = old_model
            self.model_name = old_name
            self.class_names = old_classes
            if self._model is None and fallback_to_pretrained:
                self.load_model(fallback_to_pretrained=True)
            return False

    @property
    def is_custom_model(self) -> bool:
        """Check whether the active model is a fine-tuned custom obstacle model."""
        return len(self.class_names) == 3 and "road_hazard" in self.class_names.values()

    def is_loaded(self) -> bool:
        """Check whether the model is loaded and ready for inference."""
        return self._is_loaded and self._model is not None

    @property
    def status_text(self) -> str:
        """Return human-readable status of the model."""
        if self.is_loaded():
            model_base = self.model_name.split("/")[-1]
            tag = "Custom 3-Class" if self.is_custom_model else "COCO 80-Class"
            return f"Ready ({model_base} | {tag})"
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

        # 2. Apply optional preprocessing (CLAHE, Gamma)
        processed_frame, prep_ms = self.apply_preprocessing(frame)

        # 3. Run inference locally
        start_time = time.perf_counter()
        try:
            kwargs = {
                "conf": self.conf_threshold,
                "iou": self.iou_threshold,
                "imgsz": self.imgsz,
                "max_det": self.max_det,
                "verbose": False
            }
            if self.device is not None:
                kwargs["device"] = self.device

            results = self._model(processed_frame, **kwargs)
            if self.device == "mps":
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                except Exception:
                    pass
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
        mapping_enabled: Optional[bool] = None,
        fps: Optional[float] = None,
        diagnostic_mode: bool = False
    ) -> np.ndarray:
        """Render detection bounding boxes, labels, and telemetry onto a frame copy.

        Args:
            frame: Input image array.
            detections: List of Detection instances to render.
            inference_time_ms: Optional inference latency in ms to display.
            mapping_enabled: Optional flag indicating whether perception grid mapping is ON or OFF.
            fps: Optional real-time stream processing FPS to display.
            diagnostic_mode: If True, renders comprehensive image quality diagnostics
                             (luminance, contrast, sharpness).

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
        hud_w = max(380, int(w * 0.38))
        hud_h = 106 if (mapping_enabled is not None or fps is not None) else 90
        cv2.rectangle(canvas, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (10, 10), (10 + hud_w, 10 + hud_h), (0, 200, 255), 1)

        dev_label = f" ({self.device})" if self.device else ""
        model_base = self.model_name.split("/")[-1]
        model_type_tag = "[CUSTOM 3-CLASS]" if self.is_custom_model else "[COCO 80-CLASS]"
        cv2.putText(
            canvas,
            f"YOLO: {model_base} {model_type_tag}{dev_label} | imgsz: {self.imgsz}",
            (18, 28),
            font,
            0.44,
            (0, 200, 255),
            1,
            cv2.LINE_AA
        )

        inf_str = f"{inference_time_ms:.1f} ms" if inference_time_ms is not None else "N/A"
        fps_str = f" | {fps:.1f} FPS" if fps is not None else ""

        if len(detections) > 0:
            det_text = f"Detections: {len(detections)}  |  Latency: {inf_str}{fps_str}"
            det_color = (0, 240, 160)
        else:
            det_text = f"Detections: 0 (No objects)  |  Latency: {inf_str}{fps_str}"
            det_color = (0, 165, 255)  # Amber warning color for zero detections

        cv2.putText(
            canvas,
            det_text,
            (18, 48),
            font,
            0.45,
            det_color,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            f"Conf: {self.conf_threshold:.2f} ( [ / ] ) | IoU: {self.iou_threshold:.2f} ( < / > )",
            (18, 66),
            font,
            0.40,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

        preproc_str = self.preprocess_mode.upper() if self.preprocess_mode != "none" else "NONE (Raw)"
        cv2.putText(
            canvas,
            f"Preproc: {preproc_str} ('P') | MaxDet: {self.max_det}",
            (18, 82),
            font,
            0.38,
            (160, 220, 255) if self.preprocess_mode != "none" else (150, 150, 150),
            1,
            cv2.LINE_AA
        )

        hud_next_y = 98
        if mapping_enabled is not None:
            map_status = "ON" if mapping_enabled else "OFF"
            map_color = (0, 255, 120) if mapping_enabled else (140, 140, 140)
            cv2.putText(
                canvas,
                f"Perception Mapping: {map_status} (Toggle: 'M')",
                (18, hud_next_y),
                font,
                0.38,
                map_color,
                1,
                cv2.LINE_AA
            )
            hud_next_y += 16

        # Fast subsampled image quality calculation (1/8 stride for near-zero compute overhead)
        sample_slice = frame[::8, ::8]
        if sample_slice.ndim == 3:
            gray_sample = cv2.cvtColor(sample_slice, cv2.COLOR_BGR2GRAY)
        else:
            gray_sample = sample_slice
        mean_lum = float(np.mean(gray_sample))
        std_contrast = float(np.std(gray_sample))

        # Prominent zero-detection banner when model detects zero objects
        if len(detections) == 0:
            banner_w = min(w - 40, max(460, int(w * 0.65)))
            banner_h = 44
            bx = (w - banner_w) // 2
            by = h - banner_h - 18

            # Highlight cause: Low illumination warning vs threshold filtering
            if mean_lum < 25.0:
                bg_color = (15, 15, 80)      # Dark Red / Brown
                border_color = (0, 80, 255)  # Bright Orange / Amber
                warn_title = f"[!] ZERO OBJECTS - LOW ILLUMINATION / SENSOR COVERED (Mean: {mean_lum:.1f}/255)"
                warn_subtitle = "Check camera lens, room lighting, or MacBook lid angle"
            else:
                bg_color = (25, 25, 25)
                border_color = (0, 165, 255)
                warn_title = f"[!] ZERO OBJECTS DETECTED (Threshold >= {self.conf_threshold:.2f}, imgsz={self.imgsz})"
                warn_subtitle = "No objects meeting confidence threshold in current frame"

            cv2.rectangle(canvas, (bx, by), (bx + banner_w, by + banner_h), bg_color, -1)
            cv2.rectangle(canvas, (bx, by), (bx + banner_w, by + banner_h), border_color, 2)
            cv2.putText(
                canvas,
                warn_title,
                (bx + 12, by + 18),
                font,
                0.42,
                (0, 220, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                canvas,
                warn_subtitle,
                (bx + 12, by + 34),
                font,
                0.36,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

        # Diagnostic Mode Overlay: Top-Right Camera Environment HUD
        if diagnostic_mode:
            diag_w = 260
            diag_h = 74
            dx = w - diag_w - 10
            dy = 10
            cv2.rectangle(canvas, (dx, dy), (dx + diag_w, dy + diag_h), (20, 20, 20), -1)
            cv2.rectangle(canvas, (dx, dy), (dx + diag_w, dy + diag_h), (255, 150, 0), 1)

            cv2.putText(
                canvas,
                f"DIAGNOSTIC TELEMETRY ({w}x{h})",
                (dx + 10, dy + 20),
                font,
                0.42,
                (255, 180, 0),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                canvas,
                f"Luminance: {mean_lum:.1f}/255 | Contrast: {std_contrast:.1f}",
                (dx + 10, dy + 40),
                font,
                0.38,
                (0, 255, 160) if mean_lum >= 30 else (0, 120, 255),
                1,
                cv2.LINE_AA
            )
            lighting_str = "NORMAL" if mean_lum >= 40 else ("POOR" if mean_lum >= 15 else "PITCH DARK")
            cv2.putText(
                canvas,
                f"Lighting: {lighting_str} | Aspect: {w/h:.2f}:1",
                (dx + 10, dy + 58),
                font,
                0.38,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

        return canvas
