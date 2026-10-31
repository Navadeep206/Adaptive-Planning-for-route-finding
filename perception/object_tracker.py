"""Real-time Multi-Object Tracking (MOT) and Temporal Detection Stability module.

Provides temporal association, persistent track ID assignment, track lifecycle management
(NEW -> CONFIRMED -> LOST -> EXPIRED), 2D image-plane motion estimation (velocity in px/s),
trajectory trail visualization, and configurable confidence/bounding-box smoothing.

Mathematical & Architectural Distinction:
-----------------------------------------
All motion estimations computed by this tracker represent TWO-DIMENSIONAL (2D) IMAGE-PLANE
DISPLACEMENTS (measured strictly in screen pixels and pixels per second).
Without extrinsic camera calibration (pitch, yaw, roll, camera height), flat-road planar
homography (Inverse Perspective Mapping / IPM), or metric depth sensors (LiDAR, stereo cameras),
image-plane velocity CANNOT be equated to real-world metric velocity (m/s or km/h).
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import colorsys
import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from perception.object_detector import Detection, YOLOObjectDetector

logger = logging.getLogger(__name__)


class TrackState(str, Enum):
    """Lifecycle states of a tracked object."""
    NEW = "NEW"                  # Tentative track: single detection, awaiting confirmation
    CONFIRMED = "CONFIRMED"      # Confirmed stable track: verified across multiple consecutive frames
    LOST = "LOST"                # Coasting track: temporarily missing/occluded, preserved for timeout
    EXPIRED = "EXPIRED"          # Inactive track: exceeded lost timeout, queued for eviction


@dataclass
class TrackedObject:
    """Structured representation of a temporally tracked obstacle across video frames."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]           # Smoothed bounding box (x1, y1, x2, y2)
    raw_bbox: Tuple[int, int, int, int]       # Latest raw bounding box (x1, y1, x2, y2)
    state: TrackState = TrackState.NEW
    hits: int = 1                             # Total frames matched with a valid detection
    age_frames: int = 1                       # Total frame lifespan since creation
    consecutive_misses: int = 0               # Consecutive frames without a detection match
    first_seen_ts: float = 0.0                # Epoch timestamp when track was first registered
    last_seen_ts: float = 0.0                 # Epoch timestamp of latest detection match
    velocity: Tuple[float, float] = (0.0, 0.0)  # Estimated 2D image-plane velocity (vx, vy) in px/sec
    speed_px_sec: float = 0.0                 # Scalar magnitude of image-plane velocity in px/sec
    history: deque = field(default_factory=lambda: deque(maxlen=30))  # Breadcrumbs of (cx, cy)
    timestamps: deque = field(default_factory=lambda: deque(maxlen=30))  # Corresponding timestamps

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[int, int]:
        """Current bounding box center coordinate (cx, cy)."""
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    @property
    def previous_center(self) -> Optional[Tuple[int, int]]:
        """Previous center coordinate from trajectory history, if available."""
        if len(self.history) >= 2:
            return self.history[-2]
        return None

    @property
    def is_confirmed(self) -> bool:
        """True if the track has met confirmation threshold and is active."""
        return self.state == TrackState.CONFIRMED

    @property
    def is_lost(self) -> bool:
        """True if the track is temporarily coasting without fresh detections."""
        return self.state == TrackState.LOST

    @property
    def is_active(self) -> bool:
        """True if the track is active (NEW, CONFIRMED, or LOST)."""
        return self.state in (TrackState.NEW, TrackState.CONFIRMED, TrackState.LOST)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracked object state to dictionary."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "center": self.center,
            "previous_center": self.previous_center,
            "state": self.state.value,
            "hits": self.hits,
            "age_frames": self.age_frames,
            "consecutive_misses": self.consecutive_misses,
            "velocity_px_sec": (round(self.velocity[0], 2), round(self.velocity[1], 2)),
            "speed_px_sec": round(self.speed_px_sec, 2),
            "first_seen_ts": self.first_seen_ts,
            "last_seen_ts": self.last_seen_ts,
        }


@dataclass
class TrackerConfig:
    """Configuration hyperparameters for multi-object tracking and temporal stability."""
    min_confidence: float = 0.25               # Minimum detection confidence to initiate or update tracks
    track_activation_threshold: int = 2        # Consecutive hit count required to transition NEW -> CONFIRMED
    lost_timeout_sec: float = 1.0              # Maximum time (seconds) to preserve a lost track before eviction
    max_lost_frames: int = 15                  # Maximum consecutive missed frames before track expiration
    history_len: int = 30                      # Maximum trajectory breadcrumbs retained for path trails
    iou_threshold: float = 0.30                # Minimum IoU overlap for spatial track association
    enable_box_smoothing: bool = True          # Apply exponential moving average (EMA) to bounding boxes
    smoothing_alpha: float = 0.70              # EMA alpha weight: 1.0 = raw detection, lower = heavier smoothing
    enable_confidence_smoothing: bool = True   # Smooth confidence over consecutive hits
    tracker_mode: str = "auto"                 # 'auto' (Ultralytics ByteTrack when available, fallback to IoU)


class ObjectTracker:
    """Real-time Multi-Object Tracker with ByteTrack integration and temporal stability.

    Wraps or interoperates with YOLOObjectDetector, assigning persistent track IDs across frames,
    estimating 2D image-plane motion vectors, and filtering out transient single-frame ghosts.
    """

    def __init__(
        self,
        detector: Optional[YOLOObjectDetector] = None,
        config: Optional[TrackerConfig] = None
    ):
        """Initialize the object tracker.

        Args:
            detector: Optional YOLOObjectDetector instance to drive neural inference.
            config: Optional TrackerConfig specifying tracking hyperparameters.
        """
        self.detector: Optional[YOLOObjectDetector] = detector
        self.config: TrackerConfig = config or TrackerConfig()
        self._tracks: Dict[int, TrackedObject] = {}
        self._next_track_id: int = 1
        self._last_update_ts: float = 0.0

        # Performance telemetry
        self.last_inference_time_ms: float = 0.0
        self.last_tracking_time_ms: float = 0.0
        self.last_total_time_ms: float = 0.0

        logger.info(
            "Initialized ObjectTracker (activation_hits=%d, lost_timeout=%.1fs, max_lost_frames=%d, smoothing=%s)",
            self.config.track_activation_threshold,
            self.config.lost_timeout_sec,
            self.config.max_lost_frames,
            self.config.enable_box_smoothing
        )

    def reset(self) -> None:
        """Clear all active and coasting tracks and reset track ID counter."""
        self._tracks.clear()
        self._next_track_id = 1
        self._last_update_ts = 0.0
        self.last_inference_time_ms = 0.0
        self.last_tracking_time_ms = 0.0
        self.last_total_time_ms = 0.0
        logger.info("ObjectTracker state reset.")

    def get_active_tracks(self) -> List[TrackedObject]:
        """Return all active tracks (confirmed tracks + coasting tracks within lost timeout)."""
        return [t for t in self._tracks.values() if t.is_active]

    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """Return only tracks that have passed the activation threshold."""
        return [t for t in self._tracks.values() if t.is_confirmed]

    def get_all_tracks(self) -> List[TrackedObject]:
        """Return all tracks currently in tracker memory."""
        return list(self._tracks.values())

    @property
    def active_track_count(self) -> int:
        """Number of confirmed and coasting tracks."""
        return len(self.get_active_tracks())

    @property
    def lost_track_count(self) -> int:
        """Number of coasting tracks currently missing detections."""
        return sum(1 for t in self._tracks.values() if t.is_lost)

    def update(
        self,
        frame: Optional[np.ndarray] = None,
        detections: Optional[List[Detection]] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[List[TrackedObject], Dict[str, float]]:
        """Update tracker state with a new camera frame or explicit list of detections.

        Args:
            frame: Optional camera image array (H, W, C).
            detections: Optional pre-computed Detection records (useful for testing or custom detectors).
            timestamp: Optional epoch timestamp. Defaults to time.time().

        Returns:
            Tuple[List[TrackedObject], Dict[str, float]]:
                - List of currently active TrackedObject records.
                - Telemetry dictionary with timing breakdown (inference_ms, tracking_ms, total_ms).
        """
        t_total_start = time.perf_counter()
        now = timestamp if timestamp is not None else time.time()
        inf_ms = 0.0
        trk_start = time.perf_counter()

        # Handle invalid/empty frame or detections
        if frame is not None and (frame.size == 0 or len(frame.shape) < 2):
            logger.warning("Empty or invalid frame passed to tracker. Coasting existing tracks.")
            self._coast_all_tracks(now)
            self._prune_expired_tracks(now)
            return self.get_active_tracks(), {"inference_ms": 0.0, "tracking_ms": 0.0, "total_ms": 0.0}

        # Case 1: Raw detections provided directly (headless, tests, or pre-computed)
        if detections is not None:
            active_tracks = self._update_with_detections(detections, now)

        # Case 2: Frame provided and detector is available
        elif frame is not None and self.detector is not None and self.detector.is_loaded():
            # Attempt Ultralytics native ByteTrack if supported and configured
            if self.config.tracker_mode in ("auto", "bytetrack") and hasattr(self.detector._model, "track"):
                try:
                    active_tracks, inf_ms = self._update_via_ultralytics_track(frame, now)
                except Exception as exc:
                    logger.warning("Ultralytics track() failed (%s). Falling back to detector.detect() + IoU association.", exc)
                    raw_dets, inf_ms = self.detector.detect(frame)
                    active_tracks = self._update_with_detections(raw_dets, now)
            else:
                raw_dets, inf_ms = self.detector.detect(frame)
                active_tracks = self._update_with_detections(raw_dets, now)

        # Case 3: No frame or detector not loaded -> coast all existing tracks
        else:
            self._coast_all_tracks(now)
            active_tracks = self.get_active_tracks()

        trk_ms = (time.perf_counter() - trk_start) * 1000.0
        tot_ms = (time.perf_counter() - t_total_start) * 1000.0

        self.last_inference_time_ms = inf_ms
        self.last_tracking_time_ms = trk_ms
        self.last_total_time_ms = tot_ms
        self._last_update_ts = now

        telemetry = {
            "inference_ms": round(inf_ms, 2),
            "tracking_ms": round(trk_ms, 2),
            "total_ms": round(tot_ms, 2),
        }

        return active_tracks, telemetry

    def _update_via_ultralytics_track(
        self,
        frame: np.ndarray,
        now: float
    ) -> Tuple[List[TrackedObject], float]:
        """Execute Ultralytics native ByteTrack inference and update temporal state."""
        t0 = time.perf_counter()

        # Apply preprocessing if configured in detector
        proc_frame, _ = self.detector.apply_preprocessing(frame)

        results = self.detector._model.track(
            proc_frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.config.min_confidence,
            iou=self.detector.iou_threshold,
            imgsz=self.detector.imgsz,
            max_det=self.detector.max_det,
            device=self.detector.device or None,
            verbose=False
        )
        inf_ms = (time.perf_counter() - t0) * 1000.0

        matched_track_ids: Set[int] = set()

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy_arr = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array([])
            conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array([])
            cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array([])
            id_arr = boxes.id.cpu().numpy() if boxes.id is not None and hasattr(boxes.id, "cpu") else None

            num_boxes = len(xyxy_arr)
            for i in range(num_boxes):
                x1, y1, x2, y2 = [int(round(coord)) for coord in xyxy_arr[i]]
                conf = float(conf_arr[i])
                cls_id = int(cls_arr[i])
                cls_name = self.detector.class_names.get(cls_id, f"class_{cls_id}")
                raw_bbox = (x1, y1, x2, y2)

                # Track ID assigned by ByteTrack, or allocate local ID if none returned
                if id_arr is not None and len(id_arr) > i and not np.isnan(id_arr[i]):
                    t_id = int(id_arr[i])
                else:
                    t_id = self._next_track_id
                    self._next_track_id += 1

                matched_track_ids.add(t_id)
                self._update_or_create_track(
                    track_id=t_id,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    raw_bbox=raw_bbox,
                    timestamp=now
                )

        # Coast tracks that were not matched in this frame
        for t_id, track in list(self._tracks.items()):
            if t_id not in matched_track_ids:
                self._coast_track(track, now)

        self._prune_expired_tracks(now)
        return self.get_active_tracks(), inf_ms

    def _update_with_detections(
        self,
        detections: List[Detection],
        now: float
    ) -> List[TrackedObject]:
        """Perform robust spatial IoU association on external Detection records."""
        # Filter detections by minimum confidence
        valid_dets = [d for d in detections if d.confidence >= self.config.min_confidence]

        active_tracks = [t for t in self._tracks.values() if t.is_active]

        if len(active_tracks) == 0:
            # Initialize new tracks for all detections
            for det in valid_dets:
                t_id = self._next_track_id
                self._next_track_id += 1
                self._update_or_create_track(
                    track_id=t_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    raw_bbox=det.bbox,
                    timestamp=now
                )
            self._prune_expired_tracks(now)
            return self.get_active_tracks()

        if len(valid_dets) == 0:
            # No detections -> coast all existing active tracks
            self._coast_all_tracks(now)
            self._prune_expired_tracks(now)
            return self.get_active_tracks()

        # Compute IoU cost matrix between active tracks and new detections
        # Rows: active tracks, Cols: detections
        cost_matrix = np.zeros((len(active_tracks), len(valid_dets)), dtype=np.float32)
        for r, track in enumerate(active_tracks):
            for c, det in enumerate(valid_dets):
                cost_matrix[r, c] = self._calculate_iou(track.bbox, det.bbox)

        matched_tracks: Set[int] = set()
        matched_dets: Set[int] = set()

        # Greedy maximum-IoU association
        while True:
            if cost_matrix.size == 0:
                break
            max_iou = np.max(cost_matrix)
            if max_iou < self.config.iou_threshold:
                break
            r, c = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
            track = active_tracks[r]
            det = valid_dets[c]

            # Update matched track
            self._update_or_create_track(
                track_id=track.track_id,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                raw_bbox=det.bbox,
                timestamp=now
            )
            matched_tracks.add(track.track_id)
            matched_dets.add(c)

            # Invalidate associated row and column
            cost_matrix[r, :] = -1.0
            cost_matrix[:, c] = -1.0

        # Create new tracks for unmatched detections
        for c, det in enumerate(valid_dets):
            if c not in matched_dets:
                t_id = self._next_track_id
                self._next_track_id += 1
                self._update_or_create_track(
                    track_id=t_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    raw_bbox=det.bbox,
                    timestamp=now
                )

        # Coast unmatched tracks
        for track in active_tracks:
            if track.track_id not in matched_tracks:
                self._coast_track(track, now)

        self._prune_expired_tracks(now)
        return self.get_active_tracks()

    def _update_or_create_track(
        self,
        track_id: int,
        class_id: int,
        class_name: str,
        confidence: float,
        raw_bbox: Tuple[int, int, int, int],
        timestamp: float
    ) -> TrackedObject:
        """Register a new track or update an existing matched track record."""
        if track_id in self._tracks:
            track = self._tracks[track_id]
            prev_center = track.center
            dt = max(0.001, timestamp - (track.timestamps[-1] if len(track.timestamps) > 0 else timestamp))

            # Apply Exponential Moving Average (EMA) smoothing to bounding box
            if self.config.enable_box_smoothing:
                alpha = self.config.smoothing_alpha
                smoothed_bbox = (
                    int(round(alpha * raw_bbox[0] + (1.0 - alpha) * track.bbox[0])),
                    int(round(alpha * raw_bbox[1] + (1.0 - alpha) * track.bbox[1])),
                    int(round(alpha * raw_bbox[2] + (1.0 - alpha) * track.bbox[2])),
                    int(round(alpha * raw_bbox[3] + (1.0 - alpha) * track.bbox[3])),
                )
            else:
                smoothed_bbox = raw_bbox

            # Smooth confidence
            if self.config.enable_confidence_smoothing:
                smoothed_conf = round(0.7 * confidence + 0.3 * track.confidence, 4)
            else:
                smoothed_conf = confidence

            # Compute 2D image-plane velocity (pixels/second)
            cur_center = (
                smoothed_bbox[0] + max(0, smoothed_bbox[2] - smoothed_bbox[0]) // 2,
                smoothed_bbox[1] + max(0, smoothed_bbox[3] - smoothed_bbox[1]) // 2,
            )
            dx = cur_center[0] - prev_center[0]
            dy = cur_center[1] - prev_center[1]
            raw_vx = dx / dt
            raw_vy = dy / dt

            # Smooth velocity with previous estimate
            smooth_vx = 0.6 * raw_vx + 0.4 * track.velocity[0]
            smooth_vy = 0.6 * raw_vy + 0.4 * track.velocity[1]
            speed = math.hypot(smooth_vx, smooth_vy)

            # Update lifecycle
            track.hits += 1
            track.age_frames += 1
            track.consecutive_misses = 0
            track.confidence = smoothed_conf
            track.raw_bbox = raw_bbox
            track.bbox = smoothed_bbox
            track.class_id = class_id
            track.class_name = class_name
            track.last_seen_ts = timestamp
            track.velocity = (smooth_vx, smooth_vy)
            track.speed_px_sec = speed
            track.history.append(cur_center)
            track.timestamps.append(timestamp)

            if track.hits >= self.config.track_activation_threshold:
                track.state = TrackState.CONFIRMED

            return track

        else:
            # Create a brand new track
            cur_center = (
                raw_bbox[0] + max(0, raw_bbox[2] - raw_bbox[0]) // 2,
                raw_bbox[1] + max(0, raw_bbox[3] - raw_bbox[1]) // 2,
            )
            history = deque(maxlen=self.config.history_len)
            history.append(cur_center)
            ts_history = deque(maxlen=self.config.history_len)
            ts_history.append(timestamp)

            initial_state = TrackState.CONFIRMED if self.config.track_activation_threshold <= 1 else TrackState.NEW

            new_track = TrackedObject(
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=raw_bbox,
                raw_bbox=raw_bbox,
                state=initial_state,
                hits=1,
                age_frames=1,
                consecutive_misses=0,
                first_seen_ts=timestamp,
                last_seen_ts=timestamp,
                velocity=(0.0, 0.0),
                speed_px_sec=0.0,
                history=history,
                timestamps=ts_history
            )
            self._tracks[track_id] = new_track
            return new_track

    def _coast_track(self, track: TrackedObject, now: float) -> None:
        """Mark a track as missing (LOST) and update miss counters."""
        track.consecutive_misses += 1
        track.age_frames += 1

        # Only confirmed tracks transition to LOST (coasting); unconfirmed NEW tracks expire faster
        if track.state == TrackState.CONFIRMED:
            track.state = TrackState.LOST

        # Decay velocity during coasting
        track.velocity = (track.velocity[0] * 0.8, track.velocity[1] * 0.8)
        track.speed_px_sec = math.hypot(track.velocity[0], track.velocity[1])

    def _coast_all_tracks(self, now: float) -> None:
        """Coast all currently tracked objects."""
        for track in self._tracks.values():
            if track.is_active:
                self._coast_track(track, now)

    def _prune_expired_tracks(self, now: float) -> None:
        """Evict tracks that have exceeded lost timeout or maximum consecutive misses."""
        expired_ids: List[int] = []

        for t_id, track in self._tracks.items():
            time_since_seen = now - track.last_seen_ts

            # Expire if missed for too many consecutive frames or too much wall-clock time
            if track.consecutive_misses >= self.config.max_lost_frames or time_since_seen >= self.config.lost_timeout_sec:
                track.state = TrackState.EXPIRED
                expired_ids.append(t_id)

            # Unconfirmed NEW tracks that missed immediately should be evicted right away
            elif track.state == TrackState.NEW and track.consecutive_misses >= 1:
                track.state = TrackState.EXPIRED
                expired_ids.append(t_id)

        for t_id in expired_ids:
            del self._tracks[t_id]

    @staticmethod
    def _calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def get_track_color(track_id: int) -> Tuple[int, int, int]:
        """Generate a deterministic, vibrant BGR color for a track ID using golden ratio hue spacing."""
        # Golden ratio conjugate ensures distinct, maximally spaced hues
        golden_ratio_conjugate = 0.618033988749895
        h = (track_id * golden_ratio_conjugate) % 1.0
        s = 0.85
        v = 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(b * 255), int(g * 255), int(r * 255))

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: Optional[List[TrackedObject]] = None,
        show_trails: bool = True,
        show_velocity: bool = True,
        telemetry: Optional[Dict[str, float]] = None,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """Render tracked obstacles, persistent ID badges, trajectory trails, and telemetry onto frame.

        Args:
            frame: Input image array (H, W, C) in BGR format.
            tracks: Optional list of tracks to render. If None, uses get_active_tracks().
            show_trails: If True, renders fading breadcrumb polylines of past centers.
            show_velocity: If True, renders motion vector arrow when object is moving.
            telemetry: Optional timing telemetry dict (inference_ms, tracking_ms, total_ms).
            fps: Optional effective loop frames per second.

        Returns:
            np.ndarray: Annotated frame.
        """
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()
        h, w = canvas.shape[:2]
        target_tracks = tracks if tracks is not None else self.get_active_tracks()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1

        for track in target_tracks:
            x1, y1, x2, y2 = track.bbox
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            color = self.get_track_color(track.track_id)
            box_thick = 2

            # Render trajectory breadcrumbs (fading trail)
            if show_trails and len(track.history) >= 2:
                pts = list(track.history)
                num_pts = len(pts)
                for i in range(1, num_pts):
                    alpha_factor = i / float(num_pts)
                    trail_color = (
                        int(color[0] * alpha_factor),
                        int(color[1] * alpha_factor),
                        int(color[2] * alpha_factor),
                    )
                    cv2.line(canvas, pts[i - 1], pts[i], trail_color, max(1, int(2 * alpha_factor)), cv2.LINE_AA)

            # Coasting (LOST) tracks rendered with dashed-like appearance or muted contour
            if track.is_lost:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), box_thick + 2)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (100, 160, 255), 1, cv2.LINE_4)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), box_thick + 2)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thick)

            # Motion vector arrow (if speed > 15 px/s)
            if show_velocity and track.speed_px_sec > 15.0:
                cx, cy = track.center
                # Scale velocity arrow so 100 px/s is ~30 pixels length
                scale = 0.30
                end_x = int(round(cx + track.velocity[0] * scale))
                end_y = int(round(cy + track.velocity[1] * scale))
                end_x = max(0, min(end_x, w - 1))
                end_y = max(0, min(end_y, h - 1))
                cv2.arrowedLine(canvas, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)

            # Header badge: "#ID class conf% [v=X px/s]"
            conf_pct = int(round(track.confidence * 100))
            lost_tag = " (LOST)" if track.is_lost else ""
            v_tag = f" | {track.speed_px_sec:.0f}px/s" if track.speed_px_sec > 10.0 else ""
            label = f"#{track.track_id} {track.class_name} {conf_pct}%{v_tag}{lost_tag}"

            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Top edge positioning
            if y1 - th - 8 >= 0:
                by1, by2 = y1 - th - 8, y1
                ty = y1 - 4
            else:
                by1, by2 = y1, y1 + th + 8
                ty = y1 + th + 4

            cv2.rectangle(canvas, (x1, by1), (x1 + tw + 8, by2), (20, 20, 20), -1)
            cv2.rectangle(canvas, (x1, by1), (x1 + tw + 8, by2), color, 1)
            cv2.putText(canvas, label, (x1 + 4, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Render top-left Tracker HUD overlay
        hud_w = max(390, int(w * 0.40))
        hud_h = 96
        cv2.rectangle(canvas, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (10, 10), (10 + hud_w, 10 + hud_h), (0, 200, 255), 1)

        fps_str = f" | {fps:.1f} FPS" if fps is not None else ""
        cv2.putText(
            canvas,
            f"ByteTrack Tracker: {len(target_tracks)} Active ({self.lost_track_count} Lost){fps_str}",
            (18, 30),
            font,
            0.46,
            (0, 200, 255),
            1,
            cv2.LINE_AA
        )

        inf_t = telemetry.get("inference_ms", self.last_inference_time_ms) if telemetry else self.last_inference_time_ms
        trk_t = telemetry.get("tracking_ms", self.last_tracking_time_ms) if telemetry else self.last_tracking_time_ms
        tot_t = telemetry.get("total_ms", self.last_total_time_ms) if telemetry else self.last_total_time_ms

        cv2.putText(
            canvas,
            f"Latency: Inf={inf_t:.1f}ms  Track={trk_t:.1f}ms  Total={tot_t:.1f}ms",
            (18, 52),
            font,
            0.42,
            (0, 240, 160) if tot_t < 60.0 else (0, 165, 255),
            1,
            cv2.LINE_AA
        )

        model_name = self.detector.model_name.split("/")[-1] if self.detector else "Standalone"
        cv2.putText(
            canvas,
            f"Detector: {model_name} | Motion: 2D Image-Plane (px/s)",
            (18, 74),
            font,
            0.40,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        return canvas
