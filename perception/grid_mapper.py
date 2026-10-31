"""Simplified grid-based obstacle mapping from 2D YOLO detections and persistent ByteTrack tracks.

Translates 2D perception bounding boxes and tracked obstacles into candidate terrain grid cells
for academic demonstration and laboratory robotics.

Theoretical & Architectural Assumptions:
-----------------------------------------
1. Dimensionality Gap:
   A monocular RGB camera captures a 2D projection of a 3D environment.
   A 2D bounding box (x1, y1, x2, y2) in pixel space does NOT directly provide real-world metric
   depth (Z distance) or metric lateral displacement (X distance). It is ill-posed to equate
   2D pixel coordinates to physical ground coordinates without extrinsic camera calibration
   or planar homography.

2. Planar Homography Calibration:
   When camera calibration fiducials are provided via CameraCalibrator, this module computes
   a true 3x3 projective transformation between camera pixels and the 2D tabletop reference plane.
   If uncalibrated, it gracefully falls back to Normalized Region-of-Interest (ROI) linear projection.

3. Ground-Contact Point Justification:
   For upright physical objects (bottles, boxes, barrier props, pedestrians), the visual center (cx, cy)
   is elevated in 3D space above the road. Using (cx, cy) causes severe vertical parallax / foreshortening.
   The bottom-center (cx, y2) represents the physical contact line where the object touches the ground,
   making it the mathematically and geometrically justified proxy for ground-plane projection.

4. Safety Invariants:
   - Vehicle & Goal Protection: Candidate perception obstacle cells that overlap the active
     vehicle position or navigation destination are strictly rejected and counted as conflicts.
   - Decoupled Obstacle Layers: Perception-derived cells are tracked in a dedicated set
     (`perception_obstacle_set`) and kept strictly separated from manually placed static hazards
     and dynamically spawned hazards. Clearing or refreshing perception never destroys static hazards.
   - Bounds Validation: Invalid, negative, or overflow coordinates are dropped before grid access.
   - Confidence Filtering: Detections falling below a configurable confidence threshold are ignored.
   - Safety Margins: Configurable obstacle inflation expands hazard footprints into neighboring cells.
   - Stale Detection Pruning: Detections that disappear from the camera feed expire after a
     configurable timeout duration, preventing transient ghosts from permanently locking the map.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import time

from perception.object_detector import Detection
from perception.object_tracker import TrackedObject
from perception.camera_calibrator import CameraCalibrator

logger = logging.getLogger(__name__)


class MappingMode(str, Enum):
    """Supported perception-to-grid mapping modes."""
    HOMOGRAPHY = "Planar Homography (Calibrated)"
    NORMALIZED_ROI = "Normalized ROI (Simplified Demonstration)"


@dataclass
class GridMapperConfig:
    """Configuration parameters for perception-to-grid obstacle mapping.

    Attributes:
        min_confidence: Minimum detection confidence threshold (0.0 to 1.0).
        target_classes: Optional set of COCO class names to map (None accepts all classes).
        grid_rows: Number of rows in the simulation terrain grid (default: 22).
        grid_cols: Number of columns in the simulation terrain grid (default: 26).
        roi_norm: Normalized image Region of Interest (u_min, v_min, u_max, v_max) in [0.0, 1.0].
        grid_roi: Target grid subregion (r_min, c_min, r_max, c_max) to map the ROI into.
                  Defaults to the full grid if None.
        contact_point: Point on bounding box used as ground contact proxy:
                       'bottom_center' (cx, y2) or 'center' (cx, cy).
        inflation_radius: Chebyshev inflation radius in grid cells (0 = single cell, 1 = 3x3 footprint).
        stale_timeout_sec: Time in seconds before an unrefreshed cell expires and is cleared.
        invert_row_mapping: If True, top of image maps to bottom of grid (default False: top maps to top).
        calibrator: Optional CameraCalibrator instance for projective homography.
        debounce_hits: Number of consecutive frames an obstacle must be seen before confirming on grid.
    """
    min_confidence: float = 0.35
    target_classes: Optional[Set[str]] = None
    grid_rows: int = 22
    grid_cols: int = 26
    roi_norm: Tuple[float, float, float, float] = (0.05, 0.10, 0.95, 0.95)
    grid_roi: Optional[Tuple[int, int, int, int]] = None
    contact_point: str = "bottom_center"
    inflation_radius: int = 0
    stale_timeout_sec: float = 2.0
    invert_row_mapping: bool = False
    calibrator: Optional[CameraCalibrator] = None
    debounce_hits: int = 1


@dataclass
class GridMappingResult:
    """Structured report returned after processing a perception mapping cycle.

    Attributes:
        timestamp: Time of the mapping cycle.
        raw_detections_count: Total number of raw detections / tracks provided.
        filtered_detections_count: Number of detections passing confidence & class filters.
        candidate_cells: Set of mapped base grid coordinates (row, col).
        inflated_cells: Set of candidate cells after safety margin inflation.
        vehicle_conflicts: Cells dropped because they coincided with the vehicle's position.
        goal_conflicts: Cells dropped because they coincided with the destination position.
        out_of_bounds_count: Detections rejected due to falling outside image/ROI/grid bounds.
        final_cells: Clean, validated, conflict-free obstacle cells ready for the grid.
        mapping_mode: Active projection mode (HOMOGRAPHY or NORMALIZED_ROI).
        mapped_tracks_info: Detailed metadata list for each mapped track.
    """
    timestamp: float
    raw_detections_count: int
    filtered_detections_count: int
    candidate_cells: Set[Tuple[int, int]] = field(default_factory=set)
    inflated_cells: Set[Tuple[int, int]] = field(default_factory=set)
    vehicle_conflicts: Set[Tuple[int, int]] = field(default_factory=set)
    goal_conflicts: Set[Tuple[int, int]] = field(default_factory=set)
    out_of_bounds_count: int = 0
    final_cells: Set[Tuple[int, int]] = field(default_factory=set)
    mapping_mode: str = MappingMode.NORMALIZED_ROI.value
    mapped_tracks_info: List[Dict[str, Any]] = field(default_factory=list)


class GridMapper:
    """Maps structured 2D visual detections and persistent tracks onto a discrete 2D simulation grid.

    Decoupled from camera hardware and neural network inference; operates purely
    on structured Detection or TrackedObject instances.
    """

    def __init__(self, config: Optional[GridMapperConfig] = None):
        """Initialize the grid mapper with configuration parameters.

        Args:
            config: Optional GridMapperConfig instance. Defaults to standard configuration.
        """
        self.config: GridMapperConfig = config or GridMapperConfig()
        self.calibrator: Optional[CameraCalibrator] = self.config.calibrator

        # Cache grid boundaries
        if self.config.grid_roi is not None:
            self.r_min, self.c_min, self.r_max, self.c_max = self.config.grid_roi
        else:
            self.r_min, self.c_min = 0, 0
            self.r_max = self.config.grid_rows - 1
            self.c_max = self.config.grid_cols - 1

        # Freshness tracking table: maps (row, col) -> last_seen_timestamp (seconds)
        self._tracked_cells: Dict[Tuple[int, int], float] = {}

        # Debounce hit counters: maps (row, col) -> hit_count
        self._cell_hit_counts: Dict[Tuple[int, int], int] = {}

        # Telemetry
        self.total_processed_frames: int = 0
        self.total_vehicle_conflicts: int = 0
        self.total_goal_conflicts: int = 0

    @property
    def mapping_mode(self) -> MappingMode:
        """Return active mapping mode based on calibration state."""
        if self.calibrator is not None and self.calibrator.is_calibrated:
            return MappingMode.HOMOGRAPHY
        return MappingMode.NORMALIZED_ROI

    def get_contact_point(self, target: Any) -> Tuple[int, int]:
        """Compute the 2D pixel coordinate representing the object's ground contact point.

        Justification:
        For upright obstacles standing on the ground plane, the bottom-center coordinate
        (cx, y2) eliminates vertical parallax error inherent in elevated bounding box centers.
        """
        bbox = target.bbox if hasattr(target, "bbox") else target
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) // 2)
        if self.config.contact_point == "center":
            cy = int((y1 + y2) // 2)
            return (cx, cy)
        return (cx, int(y2))

    def map_bbox_to_cell(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[float, float]]]:
        """Project a 2D bounding box to a discrete grid cell (row, col).

        Returns:
            Tuple[Optional[Tuple[int, int]], Optional[Tuple[float, float]]]:
                (cell, ground_coord) where ground_coord is (X_g, Y_g) if homography is active.
        """
        if frame_width <= 0 or frame_height <= 0:
            return None, None

        px, py = self.get_contact_point(bbox)

        # Mode A: Planar Homography Calibration
        if self.calibrator is not None and self.calibrator.is_calibrated:
            ground_pt = self.calibrator.transform_image_to_ground((float(px), float(py)))
            if ground_pt is not None:
                cell = self.calibrator.ground_to_grid(
                    ground_pt,
                    grid_rows=self.config.grid_rows,
                    grid_cols=self.config.grid_cols,
                    invert_rows=self.config.invert_row_mapping
                )
                if cell is not None and 0 <= cell[0] < self.config.grid_rows and 0 <= cell[1] < self.config.grid_cols:
                    return cell, ground_pt

        # Mode B: Normalized ROI Linear Projection (Fallback)
        u = px / float(frame_width)
        v = py / float(frame_height)

        u_min, v_min, u_max, v_max = self.config.roi_norm
        if not (u_min <= u <= u_max and v_min <= v <= v_max):
            return None, None

        roi_w = max(1e-6, u_max - u_min)
        roi_h = max(1e-6, v_max - v_min)
        norm_x = max(0.0, min(1.0, (u - u_min) / roi_w))
        norm_y = max(0.0, min(1.0, (v - v_min) / roi_h))

        col_span = self.c_max - self.c_min
        row_span = self.r_max - self.r_min

        col = int(round(self.c_min + norm_x * col_span))
        if self.config.invert_row_mapping:
            row = int(round(self.r_max - norm_y * row_span))
        else:
            row = int(round(self.r_min + norm_y * row_span))

        if not (0 <= row < self.config.grid_rows and 0 <= col < self.config.grid_cols):
            return None, None

        return (row, col), None

    def map_detection_to_cell(
        self,
        detection: Detection,
        frame_width: int,
        frame_height: int
    ) -> Optional[Tuple[int, int]]:
        """Project a single detection into a grid coordinate (row, col)."""
        if detection.confidence < self.config.min_confidence:
            return None
        if self.config.target_classes is not None and detection.class_name not in self.config.target_classes:
            return None

        cell, _ = self.map_bbox_to_cell(detection.bbox, frame_width, frame_height)
        return cell

    def inflate_cells(
        self,
        base_cells: Set[Tuple[int, int]],
        radius: Optional[int] = None
    ) -> Set[Tuple[int, int]]:
        """Inflate obstacle footprints to provide a safety margin around detected objects.

        Uses Chebyshev (square) neighborhood expansion within valid grid boundaries.
        """
        r_dist = self.config.inflation_radius if radius is None else radius
        if r_dist <= 0:
            return set(base_cells)

        inflated: Set[Tuple[int, int]] = set()
        for r, c in base_cells:
            for dr in range(-r_dist, r_dist + 1):
                for dc in range(-r_dist, r_dist + 1):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < self.config.grid_rows and 0 <= nc < self.config.grid_cols:
                        inflated.add((nr, nc))

        return inflated

    def prune_stale_cells(self, current_time: float) -> Set[Tuple[int, int]]:
        """Remove cells whose last detection timestamp has exceeded the stale timeout."""
        expired: List[Tuple[int, int]] = []
        for cell, last_seen in self._tracked_cells.items():
            if (current_time - last_seen) > self.config.stale_timeout_sec:
                expired.append(cell)

        for cell in expired:
            del self._tracked_cells[cell]
            self._cell_hit_counts.pop(cell, None)

        return set(self._tracked_cells.keys())

    def map_tracks(
        self,
        tracks: List[TrackedObject],
        frame_shape: Tuple[int, int],
        current_time: Optional[float] = None,
        vehicle_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None
    ) -> GridMappingResult:
        """Process Phase 4 TrackedObject instances into validated, conflict-free occupancy cells.

        Consumes persistent tracks, applies homography or ROI projection, filters duplicates,
        debounces coordinates against jitter, and enforces vehicle/goal safety invariants.
        """
        now = time.time() if current_time is None else current_time
        self.total_processed_frames += 1

        h, w = frame_shape
        candidate_cells: Set[Tuple[int, int]] = set()
        vehicle_conflicts: Set[Tuple[int, int]] = set()
        goal_conflicts: Set[Tuple[int, int]] = set()
        out_of_bounds_count = 0
        filtered_count = 0
        mapped_tracks_info: List[Dict[str, Any]] = []

        # Current frame candidate cells (before debounce / inflation)
        frame_candidate_cells: Set[Tuple[int, int]] = set()

        for track in tracks:
            # 1. Filter tracks by confidence and target classes
            if track.confidence < self.config.min_confidence:
                continue
            if self.config.target_classes and track.class_name not in self.config.target_classes:
                continue

            filtered_count += 1
            cell, ground_pt = self.map_bbox_to_cell(track.bbox, frame_width=w, frame_height=h)

            if cell is not None:
                frame_candidate_cells.add(cell)
                self._cell_hit_counts[cell] = self._cell_hit_counts.get(cell, 0) + 1

                # Debounce threshold check
                if self._cell_hit_counts[cell] >= self.config.debounce_hits:
                    candidate_cells.add(cell)

                mapped_tracks_info.append({
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "confidence": round(track.confidence, 4),
                    "bbox": track.bbox,
                    "ground_contact_pt": self.get_contact_point(track.bbox),
                    "ground_coord": (round(ground_pt[0], 2), round(ground_pt[1], 2)) if ground_pt else None,
                    "grid_cell": cell,
                    "track_age_frames": track.age_frames,
                    "last_seen_ts": track.last_seen_ts,
                    "is_confirmed": track.is_confirmed,
                })
            else:
                out_of_bounds_count += 1

        # 2. Safety inflation expansion
        inflated_cells = self.inflate_cells(candidate_cells)

        # 3. Freshness timestamp updates
        for cell in inflated_cells:
            self._tracked_cells[cell] = now

        # 4. Prune stale cells
        active_cells = self.prune_stale_cells(now)

        # 5. Enforce safety invariants: strictly protect vehicle & goal
        final_cells: Set[Tuple[int, int]] = set()
        for cell in active_cells:
            if vehicle_pos is not None and cell == vehicle_pos:
                vehicle_conflicts.add(cell)
                self.total_vehicle_conflicts += 1
                continue
            if goal_pos is not None and cell == goal_pos:
                goal_conflicts.add(cell)
                self.total_goal_conflicts += 1
                continue
            final_cells.add(cell)

        return GridMappingResult(
            timestamp=now,
            raw_detections_count=len(tracks),
            filtered_detections_count=filtered_count,
            candidate_cells=candidate_cells,
            inflated_cells=inflated_cells,
            vehicle_conflicts=vehicle_conflicts,
            goal_conflicts=goal_conflicts,
            out_of_bounds_count=out_of_bounds_count,
            final_cells=final_cells,
            mapping_mode=self.mapping_mode.value,
            mapped_tracks_info=mapped_tracks_info
        )

    def map_detections(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int],
        current_time: Optional[float] = None,
        vehicle_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None
    ) -> GridMappingResult:
        """Process Detection records into validated grid obstacle cells (backward compatibility)."""
        now = time.time() if current_time is None else current_time
        self.total_processed_frames += 1

        h, w = frame_shape
        candidate_cells: Set[Tuple[int, int]] = set()
        vehicle_conflicts: Set[Tuple[int, int]] = set()
        goal_conflicts: Set[Tuple[int, int]] = set()
        out_of_bounds_count = 0
        filtered_count = 0

        for det in detections:
            if det.confidence < self.config.min_confidence:
                continue
            if self.config.target_classes and det.class_name not in self.config.target_classes:
                continue

            filtered_count += 1
            cell = self.map_detection_to_cell(det, frame_width=w, frame_height=h)
            if cell is not None:
                candidate_cells.add(cell)
            else:
                out_of_bounds_count += 1

        inflated_cells = self.inflate_cells(candidate_cells)

        for cell in inflated_cells:
            self._tracked_cells[cell] = now

        active_cells = self.prune_stale_cells(now)

        final_cells: Set[Tuple[int, int]] = set()
        for cell in active_cells:
            if vehicle_pos is not None and cell == vehicle_pos:
                vehicle_conflicts.add(cell)
                self.total_vehicle_conflicts += 1
                continue
            if goal_pos is not None and cell == goal_pos:
                goal_conflicts.add(cell)
                self.total_goal_conflicts += 1
                continue
            final_cells.add(cell)

        return GridMappingResult(
            timestamp=now,
            raw_detections_count=len(detections),
            filtered_detections_count=filtered_count,
            candidate_cells=candidate_cells,
            inflated_cells=inflated_cells,
            vehicle_conflicts=vehicle_conflicts,
            goal_conflicts=goal_conflicts,
            out_of_bounds_count=out_of_bounds_count,
            final_cells=final_cells,
            mapping_mode=self.mapping_mode.value
        )

    def clear(self) -> None:
        """Clear all tracked cells, history timestamps, and reset conflict counters."""
        self._tracked_cells.clear()
        self._cell_hit_counts.clear()
        self.total_vehicle_conflicts = 0
        self.total_goal_conflicts = 0

    def get_active_cells(self) -> Set[Tuple[int, int]]:
        """Return the current set of tracked, non-stale obstacle coordinates."""
        return set(self._tracked_cells.keys())
