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

4. Non-Planar & Elevated Object Rejection:
   Homography holds strictly for contact points resting on the calibrated tabletop surface.
   Objects that do not rest on the plane (such as people, faces, hands, elevated fixtures) or
   objects whose contact points fall outside the calibrated polygon are strictly rejected
   with explicit diagnostic reasoning.

5. Safety Invariants:
   - Vehicle & Goal Protection: Candidate perception obstacle cells that overlap the active
     vehicle position or navigation destination are strictly rejected and counted as conflicts.
   - Decoupled Obstacle Layers: Perception-derived cells are tracked in a dedicated set
     (`perception_obstacle_set`) and kept strictly separated from manually placed static hazards
     and dynamically spawned hazards. Clearing or refreshing perception never destroys static hazards.
   - Bounds Validation: Invalid, negative, or overflow coordinates are dropped before grid access.
   - Confidence Filtering: Detections falling below a configurable confidence threshold are ignored.
   - Stale Detection Pruning: Detections that disappear from the camera feed expire per-track,
     ensuring expired tracks remove only their own obstacle cells while preserving other active cells.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

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
        target_classes: Optional set of COCO class names to map (None accepts all non-rejected classes).
        rejected_classes: Set of class names representing non-ground or elevated objects (default: person, hand, face).
        require_plane_containment: If True, requires contact point to lie strictly inside the calibration quad.
        grid_rows: Number of rows in the simulation terrain grid (default: 22).
        grid_cols: Number of columns in the simulation terrain grid (default: 26).
        roi_norm: Normalized image Region of Interest (u_min, v_min, u_max, v_max) in [0.0, 1.0].
        grid_roi: Target grid subregion (r_min, c_min, r_max, c_max) to map the ROI into.
        contact_point: Point on bounding box used as ground contact proxy: 'bottom_center', 'center', or 'smart'.
        inflation_radius: Chebyshev inflation radius in grid cells (0 = single cell, 1 = 3x3 footprint).
        stale_timeout_sec: Time in seconds before an unrefreshed track's cell expires.
        invert_row_mapping: If True, top of image maps to bottom of grid.
        calibrator: Optional CameraCalibrator instance for projective homography.
        preferred_mode: Optional explicit MappingMode override.
        debounce_hits: Number of consecutive frames an obstacle must be seen before confirming on grid.
    """
    min_confidence: float = 0.25
    target_classes: Optional[Set[str]] = None
    rejected_classes: Set[str] = field(default_factory=lambda: {"person", "hand", "face"})
    require_plane_containment: bool = True
    grid_rows: int = 22
    grid_cols: int = 26
    roi_norm: Tuple[float, float, float, float] = (0.01, 0.02, 0.99, 0.99)
    grid_roi: Optional[Tuple[int, int, int, int]] = None
    contact_point: str = "bottom_center"
    inflation_radius: int = 0
    stale_timeout_sec: float = 2.0
    invert_row_mapping: bool = False
    calibrator: Optional[CameraCalibrator] = None
    preferred_mode: Optional[MappingMode] = None
    debounce_hits: int = 1
    enable_footprint: bool = True
    confirmation_frames: int = 2
    disappearance_timeout_sec: float = 1.0
    max_object_depth_cm: float = 15.0


@dataclass
class GridMappingResult:
    """Structured report returned after processing a perception mapping cycle.

    Attributes:
        timestamp: Time of the mapping cycle.
        raw_detections_count: Total number of raw detections / tracks provided.
        filtered_detections_count: Number of detections passing confidence & class filters.
        candidate_cells: Set of mapped base grid coordinates (row, col) (primary contact points).
        footprint_cells: Set of physical footprint grid cells covered by detected objects.
        inflated_cells: Set of candidate cells after safety margin inflation.
        confirmed_cells: Set of grid cells belonging to confirmed tracks (hits >= confirmation_frames).
        unconfirmed_cells: Set of grid cells belonging to temporary/unconfirmed tracks.
        coasting_cells: Set of grid cells preserved during temporary missed detection/occlusion.
        vehicle_conflicts: Cells dropped because they coincided with the vehicle's position.
        goal_conflicts: Cells dropped because they coincided with the destination position.
        out_of_bounds_count: Detections rejected due to falling outside image/ROI/grid bounds.
        final_cells: Clean, validated, conflict-free obstacle cells ready for the grid.
        mapping_mode: Active projection mode (HOMOGRAPHY or NORMALIZED_ROI).
        mapped_tracks_info: Detailed diagnostic metadata list for every detection.
        cell_to_track_id: Lookup dictionary mapping each grid cell (r, c) to its track_id.
        cell_to_status: Lookup dictionary mapping each grid cell to its lifecycle status.
        cell_to_primary: Lookup dictionary mapping each grid cell to whether it is the primary contact point.
        confirmed_tracks_count: Number of currently confirmed active tracks.
        mapped_objects_count: Number of distinct physical objects actively represented on the grid.
        rejected_detections_count: Number of detections rejected during this cycle.
    """
    timestamp: float
    raw_detections_count: int
    filtered_detections_count: int
    candidate_cells: Set[Tuple[int, int]] = field(default_factory=set)
    footprint_cells: Set[Tuple[int, int]] = field(default_factory=set)
    inflated_cells: Set[Tuple[int, int]] = field(default_factory=set)
    confirmed_cells: Set[Tuple[int, int]] = field(default_factory=set)
    unconfirmed_cells: Set[Tuple[int, int]] = field(default_factory=set)
    coasting_cells: Set[Tuple[int, int]] = field(default_factory=set)
    vehicle_conflicts: Set[Tuple[int, int]] = field(default_factory=set)
    goal_conflicts: Set[Tuple[int, int]] = field(default_factory=set)
    out_of_bounds_count: int = 0
    final_cells: Set[Tuple[int, int]] = field(default_factory=set)
    mapping_mode: str = MappingMode.NORMALIZED_ROI.value
    mapped_tracks_info: List[Dict[str, Any]] = field(default_factory=list)
    cell_to_track_id: Dict[Tuple[int, int], int] = field(default_factory=dict)
    cell_to_status: Dict[Tuple[int, int], str] = field(default_factory=dict)
    cell_to_primary: Dict[Tuple[int, int], bool] = field(default_factory=dict)
    confirmed_tracks_count: int = 0
    mapped_objects_count: int = 0
    rejected_detections_count: int = 0


class GridMapper:
    """Maps structured 2D visual detections and persistent tracks onto a discrete 2D simulation grid."""

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

        # Per-track obstacle cells: maps track_id -> Set[(row, col)]
        self._track_to_cells: Dict[int, Set[Tuple[int, int]]] = {}

        # Per-track primary contact cell: maps track_id -> (row, col)
        self._track_to_primary: Dict[int, Tuple[int, int]] = {}

        # Track metadata cache: maps track_id -> last_seen_ts
        self._track_last_seen: Dict[int, float] = {}

        # Track hit counters: maps track_id -> hit count
        self._track_hit_counts: Dict[int, int] = {}

        # Track status: maps track_id -> "CONFIRMED" | "UNCONFIRMED" | "COASTING"
        self._track_status: Dict[int, str] = {}

        # Cell to track lookup: maps (row, col) -> track_id
        self._cell_to_track_id: Dict[Tuple[int, int], int] = {}

        # Cell to status lookup: maps (row, col) -> status string
        self._cell_to_status: Dict[Tuple[int, int], str] = {}

        # Cell to primary lookup: maps (row, col) -> bool (is primary contact cell)
        self._cell_to_primary: Dict[Tuple[int, int], bool] = {}

        # Debounce hit counters: maps (row, col) -> hit_count
        self._cell_hit_counts: Dict[Tuple[int, int], int] = {}

        # Telemetry
        self.total_processed_frames: int = 0
        self.total_vehicle_conflicts: int = 0
        self.total_goal_conflicts: int = 0
        self.preferred_mode: Optional[MappingMode] = self.config.preferred_mode

    @property
    def mapping_mode(self) -> MappingMode:
        """Return active mapping mode based on user preference and calibration state."""
        if self.preferred_mode is not None:
            if self.preferred_mode == MappingMode.HOMOGRAPHY and (self.calibrator is None or not self.calibrator.is_calibrated):
                return MappingMode.NORMALIZED_ROI
            return self.preferred_mode
        if self.calibrator is not None and self.calibrator.is_calibrated:
            return MappingMode.HOMOGRAPHY
        return MappingMode.NORMALIZED_ROI

    def toggle_mode(self) -> MappingMode:
        """Toggle between Planar Homography and Normalized ROI mapping modes."""
        curr = self.mapping_mode
        if curr == MappingMode.HOMOGRAPHY:
            self.preferred_mode = MappingMode.NORMALIZED_ROI
        else:
            if self.calibrator is not None and self.calibrator.is_calibrated:
                self.preferred_mode = MappingMode.HOMOGRAPHY
            else:
                self.preferred_mode = MappingMode.NORMALIZED_ROI
        logger.info("GridMapper mapping mode switched to: %s", self.preferred_mode.value)
        return self.preferred_mode

    def toggle_contact_point(self) -> str:
        """Cycle ground contact point mode: bottom_center -> center -> smart -> bottom_center."""
        modes = ["bottom_center", "center", "smart"]
        curr_idx = modes.index(self.config.contact_point) if self.config.contact_point in modes else 0
        new_mode = modes[(curr_idx + 1) % len(modes)]
        self.config.contact_point = new_mode
        logger.info("GridMapper contact point mode set to: %s", new_mode)
        return new_mode

    def refine_contact_point(
        self,
        frame: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        frame_height: Optional[int] = None
    ) -> Tuple[int, int]:
        """Refine the physical ground contact point using adaptive gradient silhouette analysis.

        Examines the bottom boundary where the object meets the tabletop surface to
        eliminate table shadow bias and bounding box padding.

        Args:
            frame: Optional BGR image frame from the camera.
            bbox: Bounding box (x1, y1, x2, y2).
            frame_height: Optional frame height for clamping.

        Returns:
            Tuple[int, int]: Refined (cx, cy) pixel coordinates.
        """
        x1, y1, x2, y2 = bbox
        default_cx = int(round((x1 + x2) / 2.0))
        default_cy = int(round(y2))

        if frame is None or frame.size == 0 or frame.ndim < 2:
            return default_cx, default_cy

        fh, fw = frame.shape[:2]
        h_box = y2 - y1
        if h_box < 12:
            return default_cx, default_cy

        # Restrict ROI to the base of the object: bottom 25% of height plus 6px margin below
        y_top = max(0, y2 - max(10, int(h_box * 0.25)))
        y_bot = min(fh, y2 + 6)
        x_left = max(0, x1 - 4)
        x_right = min(fw, x2 + 4)

        if y_bot <= y_top or x_right <= x_left:
            return default_cx, default_cy

        roi = frame[y_top:y_bot, x_left:x_right]
        if roi.size == 0:
            return default_cx, default_cy

        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
            mag_y = np.abs(sobel_y)

            if mag_y.shape[0] > 4:
                search_start = mag_y.shape[0] // 3
                lower_mag = mag_y[search_start:, :]
                max_grad_indices = np.argmax(lower_mag, axis=0) + search_start
                valid_grad = np.max(lower_mag, axis=0) > 15.0

                if np.sum(valid_grad) >= 5:
                    refined_local_y = int(np.median(max_grad_indices[valid_grad]))
                    refined_y = y_top + refined_local_y
                    if abs(refined_y - y2) <= 8:
                        return default_cx, refined_y
        except Exception:
            pass

        return default_cx, default_cy

    def get_contact_point(
        self,
        target: Any,
        frame_height: Optional[int] = None,
        frame: Optional[np.ndarray] = None
    ) -> Tuple[int, int]:
        """Compute the 2D pixel coordinate representing the object's ground contact point.

        Justification:
        For upright obstacles standing on the ground plane, the bottom-center coordinate
        (cx, y2) eliminates vertical parallax error inherent in elevated bounding box centers.
        When a video frame is provided, base gradient refinement is applied.
        """
        bbox = target.bbox if hasattr(target, "bbox") else target
        x1, y1, x2, y2 = bbox
        cx = int(round((x1 + x2) / 2.0))

        if self.config.contact_point == "center":
            cy = int(round((y1 + y2) / 2.0))
            return (cx, cy)
        elif self.config.contact_point == "smart":
            if frame_height is not None and y2 >= 0.90 * frame_height:
                cy = int(round(y1 + (y2 - y1) * 0.75))
                return (cx, cy)

        if frame is not None:
            return self.refine_contact_point(frame, bbox, frame_height)

        return (cx, int(round(y2)))

    def map_bbox_to_cell(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        class_name: Optional[str] = None,
        confidence: Optional[float] = None,
        return_diagnostics: bool = False
    ) -> Any:
        """Project a 2D bounding box to a discrete grid cell (row, col) with explicit acceptance/rejection diagnostics.

        Args:
            bbox: (x1, y1, x2, y2) bounding box in image pixels.
            frame_width: Image width in pixels.
            frame_height: Image height in pixels.
            class_name: Optional object classification label (e.g. 'box', 'person').
            confidence: Optional detection confidence score.
            return_diagnostics: If True, returns a 3-tuple (cell, ground_coord, reason).
                                If False, returns a 2-tuple (cell, ground_coord) for backwards compatibility.

        Returns:
            Tuple (cell, ground_coord) or (cell, ground_coord, reason)
        """
        if frame_width <= 0 or frame_height <= 0:
            res = (None, None, "REJECTED: Invalid frame dimensions")
            return res if return_diagnostics else (res[0], res[1])

        # 1. Non-ground / elevated object rejection (strictly enforced in calibrated homography mode)
        if class_name is not None:
            lower_cls = class_name.lower().strip()
            if self.mapping_mode == MappingMode.HOMOGRAPHY and self.config.rejected_classes and lower_cls in self.config.rejected_classes:
                res = (None, None, f"REJECTED: Class '{class_name}' is non-ground/elevated (cannot assume resting on plane)")
                return res if return_diagnostics else (res[0], res[1])

            if self.config.target_classes is not None and class_name not in self.config.target_classes:
                res = (None, None, f"REJECTED: Class '{class_name}' not in configured target classes")
                return res if return_diagnostics else (res[0], res[1])

        # 2. Confidence threshold filtering
        if confidence is not None and confidence < self.config.min_confidence:
            res = (None, None, f"REJECTED: Confidence {confidence:.2f} < {self.config.min_confidence:.2f} threshold")
            return res if return_diagnostics else (res[0], res[1])

        px, py = self.get_contact_point(bbox, frame_height=frame_height)

        # Mode A: Planar Homography Calibration (when active)
        if self.mapping_mode == MappingMode.HOMOGRAPHY and self.calibrator is not None and self.calibrator.is_calibrated:
            if hasattr(self.calibrator, "adapt_to_resolution"):
                self.calibrator.adapt_to_resolution(frame_width, frame_height)

            # Strict plane containment check
            if self.config.require_plane_containment and not self.calibrator.is_point_inside_plane((float(px), float(py))):
                res = (None, None, f"REJECTED: Contact point ({px}, {py}) outside calibrated ground plane boundary")
                return res if return_diagnostics else (res[0], res[1])

            ground_pt = self.calibrator.transform_image_to_ground((float(px), float(py)), strict_bounds=True)
            if ground_pt is not None:
                cell = self.calibrator.ground_to_grid(
                    ground_pt,
                    grid_rows=self.config.grid_rows,
                    grid_cols=self.config.grid_cols,
                    invert_rows=self.config.invert_row_mapping
                )
                if cell is not None and 0 <= cell[0] < self.config.grid_rows and 0 <= cell[1] < self.config.grid_cols:
                    res = (cell, ground_pt, f"ACCEPTED: Mapped to Cell ({cell[0]}, {cell[1]})")
                    return res if return_diagnostics else (res[0], res[1])
                else:
                    res = (None, ground_pt, "REJECTED: Grid cell coordinates outside active grid dimensions")
                    return res if return_diagnostics else (res[0], res[1])
            else:
                res = (None, None, f"REJECTED: Contact point ({px}, {py}) projects outside reference plane metric bounds")
                return res if return_diagnostics else (res[0], res[1])

        # Mode B: Normalized ROI Linear Projection (fallback or explicit ROI mode)
        u = float(px) / float(frame_width)
        v = float(py) / float(frame_height)

        u_min, v_min, u_max, v_max = self.config.roi_norm
        if not (u_min <= u <= u_max and v_min <= v <= v_max):
            res = (None, None, f"REJECTED: Contact point ({px}, {py}) outside normalized ROI [({u_min:.2f},{v_min:.2f}) - ({u_max:.2f},{v_max:.2f})]")
            return res if return_diagnostics else (res[0], res[1])

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
            res = (None, None, "REJECTED: Grid cell out of bounds")
            return res if return_diagnostics else (res[0], res[1])

        res = ((row, col), None, f"ACCEPTED: Mapped via ROI to Cell ({row}, {col})")
        return res if return_diagnostics else (res[0], res[1])

    def map_detection_to_cell(
        self,
        detection: Detection,
        frame_width: int,
        frame_height: int
    ) -> Optional[Tuple[int, int]]:
        """Project a single detection into a grid coordinate (row, col)."""
        cell, _ = self.map_bbox_to_cell(
            bbox=detection.bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            class_name=detection.class_name,
            confidence=detection.confidence,
            return_diagnostics=False
        )
        return cell

    def inflate_cells(
        self,
        base_cells: Set[Tuple[int, int]],
        radius: Optional[int] = None
    ) -> Set[Tuple[int, int]]:
        """Inflate obstacle footprints to provide a safety margin around detected objects."""
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

    def estimate_obstacle_footprint(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        class_name: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """Estimate the physical ground footprint of a detected obstacle.

        Calculates:
        - Primary ground contact point and primary grid cell.
        - Physical lateral width (W_obs) in cm via homography transform of bottom bounding edge.
        - Estimated physical depth (D_obs) in cm.
        - Set of footprint grid cells covered by the obstacle base.
        - Set of safety-inflated grid cells.

        Returns:
            Dict containing:
                'primary_cell': (row, col) or None
                'footprint_cells': Set[(row, col)]
                'inflated_cells': Set[(row, col)]
                'physical_width_cm': float or None
                'physical_depth_cm': float or None
                'ground_coord': (gx, gy) or None
                'status': 'ACCEPTED' | 'REJECTED'
                'reason': Diagnostic string
        """
        raw_res = self.map_bbox_to_cell(
            bbox=bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            class_name=class_name,
            confidence=confidence,
            return_diagnostics=True
        )
        if isinstance(raw_res, tuple) and len(raw_res) == 3:
            primary_cell, ground_pt, status_reason = raw_res
        elif isinstance(raw_res, tuple) and len(raw_res) == 2:
            primary_cell, ground_pt = raw_res
            status_reason = f"ACCEPTED: Mapped to Cell {primary_cell}" if primary_cell is not None else "REJECTED: Out of bounds"
        else:
            primary_cell, ground_pt, status_reason = None, None, "REJECTED"

        if primary_cell is None:
            return {
                "primary_cell": None,
                "footprint_cells": set(),
                "inflated_cells": set(),
                "physical_width_cm": None,
                "physical_depth_cm": None,
                "ground_coord": None,
                "status": "REJECTED",
                "reason": status_reason,
            }

        x1, y1, x2, y2 = bbox
        footprint_cells: Set[Tuple[int, int]] = {primary_cell}
        w_obs: Optional[float] = None
        d_obs: Optional[float] = None

        if self.mapping_mode == MappingMode.HOMOGRAPHY and self.calibrator is not None and self.calibrator.is_calibrated:
            # Transform left and right bottom contact corners
            g_left = self.calibrator.transform_image_to_ground((float(x1), float(y2)), strict_bounds=False)
            g_right = self.calibrator.transform_image_to_ground((float(x2), float(y2)), strict_bounds=False)

            if g_left is not None and g_right is not None:
                dx = g_right[0] - g_left[0]
                dy = g_right[1] - g_left[1]
                w_obs = math.sqrt(dx * dx + dy * dy)
                # Bounded physical depth estimate
                d_obs = max(2.0, min(w_obs, 15.0))

                cell_w, cell_h = self.calibrator.get_cell_size_cm(self.config.grid_rows, self.config.grid_cols)
                ref_w = self.calibrator.config.reference_width
                ref_h = self.calibrator.config.reference_height

                gx_min = max(0.0, min(g_left[0], g_right[0]))
                gx_max = min(ref_w, max(g_left[0], g_right[0]))
                gy_base = ground_pt[1] if ground_pt is not None else (g_left[1] + g_right[1]) / 2.0
                gy_min = max(0.0, gy_base - d_obs / 2.0)
                gy_max = min(ref_h, gy_base + d_obs / 2.0)

                c_min = max(0, min(self.config.grid_cols - 1, int(math.floor((gx_min + 1e-5) / cell_w))))
                c_max = max(0, min(self.config.grid_cols - 1, int(math.floor((gx_max + 1e-5) / cell_w))))
                r_min = max(0, min(self.config.grid_rows - 1, int(math.floor((gy_min + 1e-5) / cell_h))))
                r_max = max(0, min(self.config.grid_rows - 1, int(math.floor((gy_max + 1e-5) / cell_h))))

                if self.config.invert_row_mapping:
                    r_min, r_max = (self.config.grid_rows - 1) - r_max, (self.config.grid_rows - 1) - r_min

                footprint_cells = {
                    (r, c)
                    for r in range(min(r_min, r_max), max(r_min, r_max) + 1)
                    for c in range(min(c_min, c_max), max(c_min, c_max) + 1)
                    if 0 <= r < self.config.grid_rows and 0 <= c < self.config.grid_cols
                }

        footprint_cells.add(primary_cell)
        inflated_cells = self.inflate_cells(footprint_cells)

        return {
            "primary_cell": primary_cell,
            "footprint_cells": footprint_cells,
            "inflated_cells": inflated_cells,
            "physical_width_cm": round(w_obs, 2) if w_obs is not None else None,
            "physical_depth_cm": round(d_obs, 2) if d_obs is not None else None,
            "ground_coord": ground_pt,
            "status": "ACCEPTED",
            "reason": status_reason,
        }

    def prune_stale_cells(self, current_time: float) -> Set[Tuple[int, int]]:
        """Remove tracks and cells whose last detection timestamp has exceeded the disappearance timeout."""
        timeout = max(self.config.disappearance_timeout_sec, self.config.stale_timeout_sec)
        expired_tracks = [tid for tid, last_seen in self._track_last_seen.items()
                          if (current_time - last_seen) > timeout]
        for tid in expired_tracks:
            self._track_last_seen.pop(tid, None)
            self._track_hit_counts.pop(tid, None)
            self._track_status.pop(tid, None)
            self._track_to_primary.pop(tid, None)
            old_cells = self._track_to_cells.pop(tid, set())
            for oc in old_cells:
                self._cell_to_track_id.pop(oc, None)
                self._cell_to_status.pop(oc, None)
                self._cell_to_primary.pop(oc, None)
                self._tracked_cells.pop(oc, None)
                self._cell_hit_counts.pop(oc, None)

        expired_cells = [cell for cell, last_seen in self._tracked_cells.items()
                         if (current_time - last_seen) > timeout]
        for cell in expired_cells:
            self._tracked_cells.pop(cell, None)
            self._cell_hit_counts.pop(cell, None)
            self._cell_to_track_id.pop(cell, None)
            self._cell_to_status.pop(cell, None)
            self._cell_to_primary.pop(cell, None)

        return set(self._tracked_cells.keys())

    def map_tracks(
        self,
        tracks: List[TrackedObject],
        frame_shape: Tuple[int, int],
        current_time: Optional[float] = None,
        vehicle_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None
    ) -> GridMappingResult:
        """Process TrackedObject instances into persistent, validated occupancy cells without flickering."""
        now = time.time() if current_time is None else current_time
        self.total_processed_frames += 1

        h, w = frame_shape
        candidate_cells: Set[Tuple[int, int]] = set()
        vehicle_conflicts: Set[Tuple[int, int]] = set()
        goal_conflicts: Set[Tuple[int, int]] = set()
        out_of_bounds_count = 0
        filtered_count = 0
        mapped_tracks_info: List[Dict[str, Any]] = []

        seen_track_ids: Set[int] = set()

        for track in tracks:
            seen_track_ids.add(track.track_id)
            is_lost = hasattr(track, "state") and track.state.value == "LOST"

            # Execute footprint estimation and mapping with full diagnostic reasoning
            fp_res = self.estimate_obstacle_footprint(
                bbox=track.bbox,
                frame_width=w,
                frame_height=h,
                class_name=track.class_name,
                confidence=track.confidence
            )
            cell = fp_res["primary_cell"]
            ground_pt = fp_res["ground_coord"]
            status_reason = fp_res["reason"]
            track_footprint = fp_res["footprint_cells"]
            track_inflated = fp_res["inflated_cells"]
            w_obs = fp_res["physical_width_cm"]
            d_obs = fp_res["physical_depth_cm"]

            cx, cy = self.get_contact_point(track.bbox, frame_height=h)

            # Hit tracking and confirmation
            if not is_lost and cell is not None:
                self._track_hit_counts[track.track_id] = self._track_hit_counts.get(track.track_id, 0) + 1
            hits = getattr(track, "hits", self._track_hit_counts.get(track.track_id, 1))
            is_confirmed = (hits >= self.config.confirmation_frames) or getattr(track, "is_confirmed", False)
            track_status = "REJECTED"

            if cell is not None and not is_lost:
                filtered_count += 1

                # Safety Invariant Check: strictly protect active vehicle position and destination goal
                is_safe = True
                if vehicle_pos is not None and cell == vehicle_pos:
                    vehicle_conflicts.add(cell)
                    self.total_vehicle_conflicts += 1
                    status_reason = f"REJECTED: Safety guard - cell {cell} overlaps vehicle position"
                    cell = None
                    is_safe = False
                elif goal_pos is not None and cell == goal_pos:
                    goal_conflicts.add(cell)
                    self.total_goal_conflicts += 1
                    status_reason = f"REJECTED: Safety guard - cell {cell} overlaps destination goal"
                    cell = None
                    is_safe = False

                if is_safe and cell is not None:
                    candidate_cells.add(cell)

                    # Determine active cells for this track (footprint + optional safety inflation)
                    if self.config.enable_footprint:
                        active_track_cells = set(track_inflated if self.config.inflation_radius > 0 else track_footprint)
                    else:
                        active_track_cells = set(self.inflate_cells({cell}))

                    if vehicle_pos is not None:
                        active_track_cells.discard(vehicle_pos)
                    if goal_pos is not None:
                        active_track_cells.discard(goal_pos)

                    # Moving object support: clean up previous cells if track moved to new cells
                    old_cells = self._track_to_cells.get(track.track_id, set()) - active_track_cells
                    for oc in old_cells:
                        self._cell_to_track_id.pop(oc, None)
                        self._cell_to_status.pop(oc, None)
                        self._cell_to_primary.pop(oc, None)
                        self._tracked_cells.pop(oc, None)

                    track_status = "CONFIRMED" if is_confirmed else "UNCONFIRMED"
                    self._track_status[track.track_id] = track_status
                    self._track_to_cells[track.track_id] = active_track_cells
                    self._track_to_primary[track.track_id] = cell
                    self._track_last_seen[track.track_id] = now

                    for fc in active_track_cells:
                        self._tracked_cells[fc] = now
                        self._cell_to_track_id[fc] = track.track_id
                        self._cell_to_status[fc] = track_status
                        self._cell_to_primary[fc] = (fc == cell)
            else:
                out_of_bounds_count += 1
                # Anti-flickering: Coasting logic for temporary missed detection or occlusion
                time_since_seen = now - self._track_last_seen.get(track.track_id, 0.0)
                if track.track_id in self._track_to_cells and time_since_seen <= self.config.disappearance_timeout_sec:
                    coasting_cells = self._track_to_cells[track.track_id]
                    candidate_cells.add(self._track_to_primary.get(track.track_id, next(iter(coasting_cells))))
                    track_status = "COASTING"
                    self._track_status[track.track_id] = track_status
                    status_reason = f"COASTING: Occluded / missed for {time_since_seen:.2f}s (<= {self.config.disappearance_timeout_sec}s timeout)"
                    for fc in coasting_cells:
                        self._cell_to_status[fc] = "COASTING"
                elif track.track_id in self._track_to_cells and time_since_seen > self.config.disappearance_timeout_sec:
                    old_cells = self._track_to_cells.pop(track.track_id, set())
                    self._track_to_primary.pop(track.track_id, None)
                    self._track_last_seen.pop(track.track_id, None)
                    self._track_status.pop(track.track_id, None)
                    for oc in old_cells:
                        self._cell_to_track_id.pop(oc, None)
                        self._cell_to_status.pop(oc, None)
                        self._cell_to_primary.pop(oc, None)
                        self._tracked_cells.pop(oc, None)
                    track_status = "EXPIRED"
                else:
                    track_status = "REJECTED"

            # Record comprehensive diagnostics for EVERY track
            mapped_tracks_info.append({
                "track_id": track.track_id,
                "class_name": track.class_name,
                "confidence": round(track.confidence, 4),
                "bbox": track.bbox,
                "ground_contact_pt": (cx, cy),
                "ground_coord": (round(ground_pt[0], 2), round(ground_pt[1], 2)) if ground_pt else None,
                "grid_cell": cell or self._track_to_primary.get(track.track_id),
                "footprint_cells": sorted(list(self._track_to_cells.get(track.track_id, track_footprint))),
                "inflated_cells": sorted(list(track_inflated)),
                "physical_width_cm": w_obs,
                "physical_depth_cm": d_obs,
                "status": self._track_status.get(track.track_id, track_status),
                "reason": status_reason,
                "track_age_frames": getattr(track, "age_frames", 1),
                "track_state": track.state.value if hasattr(track, "state") else "ACTIVE",
                "is_confirmed": is_confirmed,
            })

        # Also coast active tracks in memory that were absent from tracks list this frame
        for tid, last_seen in list(self._track_last_seen.items()):
            if tid not in seen_track_ids:
                time_since_seen = now - last_seen
                if time_since_seen <= self.config.disappearance_timeout_sec:
                    if tid in self._track_to_cells:
                        coasting_cells = self._track_to_cells[tid]
                        candidate_cells.add(self._track_to_primary.get(tid, next(iter(coasting_cells))))
                        self._track_status[tid] = "COASTING"
                        for fc in coasting_cells:
                            self._cell_to_status[fc] = "COASTING"
                else:
                    old_cells = self._track_to_cells.pop(tid, set())
                    self._track_to_primary.pop(tid, None)
                    self._track_last_seen.pop(tid, None)
                    self._track_status.pop(tid, None)
                    for oc in old_cells:
                        self._cell_to_track_id.pop(oc, None)
                        self._cell_to_status.pop(oc, None)
                        self._cell_to_primary.pop(oc, None)
                        self._tracked_cells.pop(oc, None)

        # 2. Prune stale cells from expired tracks
        self.prune_stale_cells(now)

        # 3. Aggregate all non-conflicting cells across all currently active tracks
        final_cells: Set[Tuple[int, int]] = set()
        footprint_cells: Set[Tuple[int, int]] = set()
        confirmed_cells: Set[Tuple[int, int]] = set()
        unconfirmed_cells: Set[Tuple[int, int]] = set()
        coasting_cells: Set[Tuple[int, int]] = set()

        for tid, cells in self._track_to_cells.items():
            t_status = self._track_status.get(tid, "CONFIRMED")
            for c in cells:
                if vehicle_pos is not None and c == vehicle_pos:
                    continue
                if goal_pos is not None and c == goal_pos:
                    continue
                final_cells.add(c)
                footprint_cells.add(c)
                if t_status == "CONFIRMED":
                    confirmed_cells.add(c)
                elif t_status == "UNCONFIRMED":
                    unconfirmed_cells.add(c)
                elif t_status == "COASTING":
                    coasting_cells.add(c)

        inflated_cells = self.inflate_cells(footprint_cells)

        confirmed_tracks_count = sum(1 for tid in self._track_to_cells if self._track_status.get(tid) == "CONFIRMED")
        mapped_objects_count = len(self._track_to_cells)
        rejected_count = sum(1 for info in mapped_tracks_info if info.get("status") == "REJECTED")

        return GridMappingResult(
            timestamp=now,
            raw_detections_count=len(tracks),
            filtered_detections_count=filtered_count,
            candidate_cells=candidate_cells,
            footprint_cells=footprint_cells,
            inflated_cells=inflated_cells,
            confirmed_cells=confirmed_cells,
            unconfirmed_cells=unconfirmed_cells,
            coasting_cells=coasting_cells,
            vehicle_conflicts=vehicle_conflicts,
            goal_conflicts=goal_conflicts,
            out_of_bounds_count=out_of_bounds_count,
            final_cells=final_cells,
            mapping_mode=self.mapping_mode.value,
            mapped_tracks_info=mapped_tracks_info,
            cell_to_track_id=dict(self._cell_to_track_id),
            cell_to_status=dict(self._cell_to_status),
            cell_to_primary=dict(self._cell_to_primary),
            confirmed_tracks_count=confirmed_tracks_count,
            mapped_objects_count=mapped_objects_count,
            rejected_detections_count=rejected_count,
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
        mapped_tracks_info: List[Dict[str, Any]] = []

        for idx, det in enumerate(detections):
            # Route through map_detection_to_cell so mock overrides work in unit tests
            cell = self.map_detection_to_cell(det, frame_width=w, frame_height=h)
            cx, cy = self.get_contact_point(det.bbox, frame_height=h)

            if cell is not None:
                filtered_count += 1
                if vehicle_pos is not None and cell == vehicle_pos:
                    vehicle_conflicts.add(cell)
                    self.total_vehicle_conflicts += 1
                    status_reason = f"REJECTED: Safety guard - cell {cell} overlaps vehicle position"
                    cell = None
                elif goal_pos is not None and cell == goal_pos:
                    goal_conflicts.add(cell)
                    self.total_goal_conflicts += 1
                    status_reason = f"REJECTED: Safety guard - cell {cell} overlaps destination goal"
                    cell = None
                else:
                    candidate_cells.add(cell)
                    self._tracked_cells[cell] = now
                    self._cell_to_track_id[cell] = idx + 1
                    status_reason = f"ACCEPTED: Mapped to Cell {cell}"
            else:
                out_of_bounds_count += 1
                status_reason = "REJECTED: Low confidence, wrong class, or out of bounds"

            mapped_tracks_info.append({
                "track_id": idx + 1,
                "class_name": det.class_name,
                "confidence": round(det.confidence, 4),
                "bbox": det.bbox,
                "ground_contact_pt": (cx, cy),
                "ground_coord": None,
                "grid_cell": cell,
                "status": "ACCEPTED" if cell is not None else "REJECTED",
                "reason": status_reason,
            })

        inflated_cells = self.inflate_cells(candidate_cells)
        for cell in inflated_cells:
            self._tracked_cells[cell] = now

        active_cells = self.prune_stale_cells(now)
        final_cells: Set[Tuple[int, int]] = set()
        for cell in active_cells:
            if vehicle_pos is not None and cell == vehicle_pos:
                vehicle_conflicts.add(cell)
                continue
            if goal_pos is not None and cell == goal_pos:
                goal_conflicts.add(cell)
                continue
            final_cells.add(cell)

        return GridMappingResult(
            timestamp=now,
            raw_detections_count=len(detections),
            filtered_detections_count=filtered_count,
            candidate_cells=candidate_cells,
            footprint_cells=candidate_cells,
            inflated_cells=inflated_cells,
            vehicle_conflicts=vehicle_conflicts,
            goal_conflicts=goal_conflicts,
            out_of_bounds_count=out_of_bounds_count,
            final_cells=final_cells,
            mapping_mode=self.mapping_mode.value,
            mapped_tracks_info=mapped_tracks_info,
            cell_to_track_id=dict(self._cell_to_track_id)
        )

    def clear(self) -> None:
        """Clear all tracked cells, history timestamps, and reset conflict counters."""
        self._tracked_cells.clear()
        self._track_to_cells.clear()
        self._track_to_primary.clear()
        self._track_last_seen.clear()
        self._track_hit_counts.clear()
        self._track_status.clear()
        self._cell_to_track_id.clear()
        self._cell_to_status.clear()
        self._cell_to_primary.clear()
        self._cell_hit_counts.clear()
        self.total_vehicle_conflicts = 0
        self.total_goal_conflicts = 0

    def get_active_cells(self) -> Set[Tuple[int, int]]:
        """Return the current set of tracked, non-stale obstacle coordinates."""
        return set(self._tracked_cells.keys())

    def get_cell_track_id(self, cell: Tuple[int, int]) -> Optional[int]:
        """Return the Track ID associated with a mapped obstacle cell."""
        return self._cell_to_track_id.get(cell)

    def draw_mapping_hud(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        mapping_result: Optional[GridMappingResult] = None,
    ) -> np.ndarray:
        """Render contact points, grid destination coordinates, rejection reasons, and diagnostic panel."""
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Draw ROI border if in NORMALIZED_ROI mode
        if self.mapping_mode == MappingMode.NORMALIZED_ROI:
            u_min, v_min, u_max, v_max = self.config.roi_norm
            rx1 = int(round(u_min * w))
            ry1 = int(round(v_min * h))
            rx2 = int(round(u_max * w))
            ry2 = int(round(v_max * h))
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Mapping ROI (Normalized)", (rx1 + 6, ry1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

        # Build diagnostic lookup for track results
        diag_map = {}
        if mapping_result is not None:
            for info in mapping_result.mapped_tracks_info:
                diag_map[info["track_id"]] = info

        # Draw ground contact point, badges, and diagnostic reasons on each track
        for track in tracks:
            cx, cy = self.get_contact_point(track.bbox, frame_height=h)
            x1, y1, x2, y2 = track.bbox
            diag = diag_map.get(track.track_id, {})
            status = diag.get("status", "UNCONFIRMED")
            cell = diag.get("grid_cell")
            ground_pt = diag.get("ground_coord")
            reason = diag.get("reason", "")
            is_confirmed = status == "CONFIRMED"
            is_unconfirmed = status == "UNCONFIRMED"
            is_coasting = status == "COASTING"
            is_accepted = is_confirmed or is_unconfirmed or is_coasting or status == "ACCEPTED"

            # Color scheme:
            # Confirmed: Emerald green (0, 240, 100)
            # Unconfirmed: Amber / Warm yellow (0, 190, 255)
            # Coasting: Soft lavender / violet (220, 140, 255)
            # Rejected: Vibrant coral / red (50, 50, 240)
            if is_confirmed:
                marker_color = (0, 240, 100)
                sub_col = (100, 255, 160)
            elif is_unconfirmed:
                marker_color = (0, 190, 255)
                sub_col = (140, 220, 255)
            elif is_coasting:
                marker_color = (220, 140, 255)
                sub_col = (240, 190, 255)
            else:
                marker_color = (50, 50, 240)
                sub_col = (120, 140, 255)

            # Ground contact baseline: horizontal bar along the bottom of the bounding box
            cv2.line(canvas, (x1, y2), (x2, y2), marker_color, 2, cv2.LINE_AA)

            # Contact point marker (outer black border, vibrant center)
            cv2.circle(canvas, (cx, cy), 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 4, marker_color, -1, cv2.LINE_AA)

            # Line from box center to ground contact point
            mid_y = int(round((y1 + y2) / 2.0))
            cv2.line(canvas, (cx, mid_y), (cx, cy), marker_color, 1, cv2.LINE_AA)

            # Build structured badge lines
            bx = max(4, min(x1, w - 240))
            by = min(h - 30, y2 + 18)

            w_obs = diag.get("physical_width_cm")
            width_str = f" | W:{w_obs:.1f}cm" if w_obs is not None else ""

            if is_confirmed and cell is not None:
                pt_str = f"Plane ({ground_pt[0]:.1f},{ground_pt[1]:.1f})cm{width_str}" if ground_pt else "ROI"
                badge_line1 = f"T{track.track_id} {track.class_name} ({track.confidence:.2f}) -> Cell ({cell[0]}, {cell[1]})"
                badge_line2 = f"CONFIRMED | {pt_str}"
            elif is_unconfirmed and cell is not None:
                age = diag.get("track_age_frames", 1)
                req = self.config.confirmation_frames
                badge_line1 = f"T{track.track_id}? {track.class_name} ({track.confidence:.2f}) -> Cell ({cell[0]}, {cell[1]})"
                badge_line2 = f"UNCONFIRMED ({age}/{req} frames) | Contact ({cx}, {cy})"
            elif is_coasting:
                badge_line1 = f"T{track.track_id}~ {track.class_name} [COASTING] -> Cell {cell}"
                badge_line2 = f"COASTING (Occluded/Missed)"
            else:
                badge_line1 = f"T{track.track_id} {track.class_name} ({track.confidence:.2f}) [REJECTED]"
                short_reason = reason.replace("REJECTED: ", "")[:35]
                badge_line2 = f"{short_reason}"

            # Render 2-line badge card
            (tw1, th1), _ = cv2.getTextSize(badge_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            (tw2, th2), _ = cv2.getTextSize(badge_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            card_w = max(tw1, tw2) + 10
            card_h = th1 + th2 + 12

            cv2.rectangle(canvas, (bx - 3, by - th1 - 2), (bx + card_w, by + th2 + 8), (15, 20, 25), -1)
            cv2.rectangle(canvas, (bx - 3, by - th1 - 2), (bx + card_w, by + th2 + 8), marker_color, 1)

            cv2.putText(canvas, badge_line1, (bx + 2, by), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, badge_line2, (bx + 2, by + th2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, sub_col, 1, cv2.LINE_AA)

        # Header Status Banner (Bottom of Screen)
        mode_label = "HOMOGRAPHY (Calibrated)" if self.mapping_mode == MappingMode.HOMOGRAPHY else "NORMALIZED ROI (Full View)"
        raw_count = mapping_result.raw_detections_count if mapping_result else len(tracks)
        conf_count = mapping_result.confirmed_tracks_count if mapping_result else 0
        mapped_count = mapping_result.mapped_objects_count if mapping_result else len(self._track_to_cells)
        cell_count = len(mapping_result.final_cells) if mapping_result else len(self._tracked_cells)
        rej_count = mapping_result.rejected_detections_count if mapping_result else 0

        banner_text = f"Det: {raw_count} | Confirmed: {conf_count} | Mapped: {mapped_count} obj ({cell_count} cells) | Rej: {rej_count} | Mode: {mode_label}"
        cv2.rectangle(canvas, (10, h - 34), (w - 10, h - 8), (15, 20, 25), -1)
        cv2.rectangle(canvas, (10, h - 34), (w - 10, h - 8), (0, 200, 255), 1)
        cv2.putText(canvas, banner_text, (18, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        return canvas
