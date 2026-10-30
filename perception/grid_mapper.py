"""Simplified grid-based obstacle mapping from 2D YOLO detections.

Translates 2D perception bounding boxes into candidate terrain grid cells for
academic and laboratory demonstration.

Theoretical & Architectural Assumptions:
-----------------------------------------
1. Dimensionality Gap:
   A standard monocular RGB camera captures a 2D projection of a 3D environment.
   A 2D bounding box (x1, y1, x2, y2) in pixel space does NOT provide real-world metric
   depth (Z distance) or metric lateral displacement (X distance). It is mathematically
   ill-posed to directly equate 2D image pixel coordinates to physical ground-plane coordinates
   without extrinsic camera calibration (pitch, yaw, roll, camera height), flat-road planar
   homography (Inverse Perspective Mapping / IPM), or active depth sensors (LiDAR, stereo cameras).

2. Explicit Simplified Demonstration Mapping:
   For this classroom demonstration, we implement an explicitly simplified, documented mapping
   model designed for a controlled, fixed-camera scene (such as a laboratory tabletop track,
   a fixed benchtop test rig, or a controlled forward-facing viewport). An active Region
   of Interest (ROI) is defined within the normalized image viewport, and detection contact
   points (the bottom-center of bounding boxes where objects touch the ground) are projected
   onto the discrete terrain grid coordinates.
   **This mapping is explicitly labeled as APPROXIMATE DEMONSTRATION MAPPING, not real-world
   autonomous vehicle metric localization.**

3. Safety Invariants:
   - Vehicle & Goal Protection: Candidate perception obstacle cells that overlap the active
     vehicle position or navigation destination are strictly rejected and masked out. The vehicle
     is NEVER trapped inside an obstacle cell, and the goal is NEVER overwritten.
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
from typing import Dict, List, Optional, Set, Tuple
import logging
import time

from perception.object_detector import Detection

logger = logging.getLogger(__name__)


class MappingMode(str, Enum):
    """Supported perception-to-grid mapping modes."""
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


@dataclass
class GridMappingResult:
    """Structured report returned after processing a perception mapping cycle.

    Attributes:
        timestamp: Time of the mapping cycle.
        raw_detections_count: Total number of raw detections provided.
        filtered_detections_count: Number of detections passing confidence & class filters.
        candidate_cells: Set of mapped base grid coordinates (row, col).
        inflated_cells: Set of candidate cells after safety margin inflation.
        vehicle_conflicts: Cells dropped because they coincided with the vehicle's position.
        goal_conflicts: Cells dropped because they coincided with the destination position.
        out_of_bounds_count: Detections rejected due to falling outside image/ROI/grid bounds.
        final_cells: Clean, validated, conflict-free obstacle cells ready for the grid.
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


class GridMapper:
    """Maps structured 2D visual detections onto a discrete 2D simulation grid.

    Decoupled from camera hardware and neural network inference; operates purely
    on structured Detection instances and image frame dimensions.
    """

    def __init__(self, config: Optional[GridMapperConfig] = None):
        """Initialize the grid mapper with configuration parameters.

        Args:
            config: Optional GridMapperConfig instance. Defaults to standard configuration.
        """
        self.config: GridMapperConfig = config or GridMapperConfig()

        # Cache grid boundaries
        if self.config.grid_roi is not None:
            self.r_min, self.c_min, self.r_max, self.c_max = self.config.grid_roi
        else:
            self.r_min, self.c_min = 0, 0
            self.r_max = self.config.grid_rows - 1
            self.c_max = self.config.grid_cols - 1

        # Stale tracking table: maps (row, col) -> last_seen_timestamp (seconds)
        self._tracked_cells: Dict[Tuple[int, int], float] = {}

        # Telemetry
        self.total_processed_frames: int = 0
        self.total_vehicle_conflicts: int = 0
        self.total_goal_conflicts: int = 0

    def get_contact_point(self, detection: Detection) -> Tuple[int, int]:
        """Compute the 2D pixel coordinate representing the object's ground contact point.

        In camera geometry, the bottom edge of a 2D bounding box represents the contact
        boundary where an object touches the physical ground plane.

        Args:
            detection: Detection instance with bbox (x1, y1, x2, y2).

        Returns:
            Tuple[int, int]: (px, py) pixel coordinate.
        """
        x1, y1, x2, y2 = detection.bbox
        cx = (x1 + x2) // 2

        if self.config.contact_point == "center":
            cy = (y1 + y2) // 2
            return (cx, cy)

        # Default: 'bottom_center'
        return (cx, y2)

    def map_detection_to_cell(
        self,
        detection: Detection,
        frame_width: int,
        frame_height: int
    ) -> Optional[Tuple[int, int]]:
        """Project a single detection into a grid coordinate (row, col).

        Validates confidence, target classes, image boundaries, and the active ROI.

        Args:
            detection: Detection record from local YOLO detector.
            frame_width: Width of the source image in pixels.
            frame_height: Height of the source image in pixels.

        Returns:
            Optional[Tuple[int, int]]: Mapped (row, col) cell, or None if rejected.
        """
        # 1. Confidence threshold check
        if detection.confidence < self.config.min_confidence:
            return None

        # 2. Target class check
        if self.config.target_classes is not None:
            if detection.class_name not in self.config.target_classes:
                return None

        # 3. Frame dimension validation
        if frame_width <= 0 or frame_height <= 0:
            return None

        # 4. Compute contact point
        px, py = self.get_contact_point(detection)

        # 5. Normalize pixel coordinate to [0.0, 1.0]
        u = px / float(frame_width)
        v = py / float(frame_height)

        # 6. Region of Interest (ROI) filtering
        u_min, v_min, u_max, v_max = self.config.roi_norm
        if not (u_min <= u <= u_max and v_min <= v <= v_max):
            logger.debug(
                "Detection '%s' contact (%.3f, %.3f) outside ROI [%.2f, %.2f, %.2f, %.2f]",
                detection.class_name, u, v, u_min, v_min, u_max, v_max
            )
            return None

        # 7. Normalized interpolation within ROI
        roi_w = max(1e-6, u_max - u_min)
        roi_h = max(1e-6, v_max - v_min)
        norm_x = max(0.0, min(1.0, (u - u_min) / roi_w))
        norm_y = max(0.0, min(1.0, (v - v_min) / roi_h))

        # 8. Linear projection to discrete grid span [r_min, r_max], [c_min, c_max]
        col_span = self.c_max - self.c_min
        row_span = self.r_max - self.r_min

        col = int(round(self.c_min + norm_x * col_span))
        if self.config.invert_row_mapping:
            row = int(round(self.r_max - norm_y * row_span))
        else:
            row = int(round(self.r_min + norm_y * row_span))

        # 9. Strict grid bounds clamp & validation
        if not (0 <= row < self.config.grid_rows and 0 <= col < self.config.grid_cols):
            logger.warning("Projected coordinate (%d, %d) is outside grid bounds.", row, col)
            return None

        return (row, col)

    def inflate_cells(
        self,
        base_cells: Set[Tuple[int, int]],
        radius: Optional[int] = None
    ) -> Set[Tuple[int, int]]:
        """Inflate obstacle footprints to provide a safety margin around detected objects.

        Uses Chebyshev (square) neighborhood expansion within the valid grid dimensions.

        Args:
            base_cells: Set of (row, col) coordinates to inflate.
            radius: Inflation radius in cells. If None, uses config.inflation_radius.

        Returns:
            Set[Tuple[int, int]]: Union of base cells and their valid surrounding neighbors.
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
        """Remove cells whose last detection timestamp has exceeded the stale timeout.

        Args:
            current_time: Current timestamp in seconds (e.g. from time.time()).

        Returns:
            Set[Tuple[int, int]]: The remaining active, non-stale cell set.
        """
        expired: List[Tuple[int, int]] = []
        for cell, last_seen in self._tracked_cells.items():
            if (current_time - last_seen) > self.config.stale_timeout_sec:
                expired.append(cell)

        for cell in expired:
            del self._tracked_cells[cell]

        return set(self._tracked_cells.keys())

    def map_detections(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int],
        current_time: Optional[float] = None,
        vehicle_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None
    ) -> GridMappingResult:
        """Process a list of detections into validated, conflict-free grid obstacle cells.

        Guarantees:
        - Out-of-bounds detections are counted and discarded.
        - Low-confidence detections are filtered.
        - Safety margin inflation is applied if configured.
        - Vehicle cell is protected: candidate obstacles overlapping the vehicle are rejected.
        - Goal cell is protected: candidate obstacles overlapping the destination are rejected.
        - Stale detections are tracked and pruned automatically.

        Args:
            detections: List of Detection instances from local detector.
            frame_shape: (height, width) of the image frame.
            current_time: Optional timestamp in seconds. Defaults to time.time().
            vehicle_pos: Optional active (row, col) of the autonomous vehicle to protect.
            goal_pos: Optional (row, col) of the target destination to protect.

        Returns:
            GridMappingResult: Structured report with final cells and conflict diagnostics.
        """
        now = time.time() if current_time is None else current_time
        self.total_processed_frames += 1

        h, w = frame_shape
        candidate_cells: Set[Tuple[int, int]] = set()
        vehicle_conflicts: Set[Tuple[int, int]] = set()
        goal_conflicts: Set[Tuple[int, int]] = set()
        out_of_bounds_count = 0
        filtered_count = 0

        # 1. Map individual detections to base grid cells
        for det in detections:
            # Check basic filtering
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

        # 2. Apply obstacle inflation / safety margins
        inflated_cells = self.inflate_cells(candidate_cells)

        # 3. Update freshness timestamps for all currently detected/inflated cells
        for cell in inflated_cells:
            self._tracked_cells[cell] = now

        # 4. Prune stale cells whose timeout has elapsed
        active_cells = self.prune_stale_cells(now)

        # 5. Enforce safety invariants: Never overwrite vehicle or goal cells
        final_cells: Set[Tuple[int, int]] = set()
        for cell in active_cells:
            if vehicle_pos is not None and cell == vehicle_pos:
                vehicle_conflicts.add(cell)
                self.total_vehicle_conflicts += 1
                logger.warning(
                    "Perception conflict: candidate obstacle overlaps vehicle position %s. Suppressing cell.",
                    cell
                )
                continue

            if goal_pos is not None and cell == goal_pos:
                goal_conflicts.add(cell)
                self.total_goal_conflicts += 1
                logger.warning(
                    "Perception conflict: candidate obstacle overlaps navigation goal %s. Suppressing cell.",
                    cell
                )
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
            final_cells=final_cells
        )

    def clear(self) -> None:
        """Clear all tracked cells, history timestamps, and reset conflict counters."""
        self._tracked_cells.clear()
        self.total_vehicle_conflicts = 0
        self.total_goal_conflicts = 0

    def get_active_cells(self) -> Set[Tuple[int, int]]:
        """Return the current set of tracked, non-stale obstacle coordinates."""
        return set(self._tracked_cells.keys())
