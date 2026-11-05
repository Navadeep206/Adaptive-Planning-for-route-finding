"""Camera Calibration and Planar Homography Module for Robotic Perception.

Establishes a mathematical projective transformation (homography) between 2D camera image
pixel coordinates and a 2D tabletop/ground reference plane for autonomous robot navigation.

Mathematical Foundations & Assumptions:
----------------------------------------
1. Monocular Ground-Plane Projection (Homography):
   A standard monocular pinhole camera projects a 3D ray onto a 2D pixel (u, v).
   In the general 3D case, metric depth Z is lost. However, when objects are assumed to rest
   upon a common planar surface (such as a tabletop demonstration track or flat road),
   the mapping between camera pixels (u, v) and ground-plane coordinates (X_g, Y_g) is given
   by a 3x3 projective homography matrix H:

       s * [X_g, Y_g, 1]^T = H * [u, v, 1]^T

2. Ground-Contact Point Justification:
   For upright physical objects (bottles, boxes, barrier props, pedestrians) standing vertically
   on the road surface, the visual center of the 2D bounding box (cx, cy) is elevated in 3D
   space above the road. Projecting (cx, cy) would introduce severe parallax error (foreshortening),
   causing the object to appear farther away than it physically is.
   The bottom-center coordinate (cx, y_2) represents the physical boundary where the object
   makes contact with the ground plane. Therefore, (cx, y_2) is the geometrically correct proxy
   for planar homography projection.

3. Metric Boundary & Real-World Limitations:
   Homography maps strictly to the defined 2D reference plane. It does NOT produce full 3D LiDAR-like
   point clouds, cannot account for terrain elevation changes (hills, ditches), and requires
   known geometric fiducial coordinates. Objects not resting on the plane (such as people, faces,
   hands, or elevated props) must be strictly rejected.

4. Documented Coordinate Convention:
   - Image pixel space: u in [0, W_img) (horizontal), v in [0, H_img) (vertical).
   - Physical ground space: X_g in [0, W_cm] (lateral), Y_g in [0, H_cm] (longitudinal).
   - Physical origin (0.0 cm, 0.0 cm): Top-Left (Far-Left) of the calibrated surface.
   - Discrete grid space:
       cell_width_cm  = reference_width_cm / grid_cols
       cell_height_cm = reference_height_cm / grid_rows
       column = floor(X_g / cell_width_cm)
       row    = floor(Y_g / cell_height_cm)
"""

from dataclasses import dataclass, field
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationPointPair:
    """Association between an image pixel coordinate and a known ground-plane coordinate."""
    image_coord: Tuple[float, float]    # (u, v) in pixels
    ground_coord: Tuple[float, float]   # (X_g, Y_g) in reference plane units (e.g. cm or normalized [0, 1])
    label: str = "Fiducial"


@dataclass
class CalibrationConfig:
    """Configuration parameters for planar camera calibration."""
    reference_width: float = 100.0          # Metric ground plane width span (X_g in cm)
    reference_height: float = 80.0          # Metric ground plane height span (Y_g in cm)
    max_reprojection_error_px: float = 10.0 # Maximum allowed mean reprojection error in pixels
    min_points_required: int = 4            # Minimum non-collinear point pairs required for homography
    default_fiducials: List[CalibrationPointPair] = field(default_factory=list)


class CameraCalibrator:
    """Computes, validates, and applies planar homography transformations.

    Transforms detected object ground-contact points into reference plane coordinates,
    validates reprojection errors, verifies point ordering, and enforces strict boundary guards.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        """Initialize the camera calibrator.

        Args:
            config: Optional CalibrationConfig. If None, default tabletop reference is used.
        """
        self.config: CalibrationConfig = config or CalibrationConfig()
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.is_calibrated: bool = False
        self.reprojection_error_px: float = float("inf")
        self.ground_error_cm: float = float("inf")
        self.condition_number: float = 1.0
        self.calibration_warning: Optional[str] = None
        self.active_fiducials: List[CalibrationPointPair] = []
        self.show_instructions: bool = True

        self.calib_width: int = 640
        self.calib_height: int = 480
        self._base_width: int = 640
        self._base_height: int = 480
        self._base_fiducials: List[CalibrationPointPair] = []

        # ArUco Dynamic Calibration Tracking State (State-of-the-Art auto-tracking)
        self.is_aruco_tracking: bool = False
        self.aruco_detected_count: int = 0
        self.aruco_detected_markers: Dict[int, Tuple[float, float]] = {}
        self.aruco_corners_raw: List[np.ndarray] = []
        self._init_aruco_detector()

        # If default fiducials provided or empty, initialize standard laboratory fiducials
        if not self.config.default_fiducials:
            self._init_default_tabletop_fiducials()
        else:
            self.active_fiducials = list(self.config.default_fiducials)
            self._base_fiducials = [
                CalibrationPointPair(image_coord=p.image_coord, ground_coord=p.ground_coord, label=p.label)
                for p in self.active_fiducials
            ]
            self.compute_homography()

    def _init_default_tabletop_fiducials(self) -> None:
        """Initialize standard 4-corner tabletop reference calibration points for 640x480 camera."""
        # 4 corners of tabletop road track visible in standard forward-facing webcam perspective
        # Ordering: P1=Top-Left, P2=Top-Right, P3=Bottom-Right, P4=Bottom-Left
        self.active_fiducials = [
            CalibrationPointPair(image_coord=(100.0, 120.0), ground_coord=(0.0, 0.0), label="P1: TL (Far-Left)"),
            CalibrationPointPair(image_coord=(540.0, 120.0), ground_coord=(100.0, 0.0), label="P2: TR (Far-Right)"),
            CalibrationPointPair(image_coord=(580.0, 440.0), ground_coord=(100.0, 80.0), label="P3: BR (Near-Right)"),
            CalibrationPointPair(image_coord=(60.0, 440.0), ground_coord=(0.0, 80.0), label="P4: BL (Near-Left)"),
        ]
        self._base_fiducials = [
            CalibrationPointPair(image_coord=p.image_coord, ground_coord=p.ground_coord, label=p.label)
            for p in self.active_fiducials
        ]
        self._base_width = 640
        self._base_height = 480
        self.calib_width = 640
        self.calib_height = 480
        self.compute_homography()

    def _init_aruco_detector(self) -> None:
        """Initialize OpenCV ArUco detector for real-time dynamic calibration."""
        try:
            self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self._aruco_params = cv2.aruco.DetectorParameters()
            self._aruco_params.adaptiveThreshWinSizeMin = 3
            self._aruco_params.adaptiveThreshWinSizeMax = 23
            self._aruco_params.adaptiveThreshWinSizeStep = 10
            if hasattr(cv2.aruco, "ArucoDetector"):
                self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
            else:
                self._aruco_detector = None
        except Exception as exc:
            logger.warning("ArUco module initialization failed: %s", exc)
            self._aruco_dict = None
            self._aruco_detector = None

    def detect_and_update_aruco(
        self,
        frame: np.ndarray
    ) -> Tuple[bool, int, Dict[int, Tuple[float, float]]]:
        """Detect 4-corner ArUco fiducials and dynamically update homography matrix in real time.

        Fiducials:
            ID 0: Top-Left (Far-Left)      -> (0.0, 0.0) cm
            ID 1: Top-Right (Far-Right)    -> (reference_width, 0.0) cm
            ID 2: Bottom-Right (Near-Right)-> (reference_width, reference_height) cm
            ID 3: Bottom-Left (Near-Left)  -> (0.0, reference_height) cm

        Returns:
            Tuple[bool, int, Dict[int, Tuple[float, float]]]:
                (tracking_active, num_markers_found, marker_centers_dict)
        """
        if frame is None or frame.size == 0 or getattr(self, "_aruco_dict", None) is None:
            return False, 0, {}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        try:
            if self._aruco_detector is not None:
                corners, ids, _ = self._aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._aruco_params)
        except Exception as exc:
            logger.debug("ArUco detection error: %s", exc)
            return False, 0, {}

        self.aruco_corners_raw = list(corners) if corners else []
        marker_centers: Dict[int, Tuple[float, float]] = {}

        if ids is not None and len(ids) > 0:
            flat_ids = ids.flatten().tolist()
            for idx, mid in enumerate(flat_ids):
                if mid in (0, 1, 2, 3):
                    c = corners[idx][0]  # Shape: (4, 2)
                    cx = float(np.mean(c[:, 0]))
                    cy = float(np.mean(c[:, 1]))
                    marker_centers[mid] = (cx, cy)

        self.aruco_detected_count = len(marker_centers)
        self.aruco_detected_markers = dict(marker_centers)

        # When all 4 canonical markers (0, 1, 2, 3) are found, dynamically update homography
        if len(marker_centers) == 4 and all(k in marker_centers for k in (0, 1, 2, 3)):
            rw = self.config.reference_width
            rh = self.config.reference_height

            new_fiducials = [
                CalibrationPointPair(image_coord=marker_centers[0], ground_coord=(0.0, 0.0), label="ArUco 0: TL"),
                CalibrationPointPair(image_coord=marker_centers[1], ground_coord=(rw, 0.0), label="ArUco 1: TR"),
                CalibrationPointPair(image_coord=marker_centers[2], ground_coord=(rw, rh), label="ArUco 2: BR"),
                CalibrationPointPair(image_coord=marker_centers[3], ground_coord=(0.0, rh), label="ArUco 3: BL"),
            ]

            img_pts = np.array([p.image_coord for p in new_fiducials], dtype=np.float32)
            is_valid, err_msg = self.validate_point_ordering(img_pts)
            if is_valid:
                self.active_fiducials = new_fiducials
                self.calib_width = frame.shape[1]
                self.calib_height = frame.shape[0]
                self.compute_homography()
                self.is_aruco_tracking = True
                return True, 4, marker_centers
            else:
                self.is_aruco_tracking = False
                logger.debug("ArUco quadrilateral ordering invalid: %s", err_msg)
                return False, 4, marker_centers

        self.is_aruco_tracking = False
        return False, len(marker_centers), marker_centers

    def adapt_to_resolution(self, width: int, height: int) -> bool:
        """Dynamically scale fiducial coordinates to match the active camera resolution.

        Scales proportionally from base calibration fiducials to prevent drift.

        Args:
            width: Target frame width in pixels.
            height: Target frame height in pixels.

        Returns:
            bool: True if scaled and recomputed successfully.
        """
        if width <= 0 or height <= 0:
            return False
        if width == self.calib_width and height == self.calib_height:
            return True  # Already adapted

        # Compute scaling factors from base fiducials
        base_w = self._base_width if self._base_width > 0 else 640
        base_h = self._base_height if self._base_height > 0 else 480
        scale_x = float(width) / float(base_w)
        scale_y = float(height) / float(base_h)

        scaled_fiducials = []
        source_fiducials = self._base_fiducials if self._base_fiducials else self.active_fiducials
        for p in source_fiducials:
            sx = p.image_coord[0] * scale_x
            sy = p.image_coord[1] * scale_y
            scaled_fiducials.append(
                CalibrationPointPair(
                    image_coord=(sx, sy),
                    ground_coord=p.ground_coord,
                    label=p.label
                )
            )

        self.calib_width = width
        self.calib_height = height
        self.active_fiducials = scaled_fiducials
        success, _ = self.compute_homography()
        logger.info(
            "CameraCalibrator adapted to resolution %dx%d (scale: x=%.2f, y=%.2f, calibrated=%s).",
            width, height, scale_x, scale_y, self.is_calibrated
        )
        return success

    def set_fiducials(self, fiducials: List[CalibrationPointPair]) -> Tuple[bool, float]:
        """Update calibration fiducials and recompute homography matrix.

        Args:
            fiducials: List of at least 4 non-collinear CalibrationPointPair instances.

        Returns:
            Tuple[bool, float]: (is_valid, mean_reprojection_error_px)
        """
        self.active_fiducials = list(fiducials)
        self._base_fiducials = [
            CalibrationPointPair(image_coord=p.image_coord, ground_coord=p.ground_coord, label=p.label)
            for p in self.active_fiducials
        ]
        self._base_width = self.calib_width
        self._base_height = self.calib_height
        return self.compute_homography()

    def validate_point_ordering(self, img_pts: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Validate that fiducials form a strictly convex, non-self-intersecting quadrilateral in canonical order.

        Canonical order:
            P1: Top-Left (Far-Left)
            P2: Top-Right (Far-Right)
            P3: Bottom-Right (Near-Right)
            P4: Bottom-Left (Near-Left)

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message_if_invalid)
        """
        if len(img_pts) < 4:
            return False, f"Insufficient points: {len(img_pts)} provided, minimum 4 required."

        # Convert to contour format (4, 1, 2)
        contour = img_pts[:4].astype(np.float32).reshape(-1, 1, 2)

        # 1. Check convexity (detects hourglass / self-intersecting quads and collinear points)
        if not cv2.isContourConvex(contour.astype(np.int32)):
            return False, "Fiducials do not form a convex quadrilateral (crossing diagonals or concave order)."

        # 2. Check area to prevent degenerate or collapsed polygon
        area = cv2.contourArea(contour)
        if area < 100.0:
            return False, f"Degenerate polygon: area too small ({area:.1f} px^2)."

        # 3. Check canonical ordering: P1 (TL), P2 (TR), P3 (BR), P4 (BL)
        u1, v1 = img_pts[0]
        u2, v2 = img_pts[1]
        u3, v3 = img_pts[2]
        u4, v4 = img_pts[3]

        # Horizontal ordering: P1 should be left of P2, P4 left of P3
        if not (u1 < u2 and u4 < u3):
            return False, "Point ordering error: horizontal orientation inverted (must be P1:TL, P2:TR, P3:BR, P4:BL)."

        # Vertical ordering: top points (P1, P2) should be above bottom points (P4, P3) in image space
        if not (v1 < v4 and v2 < v3):
            return False, "Point ordering error: vertical perspective inverted (must be Far/Top: P1,P2 and Near/Bottom: P4,P3)."

        return True, None

    def is_point_inside_plane(self, image_pt: Tuple[float, float], tolerance_px: float = 1.0) -> bool:
        """Check whether a 2D image pixel coordinate lies within the calibrated ground quadrilateral.

        Args:
            image_pt: (u, v) image pixel coordinate.
            tolerance_px: Margin in pixels (default 1.0) to accommodate boundary points and floating-point roundoff.

        Returns:
            bool: True if strictly inside or within tolerance of the calibration quad boundary.
        """
        if not self.is_calibrated or len(self.active_fiducials) < 4:
            return False

        pts = np.array([p.image_coord for p in self.active_fiducials[:4]], dtype=np.float32)
        u, v = float(image_pt[0]), float(image_pt[1])
        # cv2.pointPolygonTest with measureDist=True returns positive distance inside, negative outside
        dist = cv2.pointPolygonTest(pts, (u, v), True)
        return dist >= -tolerance_px

    def compute_homography(self) -> Tuple[bool, float]:
        """Estimate 3x3 homography matrix H from active fiducial point pairs.

        Returns:
            Tuple[bool, float]: (success, reprojection_error_px)
        """
        if len(self.active_fiducials) < self.config.min_points_required:
            msg = f"Cannot compute homography: {len(self.active_fiducials)} fiducials provided (minimum {self.config.min_points_required} required)."
            logger.warning(msg)
            self.calibration_warning = msg
            self._invalidate_calibration()
            return False, float("inf")

        img_pts = np.array([p.image_coord for p in self.active_fiducials], dtype=np.float32)
        gnd_pts = np.array([p.ground_coord for p in self.active_fiducials], dtype=np.float32)

        # Check for collinearity / degeneracy
        if self._check_collinear(img_pts) or self._check_collinear(gnd_pts):
            msg = "Fiducial points are collinear or degenerate. Homography computation rejected."
            logger.warning(msg)
            self.calibration_warning = msg
            self._invalidate_calibration()
            return False, float("inf")

        # Validate point ordering and quadrilateral convexity
        is_order_valid, order_err = self.validate_point_ordering(img_pts)
        if not is_order_valid:
            logger.warning("Fiducial ordering validation failed: %s", order_err)
            self.calibration_warning = order_err
            self._invalidate_calibration()
            return False, float("inf")

        try:
            # If exactly 4 points, use exact perspective transform; otherwise RANSAC
            if len(self.active_fiducials) == 4:
                H = cv2.getPerspectiveTransform(img_pts, gnd_pts)
            else:
                H, _ = cv2.findHomography(img_pts, gnd_pts, cv2.RANSAC, 5.0)

            if H is None or H.shape != (3, 3):
                msg = "cv2.findHomography returned None or invalid shape."
                logger.error(msg)
                self.calibration_warning = msg
                self._invalidate_calibration()
                return False, float("inf")

            # Validate matrix conditioning (determinant check)
            det = np.linalg.det(H)
            if abs(det) < 1e-9 or math.isnan(det):
                msg = f"Homography matrix is near-singular (det={det:.2e}). Rejected."
                logger.warning(msg)
                self.calibration_warning = msg
                self._invalidate_calibration()
                return False, float("inf")

            cond = float(np.linalg.cond(H))
            self.condition_number = cond

            # Compute bidirectional reprojection error: pixel error and ground error
            pixel_err, gnd_err = self._evaluate_reprojection_error(H, img_pts, gnd_pts)
            self.reprojection_error_px = pixel_err
            self.ground_error_cm = gnd_err

            if pixel_err > self.config.max_reprojection_error_px:
                msg = f"Reprojection error ({pixel_err:.2f} px) exceeds threshold ({self.config.max_reprojection_error_px:.2f} px)."
                logger.warning(msg)
                self.calibration_warning = msg
                self._invalidate_calibration()
                return False, pixel_err

            # Successfully validated
            self.homography_matrix = H
            self.inverse_homography = np.linalg.inv(H)
            self.is_calibrated = True
            self.calibration_warning = None

            logger.info(
                "Planar homography calibration successful (%d points, error=%.2f px, det=%.2e, cond=%.1f).",
                len(self.active_fiducials),
                pixel_err,
                det,
                cond
            )
            return True, pixel_err

        except Exception as exc:
            msg = f"Exception during homography computation: {exc}"
            logger.error(msg, exc_info=True)
            self.calibration_warning = msg
            self._invalidate_calibration()
            return False, float("inf")

    def _invalidate_calibration(self) -> None:
        """Reset internal matrices upon validation failure."""
        self.homography_matrix = None
        self.inverse_homography = None
        self.is_calibrated = False
        self.reprojection_error_px = float("inf")
        self.ground_error_cm = float("inf")

    @staticmethod
    def _check_collinear(pts: np.ndarray) -> bool:
        """Check if 3 or more points in a set are collinear."""
        if len(pts) < 3:
            return True
        # If points have zero variance in X or Y
        if np.std(pts[:, 0]) < 1e-4 or np.std(pts[:, 1]) < 1e-4:
            return True
        # Triangle area check for consecutive triplets
        for i in range(len(pts) - 2):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            x3, y3 = pts[i + 2]
            area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            if area < 1e-3:
                return True
        return False

    def _evaluate_reprojection_error(
        self,
        H: np.ndarray,
        img_pts: np.ndarray,
        gnd_pts: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate mean Euclidean error between ground points and projected image points.

        Returns:
            Tuple[float, float]: (mean_pixel_reprojection_error, mean_ground_error_cm)
        """
        # Ground -> Image projection via H_inv
        H_inv = np.linalg.inv(H)
        src_gnd = gnd_pts.reshape(-1, 1, 2)
        projected_img = cv2.perspectiveTransform(src_gnd, H_inv).reshape(-1, 2)
        pixel_errors = np.linalg.norm(img_pts - projected_img, axis=1)
        mean_pixel_err = float(np.mean(pixel_errors))

        # Image -> Ground projection via H
        src_img = img_pts.reshape(-1, 1, 2)
        projected_gnd = cv2.perspectiveTransform(src_img, H).reshape(-1, 2)
        ground_errors = np.linalg.norm(gnd_pts - projected_gnd, axis=1)
        mean_ground_err = float(np.mean(ground_errors))

        return mean_pixel_err, mean_ground_err

    @staticmethod
    def get_ground_contact_point(
        bbox: Tuple[int, int, int, int],
        method: str = "bottom_center"
    ) -> Tuple[float, float]:
        """Compute the physical ground-contact proxy point from a 2D bounding box.

        Justification:
        For upright tabletop props and obstacles, the bottom-center (cx, y2) represents
        the contact locus where the object rests upon the ground plane, eliminating
        vertical parallax error.
        """
        x1, y1, x2, y2 = bbox
        cx = float((x1 + x2) / 2.0)

        if method == "center":
            cy = float((y1 + y2) / 2.0)
            return (cx, cy)

        # Default 'bottom_center'
        return (cx, float(y2))

    def transform_image_to_ground(
        self,
        image_pt: Tuple[float, float],
        strict_bounds: bool = True
    ) -> Optional[Tuple[float, float]]:
        """Project a 2D image pixel coordinate (u, v) into reference ground coordinates (X_g, Y_g).

        Args:
            image_pt: (u, v) pixel coordinate.
            strict_bounds: If True, strictly requires the point to lie inside the calibration
                           quad and reference bounds without artificial coordinate clamping.

        Returns:
            Optional[Tuple[float, float]]: (X_g, Y_g) in reference units (cm), or None if invalid.
        """
        if not self.is_calibrated or self.homography_matrix is None:
            return None

        # Verify contact point is inside calibration quadrilateral
        if strict_bounds and not self.is_point_inside_plane(image_pt):
            return None

        u, v = float(image_pt[0]), float(image_pt[1])
        pt_homogeneous = np.array([u, v, 1.0], dtype=np.float64)

        # Apply homography projection: [X, Y, W]^T = H * [u, v, 1]^T
        projected = self.homography_matrix.dot(pt_homogeneous)
        w = projected[2]

        # Reject points projecting at infinity or behind camera view
        if abs(w) < 1e-7 or w <= 0.0:
            return None

        gx = projected[0] / w
        gy = projected[1] / w

        if strict_bounds:
            eps = 1e-2
            if not (-eps <= gx <= self.config.reference_width + eps and
                    -eps <= gy <= self.config.reference_height + eps):
                return None
            return (
                max(0.0, min(self.config.reference_width, gx)),
                max(0.0, min(self.config.reference_height, gy))
            )

        # Soft margin mode (fallback / testing)
        margin_x = self.config.reference_width * 0.35
        margin_y = self.config.reference_height * 0.35

        if not (-margin_x <= gx <= self.config.reference_width + margin_x and
                -margin_y <= gy <= self.config.reference_height + margin_y):
            return None

        clamped_gx = max(0.0, min(self.config.reference_width, gx))
        clamped_gy = max(0.0, min(self.config.reference_height, gy))
        return (clamped_gx, clamped_gy)

    def transform_ground_to_image(
        self,
        ground_pt: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Project a physical ground coordinate (X_g, Y_g) in centimeters to 2D image pixel coordinates (u, v).

        Args:
            ground_pt: (X_g, Y_g) ground coordinate in cm.

        Returns:
            Optional[Tuple[float, float]]: (u, v) image pixel coordinate, or None if invalid.
        """
        if not self.is_calibrated or self.inverse_homography is None:
            return None

        gx, gy = float(ground_pt[0]), float(ground_pt[1])
        pt_homo = np.array([gx, gy, 1.0], dtype=np.float64)
        proj = self.inverse_homography.dot(pt_homo)
        w = proj[2]
        if abs(w) < 1e-7:
            return None
        u = float(proj[0] / w)
        v = float(proj[1] / w)
        return (u, v)

    def get_cell_size_cm(self, grid_rows: int = 22, grid_cols: int = 26) -> Tuple[float, float]:
        """Compute physical dimensions in centimeters of an individual simulation grid cell.

        Returns:
            Tuple[float, float]: (cell_width_cm, cell_height_cm)
        """
        ref_w = max(1e-5, self.config.reference_width)
        ref_h = max(1e-5, self.config.reference_height)
        cell_w = ref_w / float(max(1, grid_cols))
        cell_h = ref_h / float(max(1, grid_rows))
        return (cell_w, cell_h)

    def ground_to_grid(
        self,
        ground_pt: Tuple[float, float],
        grid_rows: int,
        grid_cols: int,
        invert_rows: bool = False
    ) -> Optional[Tuple[int, int]]:
        """Map physical ground coordinates (X_g, Y_g) in centimeters onto discrete simulation grid cells (row, col).

        Documented Coordinate Convention:
        - Origin (0.0 cm, 0.0 cm): Top-Left (Far-Left) of the calibrated tabletop reference surface.
        - Physical X axis (0.0 -> reference_width cm): Lateral table span (Left to Right).
        - Physical Y axis (0.0 -> reference_height cm): Longitudinal depth span (Far to Near).
        - Cell width  = reference_width / grid_cols  (e.g. 100.0 / 26 = 3.846 cm/cell)
        - Cell height = reference_height / grid_rows (e.g.  80.0 / 22 = 3.636 cm/cell)
        - Column = floor(X_g / cell_width_cm)
        - Row    = floor(Y_g / cell_height_cm)

        Args:
            ground_pt: (X_g, Y_g) ground coordinate in cm.
            grid_rows: Number of grid rows (e.g. 22).
            grid_cols: Number of grid columns (e.g. 26).
            invert_rows: If True, ground Y_g=0 maps to row=(rows-1).

        Returns:
            Optional[Tuple[int, int]]: (row, col) cell, or None if out of bounds.
        """
        gx, gy = ground_pt
        ref_w = max(1e-5, self.config.reference_width)
        ref_h = max(1e-5, self.config.reference_height)

        # Boundary check: reject coordinates outside reference plane
        eps = 1e-4
        if not (-eps <= gx <= ref_w + eps and -eps <= gy <= ref_h + eps):
            return None

        clamped_x = max(0.0, min(ref_w, gx))
        clamped_y = max(0.0, min(ref_h, gy))

        cell_w, cell_h = self.get_cell_size_cm(grid_rows, grid_cols)

        # Pure mathematical floor binning without arbitrary offsets (with 1e-5 numerical precision epsilon)
        col = int(math.floor((clamped_x + 1e-5) / cell_w))
        row = int(math.floor((clamped_y + 1e-5) / cell_h))

        # Handle boundary edge case (X_g == ref_w or Y_g == ref_h)
        col = max(0, min(grid_cols - 1, col))
        row = max(0, min(grid_rows - 1, row))

        if invert_rows:
            row = (grid_rows - 1) - row

        return (row, col)

    def save_to_json(self, filepath: str = "calibration_config.json") -> bool:
        """Persist calibration parameters, fiducials, and homography matrix to JSON."""
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            data = {
                "camera_width": self.calib_width,
                "camera_height": self.calib_height,
                "reference_width_cm": self.config.reference_width,
                "reference_height_cm": self.config.reference_height,
                "reprojection_error_px": round(self.reprojection_error_px, 4),
                "ground_error_cm": round(self.ground_error_cm, 4),
                "condition_number": round(self.condition_number, 2),
                "is_calibrated": self.is_calibrated,
                "fiducials": [
                    {
                        "label": p.label,
                        "image_coord": [float(p.image_coord[0]), float(p.image_coord[1])],
                        "ground_coord": [float(p.ground_coord[0]), float(p.ground_coord[1])]
                    }
                    for p in self.active_fiducials
                ],
                "homography_matrix": self.homography_matrix.tolist() if self.homography_matrix is not None else None,
                "timestamp": time.time()
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved camera calibration configuration to: %s", filepath)
            return True
        except Exception as exc:
            logger.error("Failed to save calibration configuration to %s: %s", filepath, exc)
            return False

    def load_from_json(self, filepath: str = "calibration_config.json") -> bool:
        """Load calibration parameters, fiducials, and matrix from JSON configuration file."""
        if not os.path.exists(filepath):
            logger.warning("Calibration file %s does not exist.", filepath)
            return False
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.calib_width = data.get("camera_width", 640)
            self.calib_height = data.get("camera_height", 480)
            self.config.reference_width = data.get("reference_width_cm", 100.0)
            self.config.reference_height = data.get("reference_height_cm", 80.0)
            fid_list = []
            for item in data.get("fiducials", []):
                fid_list.append(
                    CalibrationPointPair(
                        image_coord=tuple(item["image_coord"]),
                        ground_coord=tuple(item["ground_coord"]),
                        label=item.get("label", "Fiducial")
                    )
                )
            if len(fid_list) >= 4:
                self.active_fiducials = fid_list
                self._base_fiducials = [
                    CalibrationPointPair(image_coord=p.image_coord, ground_coord=p.ground_coord, label=p.label)
                    for p in fid_list
                ]
                self._base_width = self.calib_width
                self._base_height = self.calib_height
                success, _ = self.compute_homography()
                logger.info("Loaded camera calibration from %s (success=%s).", filepath, success)
                return success
            return False
        except Exception as exc:
            logger.error("Failed to load calibration from %s: %s", filepath, exc)
            return False

    def draw_calibration_overlay(
        self,
        frame: np.ndarray,
        show_fiducials: bool = True,
        show_instructions: Optional[bool] = None
    ) -> np.ndarray:
        """Render calibration fiducials, reference quad boundary, and status HUD onto camera frame."""
        if frame is None or frame.size == 0:
            return frame

        # Auto-adapt if frame resolution differs from current calibration dimensions
        fh, fw = frame.shape[:2]
        if fw != self.calib_width or fh != self.calib_height:
            self.adapt_to_resolution(fw, fh)

        canvas = frame.copy()
        show_instr = self.show_instructions if show_instructions is None else show_instructions

        # Draw detected ArUco marker outlines
        if show_fiducials and getattr(self, "aruco_corners_raw", None):
            for i, c in enumerate(self.aruco_corners_raw):
                c_pts = c[0].astype(np.int32)
                cv2.polylines(canvas, [c_pts], isClosed=True, color=(0, 255, 100), thickness=2, lineType=cv2.LINE_AA)

        if show_fiducials and len(self.active_fiducials) >= 4:
            pts = np.array([p.image_coord for p in self.active_fiducials[:4]], dtype=np.int32)

            if self.is_calibrated:
                # Semi-transparent fill of ground plane (cyan-emerald tone)
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts], (30, 80, 50))
                cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

                # Reference quad boundary: bright cyan-yellow with double stroke
                cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                cv2.polylines(canvas, [pts], isClosed=True, color=(0, 240, 255), thickness=2, lineType=cv2.LINE_AA)

                # Draw fiducial markers with clear badges and ground coordinates
                for i, p in enumerate(self.active_fiducials[:4]):
                    u, v = int(round(p.image_coord[0])), int(round(p.image_coord[1]))
                    # Outer black border, inner bright amber/yellow circle
                    cv2.circle(canvas, (u, v), 8, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (u, v), 6, (0, 220, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (u, v), 2, (255, 255, 255), -1, cv2.LINE_AA)

                    badge_lbl = f"{p.label} ({int(p.ground_coord[0])},{int(p.ground_coord[1])}cm)"
                    (bw, bh), _ = cv2.getTextSize(badge_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)

                    # Position badge offset to stay inside frame
                    bx = u + 10 if u < fw - 160 else u - bw - 12
                    by = v - 6 if v > bh + 10 else v + bh + 12

                    cv2.rectangle(canvas, (bx - 3, by - bh - 2), (bx + bw + 3, by + 3), (15, 20, 25), -1)
                    cv2.rectangle(canvas, (bx - 3, by - bh - 2), (bx + bw + 3, by + 3), (0, 220, 255), 1)
                    cv2.putText(canvas, badge_lbl, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            else:
                # Invalid calibration boundary in bright red
                cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                for p in self.active_fiducials[:4]:
                    u, v = int(round(p.image_coord[0])), int(round(p.image_coord[1]))
                    cv2.circle(canvas, (u, v), 7, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (u, v), 9, (255, 255, 255), 1, cv2.LINE_AA)

        # Prominent Calibration Status Card (Top Bar)
        bar_h = 36
        cv2.rectangle(canvas, (10, 8), (fw - 10, 8 + bar_h), (20, 24, 30), -1)

        if self.is_calibrated:
            cell_w, cell_h = self.get_cell_size_cm(22, 26)
            if self.is_aruco_tracking:
                cv2.rectangle(canvas, (10, 8), (fw - 10, 8 + bar_h), (0, 255, 120), 2)
                status_text = (
                    f"ARUCO AUTO-CALIB: 4/4 LOCKED (Dynamic H) | Reproj: {self.reprojection_error_px:.2f} px | "
                    f"Plane: {self.config.reference_width:.0f}x{self.config.reference_height:.0f} cm "
                    f"({cell_w:.2f}x{cell_h:.2f} cm/cell)"
                )
                text_col = (50, 255, 150)
            elif self.aruco_detected_count > 0:
                cv2.rectangle(canvas, (10, 8), (fw - 10, 8 + bar_h), (0, 200, 255), 1)
                status_text = (
                    f"ARUCO: {self.aruco_detected_count}/4 markers (Static Fallback) | Reproj: {self.reprojection_error_px:.2f} px | "
                    f"Plane: {self.config.reference_width:.0f}x{self.config.reference_height:.0f} cm"
                )
                text_col = (100, 230, 255)
            else:
                cv2.rectangle(canvas, (10, 8), (fw - 10, 8 + bar_h), (0, 200, 120), 1)
                status_text = (
                    f"CALIBRATED: Static Homography OK | Reproj Err: {self.reprojection_error_px:.2f} px | "
                    f"Plane: {self.config.reference_width:.0f}x{self.config.reference_height:.0f} cm "
                    f"({cell_w:.2f}x{cell_h:.2f} cm/cell)"
                )
                text_col = (80, 255, 160)

            cv2.putText(canvas, status_text, (20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_col, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (10, 8), (fw - 10, 8 + bar_h), (50, 50, 240), 2)
            warn_msg = self.calibration_warning or "Calibration Invalid or Degenerate"
            cv2.putText(canvas, f"WARNING: {warn_msg}", (20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (60, 60, 255), 1, cv2.LINE_AA)

        # Calibration alignment instructions box (bottom-left)
        if show_instr and self.is_calibrated:
            cell_w, cell_h = self.get_cell_size_cm(22, 26)
            instr_lines = [
                "Tabletop Calibration Setup:",
                f"1. Plane: {self.config.reference_width:.0f}x{self.config.reference_height:.0f} cm (26x22 cells, {cell_w:.2f}x{cell_h:.2f} cm/cell)",
                "2. 4 fiducials define ground plane: P1(TL), P2(TR), P3(BR), P4(BL)",
                "3. Objects outside yellow quad or non-ground (people) rejected",
                "4. ArUco markers 0(TL), 1(TR), 2(BR), 3(BL) enable real-time auto-calibration",
            ]
            box_y = fh - len(instr_lines) * 16 - 14
            cv2.rectangle(canvas, (10, box_y), (min(fw - 10, 520), fh - 8), (15, 20, 25), -1)
            cv2.rectangle(canvas, (10, box_y), (min(fw - 10, 520), fh - 8), (60, 75, 90), 1)
            for idx, line in enumerate(instr_lines):
                col = (0, 230, 255) if idx == 0 else (200, 210, 220)
                cv2.putText(canvas, line, (16, box_y + 14 + idx * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)

        return canvas
