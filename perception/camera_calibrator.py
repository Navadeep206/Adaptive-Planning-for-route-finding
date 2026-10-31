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
   known geometric fiducial coordinates.
"""

from dataclasses import dataclass, field
import logging
import math
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
    reference_width: float = 100.0          # Metric or arbitrary ground plane width span (X_g)
    reference_height: float = 80.0          # Metric or arbitrary ground plane height span (Y_g)
    max_reprojection_error_px: float = 10.0 # Maximum allowed mean reprojection error in pixels
    min_points_required: int = 4            # Minimum non-collinear point pairs required for homography
    default_fiducials: List[CalibrationPointPair] = field(default_factory=list)


class CameraCalibrator:
    """Computes, validates, and applies planar homography transformations.

    Transforms detected object ground-contact points into reference plane coordinates,
    validates reprojection errors, and enforces strict boundary guards.
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
        self.active_fiducials: List[CalibrationPointPair] = []

        # If default fiducials provided or empty, initialize standard laboratory fiducials
        if not self.config.default_fiducials:
            self._init_default_tabletop_fiducials()
        else:
            self.active_fiducials = list(self.config.default_fiducials)
            self.compute_homography()

    def _init_default_tabletop_fiducials(self) -> None:
        """Initialize standard 4-corner tabletop reference calibration points for 640x480 camera."""
        # 4 corners of tabletop road track visible in standard forward-facing webcam perspective
        # Image coordinates (trapezoidal perspective) -> Ground coordinates (rectangular reference plane)
        self.active_fiducials = [
            CalibrationPointPair(image_coord=(100.0, 120.0), ground_coord=(0.0, 0.0), label="Top-Left (Far)"),
            CalibrationPointPair(image_coord=(540.0, 120.0), ground_coord=(100.0, 0.0), label="Top-Right (Far)"),
            CalibrationPointPair(image_coord=(580.0, 440.0), ground_coord=(100.0, 80.0), label="Bottom-Right (Near)"),
            CalibrationPointPair(image_coord=(60.0, 440.0), ground_coord=(0.0, 80.0), label="Bottom-Left (Near)"),
        ]
        self.compute_homography()

    def set_fiducials(self, fiducials: List[CalibrationPointPair]) -> Tuple[bool, float]:
        """Update calibration fiducials and recompute homography matrix.

        Args:
            fiducials: List of at least 4 non-collinear CalibrationPointPair instances.

        Returns:
            Tuple[bool, float]: (is_valid, mean_reprojection_error_px)
        """
        self.active_fiducials = list(fiducials)
        return self.compute_homography()

    def compute_homography(self) -> Tuple[bool, float]:
        """Estimate 3x3 homography matrix H from active fiducial point pairs.

        Returns:
            Tuple[bool, float]: (success, reprojection_error_px)
        """
        if len(self.active_fiducials) < self.config.min_points_required:
            logger.warning(
                "Cannot compute homography: %d fiducials provided (minimum %d required).",
                len(self.active_fiducials),
                self.config.min_points_required
            )
            self._invalidate_calibration()
            return False, float("inf")

        img_pts = np.array([p.image_coord for p in self.active_fiducials], dtype=np.float32)
        gnd_pts = np.array([p.ground_coord for p in self.active_fiducials], dtype=np.float32)

        # Check for collinearity / degeneracy
        if self._check_collinear(img_pts) or self._check_collinear(gnd_pts):
            logger.warning("Fiducial points are collinear or degenerate. Homography computation rejected.")
            self._invalidate_calibration()
            return False, float("inf")

        try:
            # If exactly 4 points, use exact perspective transform; otherwise RANSAC
            if len(self.active_fiducials) == 4:
                H = cv2.getPerspectiveTransform(img_pts, gnd_pts)
            else:
                H, _ = cv2.findHomography(img_pts, gnd_pts, cv2.RANSAC, 5.0)

            if H is None or H.shape != (3, 3):
                logger.error("cv2.findHomography returned None or invalid shape.")
                self._invalidate_calibration()
                return False, float("inf")

            # Validate matrix conditioning (determinant check)
            det = np.linalg.det(H)
            if abs(det) < 1e-9 or math.isnan(det):
                logger.warning("Homography matrix is near-singular (det=%.2e). Rejected.", det)
                self._invalidate_calibration()
                return False, float("inf")

            # Compute reprojection error: map image points to ground and compare
            err = self._evaluate_reprojection_error(H, img_pts, gnd_pts)
            self.reprojection_error_px = err

            if err > self.config.max_reprojection_error_px:
                logger.warning(
                    "Calibration reprojection error (%.2f px) exceeds threshold (%.2f px).",
                    err,
                    self.config.max_reprojection_error_px
                )
                self._invalidate_calibration()
                return False, err

            # Successfully validated
            self.homography_matrix = H
            self.inverse_homography = np.linalg.inv(H)
            self.is_calibrated = True

            logger.info(
                "Planar homography calibration successful (%d points, error=%.2f px, det=%.2e).",
                len(self.active_fiducials),
                err,
                det
            )
            return True, err

        except Exception as exc:
            logger.error("Exception during homography computation: %s", exc, exc_info=True)
            self._invalidate_calibration()
            return False, float("inf")

    def _invalidate_calibration(self) -> None:
        """Reset internal matrices upon validation failure."""
        self.homography_matrix = None
        self.inverse_homography = None
        self.is_calibrated = False
        self.reprojection_error_px = float("inf")

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
    ) -> float:
        """Calculate mean Euclidean error between ground points and projected image points."""
        # Reshape for cv2.perspectiveTransform: (N, 1, 2)
        src = img_pts.reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
        distances = np.linalg.norm(gnd_pts - projected, axis=1)
        return float(np.mean(distances))

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
        image_pt: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Project a 2D image pixel coordinate (u, v) into reference ground coordinates (X_g, Y_g).

        Args:
            image_pt: (u, v) pixel coordinate.

        Returns:
            Optional[Tuple[float, float]]: (X_g, Y_g) in reference units, or None if invalid.
        """
        if not self.is_calibrated or self.homography_matrix is None:
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

        # Verify ground coordinates lie within realistic reference plane bounds (+20% margin)
        margin_x = self.config.reference_width * 0.20
        margin_y = self.config.reference_height * 0.20

        if not (-margin_x <= gx <= self.config.reference_width + margin_x and
                -margin_y <= gy <= self.config.reference_height + margin_y):
            return None

        return (gx, gy)

    def ground_to_grid(
        self,
        ground_pt: Tuple[float, float],
        grid_rows: int,
        grid_cols: int,
        invert_rows: bool = False
    ) -> Optional[Tuple[int, int]]:
        """Map reference ground coordinates (X_g, Y_g) onto discrete simulation grid cells (row, col).

        Args:
            ground_pt: (X_g, Y_g) ground coordinate.
            grid_rows: Number of grid rows.
            grid_cols: Number of grid columns.
            invert_rows: If True, ground Y_g=0 maps to row=(rows-1).

        Returns:
            Optional[Tuple[int, int]]: (row, col) cell, or None if out of bounds.
        """
        gx, gy = ground_pt

        # Normalize ground coordinates to [0.0, 1.0]
        norm_x = gx / max(1e-5, self.config.reference_width)
        norm_y = gy / max(1e-5, self.config.reference_height)

        if not (0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0):
            return None

        col = int(round(norm_x * (grid_cols - 1)))
        if invert_rows:
            row = int(round((1.0 - norm_y) * (grid_rows - 1)))
        else:
            row = int(round(norm_y * (grid_rows - 1)))

        col = max(0, min(grid_cols - 1, col))
        row = max(0, min(grid_rows - 1, row))

        return (row, col)

    def draw_calibration_overlay(
        self,
        frame: np.ndarray,
        show_fiducials: bool = True
    ) -> np.ndarray:
        """Render calibration fiducials and reference quad boundary onto camera frame."""
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()

        if show_fiducials and len(self.active_fiducials) >= 4:
            pts = np.array([p.image_coord for p in self.active_fiducials], dtype=np.int32)
            # Draw reference quad boundary
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            # Draw fiducial markers
            for i, p in enumerate(self.active_fiducials):
                u, v = int(round(p.image_coord[0])), int(round(p.image_coord[1]))
                cv2.circle(canvas, (u, v), 5, (0, 200, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (u, v), 7, (0, 0, 0), 1, cv2.LINE_AA)
                label = f"P{i+1}"
                cv2.putText(canvas, label, (u + 8, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        return canvas
