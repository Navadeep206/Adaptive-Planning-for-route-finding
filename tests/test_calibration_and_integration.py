"""End-to-end integration and calibration unit test suite for Phase 5.

Verifies:
- Planar homography calibration transformation.
- Image-to-grid mapping and coordinate bounds.
- Rejection of invalid/collinear calibration points.
- Out-of-bounds detection filtering.
- Multi-track and duplicate-track merging.
- Expired track pruning.
- Safety inflation margins.
- Vehicle position and destination goal collision guards.
- Dynamic route obstruction and A* replanning.
- No-path safe stopping when all routes are blocked.
Executes 100% locally and headlessly in CI without requiring a physical camera or downloaded weights.
"""

from unittest.mock import MagicMock
import os
import unittest
import numpy as np

# Ensure headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

from perception.camera_calibrator import CameraCalibrator, CalibrationConfig, CalibrationPointPair
from perception.grid_mapper import GridMapper, GridMapperConfig, MappingMode
from perception.object_tracker import TrackedObject, TrackState
from simulation.environment import RoadEnvironment, NavigationMode, CellType


class TestCalibrationAndIntegration(unittest.TestCase):
    """Test suite for Camera Calibration, Occupancy Grid Mapping, and A* Path Planning Integration."""

    def setUp(self):
        """Initialize calibrator, grid mapper, and environment."""
        self.calibrator = CameraCalibrator()
        self.config = GridMapperConfig(
            min_confidence=0.25,
            grid_rows=22,
            grid_cols=26,
            calibrator=self.calibrator,
            inflation_radius=0,
            stale_timeout_sec=1.0,
            debounce_hits=1
        )
        self.mapper = GridMapper(config=self.config)
        self.env = RoadEnvironment()

    def test_calibration_transformation_accuracy(self):
        """Verify that known image coordinates map accurately to ground coordinates."""
        self.assertTrue(self.calibrator.is_calibrated)
        self.assertLess(self.calibrator.reprojection_error_px, 5.0)

        # Test first fiducial: (100.0, 120.0) -> expected (0.0, 0.0)
        pt = self.calibrator.transform_image_to_ground((100.0, 120.0))
        self.assertIsNotNone(pt)
        self.assertAlmostEqual(pt[0], 0.0, delta=2.0)
        self.assertAlmostEqual(pt[1], 0.0, delta=2.0)

        # Test second fiducial: (540.0, 120.0) -> expected (100.0, 0.0)
        pt2 = self.calibrator.transform_image_to_ground((540.0, 120.0))
        self.assertIsNotNone(pt2)
        self.assertAlmostEqual(pt2[0], 100.0, delta=2.0)
        self.assertAlmostEqual(pt2[1], 0.0, delta=2.0)

    def test_image_to_grid_conversion(self):
        """Verify mapping from 2D image coordinates to discrete simulation grid coordinates."""
        # Contact point at top-left of tabletop track -> maps near top-left of grid
        cell, gnd = self.mapper.map_bbox_to_cell(
            bbox=(80, 100, 120, 120),  # cx = 100, y2 = 120
            frame_width=640,
            frame_height=480
        )
        self.assertIsNotNone(cell)
        self.assertEqual(self.mapper.mapping_mode, MappingMode.HOMOGRAPHY)
        self.assertLessEqual(cell[0], 5)  # Near row 0
        self.assertLessEqual(cell[1], 5)  # Near col 0

    def test_invalid_collinear_calibration_points(self):
        """Verify that collinear or degenerate calibration points are rejected safely."""
        collinear_fiducials = [
            CalibrationPointPair((100.0, 100.0), (0.0, 0.0)),
            CalibrationPointPair((200.0, 100.0), (10.0, 0.0)),
            CalibrationPointPair((300.0, 100.0), (20.0, 0.0)),
            CalibrationPointPair((400.0, 100.0), (30.0, 0.0)),
        ]
        calib = CameraCalibrator(config=CalibrationConfig(default_fiducials=collinear_fiducials))
        self.assertFalse(calib.is_calibrated)
        self.assertIsNone(calib.homography_matrix)

    def test_out_of_bounds_detection_rejection(self):
        """Verify that detections outside physical reference plane are rejected."""
        # Contact point at y2 = -50 or x = 2000
        cell, _ = self.mapper.map_bbox_to_cell(
            bbox=(1900, -100, 2100, -50),
            frame_width=640,
            frame_height=480
        )
        self.assertIsNone(cell)

    def test_multiple_and_duplicate_tracks_mapping(self):
        """Verify that multiple tracks are mapped and duplicates mapping to the same cell are merged."""
        # Two tracks with slightly different bboxes projecting to the same grid region
        t1 = TrackedObject(
            track_id=1, class_id=0, class_name="bottle", confidence=0.88,
            bbox=(310, 270, 330, 280), raw_bbox=(310, 270, 330, 280),
            state=TrackState.CONFIRMED, hits=5
        )
        t2 = TrackedObject(
            track_id=2, class_id=0, class_name="cup", confidence=0.85,
            bbox=(312, 272, 332, 282), raw_bbox=(312, 272, 332, 282),
            state=TrackState.CONFIRMED, hits=4
        )

        res = self.mapper.map_tracks(
            tracks=[t1, t2],
            frame_shape=(480, 640),
            current_time=100.0
        )
        self.assertEqual(res.raw_detections_count, 2)
        # Should merge gracefully without crashing or duplicating coordinates
        self.assertGreaterEqual(len(res.final_cells), 1)
        self.assertLessEqual(len(res.final_cells), 2)
        self.assertEqual(len(res.mapped_tracks_info), 2)

    def test_stale_track_expiration(self):
        """Verify that unrefreshed perception cells expire after timeout."""
        t1 = TrackedObject(
            track_id=1, class_id=0, class_name="hazard", confidence=0.90,
            bbox=(320, 260, 360, 280), raw_bbox=(320, 260, 360, 280),
            state=TrackState.CONFIRMED, hits=5
        )
        # Map at t = 100.0
        res1 = self.mapper.map_tracks([t1], frame_shape=(480, 640), current_time=100.0)
        self.assertGreaterEqual(len(res1.final_cells), 1)

        # Update at t = 102.5 (> stale_timeout_sec=1.0) with empty tracks
        res2 = self.mapper.map_tracks([], frame_shape=(480, 640), current_time=102.5)
        self.assertEqual(len(res2.final_cells), 0)
        self.assertEqual(len(self.mapper.get_active_cells()), 0)

    def test_obstacle_inflation_safety_margin(self):
        """Verify that safety inflation expands a single cell into a surrounding neighborhood."""
        mapper_inflated = GridMapper(config=GridMapperConfig(
            grid_rows=22, grid_cols=26, inflation_radius=1
        ))
        base_cells = {(10, 10)}
        inflated = mapper_inflated.inflate_cells(base_cells)
        # 1-cell radius in Chebyshev neighborhood is a 3x3 footprint (9 cells)
        self.assertEqual(len(inflated), 9)
        self.assertIn((10, 10), inflated)
        self.assertIn((9, 9), inflated)
        self.assertIn((11, 11), inflated)

    def test_vehicle_and_goal_overlap_protection(self):
        """Verify that candidate perception obstacles overlapping vehicle or goal are strictly dropped."""
        # Find pixel coordinate that maps near the vehicle start position
        self.env.vehicle_pos = (10, 13)
        self.env.destination_pos = (20, 15)

        # Manually mock map_bbox_to_cell to return the vehicle's position
        original_map_fn = self.mapper.map_bbox_to_cell
        self.mapper.map_bbox_to_cell = lambda *args, **kwargs: ((10, 13), (50.0, 40.0))

        t_hazard = TrackedObject(
            track_id=1, class_id=0, class_name="ghost_hazard", confidence=0.90,
            bbox=(300, 200, 340, 240), raw_bbox=(300, 200, 340, 240),
            state=TrackState.CONFIRMED, hits=3
        )

        res = self.mapper.map_tracks(
            tracks=[t_hazard],
            frame_shape=(480, 640),
            vehicle_pos=self.env.vehicle_pos,
            goal_pos=self.env.destination_pos
        )
        self.assertIn((10, 13), res.vehicle_conflicts)
        self.assertNotIn((10, 13), res.final_cells)
        self.assertEqual(len(res.final_cells), 0)

        # Restore original function
        self.mapper.map_bbox_to_cell = original_map_fn

    def test_route_invalidation_and_successful_replanning(self):
        """Verify that placing perception obstacles on the active route triggers successful A* replan."""
        self.env.perception_enabled = True
        plan_res = self.env.plan_path()
        self.assertTrue(plan_res.success)
        initial_path = list(self.env.planned_path)
        self.assertGreater(len(initial_path), 5)

        # Block a waypoint along the active route (e.g. index 3)
        waypoint_to_block = initial_path[3]
        self.env.set_perception_obstacles({waypoint_to_block})

        # Obstacle set should now contain the waypoint
        self.assertIn(waypoint_to_block, self.env.perception_obstacle_set)

        # A* replan should execute and produce an alternative route avoiding that cell
        replan_res = self.env.plan_path()
        self.assertTrue(replan_res.success)
        new_path = self.env.planned_path
        self.assertNotIn(waypoint_to_block, new_path)

    def test_no_path_safe_stopping(self):
        """Verify vehicle halts safely in NO_PATH mode when all corridors are fully blocked."""
        self.env.perception_enabled = True
        self.env.plan_path()
        self.env.nav_mode = NavigationMode.AUTONOMOUS

        # Block an entire horizontal cut across the road ahead of the vehicle
        cut_row = self.env.vehicle_pos[0] + 2
        blocking_wall = {(cut_row, c) for c in range(self.env.cols)}
        self.env.set_perception_obstacles(blocking_wall)

        # Try to step
        stepped = self.env.step_autonomous()
        self.assertFalse(stepped)
        self.assertEqual(self.env.nav_mode, NavigationMode.NO_PATH)
        self.assertIn("stopped safely", self.env.last_action_msg.lower())

        # Unblock wall and verify recovery
        self.env.set_perception_obstacles(set())
        res = self.env.plan_path()
        self.assertTrue(res.success)


if __name__ == "__main__":
    unittest.main()
