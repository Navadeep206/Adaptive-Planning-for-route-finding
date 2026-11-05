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
import cv2
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
            enable_footprint=False,
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

    def test_point_ordering_validation_detects_crossed_points(self):
        """Verify that self-intersecting (bow-tie) or inverted fiducial points are rejected."""
        # Crossing diagonals (P1, P3, P2, P4)
        crossed_fiducials = [
            CalibrationPointPair((100.0, 120.0), (0.0, 0.0), "P1"),
            CalibrationPointPair((580.0, 440.0), (100.0, 0.0), "P3"),  # Crossed with P2!
            CalibrationPointPair((540.0, 120.0), (100.0, 80.0), "P2"),
            CalibrationPointPair((60.0, 440.0), (0.0, 80.0), "P4"),
        ]
        calib = CameraCalibrator(config=CalibrationConfig(default_fiducials=crossed_fiducials))
        self.assertFalse(calib.is_calibrated)
        self.assertIsNotNone(calib.calibration_warning)
        self.assertIn("convex", calib.calibration_warning.lower())

    def test_non_planar_class_rejection_in_homography_mode(self):
        """Verify that non-planar classes ('person') are rejected in calibrated homography mode."""
        self.assertEqual(self.mapper.mapping_mode, MappingMode.HOMOGRAPHY)
        t_person = TrackedObject(
            track_id=1, class_id=0, class_name="person", confidence=0.92,
            bbox=(300, 200, 340, 240), raw_bbox=(300, 200, 340, 240),
            state=TrackState.CONFIRMED, hits=3
        )
        res = self.mapper.map_tracks([t_person], (480, 640))
        self.assertEqual(len(res.final_cells), 0)
        self.assertEqual(res.mapped_tracks_info[0]["status"], "REJECTED")
        self.assertIn("non-ground", res.mapped_tracks_info[0]["reason"].lower())

    def test_contact_point_outside_plane_rejection(self):
        """Verify that objects whose contact points lie outside the calibration quad are rejected."""
        # Contact point at (30, 80) is well outside the calibration quad (TL=(100,120), BL=(60,440))
        t_off_table = TrackedObject(
            track_id=2, class_id=0, class_name="box", confidence=0.88,
            bbox=(10, 40, 50, 80), raw_bbox=(10, 40, 50, 80),  # cx=30, y2=80
            state=TrackState.CONFIRMED, hits=3
        )
        res = self.mapper.map_tracks([t_off_table], (480, 640))
        self.assertEqual(len(res.final_cells), 0)
        self.assertEqual(res.mapped_tracks_info[0]["status"], "REJECTED")
        self.assertIn("outside calibrated ground plane", res.mapped_tracks_info[0]["reason"].lower())

    def test_selective_track_expiration_pruning(self):
        """Verify that expired tracks remove only their own cells, leaving other tracks untouched."""
        t1 = TrackedObject(
            track_id=10, class_id=0, class_name="box", confidence=0.90,
            bbox=(160, 150, 200, 170), raw_bbox=(160, 150, 200, 170),
            state=TrackState.CONFIRMED, hits=3
        )
        t2 = TrackedObject(
            track_id=20, class_id=0, class_name="box", confidence=0.90,
            bbox=(430, 150, 470, 170), raw_bbox=(430, 150, 470, 170),
            state=TrackState.CONFIRMED, hits=3
        )
        # Step 1: Both tracks active at t=0.0
        res1 = self.mapper.map_tracks([t1, t2], (480, 640), current_time=0.0)
        self.assertEqual(len(res1.final_cells), 2)
        cells_t1 = self.mapper._track_to_cells[10]
        cells_t2 = self.mapper._track_to_cells[20]

        # Step 2: Only track 2 is refreshed at t=1.0, track 1 is absent
        res2 = self.mapper.map_tracks([t2], (480, 640), current_time=1.0)
        # Track 1 is within stale_timeout (2.0s), so both cells remain active
        self.assertTrue(cells_t1.issubset(res2.final_cells))
        self.assertTrue(cells_t2.issubset(res2.final_cells))

        # Step 3: Track 2 refreshed at t=3.0 (>2.0s since t1 was seen)
        res3 = self.mapper.map_tracks([t2], (480, 640), current_time=3.0)
        # Track 1 has expired and is removed; Track 2 remains active!
        self.assertNotIn(list(cells_t1)[0], res3.final_cells)
        self.assertIn(list(cells_t2)[0], res3.final_cells)
        self.assertEqual(len(res3.final_cells), len(cells_t2))

    def test_grid_track_id_visualization_labeling(self):
        """Verify that perception obstacles receive track ID associations for grid rendering."""
        self.env.perception_enabled = True
        test_cells = {(5, 12), (10, 14)}
        track_map = {(5, 12): 1, (10, 14): 2}
        self.env.set_perception_obstacles(test_cells, track_map=track_map)

        self.assertEqual(self.env.perception_track_map[(5, 12)], 1)
        self.assertEqual(self.env.perception_track_map[(10, 14)], 2)
        self.assertEqual(len(self.env.perception_obstacle_set), 2)

    def test_nine_calibration_points_exact_mapping(self):
        """Verify that all 9 canonical calibration points map with zero cell error and sub-millimeter precision."""
        test_points = [
            ("TL Corner", 0.0, 0.0, 0, 0),
            ("TR Corner", 100.0, 0.0, 0, 25),
            ("BR Corner", 100.0, 80.0, 21, 25),
            ("BL Corner", 0.0, 80.0, 21, 0),
            ("Top Midpoint", 50.0, 0.0, 0, 13),
            ("Bottom Midpoint", 50.0, 80.0, 21, 13),
            ("Left Midpoint", 0.0, 40.0, 11, 0),
            ("Right Midpoint", 100.0, 40.0, 11, 25),
            ("Center Point", 50.0, 40.0, 11, 13),
        ]
        for name, gx, gy, exp_r, exp_c in test_points:
            img_pt = self.calibrator.transform_ground_to_image((gx, gy))
            self.assertIsNotNone(img_pt, f"Failed to project {name} to image")
            trans_pt = self.calibrator.transform_image_to_ground(img_pt, strict_bounds=True)
            self.assertIsNotNone(trans_pt, f"Failed to transform {name} image point to ground")
            err_cm = np.sqrt((trans_pt[0] - gx) ** 2 + (trans_pt[1] - gy) ** 2)
            self.assertLess(err_cm, 0.05, f"{name} position error {err_cm:.4f}cm exceeds 0.05cm")
            cell = self.calibrator.ground_to_grid(trans_pt, 22, 26)
            self.assertEqual(cell, (exp_r, exp_c), f"{name} mapped to {cell}, expected ({exp_r}, {exp_c})")

    def test_obstacle_footprint_estimation_and_primary_cell_separation(self):
        """Verify that footprint estimation computes physical width and isolates primary contact cell."""
        # Box centered near (50, 40) cm
        # Project 10cm base [45, 55] cm at Y=40cm to image
        pt_l = self.calibrator.transform_ground_to_image((45.0, 40.0))
        pt_r = self.calibrator.transform_ground_to_image((55.0, 40.0))
        self.assertIsNotNone(pt_l)
        self.assertIsNotNone(pt_r)

        u1, u2 = min(pt_l[0], pt_r[0]), max(pt_l[0], pt_r[0])
        v2 = max(pt_l[1], pt_r[1])
        v1 = v2 - 50.0
        bbox = (int(round(u1)), int(round(v1)), int(round(u2)), int(round(v2)))

        fp_res = self.mapper.estimate_obstacle_footprint(
            bbox=bbox,
            frame_width=640,
            frame_height=480,
            class_name="box",
            confidence=0.90
        )
        self.assertEqual(fp_res["status"], "ACCEPTED")
        primary_cell = fp_res["primary_cell"]
        self.assertEqual(primary_cell, (11, 13))

        footprint = fp_res["footprint_cells"]
        self.assertIn(primary_cell, footprint)
        self.assertGreater(len(footprint), 1)

        # Physical width should be ~10cm
        w_obs = fp_res["physical_width_cm"]
        self.assertIsNotNone(w_obs)
        self.assertAlmostEqual(w_obs, 10.0, delta=1.0)

    def test_multi_resolution_adaptation_without_drift(self):
        """Verify that calibrator adapts to 1280x720 without fiducial drift and back to 640x480."""
        initial_tl = self.calibrator.active_fiducials[0].image_coord
        self.assertEqual(initial_tl, (100.0, 120.0))

        # Scale to 1280x720 (scale_x = 2.0, scale_y = 1.5)
        success_scale = self.calibrator.adapt_to_resolution(1280, 720)
        self.assertTrue(success_scale)
        self.assertTrue(self.calibrator.is_calibrated)
        scaled_tl = self.calibrator.active_fiducials[0].image_coord
        self.assertAlmostEqual(scaled_tl[0], 200.0, delta=0.1)
        self.assertAlmostEqual(scaled_tl[1], 180.0, delta=0.1)

        # Scale back to 640x480
        success_restore = self.calibrator.adapt_to_resolution(640, 480)
        self.assertTrue(success_restore)
        restored_tl = self.calibrator.active_fiducials[0].image_coord
        self.assertAlmostEqual(restored_tl[0], 100.0, delta=0.1)
        self.assertAlmostEqual(restored_tl[1], 120.0, delta=0.1)

    def test_moving_object_trajectory_mapping(self):
        """Verify that a moving object updates its mapped grid cell dynamically across consecutive frames."""
        # Frame 1: Object at X=30cm (col 7), Y=40cm (row 11)
        pt1 = self.calibrator.transform_ground_to_image((30.0, 40.0))
        u1, v1 = int(round(pt1[0])), int(round(pt1[1]))
        t_move_1 = TrackedObject(
            track_id=42, class_id=0, class_name="box", confidence=0.90,
            bbox=(u1 - 10, v1 - 30, u1 + 10, v1),
            raw_bbox=(u1 - 10, v1 - 30, u1 + 10, v1),
            state=TrackState.CONFIRMED, hits=5
        )
        res1 = self.mapper.map_tracks([t_move_1], (480, 640), current_time=10.0)
        cell1 = res1.mapped_tracks_info[0]["grid_cell"]
        self.assertEqual(cell1[0], 11)
        self.assertEqual(cell1[1], 7)

        # Frame 2: Object moves to X=70cm (col 18), Y=40cm (row 11)
        pt2 = self.calibrator.transform_ground_to_image((70.0, 40.0))
        u2, v2 = int(round(pt2[0])), int(round(pt2[1]))
        t_move_2 = TrackedObject(
            track_id=42, class_id=0, class_name="box", confidence=0.90,
            bbox=(u2 - 10, v2 - 30, u2 + 10, v2),
            raw_bbox=(u2 - 10, v2 - 30, u2 + 10, v2),
            state=TrackState.CONFIRMED, hits=6
        )
        res2 = self.mapper.map_tracks([t_move_2], (480, 640), current_time=10.1)
        cell2 = res2.mapped_tracks_info[0]["grid_cell"]
        self.assertEqual(cell2[0], 11)
        self.assertEqual(cell2[1], 18)
        # Track 42 cells updated to new location
        self.assertIn(cell2, res2.final_cells)

    def test_calibration_json_persistence(self):
        """Verify that CameraCalibrator state persists to JSON and reloads identically."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            saved = self.calibrator.save_to_json(tmp_path)
            self.assertTrue(saved)

            new_calib = CameraCalibrator(config=CalibrationConfig(default_fiducials=[]))
            loaded = new_calib.load_from_json(tmp_path)
            self.assertTrue(loaded)
            self.assertTrue(new_calib.is_calibrated)
            self.assertEqual(len(new_calib.active_fiducials), 4)
            self.assertAlmostEqual(new_calib.reprojection_error_px, self.calibrator.reprojection_error_px, delta=1e-3)

            # Test point projection equivalence
            pt_orig = self.calibrator.transform_image_to_ground((320.0, 266.67))
            pt_new = new_calib.transform_image_to_ground((320.0, 266.67))
            self.assertAlmostEqual(pt_orig[0], pt_new[0], delta=1e-3)
            self.assertAlmostEqual(pt_orig[1], pt_new[1], delta=1e-3)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_aruco_dynamic_autocalibration(self):
        """Verify that 4-corner ArUco markers dynamically calibrate homography in real-time."""
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)

        # Place 4 markers at canonical perspective quad:
        # ID 0: TL at (100, 120), ID 1: TR at (540, 120), ID 2: BR at (580, 440), ID 3: BL at (60, 440)
        positions = [
            (0, 100, 120),
            (1, 540, 120),
            (2, 580, 440),
            (3, 60, 440),
        ]
        m_size = 60
        for mid, cx, cy in positions:
            m_img = cv2.aruco.generateImageMarker(dictionary, mid, m_size)
            m_bgr = cv2.cvtColor(m_img, cv2.COLOR_GRAY2BGR)
            x1 = cx - m_size // 2
            y1 = cy - m_size // 2
            frame[y1:y1 + m_size, x1:x1 + m_size] = m_bgr

        # Execute dynamic auto-calibration
        active, count, markers = self.calibrator.detect_and_update_aruco(frame)
        self.assertTrue(active)
        self.assertEqual(count, 4)
        self.assertTrue(self.calibrator.is_aruco_tracking)
        self.assertTrue(self.calibrator.is_calibrated)

        # Test projective mapping: Center of Marker 0 (100, 120) should map to ground origin (0, 0)
        gnd_0 = self.calibrator.transform_image_to_ground((100.0, 120.0))
        self.assertIsNotNone(gnd_0)
        self.assertAlmostEqual(gnd_0[0], 0.0, delta=1.0)
        self.assertAlmostEqual(gnd_0[1], 0.0, delta=1.0)

        # Test partial marker visibility (cover Marker 3) -> should revert to static fallback
        partial_frame = frame.copy()
        partial_frame[400:480, 0:120] = 255  # Erase BL marker
        active_part, count_part, _ = self.calibrator.detect_and_update_aruco(partial_frame)
        self.assertFalse(active_part)
        self.assertEqual(count_part, 3)
        self.assertFalse(self.calibrator.is_aruco_tracking)
        self.assertTrue(self.calibrator.is_calibrated)  # Safe static fallback remains intact

    def test_contact_point_refinement(self):
        """Verify that contact point refinement accurately detects bottom boundary."""
        frame = np.full((480, 640, 3), 220, dtype=np.uint8)  # Light table surface
        # Dark upright box
        cv2.rectangle(frame, (280, 200), (360, 350), (40, 30, 20), -1)

        refined_cx, refined_cy = self.mapper.refine_contact_point(frame, (280, 200, 360, 350), frame_height=480)
        self.assertEqual(refined_cx, 320)
        self.assertAlmostEqual(refined_cy, 350, delta=3)


if __name__ == "__main__":
    unittest.main()
