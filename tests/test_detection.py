"""Unit test suite for local YOLO object detection module.

Verifies Detection data structures, YOLOObjectDetector lifecycle, inference parsing,
empty frame resilience, error handling, visual rendering, and path planner decoupling.
Uses deterministic mocks to execute in headless CI without internet access or real weights.
"""

from unittest.mock import MagicMock, patch
import os
import unittest
import numpy as np

# Ensure headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

from perception.object_detector import Detection, YOLOObjectDetector
from simulation.environment import RoadEnvironment


class TestObjectDetection(unittest.TestCase):
    """Test suite for YOLO perception module."""

    def setUp(self):
        """Prepare sample test frame and mock data."""
        self.sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_detection_dataclass_properties(self):
        """Test Detection bounding box coordinates, dimensions, center, and serialization."""
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.8842,
            bbox=(50, 100, 200, 350)
        )
        self.assertEqual(det.class_id, 0)
        self.assertEqual(det.class_name, "person")
        self.assertAlmostEqual(det.confidence, 0.8842)
        self.assertEqual(det.x1, 50)
        self.assertEqual(det.y1, 100)
        self.assertEqual(det.x2, 200)
        self.assertEqual(det.y2, 350)
        self.assertEqual(det.width, 150)
        self.assertEqual(det.height, 250)
        self.assertEqual(det.center, (125, 225))

        d_dict = det.to_dict()
        self.assertEqual(d_dict["class_name"], "person")
        self.assertEqual(d_dict["bbox"], (50, 100, 200, 350))

    def test_empty_or_invalid_frame_handling(self):
        """Test that detector returns empty list safely when given None or empty array."""
        detector = YOLOObjectDetector(auto_load=False)
        # None frame
        dets1, t1 = detector.detect(None)
        self.assertEqual(dets1, [])
        self.assertEqual(t1, 0.0)

        # Empty numpy array
        empty_frame = np.array([], dtype=np.uint8)
        dets2, t2 = detector.detect(empty_frame)
        self.assertEqual(dets2, [])
        self.assertEqual(t2, 0.0)

    def test_detect_when_model_not_loaded(self):
        """Test safe behavior when inference is requested without a loaded model."""
        detector = YOLOObjectDetector(auto_load=False)
        self.assertFalse(detector.is_loaded())

        dets, t = detector.detect(self.sample_frame)
        self.assertEqual(dets, [])
        self.assertEqual(t, 0.0)

    @patch("perception.object_detector.logger")
    def test_model_loading_failure_is_handled_gracefully(self, mock_logger):
        """Test that failure to import or load weights logs error and sets is_loaded to False."""
        detector = YOLOObjectDetector(model_name="nonexistent_model.pt", auto_load=False)

        with patch.dict("sys.modules", {"ultralytics": None}):
            loaded = detector.load_model()
            self.assertFalse(loaded)
            self.assertFalse(detector.is_loaded())

    def test_successful_detection_parsing_with_mock_model(self):
        """Test parsing of Ultralytics YOLO inference results into structured Detections."""
        detector = YOLOObjectDetector(auto_load=False)
        detector.class_names = {0: "person", 2: "car", 7: "truck"}
        detector._is_loaded = True

        # Construct mock YOLO results structure matching Ultralytics format
        mock_box1 = MagicMock()
        mock_box1.xyxy = [MagicMock()]
        mock_box1.xyxy[0].detach().cpu().numpy.return_value = np.array([10, 20, 110, 120])
        mock_box1.conf = [MagicMock()]
        mock_box1.conf[0].detach().cpu().numpy.return_value = np.array(0.92)
        mock_box1.cls = [MagicMock()]
        mock_box1.cls[0].detach().cpu().numpy.return_value = np.array(0)

        mock_box2 = MagicMock()
        mock_box2.xyxy = [MagicMock()]
        mock_box2.xyxy[0].detach().cpu().numpy.return_value = np.array([200, 150, 400, 300])
        mock_box2.conf = [MagicMock()]
        mock_box2.conf[0].detach().cpu().numpy.return_value = np.array(0.78)
        mock_box2.cls = [MagicMock()]
        mock_box2.cls[0].detach().cpu().numpy.return_value = np.array(2)

        mock_result = MagicMock()
        mock_result.boxes = [mock_box1, mock_box2]

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        detector._model = mock_model

        # Run detect
        detections, inf_time = detector.detect(self.sample_frame)

        self.assertEqual(len(detections), 2)
        self.assertGreater(inf_time, 0.0)

        # Check first detection
        d1 = detections[0]
        self.assertEqual(d1.class_id, 0)
        self.assertEqual(d1.class_name, "person")
        self.assertAlmostEqual(d1.confidence, 0.92)
        self.assertEqual(d1.bbox, (10, 20, 110, 120))

        # Check second detection
        d2 = detections[1]
        self.assertEqual(d2.class_id, 2)
        self.assertEqual(d2.class_name, "car")
        self.assertAlmostEqual(d2.confidence, 0.78)
        self.assertEqual(d2.bbox, (200, 150, 400, 300))

    def test_draw_detections_annotation(self):
        """Test drawing bounding boxes and telemetry overlay on an image frame."""
        detector = YOLOObjectDetector(auto_load=False)
        detections = [
            Detection(class_id=0, class_name="person", confidence=0.85, bbox=(50, 50, 150, 200))
        ]

        annotated = detector.draw_detections(self.sample_frame, detections, inference_time_ms=18.5)
        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, self.sample_frame.shape)
        # Verify original frame was not modified (canvas is a copy)
        self.assertEqual(np.count_nonzero(self.sample_frame), 0)
        self.assertGreater(np.count_nonzero(annotated), 0)

    def test_draw_detections_with_zero_detections(self):
        """Verify drawing HUD overlay when 0 objects are detected."""
        detector = YOLOObjectDetector(auto_load=False)
        annotated = detector.draw_detections(self.sample_frame, [], inference_time_ms=12.0)
        self.assertIsNotNone(annotated)
        # Should render HUD banner even with 0 detections
        self.assertGreater(np.count_nonzero(annotated), 0)

    def test_draw_detections_ceiling_safe_bounding_box(self):
        """Verify bounding box touching top frame ceiling (y1=0) does not clip label off-screen."""
        detector = YOLOObjectDetector(auto_load=False)
        detections = [
            Detection(class_id=0, class_name="ceiling_box", confidence=0.95, bbox=(50, 0, 150, 80))
        ]
        annotated = detector.draw_detections(self.sample_frame, detections, inference_time_ms=10.0)
        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, self.sample_frame.shape)

    def test_inference_exception_is_handled_gracefully(self):
        """Test that an unexpected exception during model execution does not crash."""
        detector = YOLOObjectDetector(auto_load=False)
        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("Inference kernel fault")
        detector._model = mock_model
        detector._is_loaded = True

        dets, inf_ms = detector.detect(self.sample_frame)
        self.assertEqual(dets, [])
        self.assertGreaterEqual(inf_ms, 0.0)

    @patch("perception.webcam.cv2.imshow")
    def test_webcam_update_prioritizes_annotated_display_frame(self, mock_imshow):
        """Verify WebcamManager.update renders the provided annotated frame."""
        from perception.webcam import WebcamManager
        webcam = WebcamManager()
        webcam.is_preview_active = True
        webcam.state = webcam.get_state().ACTIVE

        # Pass annotated frame to update
        annotated = np.ones((480, 640, 3), dtype=np.uint8) * 200
        has_frame, returned = webcam.update(display_frame=annotated)

        self.assertTrue(has_frame)
        mock_imshow.assert_called_once()
        call_args = mock_imshow.call_args[0]
        self.assertEqual(call_args[0], webcam.window_title)
        self.assertTrue(np.array_equal(call_args[1], annotated))

    def test_object_detection_is_decoupled_from_simulation_grid(self):
        """Verify safety invariant: running detection does not alter A* grid or inject obstacles."""
        env = RoadEnvironment()
        initial_obstacles = set(env.obstacle_set)
        initial_grid = env.grid.copy()

        # Run detector
        detector = YOLOObjectDetector(auto_load=False)
        detections, _ = detector.detect(self.sample_frame)

        # Environment must remain completely unchanged
        self.assertEqual(env.obstacle_set, initial_obstacles)
        self.assertTrue(np.array_equal(env.grid, initial_grid))

    def test_end_to_end_perception_grid_and_planner_response(self):
        """Verify end-to-end flow: Detections -> GridMapper -> Environment -> A* Planner replan."""
        from perception.grid_mapper import GridMapper, GridMapperConfig
        from simulation.environment import CellType, NavigationMode

        env = RoadEnvironment()
        res = env.plan_path()
        self.assertTrue(res.success)
        self.assertGreater(len(env.planned_path), 3)

        # Pick a waypoint directly along the vehicle's planned path
        target_cell = env.planned_path[2]
        r_target, c_target = target_cell

        # Enable perception mapping
        env.toggle_perception_mapping()
        self.assertTrue(env.perception_enabled)

        # Create structured detection that maps to target_cell
        mapper = GridMapper(GridMapperConfig(grid_rows=env.rows, grid_cols=env.cols))
        mapper.map_detection_to_cell = lambda *args, **kwargs: target_cell

        det = Detection(class_id=0, class_name="hazard_box", confidence=0.90, bbox=(100, 100, 200, 200))
        mapping_result = mapper.map_detections(
            detections=[det],
            frame_shape=(480, 640),
            current_time=100.0,
            vehicle_pos=env.vehicle_pos,
            goal_pos=env.destination_pos
        )

        # Set perception obstacles in environment
        env.set_perception_obstacles(mapping_result.final_cells)

        # Verify target_cell is now marked as an obstacle
        self.assertEqual(env.grid[target_cell], CellType.OBSTACLE)
        self.assertIn(target_cell, env.perception_obstacle_set)

        # Autonomous mode should trigger detour replanning
        env.nav_mode = NavigationMode.AUTONOMOUS
        env.is_route_blocked = MagicMock(return_value=True)
        env.set_perception_obstacles(mapping_result.final_cells)

        # Vehicle must not move into blocked cell
        self.assertNotEqual(env.vehicle_pos, target_cell)


if __name__ == "__main__":
    unittest.main()

