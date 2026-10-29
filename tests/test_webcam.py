"""Automated test suite for reliable MacBook webcam integration.

Tests camera initialization, device selection, frame acquisition, continuous streaming,
error recovery, idempotence, resource cleanup, and decoupling from A* path planning.
Uses deterministic mocks for hardware-independent CI execution.
"""

from unittest.mock import MagicMock, patch
import os
import sys
import time
import unittest
import numpy as np

# Ensure headless execution for SDL/Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

from perception.webcam import (
    WebcamManager,
    CameraState,
    get_macbook_builtin_camera_index
)
from simulation.environment import RoadEnvironment, NavigationMode


class TestMacBookWebcam(unittest.TestCase):
    """Unit tests for MacBook webcam selection, streaming, and error handling."""

    def setUp(self):
        """Prepare fresh mock frame for testing."""
        # 1080p dummy frame with distinct color channels
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.dummy_frame[:, :, 0] = 255  # Blue in BGR

    @patch("perception.webcam.cv2.VideoCapture")
    def test_successful_camera_initialization_and_frame_capture(self, mock_vcap_cls):
        """Test successful initialization and live frame capture with verified state transition."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self.dummy_frame)
        mock_vcap_cls.return_value = mock_cap

        cam = WebcamManager(device_index=0)
        self.assertEqual(cam.get_state(), CameraState.OFF)
        self.assertFalse(cam.is_active())

        # Start preview
        started = cam.start_preview()
        self.assertTrue(started)

        # Allow worker thread to capture at least one frame
        time.sleep(0.1)

        self.assertTrue(cam.is_active())
        self.assertEqual(cam.get_state(), CameraState.ACTIVE)
        self.assertIn("ACTIVE", cam.get_status())
        self.assertGreater(cam.frame_count, 0)
        self.assertEqual(cam.resolution, (640, 480))

        # Test frame access
        bgr_frame = cam.get_current_frame()
        self.assertIsNotNone(bgr_frame)
        self.assertEqual(bgr_frame.shape, (480, 640, 3))
        # Blue channel in BGR
        self.assertEqual(bgr_frame[0, 0, 0], 255)

        # Test RGB color conversion
        rgb_frame = cam.get_current_frame_rgb()
        self.assertIsNotNone(rgb_frame)
        # Red channel in RGB should now have the 255 from blue in BGR
        self.assertEqual(rgb_frame[0, 0, 2], 255)

        # Cleanup
        cam.release()
        self.assertFalse(cam.is_active())
        self.assertIsNone(cam.cap)

    @patch("perception.webcam.cv2.VideoCapture")
    def test_camera_unavailable_error_handling(self, mock_vcap_cls):
        """Test graceful handling when camera hardware is unavailable or permissions denied."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_vcap_cls.return_value = mock_cap

        cam = WebcamManager(device_index=0)
        started = cam.start_preview()

        self.assertFalse(started)
        self.assertFalse(cam.is_active())
        self.assertEqual(cam.get_state(), CameraState.ERROR)
        self.assertIn("Unavailable", cam.get_status())
        self.assertIsNone(cam.cap)

    @patch("perception.webcam.cv2.VideoCapture")
    def test_sustained_frame_read_failure_detection(self, mock_vcap_cls):
        """Test detection of sustained frame read errors and disconnection."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Read initially succeeds, then fails continuously
        mock_cap.read.side_effect = [(True, self.dummy_frame)] + [(False, None)] * 100
        mock_vcap_cls.return_value = mock_cap

        cam = WebcamManager(device_index=0)
        cam.start_preview()
        time.sleep(0.05)
        self.assertTrue(cam.is_active())

        # Fast-forward worker thread reading failures
        # In webcam.py, consecutive_errors > 45 triggers disconnection
        time.sleep(0.8)

        # Update should process stream error
        ready, frame = cam.update()
        self.assertFalse(ready)
        self.assertEqual(cam.get_state(), CameraState.ERROR)
        self.assertIn("Error", cam.get_status())

        cam.release()

    @patch("perception.webcam.cv2.VideoCapture")
    def test_repeated_toggle_preview_idempotence(self, mock_vcap_cls):
        """Test that repeatedly toggling preview is smooth, safe, and idempotent."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self.dummy_frame)
        mock_vcap_cls.return_value = mock_cap

        cam = WebcamManager(device_index=0)

        # Toggle 1: OFF -> ACTIVE
        state1 = cam.toggle_preview()
        self.assertTrue(state1)
        self.assertTrue(cam.is_preview_active)

        # Toggle 2: ACTIVE -> OFF
        state2 = cam.toggle_preview()
        self.assertFalse(state2)
        self.assertFalse(cam.is_preview_active)
        self.assertEqual(cam.get_state(), CameraState.OFF)

        # Toggle 3: OFF -> ACTIVE
        state3 = cam.toggle_preview()
        self.assertTrue(state3)
        self.assertTrue(cam.is_preview_active)

        # Clean release
        cam.release()
        self.assertFalse(cam.is_active())

    @patch("perception.webcam.cv2.VideoCapture")
    def test_no_duplicate_camera_sessions_created(self, mock_vcap_cls):
        """Test that calling start_preview when already open does not open duplicate handles."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self.dummy_frame)
        mock_vcap_cls.return_value = mock_cap

        cam = WebcamManager(device_index=0)
        cam.start_preview()
        first_cap = cam.cap

        # Call start_preview a second time while active
        cam.start_preview()
        self.assertIs(cam.cap, first_cap)
        # VideoCapture should have been instantiated exactly once
        self.assertEqual(mock_vcap_cls.call_count, 1)

        cam.release()

    def test_resource_cleanup_is_idempotent(self):
        """Test that release() can be called multiple times without throwing errors."""
        cam = WebcamManager(device_index=999)
        # Calling release on uninitialized camera
        cam.release()
        cam.release()
        cam.release()
        self.assertFalse(cam.is_active())
        self.assertIsNone(cam.cap)

    @patch("perception.webcam.cv2.VideoCapture")
    def test_camera_failure_does_not_affect_astar_or_autonomous_nav(self, mock_vcap_cls):
        """Verify safety invariant: camera errors never disrupt A* path planning or vehicle motion."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_vcap_cls.return_value = mock_cap

        # Camera fails completely
        cam = WebcamManager(device_index=999)
        cam.start_preview()
        self.assertEqual(cam.get_state(), CameraState.ERROR)

        # RoadEnvironment should plan and navigate independently without error
        env = RoadEnvironment()
        result = env.plan_path()
        self.assertTrue(result.success)
        self.assertGreater(result.path_length, 0)

        # Enable autonomous navigation and step vehicle along path
        initial_pos = env.vehicle_pos
        env.toggle_autonomous_navigation()
        self.assertEqual(env.nav_mode, NavigationMode.AUTONOMOUS)
        stepped = env.step_autonomous()
        self.assertTrue(stepped)
        self.assertNotEqual(env.vehicle_pos, initial_pos)
        self.assertEqual(env.collision_count, 0)

        cam.release()

    def test_device_index_configuration_via_environment_variable(self):
        """Test overriding camera device index with MACBOOK_CAMERA_INDEX env var."""
        with patch.dict(os.environ, {"MACBOOK_CAMERA_INDEX": "2"}):
            idx = get_macbook_builtin_camera_index()
            self.assertEqual(idx, 2)

            cam = WebcamManager()
            self.assertEqual(cam.device_index, 2)

    @patch("perception.webcam.subprocess.check_output")
    def test_device_detection_filters_iphone_and_picks_builtin_camera(self, mock_subprocess):
        """Test discovery parser prioritizes FaceTime camera over paired iPhone."""
        # System profiler JSON mock matching real macOS output
        mock_json = """
        {
          "SPCameraDataType" : [
            {
              "_name" : "Navadeep Camera",
              "spcamera_model-id" : "iPhone14,7",
              "spcamera_unique-id" : "B3F57152-F46C-4AFD-92DA-E02300000001"
            },
            {
              "_name" : "FaceTime HD Camera",
              "spcamera_model-id" : "FaceTime HD Camera",
              "spcamera_unique-id" : "EAB7A68F-EC2B-4487-AADF-D8A91C1CB782"
            }
          ]
        }
        """
        mock_subprocess.return_value = mock_json

        # Ensure environment variable does not override
        with patch.dict(os.environ, {}, clear=True):
            with patch("sys.platform", "darwin"):
                idx = get_macbook_builtin_camera_index()
                # FaceTime camera is at index 1 in this mock JSON; iPhone at 0 should be skipped
                self.assertEqual(idx, 1)


if __name__ == "__main__":
    unittest.main()
