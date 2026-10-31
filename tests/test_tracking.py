"""Unit test suite for real-time multi-object tracking and temporal detection stability.

Tests track ID assignment, ID persistence, multiple object tracking, temporary missed
detection coasting, track expiration timeouts, image-plane velocity estimation, empty frame
resilience, visual rendering, and tracker reset.
Executes 100% locally and headlessly in CI without requiring a physical camera or downloaded weights.
"""

from collections import deque
from unittest.mock import MagicMock, patch
import os
import unittest
import numpy as np

# Ensure headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

from perception.object_detector import Detection, YOLOObjectDetector
from perception.object_tracker import ObjectTracker, TrackedObject, TrackerConfig, TrackState


class TestObjectTracker(unittest.TestCase):
    """Test suite for ObjectTracker temporal stability and motion estimation."""

    def setUp(self):
        """Prepare tracker and deterministic test detections."""
        self.config = TrackerConfig(
            min_confidence=0.25,
            track_activation_threshold=2,
            lost_timeout_sec=0.5,
            max_lost_frames=5,
            history_len=10,
            iou_threshold=0.3,
            enable_box_smoothing=True,
            smoothing_alpha=0.7
        )
        self.tracker = ObjectTracker(config=self.config)
        self.sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_track_id_assignment(self):
        """Verify that newly detected objects receive unique, persistent track IDs."""
        det1 = Detection(class_id=0, class_name="bottle", confidence=0.85, bbox=(100, 100, 160, 220))
        tracks, _ = self.tracker.update(detections=[det1], timestamp=100.0)

        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.class_name, "bottle")
        self.assertEqual(track.state, TrackState.NEW)
        self.assertEqual(track.hits, 1)

    def test_id_persistence_across_consecutive_frames(self):
        """Verify that an object moving smoothly maintains its track ID and confirms."""
        # Frame 1: Detection at (100, 100, 160, 220)
        d1 = Detection(class_id=0, class_name="bottle", confidence=0.85, bbox=(100, 100, 160, 220))
        self.tracker.update(detections=[d1], timestamp=100.0)

        # Frame 2: Slightly shifted detection at (104, 102, 164, 222)
        d2 = Detection(class_id=0, class_name="bottle", confidence=0.88, bbox=(104, 102, 164, 222))
        tracks, _ = self.tracker.update(detections=[d2], timestamp=100.05)

        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.hits, 2)
        self.assertEqual(track.state, TrackState.CONFIRMED)
        self.assertTrue(track.is_confirmed)

    def test_multiple_objects_tracked_simultaneously(self):
        """Verify tracking multiple spatially distinct objects simultaneously."""
        d1 = Detection(class_id=0, class_name="bottle", confidence=0.80, bbox=(50, 50, 100, 150))
        d2 = Detection(class_id=1, class_name="person", confidence=0.90, bbox=(300, 100, 420, 350))

        tracks, _ = self.tracker.update(detections=[d1, d2], timestamp=100.0)
        self.assertEqual(len(tracks), 2)
        ids = {t.track_id for t in tracks}
        self.assertEqual(ids, {1, 2})

        # Frame 2: Both objects present with slight shift
        d1_next = Detection(class_id=0, class_name="bottle", confidence=0.82, bbox=(52, 52, 102, 152))
        d2_next = Detection(class_id=1, class_name="person", confidence=0.91, bbox=(305, 102, 425, 352))

        tracks2, _ = self.tracker.update(detections=[d1_next, d2_next], timestamp=100.05)
        self.assertEqual(len(tracks2), 2)
        ids2 = {t.track_id for t in tracks2}
        self.assertEqual(ids2, {1, 2})
        for t in tracks2:
            self.assertTrue(t.is_confirmed)

    def test_temporary_missed_detections_coasting(self):
        """Verify that a confirmed track enters LOST state when missed, preserving state."""
        # 2 frames to confirm track
        d1 = Detection(class_id=0, class_name="bottle", confidence=0.85, bbox=(100, 100, 160, 220))
        self.tracker.update(detections=[d1], timestamp=100.0)
        self.tracker.update(detections=[d1], timestamp=100.05)

        confirmed_tracks = self.tracker.get_confirmed_tracks()
        self.assertEqual(len(confirmed_tracks), 1)

        # Frame 3: Temporary occlusion / missing detection
        tracks, _ = self.tracker.update(detections=[], timestamp=100.10)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)
        self.assertEqual(tracks[0].state, TrackState.LOST)
        self.assertTrue(tracks[0].is_lost)
        self.assertEqual(tracks[0].consecutive_misses, 1)

    def test_reappearing_object_recovery(self):
        """Verify that an object missing for 1 frame recovers without an ID switch."""
        d1 = Detection(class_id=0, class_name="bottle", confidence=0.85, bbox=(100, 100, 160, 220))
        self.tracker.update(detections=[d1], timestamp=100.0)
        self.tracker.update(detections=[d1], timestamp=100.05)

        # Missing detection in frame 3
        self.tracker.update(detections=[], timestamp=100.10)
        self.assertTrue(self.tracker.get_active_tracks()[0].is_lost)

        # Reappears in frame 4
        d_reappear = Detection(class_id=0, class_name="bottle", confidence=0.84, bbox=(102, 101, 162, 221))
        tracks, _ = self.tracker.update(detections=[d_reappear], timestamp=100.15)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)
        self.assertEqual(tracks[0].state, TrackState.CONFIRMED)
        self.assertEqual(tracks[0].consecutive_misses, 0)

    def test_track_expiration_after_timeout(self):
        """Verify that a lost track expires and is pruned after max_lost_frames / timeout."""
        d1 = Detection(class_id=0, class_name="bottle", confidence=0.85, bbox=(100, 100, 160, 220))
        self.tracker.update(detections=[d1], timestamp=100.0)
        self.tracker.update(detections=[d1], timestamp=100.05)

        # Deliver 5 consecutive missed frames (matching max_lost_frames=5)
        for i in range(1, 6):
            self.tracker.update(detections=[], timestamp=100.05 + i * 0.05)

        # Track should now be expired and evicted
        active_tracks = self.tracker.get_active_tracks()
        self.assertEqual(len(active_tracks), 0)
        self.assertEqual(self.tracker.active_track_count, 0)

    def test_image_plane_velocity_estimation(self):
        """Verify 2D image-plane velocity estimation (pixels/second) from consecutive detections."""
        # Frame 1 at t = 100.0, center = (100, 100)
        d1 = Detection(class_id=0, class_name="car", confidence=0.90, bbox=(80, 80, 120, 120))
        self.tracker.update(detections=[d1], timestamp=100.0)

        # Frame 2 at t = 100.1 (0.1s later), moved by dx = +8px, dy = +5px -> center = (108, 105)
        d2 = Detection(class_id=0, class_name="car", confidence=0.90, bbox=(88, 85, 128, 125))
        tracks, _ = self.tracker.update(detections=[d2], timestamp=100.1)

        track = tracks[0]
        # Expect image-plane velocity in positive direction
        self.assertGreater(track.velocity[0], 0.0)
        self.assertGreater(track.velocity[1], 0.0)
        self.assertGreater(track.speed_px_sec, 0.0)
        self.assertIsNotNone(track.previous_center)

    def test_empty_or_invalid_frame_resilience(self):
        """Verify that None, 0-sized, or invalid frames do not cause unhandled exceptions."""
        # Empty array
        empty_frame = np.array([], dtype=np.uint8)
        tracks, telemetry = self.tracker.update(frame=empty_frame)
        self.assertEqual(tracks, [])

        # None frame
        tracks2, telemetry2 = self.tracker.update(frame=None)
        self.assertEqual(tracks2, [])

    def test_tracker_reset(self):
        """Verify that reset() clears all tracks and resets track ID sequence."""
        d1 = Detection(class_id=0, class_name="bottle", confidence=0.85, bbox=(100, 100, 160, 220))
        self.tracker.update(detections=[d1], timestamp=100.0)
        self.assertEqual(self.tracker.active_track_count, 1)

        self.tracker.reset()
        self.assertEqual(self.tracker.active_track_count, 0)
        self.assertEqual(self.tracker.lost_track_count, 0)

        # Next detection should start with track_id = 1 again
        tracks, _ = self.tracker.update(detections=[d1], timestamp=200.0)
        self.assertEqual(tracks[0].track_id, 1)

    def test_visual_rendering(self):
        """Verify draw_tracks renders bounding boxes, badges, trails, and HUD without error."""
        d1 = Detection(class_id=0, class_name="pedestrian", confidence=0.88, bbox=(50, 50, 120, 200))
        self.tracker.update(detections=[d1], timestamp=100.0)
        self.tracker.update(detections=[d1], timestamp=100.05)

        active = self.tracker.get_active_tracks()
        rendered = self.tracker.draw_tracks(
            frame=self.sample_frame,
            tracks=active,
            show_trails=True,
            show_velocity=True,
            fps=30.0
        )
        self.assertEqual(rendered.shape, self.sample_frame.shape)
        self.assertEqual(rendered.dtype, np.uint8)
        # Rendered image should have modifications (non-zero pixels)
        self.assertGreater(np.count_nonzero(rendered), 0)

    @patch("perception.object_detector.logger")
    def test_ultralytics_track_fallback_on_exception(self, mock_logger):
        """Verify that if model.track() throws an exception, tracker falls back to detect()."""
        mock_detector = MagicMock()
        mock_detector.is_loaded.return_value = True
        mock_detector.apply_preprocessing.return_value = (self.sample_frame, 0.0)
        mock_detector.iou_threshold = 0.45
        mock_detector.imgsz = 640
        mock_detector.max_det = 50
        mock_detector.device = "cpu"

        # Simulate exception in model.track()
        mock_detector._model.track.side_effect = RuntimeError("Kernel failure")
        mock_detector.detect.return_value = ([
            Detection(class_id=0, class_name="hazard", confidence=0.75, bbox=(10, 10, 60, 60))
        ], 12.5)

        tracker = ObjectTracker(detector=mock_detector)
        tracks, telemetry = tracker.update(frame=self.sample_frame)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].class_name, "hazard")
        mock_detector.detect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
