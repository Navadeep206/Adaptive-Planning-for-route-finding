"""Unit tests for Day 6 Perception-to-Grid Obstacle Mapping.

Verifies coordinate transformation, bounds validation, confidence filtering,
inflation margins, stale detection timeout, vehicle/goal cell protection,
and seamless integration with RoadEnvironment.
Zero reliance on webcam hardware or downloaded YOLO weights.
"""

import unittest
import time
import os
import pygame

from perception.object_detector import Detection
from perception.grid_mapper import (
    GridMapper,
    GridMapperConfig,
    GridMappingResult,
    MappingMode
)
from simulation.environment import RoadEnvironment, CellType, NavigationMode


class TestGridMapperUnit(unittest.TestCase):
    """Test suite for standalone GridMapper algorithm and safety rules."""

    def setUp(self) -> None:
        """Set up standard configuration and mapper for testing."""
        self.config = GridMapperConfig(
            min_confidence=0.35,
            grid_rows=22,
            grid_cols=26,
            roi_norm=(0.10, 0.10, 0.90, 0.90),
            contact_point="bottom_center",
            inflation_radius=0,
            stale_timeout_sec=2.0
        )
        self.mapper = GridMapper(config=self.config)
        self.frame_w = 640
        self.frame_h = 480

    def test_basic_coordinate_mapping(self) -> None:
        """Test projecting a detection within the ROI into the correct grid coordinate."""
        # Detection centered in ROI
        # ROI is [0.10, 0.10, 0.90, 0.90].
        # Frame center: x = 320 (u=0.5), y = 240 (v=0.5).
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=(300, 200, 340, 240)  # cx=320, bottom y2=240
        )
        cell = self.mapper.map_detection_to_cell(det, self.frame_w, self.frame_h)
        self.assertIsNotNone(cell)
        r, c = cell
        # u=0.5 is exactly midway in [0.10, 0.90], so norm_x=0.5 -> col = round(0 + 0.5*25) = 12 or 13
        self.assertTrue(11 <= c <= 13)
        self.assertTrue(9 <= r <= 12)

    def test_contact_point_bottom_center_vs_center(self) -> None:
        """Verify contact point calculation for bottom_center vs center modes."""
        det = Detection(
            class_id=2,
            class_name="car",
            confidence=0.90,
            bbox=(100, 100, 200, 300)  # cx = 150, cy = 200, y2 = 300
        )
        # Default: bottom_center
        pt_bottom = self.mapper.get_contact_point(det)
        self.assertEqual(pt_bottom, (150, 300))

        # Center mode
        self.mapper.config.contact_point = "center"
        pt_center = self.mapper.get_contact_point(det)
        self.assertEqual(pt_center, (150, 200))

    def test_confidence_filtering(self) -> None:
        """Test that detections below min_confidence are rejected."""
        det_low = Detection(class_id=0, class_name="person", confidence=0.20, bbox=(300, 200, 340, 240))
        det_high = Detection(class_id=0, class_name="person", confidence=0.80, bbox=(300, 200, 340, 240))

        result = self.mapper.map_detections(
            [det_low, det_high],
            (self.frame_h, self.frame_w),
            current_time=100.0
        )

        self.assertEqual(result.raw_detections_count, 2)
        self.assertEqual(result.filtered_detections_count, 1)
        self.assertEqual(len(result.final_cells), 1)

    def test_target_class_filtering(self) -> None:
        """Test filtering by specific target classes."""
        self.mapper.config.target_classes = {"car", "truck"}

        det_person = Detection(class_id=0, class_name="person", confidence=0.9, bbox=(300, 200, 340, 240))
        det_car = Detection(class_id=2, class_name="car", confidence=0.9, bbox=(300, 200, 340, 240))

        result = self.mapper.map_detections([det_person, det_car], (self.frame_h, self.frame_w))
        self.assertEqual(result.filtered_detections_count, 1)
        self.assertEqual(len(result.final_cells), 1)

    def test_out_of_bounds_rejection(self) -> None:
        """Verify detections with contact points outside frame or ROI are rejected."""
        # Detection outside ROI at top-left: cx=20, y2=20 -> u=0.031, v=0.041 (ROI min is 0.10)
        det_outside_roi = Detection(class_id=0, class_name="person", confidence=0.9, bbox=(10, 10, 30, 20))
        cell = self.mapper.map_detection_to_cell(det_outside_roi, self.frame_w, self.frame_h)
        self.assertIsNone(cell)

        # Invalid frame dimensions
        cell_invalid_frame = self.mapper.map_detection_to_cell(det_outside_roi, 0, 0)
        self.assertIsNone(cell_invalid_frame)

    def test_obstacle_inflation_safety_margin(self) -> None:
        """Verify inflation margin expands obstacle cells by radius without overflowing grid."""
        # Base cell at (10, 10)
        base = {(10, 10)}
        inflated_1 = self.mapper.inflate_cells(base, radius=1)
        # 3x3 square = 9 cells
        self.assertEqual(len(inflated_1), 9)
        self.assertIn((10, 10), inflated_1)
        self.assertIn((9, 9), inflated_1)
        self.assertIn((11, 11), inflated_1)

        # Corner cell at (0, 0) - should clamp to grid bounds (no negative coordinates)
        corner_base = {(0, 0)}
        inflated_corner = self.mapper.inflate_cells(corner_base, radius=1)
        # 2x2 = 4 cells within bounds
        self.assertEqual(len(inflated_corner), 4)
        for r, c in inflated_corner:
            self.assertTrue(0 <= r < self.config.grid_rows)
            self.assertTrue(0 <= c < self.config.grid_cols)

    def test_stale_detection_expiration(self) -> None:
        """Verify detections expire after stale_timeout_sec if not refreshed."""
        det = Detection(class_id=0, class_name="box", confidence=0.9, bbox=(300, 200, 340, 240))

        # 1. First observation at t=10.0
        res1 = self.mapper.map_detections([det], (self.frame_h, self.frame_w), current_time=10.0)
        self.assertEqual(len(res1.final_cells), 1)
        active_cell = list(res1.final_cells)[0]

        # 2. Check at t=11.0 (within 2.0s timeout) without new detections
        active_11 = self.mapper.prune_stale_cells(current_time=11.0)
        self.assertIn(active_cell, active_11)

        # 3. Check at t=12.5 (>2.0s timeout) without new detections -> should expire
        active_12_5 = self.mapper.prune_stale_cells(current_time=12.5)
        self.assertEqual(len(active_12_5), 0)
        self.assertNotIn(active_cell, active_12_5)

    def test_vehicle_cell_protection_invariant(self) -> None:
        """CRITICAL INVARIANT: Candidate obstacle overlapping vehicle position is strictly rejected."""
        # Set vehicle position
        veh_pos = (5, 5)

        # Force a candidate cell directly at veh_pos
        det = Detection(class_id=0, class_name="hazard", confidence=0.9, bbox=(300, 200, 340, 240))
        # Mock map_detection_to_cell to return veh_pos
        self.mapper.map_detection_to_cell = lambda *args, **kwargs: veh_pos

        result = self.mapper.map_detections(
            [det],
            (self.frame_h, self.frame_w),
            current_time=50.0,
            vehicle_pos=veh_pos,
            goal_pos=(20, 20)
        )

        # Vehicle cell must NOT be in final obstacle cells
        self.assertNotIn(veh_pos, result.final_cells)
        self.assertIn(veh_pos, result.vehicle_conflicts)
        self.assertEqual(self.mapper.total_vehicle_conflicts, 1)

    def test_goal_cell_protection_invariant(self) -> None:
        """CRITICAL INVARIANT: Candidate obstacle overlapping goal position is strictly rejected."""
        goal_pos = (20, 15)

        det = Detection(class_id=0, class_name="hazard", confidence=0.9, bbox=(300, 200, 340, 240))
        self.mapper.map_detection_to_cell = lambda *args, **kwargs: goal_pos

        result = self.mapper.map_detections(
            [det],
            (self.frame_h, self.frame_w),
            current_time=50.0,
            vehicle_pos=(2, 12),
            goal_pos=goal_pos
        )

        # Goal cell must NOT be in final obstacle cells
        self.assertNotIn(goal_pos, result.final_cells)
        self.assertIn(goal_pos, result.goal_conflicts)
        self.assertEqual(self.mapper.total_goal_conflicts, 1)

    def test_clear_resets_tracked_cells(self) -> None:
        """Test that clear() purges all tracked cells and conflict counters."""
        det = Detection(class_id=0, class_name="box", confidence=0.9, bbox=(300, 200, 340, 240))
        self.mapper.map_detections([det], (self.frame_h, self.frame_w), current_time=1.0)
        self.assertTrue(len(self.mapper.get_active_cells()) > 0)

        self.mapper.clear()
        self.assertEqual(len(self.mapper.get_active_cells()), 0)
        self.assertEqual(self.mapper.total_vehicle_conflicts, 0)
        self.assertEqual(self.mapper.total_goal_conflicts, 0)


class TestEnvironmentPerceptionIntegration(unittest.TestCase):
    """Integration test suite for RoadEnvironment perception handling."""

    def setUp(self) -> None:
        """Initialize RoadEnvironment without GUI window."""
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.env = RoadEnvironment()

    def tearDown(self) -> None:
        """Clean up pygame."""
        pygame.quit()

    def test_toggle_perception_mapping(self) -> None:
        """Test toggling perception mapping enabled/disabled."""
        self.assertFalse(self.env.perception_enabled)

        # Toggle on
        state_on = self.env.toggle_perception_mapping()
        self.assertTrue(state_on)
        self.assertTrue(self.env.perception_enabled)

        # Toggle off
        state_off = self.env.toggle_perception_mapping()
        self.assertFalse(state_off)
        self.assertFalse(self.env.perception_enabled)

    def test_set_and_clear_perception_obstacles(self) -> None:
        """Test applying perception obstacles to grid and clearing them cleanly."""
        self.env.perception_enabled = True
        test_cells = {(3, 12), (3, 13)}

        # Both cells should be drivable road before perception
        for cell in test_cells:
            self.assertEqual(self.env.grid[cell], CellType.DRIVABLE_ROAD)

        # Apply perception obstacles
        self.env.set_perception_obstacles(test_cells)
        for cell in test_cells:
            self.assertEqual(self.env.grid[cell], CellType.OBSTACLE)
            self.assertIn(cell, self.env.perception_obstacle_set)
            self.assertIn(cell, self.env.obstacle_set)

        # Clear perception obstacles
        self.env.clear_perception_obstacles()
        for cell in test_cells:
            self.assertEqual(self.env.grid[cell], CellType.DRIVABLE_ROAD)
            self.assertNotIn(cell, self.env.perception_obstacle_set)
            self.assertNotIn(cell, self.env.obstacle_set)

    def test_perception_does_not_destroy_static_or_dynamic_obstacles(self) -> None:
        """Perception clearing must never destroy static hazards or dynamically spawned obstacles."""
        self.env.perception_enabled = True

        # Pick an existing static obstacle
        static_cell = list(self.env.initial_obstacle_coords)[0]
        self.assertEqual(self.env.grid[static_cell], CellType.OBSTACLE)

        # Spawn a dynamic obstacle
        dynamic_cell = self.env.insert_dynamic_obstacle(ahead_steps=1)
        self.assertIsNotNone(dynamic_cell)
        self.assertEqual(self.env.grid[dynamic_cell], CellType.OBSTACLE)

        # Include both in perception cells
        self.env.set_perception_obstacles({static_cell, dynamic_cell, (3, 12)})
        self.assertEqual(self.env.grid[static_cell], CellType.OBSTACLE)
        self.assertEqual(self.env.grid[dynamic_cell], CellType.OBSTACLE)

        # Clear perception obstacles
        self.env.clear_perception_obstacles()

        # Static and dynamic hazards MUST remain OBSTACLE
        self.assertEqual(self.env.grid[static_cell], CellType.OBSTACLE)
        self.assertEqual(self.env.grid[dynamic_cell], CellType.OBSTACLE)
        self.assertIn(static_cell, self.env.obstacle_set)
        self.assertIn(dynamic_cell, self.env.obstacle_set)
        # Only the pure perception cell is restored to DRIVABLE_ROAD
        self.assertEqual(self.env.grid[(3, 12)], CellType.DRIVABLE_ROAD)

    def test_perception_obstacle_replan_on_active_route(self) -> None:
        """Placing a perception obstacle on the planned route triggers adaptive replanning."""
        self.env.perception_enabled = True
        plan_res = self.env.plan_path()
        self.assertTrue(plan_res.success)
        self.assertTrue(len(self.env.planned_path) > 3)

        # Obstacle placed on an upcoming waypoint
        blocked_waypoint = self.env.planned_path[2]
        old_replans = self.env.replanning_events

        # Autonomous mode active
        self.env.nav_mode = NavigationMode.AUTONOMOUS
        self.env.set_perception_obstacles({blocked_waypoint})

        # Replanning should have triggered
        self.assertGreater(self.env.replanning_events, old_replans)
        # Blocked waypoint should no longer be in new planned path
        self.assertNotIn(blocked_waypoint, self.env.planned_path)

    def test_reset_vehicle_clears_perception_obstacles(self) -> None:
        """Resetting vehicle clears perception-derived cells."""
        self.env.perception_enabled = True
        test_cell = (3, 12)
        self.env.set_perception_obstacles({test_cell})
        self.assertIn(test_cell, self.env.perception_obstacle_set)

        self.env.reset_vehicle()
        self.assertEqual(len(self.env.perception_obstacle_set), 0)
        self.assertEqual(self.env.grid[test_cell], CellType.DRIVABLE_ROAD)

    def test_headless_render_with_perception_active(self) -> None:
        """Render complete simulation frame with perception active in headless mode."""
        from main import initialize_fonts
        fonts = initialize_fonts()
        screen = pygame.Surface((RoadEnvironment.WINDOW_WIDTH, RoadEnvironment.WINDOW_HEIGHT))

        self.env.perception_enabled = True
        self.env.set_perception_obstacles({(3, 12), (4, 12)})

        # Render should complete with zero exceptions
        self.env.render(screen, fonts, webcam_status="ACTIVE (30.0 FPS • 640x480)")
        self.assertEqual(screen.get_size(), (1000, 700))


if __name__ == "__main__":
    unittest.main()
