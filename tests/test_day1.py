"""Automated verification suite for Day 1 baseline components.

Tests grid geometry, obstacle seeding, vehicle boundary enforcement,
collision prevention, reset behavior, and dummy headless rendering.
"""

import os
import unittest
import numpy as np
import pygame

# Set dummy audio and video driver so tests can run in headless CI environments
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from simulation.environment import RoadEnvironment, CellType
from perception.webcam import WebcamManager
from main import initialize_fonts


class TestDay1Baseline(unittest.TestCase):
    """Test suite verifying Day 1 core requirements."""

    def setUp(self):
        pygame.init()
        self.env = RoadEnvironment(rows=22, cols=26, cell_size=25)

    def tearDown(self):
        pygame.quit()

    def test_environment_dimensions(self):
        """Verify grid size and numpy array types."""
        self.assertEqual(self.env.grid.shape, (22, 26))
        self.assertEqual(self.env.grid.dtype, np.int8)

    def test_start_and_destination_validity(self):
        """Start and destination positions must be on drivable road and unobstructed."""
        sr, sc = self.env.start_pos
        dr, dc = self.env.destination_pos

        self.assertEqual(self.env.vehicle_pos, (sr, sc))
        self.assertEqual(self.env.grid[sr, sc], CellType.DRIVABLE_ROAD)
        self.assertEqual(self.env.grid[dr, dc], CellType.DRIVABLE_ROAD)
        self.assertNotIn((sr, sc), self.env.obstacle_set)
        self.assertNotIn((dr, dc), self.env.obstacle_set)

    def test_obstacles_seeded(self):
        """Ensure static obstacles are placed and counted."""
        self.assertGreater(len(self.env.obstacle_set), 5)
        for r, c in self.env.obstacle_set:
            self.assertEqual(self.env.grid[r, c], CellType.OBSTACLE)

    def test_manual_movement_and_collision_prevention(self):
        """Test movement into road, into obstacle, into boundary, and out-of-bounds."""
        sr, sc = self.env.start_pos

        # Find adjacent cell states around start_pos
        moved = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            target_r, target_c = sr + dr, sc + dc
            if 0 <= target_r < self.env.rows and 0 <= target_c < self.env.cols:
                cell_type = self.env.grid[target_r, target_c]
                if cell_type == CellType.DRIVABLE_ROAD:
                    res = self.env.move_vehicle(dr, dc)
                    self.assertTrue(res)
                    self.assertEqual(self.env.vehicle_pos, (target_r, target_c))
                    moved = True
                    break

        self.assertTrue(moved, "Should have at least one drivable adjacent cell from start")

        # Reset back
        self.env.reset_vehicle()
        self.assertEqual(self.env.vehicle_pos, self.env.start_pos)

    def test_boundary_and_out_of_bounds_blocking(self):
        """Test that out of bounds movement is strictly rejected."""
        self.env.vehicle_pos = (0, 0)
        # Attempt moving further up and left (out of bounds)
        res_up = self.env.move_vehicle(-1, 0)
        self.assertFalse(res_up)
        self.assertEqual(self.env.vehicle_pos, (0, 0))

        res_left = self.env.move_vehicle(0, -1)
        self.assertFalse(res_left)
        self.assertEqual(self.env.vehicle_pos, (0, 0))

    def test_obstacle_blocking(self):
        """Test that moving directly into an obstacle is rejected."""
        # Pick an obstacle
        obs_r, obs_c = next(iter(self.env.obstacle_set))
        # Place vehicle adjacent to obstacle
        adj_candidates = [
            (obs_r - 1, obs_c, 1, 0),
            (obs_r + 1, obs_c, -1, 0),
            (obs_r, obs_c - 1, 0, 1),
            (obs_r, obs_c + 1, 0, -1),
        ]
        tested = False
        for vr, vc, dr, dc in adj_candidates:
            if 0 <= vr < self.env.rows and 0 <= vc < self.env.cols:
                self.env.vehicle_pos = (vr, vc)
                res = self.env.move_vehicle(dr, dc)
                self.assertFalse(res, "Vehicle must not enter obstacle cell")
                self.assertEqual(self.env.vehicle_pos, (vr, vc))
                self.assertEqual(self.env.last_action_status, "blocked")
                tested = True
                break
        self.assertTrue(tested)

    def test_headless_rendering(self):
        """Verify rendering pipeline does not throw exceptions with dummy display."""
        screen = pygame.display.set_mode((RoadEnvironment.WINDOW_WIDTH, RoadEnvironment.WINDOW_HEIGHT))
        fonts = initialize_fonts()
        # Should render cleanly without exception
        self.env.render(screen, fonts, "Standby (Press 'C' to toggle)")

    def test_webcam_manager_lifecycle(self):
        """Verify WebcamManager handles non-existent camera or release cleanly."""
        # Use an unlikely device index to test graceful failure handling
        cam = WebcamManager(device_index=999)
        self.assertFalse(cam.is_active())
        self.assertIn("Standby", cam.get_status())

        # Attempt start
        success = cam.start_preview()
        self.assertFalse(success)
        self.assertFalse(cam.is_active())
        self.assertIn("Unavailable", cam.get_status())

        # Clean release
        cam.release()
        self.assertFalse(cam.is_active())


if __name__ == "__main__":
    unittest.main()
