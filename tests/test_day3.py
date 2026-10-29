"""Automated verification suite for Day 3: Autonomous Traversal & Dynamic Replanning.

Tests autonomous waypoint following, pause/resume behavior, dynamic hazard insertion,
collision prevention, adaptive replanning, state transitions, and clean resetting.
"""

import os
import unittest
import pygame

# Set dummy SDL drivers for headless test environments
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from simulation.environment import RoadEnvironment, CellType, NavigationMode
from main import initialize_fonts


class TestDay3AutonomousMovement(unittest.TestCase):
    """Test suite verifying Day 3 autonomous movement and dynamic replanning."""

    def setUp(self):
        """Initialize fresh test environment."""
        pygame.init()
        self.env = RoadEnvironment()

    def tearDown(self):
        """Clean up pygame display."""
        pygame.quit()

    def test_autonomous_navigation_reaches_goal(self):
        """Vehicle follows the A* path step-by-step and reaches the destination."""
        self.assertEqual(self.nav_mode_value(), NavigationMode.MANUAL)
        self.env.toggle_autonomous_navigation()
        self.assertEqual(self.env.nav_mode, NavigationMode.AUTONOMOUS)
        self.assertGreater(len(self.env.planned_path), 0)

        # Step along path until goal or upper bound
        max_steps = 100
        steps = 0
        while self.env.nav_mode == NavigationMode.AUTONOMOUS and steps < max_steps:
            moved = self.env.step_autonomous()
            if not moved:
                break
            steps += 1

        self.assertEqual(self.env.nav_mode, NavigationMode.GOAL_REACHED)
        self.assertEqual(self.env.vehicle_pos, self.env.destination_pos)
        self.assertEqual(self.env.collision_count, 0)
        self.assertGreater(steps, 15)

    def test_space_pauses_and_resumes_autonomous_movement(self):
        """SPACE toggles between Autonomous and Paused navigation."""
        self.env.toggle_autonomous_navigation()
        self.assertEqual(self.env.nav_mode, NavigationMode.AUTONOMOUS)

        # Step vehicle forward once
        self.assertTrue(self.env.step_autonomous())
        pos_after_one = self.env.vehicle_pos

        # Toggle to pause
        self.env.toggle_autonomous_navigation()
        self.assertEqual(self.env.nav_mode, NavigationMode.PAUSED)

        # In paused state, step_autonomous must reject movement
        self.assertFalse(self.env.step_autonomous())
        self.assertEqual(self.env.vehicle_pos, pos_after_one)

        # Toggle to resume
        self.env.toggle_autonomous_navigation()
        self.assertEqual(self.env.nav_mode, NavigationMode.AUTONOMOUS)
        self.assertTrue(self.env.step_autonomous())
        self.assertNotEqual(self.env.vehicle_pos, pos_after_one)

    def test_obstacle_ahead_triggers_adaptive_replanning(self):
        """Inserting a dynamic obstacle ahead on the route triggers automatic replanning."""
        self.env.toggle_autonomous_navigation()
        self.assertEqual(self.env.nav_mode, NavigationMode.AUTONOMOUS)

        initial_replans = self.env.replanning_events

        # Advance 2 steps
        self.env.step_autonomous()
        self.env.step_autonomous()

        # Insert dynamic obstacle 3 steps ahead on the planned route
        chosen_hazard = self.env.insert_dynamic_obstacle(ahead_steps=3)
        self.assertIsNotNone(chosen_hazard)
        self.assertIn(chosen_hazard, self.env.dynamic_obstacle_set)
        self.assertEqual(self.env.grid[chosen_hazard], CellType.OBSTACLE)

        # Verify replanning event occurred
        self.assertGreater(self.env.replanning_events, initial_replans)

        # The newly replanned route must not pass through the inserted hazard
        self.assertNotIn(chosen_hazard, self.env.planned_path)

        # Vehicle continues navigation around the new obstacle without collision
        steps = 0
        while self.env.nav_mode == NavigationMode.AUTONOMOUS and steps < 80:
            self.env.step_autonomous()
            self.assertNotEqual(self.env.vehicle_pos, chosen_hazard)
            steps += 1

        self.assertEqual(self.env.collision_count, 0)
        self.assertEqual(self.env.vehicle_pos, self.env.destination_pos)

    def test_vehicle_never_enters_obstacle_or_non_drivable(self):
        """Verify safety invariant: vehicle never steps onto obstacles or non-drivable verge."""
        self.env.toggle_autonomous_navigation()
        visited = []

        max_steps = 100
        steps = 0
        while self.env.nav_mode == NavigationMode.AUTONOMOUS and steps < max_steps:
            visited.append(self.env.vehicle_pos)
            self.env.step_autonomous()
            steps += 1

        # Check every visited cell
        for cell in visited:
            self.assertNotIn(cell, self.env.obstacle_set)
            self.assertEqual(self.env.grid[cell], CellType.DRIVABLE_ROAD)

        self.assertEqual(self.env.collision_count, 0)

    def test_stops_safely_when_completely_blocked(self):
        """Simulation stops safely in NO_PATH mode when no alternate route exists."""
        self.env.plan_path()

        # Enclose the destination completely with obstacles
        dr, dc = self.env.destination_pos
        for r_offset, c_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = dr + r_offset, dc + c_offset
            if 0 <= nr < self.env.rows and 0 <= nc < self.env.cols:
                self.env.grid[nr, nc] = CellType.OBSTACLE
                self.env.obstacle_set.add((nr, nc))

        # Attempt autonomous movement
        self.env.toggle_autonomous_navigation()

        # Navigation mode must indicate No Path
        self.assertEqual(self.env.nav_mode, NavigationMode.NO_PATH)
        self.assertEqual(self.env.planned_path, [])
        self.assertFalse(self.env.step_autonomous())
        self.assertEqual(self.env.collision_count, 0)

    def test_dynamic_obstacle_never_overwrites_vehicle_or_destination(self):
        """Dynamic obstacles must never be placed on the vehicle position or destination."""
        for _ in range(8):
            hazard = self.env.insert_dynamic_obstacle()
            if hazard:
                self.assertNotEqual(hazard, self.env.vehicle_pos)
                self.assertNotEqual(hazard, self.env.destination_pos)
                self.assertEqual(self.env.grid[self.env.vehicle_pos], CellType.DRIVABLE_ROAD)
                self.assertEqual(self.env.grid[self.env.destination_pos], CellType.DRIVABLE_ROAD)

    def test_reset_restores_initial_state(self):
        """Reset clears dynamic obstacles, resets vehicle position, and resets counters."""
        self.env.toggle_autonomous_navigation()
        self.env.step_autonomous()
        self.env.step_autonomous()
        self.env.insert_dynamic_obstacle()
        self.assertGreater(len(self.env.dynamic_obstacle_set), 0)

        # Reset environment
        self.env.reset_vehicle()

        self.assertEqual(self.env.vehicle_pos, self.env.start_pos)
        self.assertEqual(self.env.nav_mode, NavigationMode.MANUAL)
        self.assertEqual(len(self.env.dynamic_obstacle_set), 0)
        self.assertEqual(self.env.obstacle_set, self.env.initial_obstacle_coords)
        self.assertEqual(self.env.replanning_events, 0)
        self.assertEqual(self.env.collision_count, 0)
        self.assertEqual(self.env.planned_path, [])
        self.assertEqual(self.env.planning_status, "Press P to plan")

    def test_manual_controls_locked_during_autonomous_movement(self):
        """Manual movement is locked during autonomous traversal and unlocked when paused/manual."""
        self.env.toggle_autonomous_navigation()
        self.assertEqual(self.env.nav_mode, NavigationMode.AUTONOMOUS)

        pos_before = self.env.vehicle_pos
        # Attempt manual move (must be rejected)
        result = self.env.move_vehicle(1, 0)
        self.assertFalse(result)
        self.assertEqual(self.env.vehicle_pos, pos_before)
        self.assertEqual(self.env.nav_mode, NavigationMode.AUTONOMOUS)

        # Pause autonomous movement
        self.env.toggle_autonomous_navigation()
        self.assertEqual(self.env.nav_mode, NavigationMode.PAUSED)

        # Find a valid adjacent drivable cell to move to manually
        moved_manually = False
        for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            tr, tc = self.env.vehicle_pos[0] + dr, self.env.vehicle_pos[1] + dc
            if self.env.grid[tr, tc] == CellType.DRIVABLE_ROAD:
                res = self.env.move_vehicle(dr, dc)
                if res:
                    moved_manually = True
                    break

        self.assertTrue(moved_manually)
        self.assertEqual(self.env.nav_mode, NavigationMode.MANUAL)
        self.assertEqual(self.env.planned_path, [])

    def test_unnecessary_replanning_avoided_for_off_path_obstacles(self):
        """Obstacles placed off the active route do not trigger redundant replanning."""
        self.env.plan_path()
        path_set = set(self.env.planned_path)
        replans_before = self.env.replanning_events

        # Find an open road cell that is NOT on the planned path
        off_path_cell = None
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                pos = (r, c)
                if (
                    self.env.grid[pos] == CellType.DRIVABLE_ROAD
                    and pos not in path_set
                    and pos != self.env.vehicle_pos
                    and pos != self.env.destination_pos
                ):
                    off_path_cell = pos
                    break
            if off_path_cell:
                break

        self.assertIsNotNone(off_path_cell)

        # Place hazard directly on off-path cell
        r, c = off_path_cell
        self.env.grid[r, c] = CellType.OBSTACLE
        self.env.obstacle_set.add(off_path_cell)
        self.env.dynamic_obstacle_set.add(off_path_cell)

        # Check route blocked check
        self.assertFalse(self.env.is_route_blocked())
        # Replanning count should remain unchanged
        self.assertEqual(self.env.replanning_events, replans_before)

    def test_update_loop_timed_stepping(self):
        """env.update() respects step_delay_ms timer to pace movement."""
        self.env.toggle_autonomous_navigation()
        self.assertEqual(self.env.nav_mode, NavigationMode.AUTONOMOUS)
        pos_init = self.env.vehicle_pos

        # Update with only 100 ms (less than 250 ms threshold)
        self.env.update(dt_ms=100)
        self.assertEqual(self.env.vehicle_pos, pos_init)

        # Update with remaining 160 ms (total 260 ms >= 250 ms)
        self.env.update(dt_ms=160)
        self.assertNotEqual(self.env.vehicle_pos, pos_init)

    def test_headless_rendering_with_dynamic_hazards(self):
        """Verify rendering pipeline displays vehicle and dynamic hazards without exception."""
        screen = pygame.display.set_mode((RoadEnvironment.WINDOW_WIDTH, RoadEnvironment.WINDOW_HEIGHT))
        fonts = initialize_fonts()
        self.env.insert_dynamic_obstacle()
        self.env.toggle_autonomous_navigation()
        self.env.render(screen, fonts, "Standby")

    def nav_mode_value(self) -> str:
        """Helper to get current navigation mode value."""
        return self.env.nav_mode.value


if __name__ == "__main__":
    unittest.main()
