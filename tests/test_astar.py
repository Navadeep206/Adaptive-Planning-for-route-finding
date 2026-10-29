"""Comprehensive test suite for A* Path Planning algorithm.

Validates algorithm correctness, admissibility, edge cases, obstacle avoidance,
optimality on deterministic benchmark grids, and integration with RoadEnvironment.
"""

import os
import unittest
import numpy as np
import pygame

# Set dummy audio and video driver so tests can run in headless environments
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from planning.astar import AStarPlanner, AStarResult, manhattan_distance, astar_search
from simulation.environment import RoadEnvironment, CellType
from main import initialize_fonts


class TestAStarPlanning(unittest.TestCase):
    """Test suite covering A* path planning functional requirements and edge cases."""

    def setUp(self):
        """Set up standard planner instance."""
        self.planner = AStarPlanner(default_drivable_value=1)

    def test_manhattan_distance_properties(self):
        """Verify Manhattan distance calculation and triangle inequality consistency."""
        p1 = (2, 3)
        p2 = (5, 9)
        # |2 - 5| + |3 - 9| = 3 + 6 = 9
        self.assertEqual(manhattan_distance(p1, p2), 9)
        self.assertEqual(manhattan_distance(p2, p1), 9)
        self.assertEqual(manhattan_distance(p1, p1), 0)

        # Admissibility property: Manhattan distance is <= exact shortest path on grid
        p3 = (4, 6)
        self.assertLessEqual(manhattan_distance(p1, p2), manhattan_distance(p1, p3) + manhattan_distance(p3, p2))

    def test_straight_line_unobstructed_path(self):
        """Test simple path planning along an open corridor."""
        # 5x5 grid with all drivable road (1)
        grid = np.ones((5, 5), dtype=int)
        start = (0, 0)
        goal = (0, 4)

        result = self.planner.plan(grid, start, goal)

        self.assertTrue(result.success)
        self.assertEqual(result.path[0], start)
        self.assertEqual(result.path[-1], goal)
        # Expected moves = 4, path cells = 5
        self.assertEqual(result.path_length, 4)
        self.assertEqual(len(result.path), 5)

        # Confirm step-by-step 4-connected continuity
        for i in range(len(result.path) - 1):
            r1, c1 = result.path[i]
            r2, c2 = result.path[i + 1]
            self.assertEqual(abs(r1 - r2) + abs(c1 - c2), 1)

    def test_wall_blocks_direct_route_alternate_path_exists(self):
        """Test that A* detours around a wall and finds the optimal detour.

        Grid layout (5x5):
        S  .  .  .  .  (0,0) to (0,4)
        #  #  #  #  .  Row 1 blocked except at col 4
        .  .  .  .  .
        .  .  .  .  .
        G  .  .  .  .  (4,0)
        """
        grid = np.ones((5, 5), dtype=int)
        grid[1, 0] = 0
        grid[1, 1] = 0
        grid[1, 2] = 0
        grid[1, 3] = 0
        # Only (1, 4) is open in row 1

        start = (0, 0)
        goal = (4, 0)

        result = self.planner.plan(grid, start, goal)

        self.assertTrue(result.success)
        self.assertEqual(result.path[0], start)
        self.assertEqual(result.path[-1], goal)

        # Verify no path node touches any wall cell
        wall_cells = {(1, 0), (1, 1), (1, 2), (1, 3)}
        for cell in result.path:
            self.assertNotIn(cell, wall_cells)
            self.assertEqual(grid[cell], 1)

        # Expected path must go through (0, 4) -> (1, 4) then down and left to (4, 0):
        # (0,0)->(0,4) is 4 moves, (0,4)->(4,4) is 4 moves, (4,4)->(4,0) is 4 moves. Total = 12 moves.
        self.assertEqual(result.path_length, 12)

    def test_enclosed_destination_no_path_terminates(self):
        """Test algorithm termination and clean failure reporting when goal is boxed in."""
        # 5x5 grid where goal (2, 2) is boxed in by obstacles (0)
        grid = np.ones((5, 5), dtype=int)
        grid[1, 2] = 0
        grid[3, 2] = 0
        grid[2, 1] = 0
        grid[2, 3] = 0

        start = (0, 0)
        goal = (2, 2)

        result = self.planner.plan(grid, start, goal)

        self.assertFalse(result.success)
        self.assertEqual(result.path, [])
        self.assertEqual(result.path_length, 0)
        self.assertGreater(result.explored_nodes, 0)
        self.assertIn("No drivable path", result.message)

    def test_start_equals_goal_valid(self):
        """Test when start and goal are the same valid cell."""
        grid = np.ones((3, 3), dtype=int)
        start = (1, 1)
        goal = (1, 1)

        result = self.planner.plan(grid, start, goal)

        self.assertTrue(result.success)
        self.assertEqual(result.path, [start])
        self.assertEqual(result.path_length, 0)
        self.assertEqual(result.explored_nodes, 1)

    def test_start_or_goal_on_obstacle_or_boundary_rejected(self):
        """Test that invalid start or goal positions are rejected immediately."""
        grid = np.ones((4, 4), dtype=int)
        grid[1, 1] = 0  # Blocked

        # Start on blocked cell
        res_start_blocked = self.planner.plan(grid, (1, 1), (3, 3))
        self.assertFalse(res_start_blocked.success)
        self.assertEqual(res_start_blocked.path, [])
        self.assertEqual(res_start_blocked.explored_nodes, 0)
        self.assertIn("Start coordinate", res_start_blocked.message)

        # Goal on blocked cell
        res_goal_blocked = self.planner.plan(grid, (0, 0), (1, 1))
        self.assertFalse(res_goal_blocked.success)
        self.assertEqual(res_goal_blocked.path, [])
        self.assertEqual(res_goal_blocked.explored_nodes, 0)
        self.assertIn("Goal coordinate", res_goal_blocked.message)

        # Start out of bounds
        res_start_oob = self.planner.plan(grid, (-1, 0), (3, 3))
        self.assertFalse(res_start_oob.success)
        self.assertEqual(res_start_oob.explored_nodes, 0)

        # Goal out of bounds
        res_goal_oob = self.planner.plan(grid, (0, 0), (10, 10))
        self.assertFalse(res_goal_oob.success)
        self.assertEqual(res_goal_oob.explored_nodes, 0)

        # Start == Goal on blocked cell must be rejected
        res_both_blocked = self.planner.plan(grid, (1, 1), (1, 1))
        self.assertFalse(res_both_blocked.success)
        self.assertEqual(res_both_blocked.path, [])

    def test_four_directional_movement_only_no_diagonals(self):
        """Ensure no diagonal transitions occur along the planned path."""
        # 3x3 open grid, start at (0, 0), goal at (2, 2)
        grid = np.ones((3, 3), dtype=int)
        result = self.planner.plan(grid, (0, 0), (2, 2))

        self.assertTrue(result.success)
        # Diagonal would be 2 moves. Orthogonal only must be |2-0| + |2-0| = 4 moves.
        self.assertEqual(result.path_length, 4)
        for i in range(len(result.path) - 1):
            r1, c1 = result.path[i]
            r2, c2 = result.path[i + 1]
            diff_r = abs(r1 - r2)
            diff_c = abs(c1 - c2)
            # Only one coordinate may change by exactly 1
            self.assertTrue((diff_r == 1 and diff_c == 0) or (diff_r == 0 and diff_c == 1))

    def test_path_length_matches_moves_not_cell_count(self):
        """Confirm requirement: path_length == number of moves == len(path) - 1."""
        grid = np.ones((6, 6), dtype=int)
        result = self.planner.plan(grid, (1, 1), (1, 4))

        self.assertTrue(result.success)
        self.assertEqual(len(result.path), 4)  # (1,1), (1,2), (1,3), (1,4)
        self.assertEqual(result.path_length, 3)  # 3 moves
        self.assertEqual(result.path_length, len(result.path) - 1)

    def test_repeated_planning_is_deterministic(self):
        """Repeated planning queries must yield identical routes and costs."""
        grid = np.ones((7, 7), dtype=int)
        # Scatter some deterministic obstacles
        grid[2, 3] = 0
        grid[3, 3] = 0
        grid[4, 3] = 0

        res1 = self.planner.plan(grid, (3, 1), (3, 5))
        res2 = self.planner.plan(grid, (3, 1), (3, 5))

        self.assertTrue(res1.success)
        self.assertTrue(res2.success)
        self.assertEqual(res1.path, res2.path)
        self.assertEqual(res1.path_length, res2.path_length)
        self.assertEqual(res1.explored_nodes, res2.explored_nodes)

    def test_convenience_function_astar_search(self):
        """Verify the module-level astar_search function wrapper."""
        grid = [[1, 1, 1], [0, 0, 1], [1, 1, 1]]
        res = astar_search(grid, (0, 0), (2, 0), drivable_values=1)
        self.assertTrue(res.success)
        self.assertEqual(res.path_length, 6)

    def test_road_environment_integration(self):
        """Verify A* operates correctly on the active RoadEnvironment grid."""
        env = RoadEnvironment()
        result = self.planner.plan(
            grid=env.grid,
            start=env.vehicle_pos,
            goal=env.destination_pos,
            drivable_values={CellType.DRIVABLE_ROAD}
        )

        self.assertTrue(result.success, f"A* failed on RoadEnvironment: {result.message}")
        self.assertGreater(result.path_length, 0)
        self.assertEqual(result.path[0], env.vehicle_pos)
        self.assertEqual(result.path[-1], env.destination_pos)

        # Assert no cell on path is border or static obstacle
        for (r, c) in result.path:
            self.assertEqual(env.grid[r, c], CellType.DRIVABLE_ROAD)
            self.assertNotIn((r, c), env.obstacle_set)

        # Performance assertion: path search on this 22x26 grid should take < 50ms
        self.assertLess(result.execution_time_ms, 50.0)

    def test_environment_plan_path_method(self):
        """Test RoadEnvironment.plan_path() updates internal state correctly."""
        env = RoadEnvironment()
        self.assertEqual(env.planning_status, "Press P to plan")
        self.assertEqual(len(env.planned_path), 0)

        result = env.plan_path()

        self.assertTrue(result.success)
        self.assertEqual(env.planning_status, "Path found")
        self.assertGreater(len(env.planned_path), 0)
        self.assertEqual(env.path_length, len(env.planned_path) - 1)
        self.assertGreater(env.explored_nodes_count, 0)
        self.assertGreater(env.planning_time_ms, 0.0)

    def test_environment_reset_clears_route(self):
        """Test RoadEnvironment.reset_vehicle() clears any active route."""
        env = RoadEnvironment()
        env.plan_path()
        self.assertGreater(len(env.planned_path), 0)

        env.reset_vehicle()

        self.assertEqual(env.planned_path, [])
        self.assertIsNone(env.planning_result)
        self.assertEqual(env.planning_status, "Press P to plan")
        self.assertEqual(env.path_length, 0)
        self.assertEqual(env.vehicle_pos, env.start_pos)

    def test_environment_manual_move_clears_stale_route(self):
        """Test that moving the vehicle manually clears any displayed route."""
        env = RoadEnvironment()
        env.plan_path()
        self.assertGreater(len(env.planned_path), 0)
        self.assertEqual(env.planning_status, "Path found")

        # Find a valid move direction
        sr, sc = env.vehicle_pos
        moved = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if env.move_vehicle(dr, dc):
                moved = True
                break

        self.assertTrue(moved)
        # Route must now be cleared to avoid being stale
        self.assertEqual(env.planned_path, [])
        self.assertIsNone(env.planning_result)
        self.assertEqual(env.planning_status, "Press P to plan")

    def test_environment_rendering_with_planned_route(self):
        """Verify rendering pipeline displays planned path without throwing exceptions."""
        pygame.init()
        screen = pygame.display.set_mode((RoadEnvironment.WINDOW_WIDTH, RoadEnvironment.WINDOW_HEIGHT))
        fonts = initialize_fonts()
        env = RoadEnvironment()
        env.plan_path()

        # Render with planned route active
        env.render(screen, fonts, "Standby (Press 'C' to toggle)")
        pygame.quit()


if __name__ == "__main__":
    unittest.main()
