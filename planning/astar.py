"""A* Path Planning implementation for grid-based autonomous navigation.

Calculates optimal 4-connected orthogonal paths avoiding static obstacles and
non-drivable road shoulders using the A* heuristic search algorithm.
"""

from dataclasses import dataclass
import heapq
import itertools
import time
from typing import Callable, Iterable, List, Optional, Set, Tuple, Union
import numpy as np


@dataclass
class AStarResult:
    """Encapsulates the complete outcome of an A* path planning query.

    Attributes:
        success: True if an unobstructed path from start to goal was discovered.
        path: Ordered list of (row, col) grid coordinates from start to goal.
        path_length: Number of directional step moves along the path (len(path) - 1).
        explored_nodes: Total number of unique nodes expanded (closed) during search.
        execution_time_ms: Computation runtime of the search algorithm in milliseconds.
        message: Descriptive telemetry string summarizing the planning status.
    """
    success: bool
    path: List[Tuple[int, int]]
    path_length: int
    explored_nodes: int
    execution_time_ms: float
    message: str


def manhattan_distance(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> int:
    """Calculate the L1 Manhattan distance between two grid cells.

    For 4-connected grid movement with uniform step cost c=1, Manhattan distance
    represents the exact minimal traversal distance in unconstrained space.
    Because obstacles can only increase the actual path length, h(n) <= h*(n)
    is strictly satisfied, making this heuristic both admissible and consistent.

    Args:
        pos_a: (row, col) coordinates of the first point.
        pos_b: (row, col) coordinates of the second point.

    Returns:
        int: The Manhattan distance |r1 - r2| + |c1 - c2|.
    """
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


class AStarPlanner:
    """Modular A* Path Planner operating on 2D grid environments."""

    # 4 orthogonal movement directions: Up, Down, Left, Right
    ORTHOGONAL_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
        (-1, 0),  # North / Up
        (1, 0),   # South / Down
        (0, -1),  # West  / Left
        (0, 1),   # East  / Right
    )
    STEP_COST: int = 1

    def __init__(self, default_drivable_value: int = 1):
        """Initialize the A* planner.

        Args:
            default_drivable_value: Default integer value representing drivable cells.
        """
        self.default_drivable_value = default_drivable_value

    def plan(
        self,
        grid: Union[np.ndarray, List[List[int]]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        drivable_values: Optional[Union[int, Iterable[int], Callable[[int, int], bool]]] = None
    ) -> AStarResult:
        """Execute A* search on the given 2D grid from start to goal.

        Args:
            grid: 2D array-like structure (rows, cols) representing terrain cell states.
            start: (row, col) coordinate of the vehicle.
            goal: (row, col) coordinate of the navigation destination.
            drivable_values: Optional definition of walkable terrain. Can be an integer,
                collection of integers, or a custom predicate function `is_walkable(r, c)`.

        Returns:
            AStarResult: Comprehensive planning result including path, cost, and telemetry.
        """
        start_time = time.perf_counter()

        # Determine grid dimensions
        if isinstance(grid, np.ndarray):
            rows, cols = grid.shape[0], grid.shape[1]
        else:
            rows = len(grid)
            cols = len(grid[0]) if rows > 0 else 0

        # Construct drivability predicate
        is_drivable: Callable[[int, int], bool]
        if callable(drivable_values):
            is_drivable = drivable_values
        elif drivable_values is not None:
            if isinstance(drivable_values, (int, np.integer)):
                allowed_vals = {int(drivable_values)}
            else:
                allowed_vals = set(int(v) for v in drivable_values)
            is_drivable = lambda r, c: int(grid[r][c]) in allowed_vals
        else:
            allowed_vals = {self.default_drivable_value}
            is_drivable = lambda r, c: int(grid[r][c]) in allowed_vals

        # Helper for boundary validation
        def in_bounds(r: int, c: int) -> bool:
            return 0 <= r < rows and 0 <= c < cols

        sr, sc = start
        gr, gc = goal

        # 1. Start coordinate boundary & drivability validation
        if not in_bounds(sr, sc):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AStarResult(
                success=False,
                path=[],
                path_length=0,
                explored_nodes=0,
                execution_time_ms=elapsed_ms,
                message=f"Start coordinate ({sc}, {sr}) is out of grid bounds."
            )

        if not is_drivable(sr, sc):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AStarResult(
                success=False,
                path=[],
                path_length=0,
                explored_nodes=0,
                execution_time_ms=elapsed_ms,
                message=f"Start coordinate ({sc}, {sr}) is on a non-drivable cell or obstacle."
            )

        # 2. Goal coordinate boundary & drivability validation
        if not in_bounds(gr, gc):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AStarResult(
                success=False,
                path=[],
                path_length=0,
                explored_nodes=0,
                execution_time_ms=elapsed_ms,
                message=f"Goal coordinate ({gc}, {gr}) is out of grid bounds."
            )

        if not is_drivable(gr, gc):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AStarResult(
                success=False,
                path=[],
                path_length=0,
                explored_nodes=0,
                execution_time_ms=elapsed_ms,
                message=f"Goal coordinate ({gc}, {gr}) is on a non-drivable cell or obstacle."
            )

        # 3. Trivial edge case: Start equals Goal
        if start == goal:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AStarResult(
                success=True,
                path=[start],
                path_length=0,
                explored_nodes=1,
                execution_time_ms=elapsed_ms,
                message="Vehicle is already at destination (0 moves required)."
            )

        # 4. Standard A* Graph Search Initialization
        # Priority Queue entries: (f_score, h_score, tie_breaker_counter, current_node)
        # Using h_score in the tuple biases selection towards nodes closer to goal on f_score ties.
        open_heap: List[Tuple[int, int, int, Tuple[int, int]]] = []
        counter = itertools.count()

        h_start = manhattan_distance(start, goal)
        f_start = 0 + h_start
        heapq.heappush(open_heap, (f_start, h_start, next(counter), start))

        g_score: dict[Tuple[int, int], int] = {start: 0}
        parent: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        closed_set: Set[Tuple[int, int]] = set()

        goal_reached = False

        # 5. Core Search Loop
        while open_heap:
            _, _, _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue

            closed_set.add(current)

            # Destination check
            if current == goal:
                goal_reached = True
                break

            curr_g = g_score[current]

            # Expand 4-directional orthogonal neighbors
            cr, cc = current
            for dr, dc in self.ORTHOGONAL_DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                neighbor = (nr, nc)

                # Strictly enforce boundaries
                if not in_bounds(nr, nc):
                    continue

                # Strictly enforce drivable terrain and obstacle avoidance
                if not is_drivable(nr, nc):
                    continue

                # Skip already expanded / closed nodes
                if neighbor in closed_set:
                    continue

                tentative_g = curr_g + self.STEP_COST

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    h_neighbor = manhattan_distance(neighbor, goal)
                    f_neighbor = tentative_g + h_neighbor
                    heapq.heappush(open_heap, (f_neighbor, h_neighbor, next(counter), neighbor))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 6. Path Reconstruction or Failure Delivery
        if goal_reached:
            path: List[Tuple[int, int]] = []
            curr: Optional[Tuple[int, int]] = goal
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()

            path_length = len(path) - 1
            return AStarResult(
                success=True,
                path=path,
                path_length=path_length,
                explored_nodes=len(closed_set),
                execution_time_ms=elapsed_ms,
                message=f"Optimal path discovered with {path_length} moves."
            )

        return AStarResult(
            success=False,
            path=[],
            path_length=0,
            explored_nodes=len(closed_set),
            execution_time_ms=elapsed_ms,
            message="No drivable path found: destination is unreachable or enclosed."
        )


def astar_search(
    grid: Union[np.ndarray, List[List[int]]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    drivable_values: Optional[Union[int, Iterable[int], Callable[[int, int], bool]]] = None
) -> AStarResult:
    """Convenience function wrapper for AStarPlanner."""
    planner = AStarPlanner()
    return planner.plan(grid, start, goal, drivable_values)
