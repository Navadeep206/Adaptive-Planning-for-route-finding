"""Simulation environment for unstructured road navigation.

Provides a 2D Pygame simulation grid featuring irregular road boundaries,
static hazards, vehicle tracking, legend HUD, and manual keyboard testing.
"""

from enum import IntEnum, Enum
from typing import List, Tuple, Set, Optional
import math
import logging
import numpy as np
import pygame

from planning.astar import AStarPlanner, AStarResult

logger = logging.getLogger(__name__)


class CellType(IntEnum):
    """Enumeration of cell types across the terrain grid."""
    NON_DRIVABLE_BORDER = 0  # Off-road rough verge, rocks, ditches
    DRIVABLE_ROAD = 1        # Winding, unpaved unstructured roadway
    OBSTACLE = 2             # Static road hazard (pothole, fallen rock, debris)


class NavigationMode(str, Enum):
    """Operational states for vehicle navigation control."""
    MANUAL = "Manual"
    AUTONOMOUS = "Autonomous"
    PAUSED = "Paused"
    REPLANNING = "Replanning"
    GOAL_REACHED = "Goal Reached"
    NO_PATH = "No Path"


class RoadEnvironment:
    """Represents the 2D unstructured-road simulation world and handles rendering."""

    WINDOW_WIDTH: int = 1000
    WINDOW_HEIGHT: int = 700
    WINDOW_TITLE: str = "Adaptive Path Planning and Collision Avoidance"

    # Color Palette (Tailored, high-contrast, modern UI)
    COLOR_BG: Tuple[int, int, int] = (18, 22, 28)              # Dark slate canvas
    COLOR_PANEL_BG: Tuple[int, int, int] = (28, 34, 43)        # Card backgrounds
    COLOR_PANEL_BORDER: Tuple[int, int, int] = (45, 55, 72)    # Card borders
    COLOR_GRID_LINE: Tuple[int, int, int] = (35, 42, 54)       # Subtle grid subdivisions

    # Terrain & Entity Colors
    COLOR_BORDER_TERRAIN: Tuple[int, int, int] = (42, 48, 38)  # Rough shoulder / off-road verge
    COLOR_DRIVABLE_ROAD: Tuple[int, int, int] = (148, 128, 98) # Unpaved dirt/gravel road surface
    COLOR_ROAD_ACCENT: Tuple[int, int, int] = (165, 145, 114)  # Texture accent for unpaved ground
    COLOR_VEHICLE: Tuple[int, int, int] = (33, 150, 243)       # Electric blue vehicle marker
    COLOR_DESTINATION: Tuple[int, int, int] = (255, 214, 10)   # Yellow destination goal
    COLOR_OBSTACLE: Tuple[int, int, int] = (239, 68, 68)       # Coral red obstacle marker
    COLOR_DYNAMIC_BORDER: Tuple[int, int, int] = (251, 146, 60) # Amber-orange dynamic hazard highlight
    COLOR_PERCEPTION_OBSTACLE: Tuple[int, int, int] = (168, 85, 247) # Electric purple perception hazard
    COLOR_PERCEPTION_BORDER: Tuple[int, int, int] = (217, 70, 239)   # Neon fuchsia perception border
    COLOR_PERCEPTION_GLOW: Tuple[int, int, int] = (192, 132, 252)     # Luminous sensor bead
    COLOR_MODE_REPLANNING: Tuple[int, int, int] = (217, 70, 239) # Magenta replanning accent
    COLOR_PATH: Tuple[int, int, int] = (34, 197, 94)          # Emerald green planned path
    COLOR_PATH_ACCENT: Tuple[int, int, int] = (134, 239, 172) # Luminous light green path beads
    COLOR_PATH_LINE: Tuple[int, int, int] = (74, 222, 128)    # Route trajectory line

    # Typography Colors
    COLOR_TEXT_PRIMARY: Tuple[int, int, int] = (243, 244, 246)
    COLOR_TEXT_SECONDARY: Tuple[int, int, int] = (156, 163, 175)
    COLOR_ACCENT_CYAN: Tuple[int, int, int] = (56, 189, 248)
    COLOR_ACCENT_GREEN: Tuple[int, int, int] = (74, 222, 128)
    COLOR_ACCENT_AMBER: Tuple[int, int, int] = (251, 191, 36)

    def __init__(self, rows: int = 22, cols: int = 26, cell_size: int = 25):
        """Initialize the road environment grid and coordinates.

        Args:
            rows: Number of grid rows.
            cols: Number of grid columns.
            cell_size: Width and height of each grid cell in pixels.
        """
        self.rows: int = rows
        self.cols: int = cols
        self.cell_size: int = cell_size

        # Layout positioning
        self.grid_origin_x: int = 24
        self.grid_origin_y: int = 90
        self.sidebar_x: int = self.grid_origin_x + (self.cols * self.cell_size) + 24
        self.sidebar_width: int = self.WINDOW_WIDTH - self.sidebar_x - 24

        # Grid state representation using NumPy
        self.grid: np.ndarray = np.full((self.rows, self.cols), CellType.NON_DRIVABLE_BORDER, dtype=np.int8)

        # Vehicle navigation entities (grid row, grid col)
        self.start_pos: Tuple[int, int] = (2, 12)
        self.destination_pos: Tuple[int, int] = (20, 15)
        self.vehicle_pos: Tuple[int, int] = self.start_pos

        # Feedback & interaction telemetry
        self.last_action_msg: str = "Ready. Press SPACE to navigate, 'O' to spawn hazard, WASD for manual."
        self.last_action_status: str = "normal"  # "normal", "success", "blocked"

        # A* Path planning telemetry & state (Day 2)
        self.planner: AStarPlanner = AStarPlanner(default_drivable_value=int(CellType.DRIVABLE_ROAD))
        self.planned_path: List[Tuple[int, int]] = []
        self.planning_result: Optional[AStarResult] = None
        self.planning_status: str = "Press P to plan"  # "Path found", "No valid path", or "Press P to plan"
        self.path_length: int = 0
        self.explored_nodes_count: int = 0
        self.planning_time_ms: float = 0.0

        # Autonomous Navigation & Dynamic Replanning state (Day 3)
        self.nav_mode: NavigationMode = NavigationMode.MANUAL
        self.replanning_events: int = 0
        self.collision_count: int = 0
        self.dynamic_obstacle_set: Set[Tuple[int, int]] = set()
        self.initial_obstacle_coords: Set[Tuple[int, int]] = set()
        self.step_delay_ms: int = 250  # 4 steps per second for clear visualization
        self.step_timer_ms: int = 0

        # Perception-derived obstacle mapping state (Day 6)
        self.perception_enabled: bool = False
        self.perception_obstacle_set: Set[Tuple[int, int]] = set()
        self.perception_conflict_count: int = 0
        self.perception_last_update_count: int = 0

        # Generate unstructured road contours and obstacles
        self._build_unstructured_road()
        self._seed_static_obstacles()

    def _build_unstructured_road(self) -> None:
        """Carve an irregular, winding drivable road corridor across the terrain."""
        for r in range(self.rows):
            # Sinusoidal centerline to simulate winding rural / unpaved terrain
            center_col = 13.0 + 5.5 * math.sin(r * 0.32) + 1.5 * math.cos(r * 0.7)
            # Irregular road width that varies along the path
            road_width = 8.5 + 2.0 * math.sin(r * 0.45)

            half_width = road_width / 2.0
            left_col = max(0, int(center_col - half_width))
            right_col = min(self.cols - 1, int(center_col + half_width))

            for c in range(left_col, right_col + 1):
                self.grid[r, c] = CellType.DRIVABLE_ROAD

        # Guarantee start and destination areas are drivable
        sr, sc = self.start_pos
        dr, dc = self.destination_pos
        self.grid[sr, sc] = CellType.DRIVABLE_ROAD
        self.grid[dr, dc] = CellType.DRIVABLE_ROAD

    def _seed_static_obstacles(self) -> None:
        """Place static hazards (boulders, potholes) on the drivable corridor."""
        # Curated list of hazards that test collision prevention while leaving paths open
        obstacle_coords: List[Tuple[int, int]] = [
            (4, 11), (5, 12),
            (7, 16), (8, 17),
            (10, 10), (10, 11),
            (12, 14), (13, 13),
            (15, 8), (15, 9),
            (17, 14), (18, 15)
        ]

        self.obstacle_set: Set[Tuple[int, int]] = set()
        for r, c in obstacle_coords:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                # Never place obstacles on the vehicle start or destination points
                if (r, c) != self.start_pos and (r, c) != self.destination_pos:
                    if self.grid[r, c] == CellType.DRIVABLE_ROAD:
                        self.grid[r, c] = CellType.OBSTACLE
                        self.obstacle_set.add((r, c))

        # Preserve the initial static obstacle baseline for clean resets
        self.initial_obstacle_coords = set(self.obstacle_set)

    def reset_vehicle(self) -> None:
        """Reset vehicle, restore initial environment, clear dynamic and perception obstacles and routes."""
        # Clear perception-derived hazards from the grid
        self.clear_perception_obstacles()

        # Remove dynamically inserted hazards from the grid
        for r, c in self.dynamic_obstacle_set:
            if (r, c) not in self.initial_obstacle_coords:
                self.grid[r, c] = CellType.DRIVABLE_ROAD

        # Restore static obstacle set
        self.obstacle_set = set(self.initial_obstacle_coords)
        for r, c in self.obstacle_set:
            self.grid[r, c] = CellType.OBSTACLE

        self.dynamic_obstacle_set.clear()

        # Reset navigation & kinematics state
        self.vehicle_pos = self.start_pos
        self.nav_mode = NavigationMode.MANUAL
        self.planned_path = []
        self.planning_result = None
        self.planning_status = "Press P to plan"
        self.path_length = 0
        self.explored_nodes_count = 0
        self.planning_time_ms = 0.0
        self.replanning_events = 0
        self.collision_count = 0
        self.step_timer_ms = 0
        self.last_action_msg = "Simulation reset to initial state. Vehicle at start."
        self.last_action_status = "normal"

    def move_vehicle(self, d_row: int, d_col: int) -> bool:
        """Attempt manual vehicle movement with collision checking and autonomous lock.

        Args:
            d_row: Row offset (-1, 0, +1).
            d_col: Column offset (-1, 0, +1).

        Returns:
            bool: True if move was valid and executed, False if rejected or blocked.
        """
        # Manual movement strictly rejected while autonomous mode is running
        if self.nav_mode == NavigationMode.AUTONOMOUS:
            self.last_action_msg = "Manual movement locked while Autonomous mode is active. Press SPACE to pause."
            self.last_action_status = "blocked"
            return False

        curr_r, curr_c = self.vehicle_pos
        target_r = curr_r + d_row
        target_c = curr_c + d_col

        # 1. Bounds check
        if not (0 <= target_r < self.rows and 0 <= target_c < self.cols):
            self.last_action_msg = f"Movement blocked: Target ({target_c}, {target_r}) is out of bounds."
            self.last_action_status = "blocked"
            return False

        cell_state = self.grid[target_r, target_c]

        # 2. Road boundary check
        if cell_state == CellType.NON_DRIVABLE_BORDER:
            self.last_action_msg = f"Movement blocked: Off-road boundary / ditch at ({target_c}, {target_r})."
            self.last_action_status = "blocked"
            return False

        # 3. Static/Dynamic obstacle check
        if cell_state == CellType.OBSTACLE:
            self.last_action_msg = f"Collision prevented: Hazard detected at ({target_c}, {target_r})."
            self.last_action_status = "blocked"
            return False

        # Valid movement executed
        self.vehicle_pos = (target_r, target_c)
        self.nav_mode = NavigationMode.MANUAL

        # Clear stale path when vehicle is moved manually
        if self.planned_path:
            self.planned_path = []
            self.planning_result = None
            self.planning_status = "Press P to plan"
            self.path_length = 0
            self.explored_nodes_count = 0
            self.planning_time_ms = 0.0

        if self.vehicle_pos == self.destination_pos:
            self.nav_mode = NavigationMode.GOAL_REACHED
            self.last_action_msg = "Goal Reached! Vehicle arrived at destination marker."
            self.last_action_status = "success"
        else:
            self.last_action_msg = f"Vehicle moved to cell ({target_c}, {target_r}). Press SPACE or 'P' to plan."
            self.last_action_status = "normal"

        return True

    def plan_path(self) -> AStarResult:
        """Calculate an optimal A* path from current vehicle position to destination.

        Returns:
            AStarResult: Result of search query containing path, cost, and telemetry.
        """
        result = self.planner.plan(
            grid=self.grid,
            start=self.vehicle_pos,
            goal=self.destination_pos,
            drivable_values={int(CellType.DRIVABLE_ROAD)}
        )
        self.planning_result = result
        self.planning_time_ms = result.execution_time_ms
        self.explored_nodes_count = result.explored_nodes
        self.path_length = result.path_length

        if result.success:
            self.planned_path = result.path
            self.planning_status = "Path found"
            self.last_action_msg = f"A* Path found: {result.path_length} moves, {result.explored_nodes} explored ({result.execution_time_ms:.2f} ms)."
            self.last_action_status = "success"
        else:
            self.planned_path = []
            self.planning_status = "No valid path"
            self.nav_mode = NavigationMode.NO_PATH
            self.last_action_msg = f"No valid path: {result.message}"
            self.last_action_status = "blocked"

        return result

    def is_route_blocked(self) -> bool:
        """Check if any remaining waypoint along the planned route is obstructed."""
        if not self.planned_path or len(self.planned_path) < 2:
            return False
        # Inspect upcoming waypoints along remaining trajectory
        for r, c in self.planned_path[1:]:
            if self.grid[r, c] != CellType.DRIVABLE_ROAD:
                return True
        return False

    def replan_route(self, reason: str = "Obstacle detected") -> bool:
        """Recompute an optimal collision-free path from current vehicle position.

        Args:
            reason: Telemetry explanation for the replanning trigger.

        Returns:
            bool: True if an alternate safe route was discovered, False if blocked.
        """
        old_mode = self.nav_mode
        self.nav_mode = NavigationMode.REPLANNING
        self.replanning_events += 1

        result = self.planner.plan(
            grid=self.grid,
            start=self.vehicle_pos,
            goal=self.destination_pos,
            drivable_values={int(CellType.DRIVABLE_ROAD)}
        )
        self.planning_result = result
        self.planning_time_ms = result.execution_time_ms
        self.explored_nodes_count = result.explored_nodes

        if result.success:
            self.planned_path = result.path
            self.path_length = result.path_length
            self.planning_status = "Path found"
            # Resume autonomous or paused mode if previously active
            if old_mode in (NavigationMode.AUTONOMOUS, NavigationMode.PAUSED):
                self.nav_mode = old_mode
            else:
                self.nav_mode = NavigationMode.MANUAL
            self.last_action_msg = f"Route replanned: {result.path_length} moves ({reason}, #{self.replanning_events})."
            self.last_action_status = "success"
            return True
        else:
            self.planned_path = []
            self.path_length = 0
            self.planning_status = "No valid path"
            self.nav_mode = NavigationMode.NO_PATH
            self.last_action_msg = "No safe path available. Vehicle stopped safely."
            self.last_action_status = "blocked"
            return False

    def toggle_autonomous_navigation(self) -> None:
        """Start, pause, or resume autonomous waypoint following."""
        # 1. Active Autonomous -> Pause
        if self.nav_mode == NavigationMode.AUTONOMOUS:
            self.nav_mode = NavigationMode.PAUSED
            self.last_action_msg = "Autonomous navigation PAUSED. Press SPACE to resume."
            self.last_action_status = "normal"
            return

        # 2. Paused -> Resume
        if self.nav_mode == NavigationMode.PAUSED:
            if self.vehicle_pos == self.destination_pos:
                self.nav_mode = NavigationMode.GOAL_REACHED
                self.last_action_msg = "Goal reached! Vehicle at destination marker."
                self.last_action_status = "success"
                return

            if self.is_route_blocked() or not self.planned_path or len(self.planned_path) < 2:
                replan_success = self.replan_route(reason="Resuming autonomous movement")
                if not replan_success:
                    return

            self.nav_mode = NavigationMode.AUTONOMOUS
            self.step_timer_ms = 0
            self.last_action_msg = f"Autonomous navigation RESUMED ({self.path_length} moves remaining)."
            self.last_action_status = "success"
            return

        # 3. Manual, Goal Reached, or No Path -> Start
        if self.vehicle_pos == self.destination_pos:
            self.last_action_msg = "Already at destination. Press 'R' to reset."
            self.last_action_status = "normal"
            return

        # Ensure valid route exists
        if not self.planned_path or self.is_route_blocked() or len(self.planned_path) < 2:
            res = self.plan_path()
            if not res.success:
                self.nav_mode = NavigationMode.NO_PATH
                self.last_action_msg = "Cannot start autonomous navigation: No valid path."
                self.last_action_status = "blocked"
                return

        self.nav_mode = NavigationMode.AUTONOMOUS
        self.step_timer_ms = 0
        self.last_action_msg = f"Autonomous navigation STARTED ({self.path_length} moves)."
        self.last_action_status = "success"

    def step_autonomous(self) -> bool:
        """Advance the vehicle one waypoint step along the planned route.

        Returns:
            bool: True if vehicle advanced, False if halted, paused, or arrived.
        """
        if self.nav_mode != NavigationMode.AUTONOMOUS:
            return False

        # Destination check
        if self.vehicle_pos == self.destination_pos:
            self.nav_mode = NavigationMode.GOAL_REACHED
            self.last_action_msg = "Goal Reached! Vehicle arrived at destination marker."
            self.last_action_status = "success"
            return False

        # Verify remaining path
        if not self.planned_path or len(self.planned_path) < 2:
            success = self.replan_route(reason="End of path or empty route")
            if not success or len(self.planned_path) < 2:
                return False

        # Check if route ahead is blocked
        if self.is_route_blocked():
            success = self.replan_route(reason="Obstacle detected on remaining route")
            if not success or len(self.planned_path) < 2:
                return False

        next_cell = self.planned_path[1]
        nr, nc = next_cell

        # Strict safety check before moving into cell
        if self.grid[nr, nc] != CellType.DRIVABLE_ROAD:
            success = self.replan_route(reason=f"Waypoint ({nc}, {nr}) blocked")
            if not success or len(self.planned_path) < 2:
                return False
            next_cell = self.planned_path[1]
            nr, nc = next_cell
            if self.grid[nr, nc] != CellType.DRIVABLE_ROAD:
                self.nav_mode = NavigationMode.NO_PATH
                self.last_action_msg = "Collision prevented: Path blocked. Stopped safely."
                self.last_action_status = "blocked"
                return False

        # Advance vehicle to next waypoint
        self.vehicle_pos = next_cell
        self.planned_path.pop(0)
        self.path_length = max(0, len(self.planned_path) - 1)

        # Arrival verification
        if self.vehicle_pos == self.destination_pos:
            self.nav_mode = NavigationMode.GOAL_REACHED
            self.last_action_msg = "Goal Reached! Vehicle arrived at destination marker."
            self.last_action_status = "success"
        else:
            self.last_action_msg = f"Autonomous step to ({nc}, {nr}). {self.path_length} moves remaining."
            self.last_action_status = "normal"

        return True

    def update(self, dt_ms: int = 16) -> None:
        """Periodic tick update managing timed autonomous stepping.

        Args:
            dt_ms: Milliseconds elapsed since the previous update frame.
        """
        if self.nav_mode == NavigationMode.AUTONOMOUS:
            self.step_timer_ms += dt_ms
            while self.step_timer_ms >= self.step_delay_ms and self.nav_mode == NavigationMode.AUTONOMOUS:
                self.step_timer_ms -= self.step_delay_ms
                stepped = self.step_autonomous()
                if not stepped:
                    break

    def insert_dynamic_obstacle(self, ahead_steps: int = 3) -> Optional[Tuple[int, int]]:
        """Insert a new dynamic obstacle on the terrain corridor during simulation.

        Prefers a valid drivable cell on the remaining planned route ahead of the vehicle.
        Falls back to any safe drivable road cell if no route exists or route cells are unavailable.

        Args:
            ahead_steps: Target distance in steps ahead of the vehicle along the path.

        Returns:
            Optional[Tuple[int, int]]: Coordinate (r, c) of inserted obstacle, or None if failed.
        """
        chosen_cell: Optional[Tuple[int, int]] = None

        # 1. Prefer a cell on the remaining planned route ahead of the vehicle
        if self.planned_path and len(self.planned_path) > 1:
            route_candidates = [
                pos for pos in self.planned_path[1:]
                if pos != self.vehicle_pos
                and pos != self.destination_pos
                and pos not in self.obstacle_set
                and self.grid[pos] == CellType.DRIVABLE_ROAD
            ]
            if route_candidates:
                target_idx = min(ahead_steps - 1, len(route_candidates) - 1)
                chosen_cell = route_candidates[target_idx]

        # 2. Fallback: Search safe drivable road cells
        if chosen_cell is None:
            road_candidates: List[Tuple[int, int]] = []
            for r in range(self.rows):
                for c in range(self.cols):
                    pos = (r, c)
                    if (
                        self.grid[r, c] == CellType.DRIVABLE_ROAD
                        and pos != self.vehicle_pos
                        and pos != self.destination_pos
                        and pos not in self.obstacle_set
                    ):
                        road_candidates.append(pos)

            if road_candidates:
                # Pick a candidate ahead of the vehicle row if possible
                ahead_candidates = [pos for pos in road_candidates if pos[0] >= self.vehicle_pos[0]]
                chosen_cell = ahead_candidates[0] if ahead_candidates else road_candidates[0]

        # 3. If no drivable cell could be found
        if chosen_cell is None:
            self.last_action_msg = "Dynamic obstacle failed: No available drivable cells."
            self.last_action_status = "blocked"
            return None

        # 4. Apply dynamic obstacle to grid and tracking sets
        r, c = chosen_cell
        self.grid[r, c] = CellType.OBSTACLE
        self.obstacle_set.add(chosen_cell)
        self.dynamic_obstacle_set.add(chosen_cell)

        # 5. Check if the inserted obstacle obstructs the active route
        blocks_route = self.planned_path and (chosen_cell in self.planned_path)
        if blocks_route:
            if self.nav_mode == NavigationMode.AUTONOMOUS:
                self.replan_route(reason=f"Hazard spawned on active route at ({c}, {r})")
            elif self.planned_path:
                self.replan_route(reason=f"Hazard spawned on route at ({c}, {r})")
        else:
            self.last_action_msg = f"Dynamic hazard placed at ({c}, {r}) off active path. No replan needed."
            self.last_action_status = "normal"

        return chosen_cell

    def toggle_perception_mapping(self) -> bool:
        """Toggle perception-derived obstacle mapping on or off.

        When disabled, all perception-derived cells are cleared from the grid.

        Returns:
            bool: True if now enabled, False if disabled.
        """
        self.perception_enabled = not self.perception_enabled
        state_str = "ON" if self.perception_enabled else "OFF"
        if not self.perception_enabled:
            self.clear_perception_obstacles()
            self.last_action_msg = f"Perception Mapping: {state_str}. Vision-derived obstacle cells cleared."
            self.last_action_status = "normal"
        else:
            self.last_action_msg = f"Perception Mapping: {state_str}. Streaming camera detections to grid."
            self.last_action_status = "success"
        logger.info("Perception mapping toggled: state=%s", state_str)
        return self.perception_enabled

    def set_perception_obstacles(self, cells: Set[Tuple[int, int]]) -> None:
        """Update perception-derived obstacle cells on the terrain grid.

        Guarantees:
        - Decoupled from static & dynamic hazards: only perception cells are updated.
        - Safety invariant: Never overwrites vehicle or destination cell.
        - Triggers adaptive replanning if a new perception hazard obstructs the active route.

        Args:
            cells: Set of (row, col) coordinates to set as perception obstacles.
        """
        if not self.perception_enabled:
            return

        # 1. Prune removed cells (cells previously in perception_obstacle_set but not in new cells)
        cells_to_remove = self.perception_obstacle_set - cells
        for r, c in cells_to_remove:
            # Preserve cell if it was independently a static or dynamic hazard
            if (r, c) not in self.initial_obstacle_coords and (r, c) not in self.dynamic_obstacle_set:
                self.grid[r, c] = CellType.DRIVABLE_ROAD
                self.obstacle_set.discard((r, c))

        # 2. Add new valid perception cells
        cells_to_add = cells - self.perception_obstacle_set
        for r, c in cells_to_add:
            # Strict safety invariant: never overwrite vehicle or destination
            if (r, c) == self.vehicle_pos:
                self.perception_conflict_count += 1
                continue
            if (r, c) == self.destination_pos:
                self.perception_conflict_count += 1
                continue

            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r, c] = CellType.OBSTACLE
                self.obstacle_set.add((r, c))

        # 3. Store the sanitized perception obstacle set
        self.perception_obstacle_set = {
            (r, c) for (r, c) in cells
            if (r, c) != self.vehicle_pos and (r, c) != self.destination_pos
            and 0 <= r < self.rows and 0 <= c < self.cols
        }
        self.perception_last_update_count = len(self.perception_obstacle_set)

        # 4. Check if newly marked perception obstacles block active route
        if self.is_route_blocked():
            if self.nav_mode == NavigationMode.AUTONOMOUS:
                self.replan_route(reason="Perception obstacle detected on active route")
            elif self.planned_path:
                self.replan_route(reason="Perception obstacle detected on route")
        elif cells_to_remove and self.nav_mode == NavigationMode.NO_PATH:
            res = self.plan_path()
            if res.success:
                self.nav_mode = NavigationMode.AUTONOMOUS
                self.last_action_msg = "Route unblocked: Perception obstacle cleared. Autonomous driving resumed."
                self.last_action_status = "success"

    def clear_perception_obstacles(self) -> None:
        """Clear all perception-derived cells from grid and restore drivable terrain."""
        for r, c in self.perception_obstacle_set:
            if (r, c) not in self.initial_obstacle_coords and (r, c) not in self.dynamic_obstacle_set:
                self.grid[r, c] = CellType.DRIVABLE_ROAD
                self.obstacle_set.discard((r, c))

        self.perception_obstacle_set.clear()
        self.perception_last_update_count = 0

    def render(
        self,
        screen: pygame.Surface,
        fonts: dict,
        webcam_status: str = "Standby",
        model_status: str = "Ready (YOLO11n)",
        detection_count: int = 0,
        inference_latency_ms: Optional[float] = None
    ) -> None:
        """Render the complete simulation frame: header, terrain grid, entities, and sidebar HUD.

        Args:
            screen: Pygame surface destination.
            fonts: Dictionary containing initialized pygame Font objects.
            webcam_status: Live status string from WebcamManager.
            model_status: Status text for the YOLO object detection model.
            detection_count: Number of objects detected in the current perception frame.
            inference_latency_ms: Latency of the most recent neural network inference in ms.
        """
        screen.fill(self.COLOR_BG)

        self._render_header(screen, fonts)
        self._render_grid(screen)
        self._render_path(screen)
        self._render_entities(screen, fonts)
        self._render_sidebar(screen, fonts, webcam_status, model_status, detection_count, inference_latency_ms)
        self._render_footer(screen, fonts)

    def _render_header(self, screen: pygame.Surface, fonts: dict) -> None:
        """Render title and subheadings at top of canvas."""
        title_surf = fonts["title"].render(self.WINDOW_TITLE, True, self.COLOR_TEXT_PRIMARY)
        screen.blit(title_surf, (self.grid_origin_x, 18))

        sub_text = "Adaptive Path Planning & Collision Avoidance  |  Autonomous Off-Road Navigation System"
        sub_surf = fonts["small"].render(sub_text, True, self.COLOR_ACCENT_CYAN)
        screen.blit(sub_surf, (self.grid_origin_x, 52))

        # Thin header accent line
        pygame.draw.line(screen, self.COLOR_PANEL_BORDER, (self.grid_origin_x, 76), (self.WINDOW_WIDTH - 24, 76), 1)

    def _render_grid(self, screen: pygame.Surface) -> None:
        """Render grid cells distinguishing drivable unpaved road from rough shoulders."""
        for r in range(self.rows):
            for c in range(self.cols):
                px = self.grid_origin_x + c * self.cell_size
                py = self.grid_origin_y + r * self.cell_size
                rect = pygame.Rect(px, py, self.cell_size, self.cell_size)

                cell_state = self.grid[r, c]

                if cell_state == CellType.DRIVABLE_ROAD:
                    # Draw unpaved dirt road surface with subtle checker alternation for visual depth
                    base_color = self.COLOR_DRIVABLE_ROAD if (r + c) % 2 == 0 else self.COLOR_ROAD_ACCENT
                    pygame.draw.rect(screen, base_color, rect)
                elif cell_state == CellType.OBSTACLE:
                    # Obstacle base is road with red hazard marker drawn later
                    pygame.draw.rect(screen, self.COLOR_DRIVABLE_ROAD, rect)
                else:
                    # Non-drivable rough verge / ditch
                    pygame.draw.rect(screen, self.COLOR_BORDER_TERRAIN, rect)

                # Grid cell outlines
                pygame.draw.rect(screen, self.COLOR_GRID_LINE, rect, 1)

        # Border outline around entire grid area
        grid_boundary = pygame.Rect(
            self.grid_origin_x,
            self.grid_origin_y,
            self.cols * self.cell_size,
            self.rows * self.cell_size
        )
        pygame.draw.rect(screen, self.COLOR_PANEL_BORDER, grid_boundary, 2)

    def _render_path(self, screen: pygame.Surface) -> None:
        """Render the planned A* route as visible green cells and a continuous trajectory line."""
        if not self.planned_path:
            return

        # 1. Draw glowing transparent green highlight over all path cells
        path_cell_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        path_cell_surf.fill((34, 197, 94, 90))  # Emerald green with subtle transparency

        for r, c in self.planned_path:
            px = self.grid_origin_x + c * self.cell_size
            py = self.grid_origin_y + r * self.cell_size
            screen.blit(path_cell_surf, (px, py))
            inner_rect = pygame.Rect(px + 1, py + 1, self.cell_size - 2, self.cell_size - 2)
            pygame.draw.rect(screen, (74, 222, 128, 140), inner_rect, 1)

        # 2. Draw continuous trajectory line connecting cell center points
        if len(self.planned_path) > 1:
            points = [
                (
                    self.grid_origin_x + c * self.cell_size + self.cell_size // 2,
                    self.grid_origin_y + r * self.cell_size + self.cell_size // 2,
                )
                for r, c in self.planned_path
            ]
            # Outer dark green line for contrast
            pygame.draw.lines(screen, (20, 83, 45), False, points, 5)
            # Inner bright emerald core line
            pygame.draw.lines(screen, self.COLOR_PATH_LINE, False, points, 3)

            # Node beads along trajectory
            for pt in points:
                pygame.draw.circle(screen, self.COLOR_PATH_ACCENT, pt, 3)

    def _render_entities(self, screen: pygame.Surface, fonts: dict) -> None:
        """Render static obstacles, dynamic hazards, perception obstacles, destination marker, and vehicle."""
        # 1. Obstacles with distinct visual hierarchy
        for (r, c) in self.obstacle_set:
            px = self.grid_origin_x + c * self.cell_size
            py = self.grid_origin_y + r * self.cell_size
            pad = 2
            obs_rect = pygame.Rect(px + pad, py + pad, self.cell_size - 2 * pad, self.cell_size - 2 * pad)
            center_pt = (px + self.cell_size // 2, py + self.cell_size // 2)

            # Perception-derived cells: Electric purple with neon fuchsia border and bright sensor pulse
            if (r, c) in self.perception_obstacle_set:
                pygame.draw.rect(screen, self.COLOR_PERCEPTION_OBSTACLE, obs_rect, border_radius=4)
                pygame.draw.rect(screen, self.COLOR_PERCEPTION_BORDER, obs_rect, width=2, border_radius=4)
                pygame.draw.circle(screen, self.COLOR_PERCEPTION_GLOW, center_pt, 4)
                pygame.draw.circle(screen, (255, 255, 255), center_pt, 2)

            # Dynamic hazards ('O' key): Vibrant safety orange with amber border and warning bead
            elif (r, c) in self.dynamic_obstacle_set:
                pygame.draw.rect(screen, (234, 88, 12), obs_rect, border_radius=4)
                pygame.draw.rect(screen, self.COLOR_DYNAMIC_BORDER, obs_rect, width=2, border_radius=4)
                pygame.draw.circle(screen, (254, 240, 138), center_pt, 3)

            # Static obstacles: Deep coral red with cross-hatch warning pattern
            else:
                pygame.draw.rect(screen, self.COLOR_OBSTACLE, obs_rect, border_radius=4)
                pygame.draw.rect(screen, (185, 28, 28), obs_rect, width=1, border_radius=4)
                pygame.draw.line(screen, (120, 20, 20), (px + pad + 3, py + pad + 3), (px + self.cell_size - pad - 4, py + self.cell_size - pad - 4), 2)
                pygame.draw.line(screen, (120, 20, 20), (px + self.cell_size - pad - 4, py + pad + 3), (px + pad + 3, py + self.cell_size - pad - 4), 2)

        # 2. Destination marker (Golden yellow target with concentric guide)
        dest_r, dest_c = self.destination_pos
        dest_px = self.grid_origin_x + dest_c * self.cell_size
        dest_py = self.grid_origin_y + dest_r * self.cell_size
        dest_rect = pygame.Rect(dest_px + 2, dest_py + 2, self.cell_size - 4, self.cell_size - 4)
        pygame.draw.rect(screen, self.COLOR_DESTINATION, dest_rect, border_radius=5)
        dest_center = (dest_px + self.cell_size // 2, dest_py + self.cell_size // 2)
        pygame.draw.circle(screen, (180, 140, 0), dest_center, 5, 2)

        # 3. Vehicle marker (Electric blue with status halo)
        veh_r, veh_c = self.vehicle_pos
        veh_px = self.grid_origin_x + veh_c * self.cell_size
        veh_py = self.grid_origin_y + veh_r * self.cell_size
        veh_rect = pygame.Rect(veh_px + 2, veh_py + 2, self.cell_size - 4, self.cell_size - 4)
        halo_rect = pygame.Rect(veh_px - 1, veh_py - 1, self.cell_size + 2, self.cell_size + 2)

        # Active state halos around vehicle
        if self.nav_mode == NavigationMode.AUTONOMOUS:
            pygame.draw.rect(screen, self.COLOR_ACCENT_CYAN, halo_rect, 2, border_radius=6)
        elif self.nav_mode == NavigationMode.PAUSED:
            pygame.draw.rect(screen, self.COLOR_ACCENT_AMBER, halo_rect, 2, border_radius=6)
        elif self.nav_mode == NavigationMode.REPLANNING:
            pygame.draw.rect(screen, self.COLOR_MODE_REPLANNING, halo_rect, 2, border_radius=6)
        elif self.nav_mode == NavigationMode.GOAL_REACHED:
            pygame.draw.rect(screen, self.COLOR_ACCENT_GREEN, halo_rect, 2, border_radius=6)
        elif self.nav_mode == NavigationMode.NO_PATH:
            pygame.draw.rect(screen, self.COLOR_OBSTACLE, halo_rect, 2, border_radius=6)

        pygame.draw.rect(screen, self.COLOR_VEHICLE, veh_rect, border_radius=5)
        veh_center = (veh_px + self.cell_size // 2, veh_py + self.cell_size // 2)
        pygame.draw.circle(screen, (255, 255, 255), veh_center, 4)

    def _render_sidebar(
        self,
        screen: pygame.Surface,
        fonts: dict,
        webcam_status: str,
        model_status: str = "Ready (YOLO11n)",
        detection_count: int = 0,
        inference_latency_ms: Optional[float] = None
    ) -> None:
        """Render side telemetry, legend, perception status, and control guides."""
        card_x = self.sidebar_x
        card_w = self.sidebar_width
        curr_y = self.grid_origin_y

        # ======================================================================
        # Card 1: Autonomous Navigation HUD
        # ======================================================================
        card1_h = 188
        self._draw_panel_box(screen, card_x, curr_y, card_w, card1_h, "AUTONOMOUS NAVIGATION HUD")

        veh_r, veh_c = self.vehicle_pos
        dest_r, dest_c = self.destination_pos

        path_length_str = f"{self.path_length} moves" if self.planned_path else "0"
        explored_str = f"{self.explored_nodes_count} cells" if self.explored_nodes_count > 0 else "0"
        time_str = f"{self.planning_time_ms:.2f} ms" if self.planning_time_ms > 0 else "N/A"

        # Prominent Navigation State Banner / Badge
        badge_rect = pygame.Rect(card_x + 12, curr_y + 25, card_w - 24, 20)
        if self.nav_mode == NavigationMode.AUTONOMOUS:
            badge_bg = (12, 74, 110)
            badge_border = (56, 189, 248)
            badge_text = "● AUTONOMOUS RUNNING"
            text_color = (224, 242, 254)
        elif self.nav_mode == NavigationMode.PAUSED:
            badge_bg = (120, 53, 15)
            badge_border = (251, 191, 36)
            badge_text = "⏸ PAUSED (SPACE to resume)"
            text_color = (254, 243, 199)
        elif self.nav_mode == NavigationMode.REPLANNING:
            badge_bg = (112, 26, 117)
            badge_border = (217, 70, 239)
            badge_text = "⟳ ADAPTIVE REPLANNING"
            text_color = (250, 232, 255)
        elif self.nav_mode == NavigationMode.GOAL_REACHED:
            badge_bg = (20, 83, 45)
            badge_border = (74, 222, 128)
            badge_text = "★ DESTINATION REACHED"
            text_color = (220, 252, 231)
        elif self.nav_mode == NavigationMode.NO_PATH:
            badge_bg = (127, 29, 29)
            badge_border = (248, 113, 113)
            badge_text = "⚠ NO PATH - SAFE STOP"
            text_color = (254, 226, 226)
        else:  # MANUAL
            badge_bg = (30, 41, 59)
            badge_border = (148, 163, 184)
            badge_text = "▶ MANUAL CONTROL (WASD)"
            text_color = (241, 245, 249)

        pygame.draw.rect(screen, badge_bg, badge_rect, border_radius=4)
        pygame.draw.rect(screen, badge_border, badge_rect, 1, border_radius=4)
        badge_surf = fonts["mono_small"].render(badge_text, True, text_color)
        badge_text_x = badge_rect.centerx - badge_surf.get_width() // 2
        badge_text_y = badge_rect.centery - badge_surf.get_height() // 2
        screen.blit(badge_surf, (badge_text_x, badge_text_y))

        lines_telemetry = [
            ("Vehicle Pos (X, Y):", f"({veh_c}, {veh_r})", self.COLOR_TEXT_PRIMARY),
            ("Destination (X, Y):", f"({dest_c}, {dest_r})", self.COLOR_TEXT_PRIMARY),
            ("Remaining Route:", path_length_str, self.COLOR_ACCENT_GREEN if self.planned_path else self.COLOR_TEXT_PRIMARY),
            ("Replanning Events:", f"{self.replanning_events} dynamic", self.COLOR_MODE_REPLANNING if self.replanning_events > 0 else self.COLOR_TEXT_PRIMARY),
            ("Collisions (Safety):", f"{self.collision_count} (Safe)", self.COLOR_ACCENT_GREEN),
            ("A* Compute Latency:", time_str, self.COLOR_TEXT_PRIMARY),
            ("Explored Search Cells:", explored_str, self.COLOR_TEXT_PRIMARY),
        ]

        ty = curr_y + 50
        for label, val, val_color in lines_telemetry:
            lbl_surf = fonts["small"].render(label, True, self.COLOR_TEXT_SECONDARY)
            val_surf = fonts["mono_small"].render(val, True, val_color)
            screen.blit(lbl_surf, (card_x + 14, ty))
            screen.blit(val_surf, (card_x + card_w - val_surf.get_width() - 14, ty))
            ty += 19

        curr_y += card1_h + 6

        # ======================================================================
        # Card 2: Perception & Sensors Status
        # ======================================================================
        card2_h = 126
        self._draw_panel_box(screen, card_x, curr_y, card_w, card2_h, "PERCEPTION & SENSORS")

        # 1. Webcam Stream Status
        cam_color = self.COLOR_TEXT_SECONDARY
        cam_upper = webcam_status.upper()
        if "ACTIVE" in cam_upper:
            cam_color = self.COLOR_ACCENT_GREEN
        elif "STARTING" in cam_upper or "INITIALIZING" in cam_upper:
            cam_color = self.COLOR_ACCENT_AMBER
        elif "ERROR" in cam_upper or "UNAVAILABLE" in cam_upper or "DENIED" in cam_upper:
            cam_color = self.COLOR_OBSTACLE

        cam_lbl = fonts["small"].render("Webcam Feed:", True, self.COLOR_TEXT_SECONDARY)
        cam_val = fonts["mono_small"].render(webcam_status[:17], True, cam_color)
        screen.blit(cam_lbl, (card_x + 14, curr_y + 24))
        screen.blit(cam_val, (card_x + card_w - cam_val.get_width() - 14, curr_y + 24))

        # 2. YOLO Model Status
        model_upper = model_status.upper()
        mod_color = self.COLOR_ACCENT_GREEN if "READY" in model_upper or "LOADED" in model_upper else self.COLOR_OBSTACLE
        mod_lbl = fonts["small"].render("YOLO Model:", True, self.COLOR_TEXT_SECONDARY)
        mod_val = fonts["mono_small"].render(model_status, True, mod_color)
        screen.blit(mod_lbl, (card_x + 14, curr_y + 43))
        screen.blit(mod_val, (card_x + card_w - mod_val.get_width() - 14, curr_y + 43))

        # 3. Vision Detections & Inference Latency
        det_lbl = fonts["small"].render("Vision Detections:", True, self.COLOR_TEXT_SECONDARY)
        lat_str = f"{inference_latency_ms:.1f}ms" if inference_latency_ms is not None else "--"
        det_str = f"{detection_count} obj ({lat_str})"
        det_color = self.COLOR_ACCENT_CYAN if detection_count > 0 else self.COLOR_TEXT_PRIMARY
        det_val = fonts["mono_small"].render(det_str, True, det_color)
        screen.blit(det_lbl, (card_x + 14, curr_y + 62))
        screen.blit(det_val, (card_x + card_w - det_val.get_width() - 14, curr_y + 62))

        # 4. Perception Mapping State
        map_lbl = fonts["small"].render("Perception Mapping:", True, self.COLOR_TEXT_SECONDARY)
        if self.perception_enabled:
            map_status_str = f"ON ({len(self.perception_obstacle_set)} cells)"
            map_color = self.COLOR_ACCENT_GREEN
        else:
            map_status_str = "OFF (Press 'M')"
            map_color = (156, 163, 175)
        map_val = fonts["mono_small"].render(map_status_str, True, map_color)
        screen.blit(map_lbl, (card_x + 14, curr_y + 81))
        screen.blit(map_val, (card_x + card_w - map_val.get_width() - 14, curr_y + 81))

        # 5. Safety Invariant Guard
        guard_lbl = fonts["small"].render("Safety Guard:", True, self.COLOR_TEXT_SECONDARY)
        guard_val = fonts["mono_small"].render("Vehicle & Goal Safe", True, self.COLOR_ACCENT_GREEN)
        screen.blit(guard_lbl, (card_x + 14, curr_y + 100))
        screen.blit(guard_val, (card_x + card_w - guard_val.get_width() - 14, curr_y + 100))

        curr_y += card2_h + 6

        # ======================================================================
        # Card 3: Color Legend
        # ======================================================================
        card3_h = 138
        self._draw_panel_box(screen, card_x, curr_y, card_w, card3_h, "TERRAIN & ROUTE LEGEND")
        legend_items = [
            (self.COLOR_VEHICLE, "Vehicle (Autonomous Agent)"),
            (self.COLOR_DESTINATION, "Target Navigation Destination"),
            (self.COLOR_OBSTACLE, "Static Hazards (Road Corridors)"),
            (self.COLOR_DYNAMIC_BORDER, "Dynamic Hazards ('O' Key)"),
            (self.COLOR_PERCEPTION_OBSTACLE, "Perception Hazards (Vision/YOLO)"),
            (self.COLOR_PATH, "Planned Route (A* Path)"),
            (self.COLOR_DRIVABLE_ROAD, "Drivable Unstructured Road"),
        ]

        ly = curr_y + 24
        for color, name in legend_items:
            swatch_rect = pygame.Rect(card_x + 14, ly + 2, 11, 11)
            pygame.draw.rect(screen, color, swatch_rect, border_radius=3)
            pygame.draw.rect(screen, self.COLOR_PANEL_BORDER, swatch_rect, 1)

            name_surf = fonts["small"].render(name, True, self.COLOR_TEXT_PRIMARY)
            screen.blit(name_surf, (card_x + 32, ly))
            ly += 15

        curr_y += card3_h + 6

        # ======================================================================
        # Card 4: Keyboard Controls Guide
        # ======================================================================
        card4_h = 130
        self._draw_panel_box(screen, card_x, curr_y, card_w, card4_h, "KEYBOARD SHORTCUTS")
        controls = [
            ("SPACE", "Start / Pause autonomous"),
            ("M Key", "Toggle perception mapping"),
            ("O Key", "Insert dynamic hazard"),
            ("P Key", "Compute / refresh route"),
            ("R Key", "Reset vehicle & hazards"),
            ("W/A/S/D", "Manual step (when inactive)"),
            ("C / ESC", "Webcam feed / Safe exit"),
        ]

        cy = curr_y + 23
        for key_str, desc in controls:
            k_surf = fonts["mono_small"].render(key_str, True, self.COLOR_ACCENT_AMBER)
            d_surf = fonts["small"].render(desc, True, self.COLOR_TEXT_SECONDARY)
            screen.blit(k_surf, (card_x + 14, cy))
            screen.blit(d_surf, (card_x + 85, cy))
            cy += 14

    def _render_footer(self, screen: pygame.Surface, fonts: dict) -> None:
        """Render status action feedback bar at the bottom."""
        footer_y = self.grid_origin_y + (self.rows * self.cell_size) + 12
        footer_rect = pygame.Rect(self.grid_origin_x, footer_y, self.cols * self.cell_size, 40)
        pygame.draw.rect(screen, self.COLOR_PANEL_BG, footer_rect, border_radius=6)
        pygame.draw.rect(screen, self.COLOR_PANEL_BORDER, footer_rect, 1)

        status_prefix = "SYSTEM STATUS: "
        prefix_surf = fonts["small"].render(status_prefix, True, self.COLOR_ACCENT_CYAN)
        screen.blit(prefix_surf, (self.grid_origin_x + 14, footer_y + 12))

        msg_color = self.COLOR_TEXT_PRIMARY
        if self.last_action_status == "blocked":
            msg_color = self.COLOR_OBSTACLE
        elif self.last_action_status == "success":
            msg_color = self.COLOR_ACCENT_GREEN

        msg_surf = fonts["small"].render(self.last_action_msg, True, msg_color)
        screen.blit(msg_surf, (self.grid_origin_x + 14 + prefix_surf.get_width(), footer_y + 12))

    def _draw_panel_box(self, screen: pygame.Surface, x: int, y: int, w: int, h: int, title: str) -> None:
        """Draw a styled card container with a subtle header label."""
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, self.COLOR_PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(screen, self.COLOR_PANEL_BORDER, rect, 1)

        # Title banner
        header_font = pygame.font.SysFont("Helvetica", 11, bold=True)
        t_surf = header_font.render(title, True, self.COLOR_TEXT_SECONDARY)
        screen.blit(t_surf, (x + 14, y + 7))
        pygame.draw.line(screen, self.COLOR_PANEL_BORDER, (x + 10, y + 22), (x + w - 10, y + 22), 1)
