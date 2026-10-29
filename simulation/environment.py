"""Simulation environment for unstructured road navigation.

Provides a 2D Pygame simulation grid featuring irregular road boundaries,
static hazards, vehicle tracking, legend HUD, and manual keyboard testing.
"""

from enum import IntEnum
from typing import List, Tuple, Set, Optional
import math
import numpy as np
import pygame


class CellType(IntEnum):
    """Enumeration of cell types across the terrain grid."""
    NON_DRIVABLE_BORDER = 0  # Off-road rough verge, rocks, ditches
    DRIVABLE_ROAD = 1        # Winding, unpaved unstructured roadway
    OBSTACLE = 2             # Static road hazard (pothole, fallen rock, debris)


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
        self.last_action_msg: str = "Ready. Use WASD / Arrow keys to test manual movement."
        self.last_action_status: str = "normal"  # "normal", "success", "blocked"

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

    def reset_vehicle(self) -> None:
        """Reset the vehicle to its designated starting position."""
        self.vehicle_pos = self.start_pos
        self.last_action_msg = "Vehicle reset to starting coordinates."
        self.last_action_status = "normal"

    def move_vehicle(self, d_row: int, d_col: int) -> bool:
        """Attempt to move the vehicle by (d_row, d_col) with collision checking.

        Args:
            d_row: Row offset (-1, 0, +1).
            d_col: Column offset (-1, 0, +1).

        Returns:
            bool: True if move was valid and executed, False if blocked by boundary or obstacle.
        """
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

        # 3. Static obstacle check
        if cell_state == CellType.OBSTACLE:
            self.last_action_msg = f"Collision prevented: Static hazard detected at ({target_c}, {target_r})."
            self.last_action_status = "blocked"
            return False

        # Valid movement executed
        self.vehicle_pos = (target_r, target_c)
        if self.vehicle_pos == self.destination_pos:
            self.last_action_msg = "Goal Reached! Vehicle arrived at destination marker."
            self.last_action_status = "success"
        else:
            self.last_action_msg = f"Vehicle moved to cell ({target_c}, {target_r})."
            self.last_action_status = "normal"

        return True

    def render(self, screen: pygame.Surface, fonts: dict, webcam_status: str) -> None:
        """Render the complete simulation frame: header, terrain grid, entities, and sidebar HUD.

        Args:
            screen: Pygame surface destination.
            fonts: Dictionary containing initialized pygame Font objects.
            webcam_status: Live status string from WebcamManager.
        """
        screen.fill(self.COLOR_BG)

        self._render_header(screen, fonts)
        self._render_grid(screen)
        self._render_entities(screen, fonts)
        self._render_sidebar(screen, fonts, webcam_status)
        self._render_footer(screen, fonts)

    def _render_header(self, screen: pygame.Surface, fonts: dict) -> None:
        """Render title and subheadings at top of canvas."""
        title_surf = fonts["title"].render(self.WINDOW_TITLE, True, self.COLOR_TEXT_PRIMARY)
        screen.blit(title_surf, (self.grid_origin_x, 18))

        sub_text = "Autonomous Navigation on Unstructured Roads  |  Day 1: System Baseline & Environment Setup"
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

    def _render_entities(self, screen: pygame.Surface, fonts: dict) -> None:
        """Render static obstacles, destination marker, and the vehicle."""
        # 1. Static obstacles (Red blocks with distinct cross/warning icon)
        for (r, c) in self.obstacle_set:
            px = self.grid_origin_x + c * self.cell_size
            py = self.grid_origin_y + r * self.cell_size
            pad = 3
            obs_rect = pygame.Rect(px + pad, py + pad, self.cell_size - 2 * pad, self.cell_size - 2 * pad)
            pygame.draw.rect(screen, self.COLOR_OBSTACLE, obs_rect, border_radius=4)
            # Inner dark warning detail
            pygame.draw.line(screen, (120, 20, 20), (px + pad + 3, py + pad + 3), (px + self.cell_size - pad - 4, py + self.cell_size - pad - 4), 2)
            pygame.draw.line(screen, (120, 20, 20), (px + self.cell_size - pad - 4, py + pad + 3), (px + pad + 3, py + self.cell_size - pad - 4), 2)

        # 2. Destination marker (Yellow target with ring)
        dest_r, dest_c = self.destination_pos
        dest_px = self.grid_origin_x + dest_c * self.cell_size
        dest_py = self.grid_origin_y + dest_r * self.cell_size
        dest_rect = pygame.Rect(dest_px + 2, dest_py + 2, self.cell_size - 4, self.cell_size - 4)
        pygame.draw.rect(screen, self.COLOR_DESTINATION, dest_rect, border_radius=5)
        # Inner target circle
        center = (dest_px + self.cell_size // 2, dest_py + self.cell_size // 2)
        pygame.draw.circle(screen, (180, 140, 0), center, 5, 2)

        # 3. Vehicle marker (Electric blue rounded rectangle with headlights/heading indicator)
        veh_r, veh_c = self.vehicle_pos
        veh_px = self.grid_origin_x + veh_c * self.cell_size
        veh_py = self.grid_origin_y + veh_r * self.cell_size
        veh_rect = pygame.Rect(veh_px + 2, veh_py + 2, self.cell_size - 4, self.cell_size - 4)
        pygame.draw.rect(screen, self.COLOR_VEHICLE, veh_rect, border_radius=5)
        # Center vehicle core
        veh_center = (veh_px + self.cell_size // 2, veh_py + self.cell_size // 2)
        pygame.draw.circle(screen, (255, 255, 255), veh_center, 4)

    def _render_sidebar(self, screen: pygame.Surface, fonts: dict, webcam_status: str) -> None:
        """Render side telemetry, legend, and control guides."""
        card_x = self.sidebar_x
        card_w = self.sidebar_width
        curr_y = self.grid_origin_y

        # Card 1: System Telemetry
        card1_h = 175
        self._draw_panel_box(screen, card_x, curr_y, card_w, card1_h, "SYSTEM TELEMETRY")
        
        veh_r, veh_c = self.vehicle_pos
        dest_r, dest_c = self.destination_pos
        manhattan_dist = abs(dest_r - veh_r) + abs(dest_c - veh_c)

        lines_telemetry = [
            ("Execution Mode:", "Manual Test (Day 1)"),
            ("Vehicle Pos (X, Y):", f"({veh_c}, {veh_r})"),
            ("Destination (X, Y):", f"({dest_c}, {dest_r})"),
            ("Distance to Goal:", f"{manhattan_dist} cells (L1)"),
            ("Static Obstacles:", f"{len(self.obstacle_set)} hazards"),
        ]

        ty = curr_y + 36
        for label, val in lines_telemetry:
            lbl_surf = fonts["small"].render(label, True, self.COLOR_TEXT_SECONDARY)
            val_surf = fonts["mono_small"].render(val, True, self.COLOR_TEXT_PRIMARY)
            screen.blit(lbl_surf, (card_x + 14, ty))
            screen.blit(val_surf, (card_x + card_w - val_surf.get_width() - 14, ty))
            ty += 23

        curr_y += card1_h + 12

        # Card 2: Perception / Webcam Status
        card2_h = 75
        self._draw_panel_box(screen, card_x, curr_y, card_w, card2_h, "PERCEPTION & SENSORS")
        status_color = self.COLOR_ACCENT_GREEN if "Active" in webcam_status else self.COLOR_TEXT_SECONDARY
        if "Unavailable" in webcam_status or "Denied" in webcam_status or "Error" in webcam_status:
            status_color = self.COLOR_OBSTACLE

        cam_lbl = fonts["small"].render("Webcam Feed:", True, self.COLOR_TEXT_SECONDARY)
        cam_val = fonts["small"].render(webcam_status, True, status_color)
        screen.blit(cam_lbl, (card_x + 14, curr_y + 32))
        screen.blit(cam_val, (card_x + 14, curr_y + 50))

        curr_y += card2_h + 12

        # Card 3: Color Legend
        card3_h = 135
        self._draw_panel_box(screen, card_x, curr_y, card_w, card3_h, "TERRAIN LEGEND")
        legend_items = [
            (self.COLOR_VEHICLE, "Vehicle (Autonomous Agent)"),
            (self.COLOR_DESTINATION, "Target Navigation Destination"),
            (self.COLOR_OBSTACLE, "Static Hazard (Pothole / Boulder)"),
            (self.COLOR_DRIVABLE_ROAD, "Drivable Unstructured Road"),
            (self.COLOR_BORDER_TERRAIN, "Non-Drivable Verge / Shoulder"),
        ]

        ly = curr_y + 32
        for color, name in legend_items:
            swatch_rect = pygame.Rect(card_x + 14, ly + 2, 14, 14)
            pygame.draw.rect(screen, color, swatch_rect, border_radius=3)
            pygame.draw.rect(screen, self.COLOR_PANEL_BORDER, swatch_rect, 1)

            name_surf = fonts["small"].render(name, True, self.COLOR_TEXT_PRIMARY)
            screen.blit(name_surf, (card_x + 36, ly))
            ly += 20

        curr_y += card3_h + 12

        # Card 4: Keyboard Controls Guide
        card4_h = 140
        self._draw_panel_box(screen, card_x, curr_y, card_w, card4_h, "KEYBOARD CONTROLS")
        controls = [
            ("W / A / S / D", "Move vehicle (Grid steps)"),
            ("Arrow Keys", "Alternate movement"),
            ("R Key", "Reset vehicle position"),
            ("C Key", "Toggle webcam stream"),
            ("ESC Key", "Safe application exit"),
        ]

        cy = curr_y + 30
        for key_str, desc in controls:
            k_surf = fonts["mono_small"].render(key_str, True, self.COLOR_ACCENT_AMBER)
            d_surf = fonts["small"].render(desc, True, self.COLOR_TEXT_SECONDARY)
            screen.blit(k_surf, (card_x + 14, cy))
            screen.blit(d_surf, (card_x + 115, cy))
            cy += 21

    def _render_footer(self, screen: pygame.Surface, fonts: dict) -> None:
        """Render status action feedback bar at the bottom."""
        footer_y = self.grid_origin_y + (self.rows * self.cell_size) + 12
        footer_rect = pygame.Rect(self.grid_origin_x, footer_y, self.cols * self.cell_size, 38)
        pygame.draw.rect(screen, self.COLOR_PANEL_BG, footer_rect, border_radius=6)
        pygame.draw.rect(screen, self.COLOR_PANEL_BORDER, footer_rect, 1)

        status_prefix = "STATUS: "
        prefix_surf = fonts["small"].render(status_prefix, True, self.COLOR_ACCENT_CYAN)
        screen.blit(prefix_surf, (self.grid_origin_x + 14, footer_y + 10))

        msg_color = self.COLOR_TEXT_PRIMARY
        if self.last_action_status == "blocked":
            msg_color = self.COLOR_OBSTACLE
        elif self.last_action_status == "success":
            msg_color = self.COLOR_ACCENT_GREEN

        msg_surf = fonts["small"].render(self.last_action_msg, True, msg_color)
        screen.blit(msg_surf, (self.grid_origin_x + 14 + prefix_surf.get_width(), footer_y + 10))

    def _draw_panel_box(self, screen: pygame.Surface, x: int, y: int, w: int, h: int, title: str) -> None:
        """Draw a styled card container with a subtle header label."""
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, self.COLOR_PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(screen, self.COLOR_PANEL_BORDER, rect, 1)

        # Title banner
        header_font = pygame.font.SysFont("Helvetica", 11, bold=True)
        t_surf = header_font.render(title, True, self.COLOR_TEXT_SECONDARY)
        screen.blit(t_surf, (x + 14, y + 8))
        pygame.draw.line(screen, self.COLOR_PANEL_BORDER, (x + 10, y + 24), (x + w - 10, y + 24), 1)
