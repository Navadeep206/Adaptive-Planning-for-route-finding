"""Main entry point for Adaptive Path Planning and Simulation Environment.

Coordinates Pygame rendering loop, interactive keyboard controls (including
autonomous navigation via SPACE, dynamic hazard insertion via 'O', A* computation via 'P',
and reset via 'R'), reliable MacBook webcam stream via 'C', and manages clean system shutdown.
"""

import argparse
import sys
import logging
import pygame

from simulation.environment import RoadEnvironment
from perception.webcam import WebcamManager

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Main")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for simulation configuration."""
    parser = argparse.ArgumentParser(
        description="Adaptive Path Planning and Collision Avoidance - Simulation Baseline"
    )
    parser.add_argument(
        "--camera-index",
        "-c",
        type=int,
        default=None,
        help="Camera device index (default: auto-detect Mac built-in FaceTime camera, bypassing iPhone)"
    )
    return parser.parse_args()


def initialize_fonts() -> dict:
    """Initialize system fonts for the simulation user interface."""
    pygame.font.init()

    # Fallback to system fonts available cross-platform (macOS, Linux, Windows)
    title_font = pygame.font.SysFont(["Arial", "DejaVu Sans", "Helvetica"], 22, bold=True)
    regular_font = pygame.font.SysFont(["Arial", "DejaVu Sans", "Helvetica"], 14)
    small_font = pygame.font.SysFont(["Arial", "DejaVu Sans", "Helvetica"], 12)
    mono_small = pygame.font.SysFont(["Menlo", "Courier New", "Consolas"], 12, bold=True)

    return {
        "title": title_font,
        "regular": regular_font,
        "small": small_font,
        "mono_small": mono_small,
    }


def main() -> None:
    """Run the Day 3 simulation loop."""
    args = parse_arguments()
    logger.info("Initializing Day 3 Autonomous Navigation & Dynamic Replanning Environment...")

    # Initialize Pygame core
    pygame.init()
    pygame.display.set_caption(RoadEnvironment.WINDOW_TITLE)

    screen = pygame.display.set_mode((RoadEnvironment.WINDOW_WIDTH, RoadEnvironment.WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    fonts = initialize_fonts()

    # Instantiate Environment and Perception Manager
    env = RoadEnvironment()
    webcam = WebcamManager(device_index=args.camera_index)
    logger.info("Webcam manager configured for device: %s (Index %d)", webcam.device_name, webcam.device_index)

    running = True
    logger.info("Simulation initialized successfully. Entering main tick loop.")

    try:
        while running:
            # 1. Cap frame rate (60 FPS) and retrieve elapsed delta time
            dt_ms = clock.tick(60)

            # 2. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Exit requested via window close button.")
                    running = False

                elif event.type == pygame.KEYDOWN:
                    # Safe exit via ESC
                    if event.key == pygame.K_ESCAPE:
                        logger.info("Exit requested via ESC key.")
                        running = False

                    # Toggle Autonomous Navigation: Start / Pause / Resume
                    elif event.key == pygame.K_SPACE:
                        env.toggle_autonomous_navigation()
                        logger.info("Autonomous mode toggled: state=%s", env.nav_mode.value)

                    # Dynamic obstacle insertion ahead of the vehicle
                    elif event.key == pygame.K_o:
                        inserted = env.insert_dynamic_obstacle(ahead_steps=3)
                        if inserted:
                            logger.info("Dynamic hazard inserted at coordinate: %s", inserted)
                        else:
                            logger.warning("Could not insert dynamic hazard: no valid cell available.")

                    # A* Path planning calculation
                    elif event.key == pygame.K_p:
                        result = env.plan_path()
                        logger.info(
                            "A* planning triggered: success=%s, moves=%d, explored=%d, time=%.2f ms",
                            result.success,
                            result.path_length,
                            result.explored_nodes,
                            result.execution_time_ms
                        )

                    # Manual vehicle navigation: W/A/S/D or Arrow keys (active only when autonomous mode is off)
                    elif event.key in (pygame.K_w, pygame.K_UP):
                        env.move_vehicle(-1, 0)
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        env.move_vehicle(1, 0)
                    elif event.key in (pygame.K_a, pygame.K_LEFT):
                        env.move_vehicle(0, -1)
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        env.move_vehicle(0, 1)

                    # Reset vehicle position, hazards, and clear route
                    elif event.key == pygame.K_r:
                        env.reset_vehicle()
                        logger.info("Vehicle, dynamic hazards, and route reset to initial state.")

                    # Toggle OpenCV webcam preview
                    elif event.key == pygame.K_c:
                        is_active = webcam.toggle_preview()
                        logger.info("Webcam toggled: active=%s (%s, state=%s)", is_active, webcam.device_name, webcam.get_state().value)

            # 3. Update Autonomous Navigation & Timing
            env.update(dt_ms)

            # 4. Update Perception Feed (Non-blocking)
            webcam.update()

            # 5. Render Graphics
            env.render(screen, fonts, webcam.get_status())
            pygame.display.flip()

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user (KeyboardInterrupt).")

    except Exception as exc:
        logger.error("Unhandled runtime exception in main loop: %s", exc, exc_info=True)

    finally:
        # Guaranteed cleanup of all hardware and graphic resources
        logger.info("Shutting down resources...")
        webcam.release()
        pygame.quit()
        logger.info("Application exited safely.")


if __name__ == "__main__":
    main()
