"""Main entry point for Day 2: Adaptive Path Planning Simulation Environment.

Sets up the Pygame rendering loop, handles interactive keyboard controls (including
A* path computation via 'P' and reset via 'R'), coordinates the OpenCV webcam stream,
and manages clean system shutdown.
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
    """Run the Day 1 simulation loop."""
    args = parse_arguments()
    logger.info("Initializing Day 1 Adaptive Path Planning Environment...")

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
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Exit requested via window close button.")
                    running = False

                elif event.type == pygame.KEYDOWN:
                    # Safe exit via ESC
                    if event.key == pygame.K_ESCAPE:
                        logger.info("Exit requested via ESC key.")
                        running = False

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

                    # Manual vehicle navigation: W/A/S/D or Arrow keys
                    elif event.key in (pygame.K_w, pygame.K_UP):
                        env.move_vehicle(-1, 0)
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        env.move_vehicle(1, 0)
                    elif event.key in (pygame.K_a, pygame.K_LEFT):
                        env.move_vehicle(0, -1)
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        env.move_vehicle(0, 1)

                    # Reset vehicle position and clear route
                    elif event.key == pygame.K_r:
                        env.reset_vehicle()
                        logger.info("Vehicle and route reset to initial state.")

                    # Toggle OpenCV webcam preview
                    elif event.key == pygame.K_c:
                        is_active = webcam.toggle_preview()
                        logger.info("Webcam toggled: active=%s (%s)", is_active, webcam.device_name)

            # 2. Update Perception Feed (Non-blocking)
            webcam.update()

            # 3. Render Graphics
            env.render(screen, fonts, webcam.get_status())
            pygame.display.flip()

            # 4. Cap frame rate (60 FPS)
            clock.tick(60)

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
