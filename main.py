"""Main entry point for Adaptive Path Planning and Simulation Environment.

Coordinates Pygame rendering loop, interactive keyboard controls (including
autonomous navigation via SPACE, dynamic hazard insertion via 'O', A* computation via 'P',
and reset via 'R'), reliable MacBook webcam stream via 'C', and manages clean system shutdown.
"""

import argparse
import os
import sys
import time
import logging
import pygame

from simulation.environment import RoadEnvironment
from perception.webcam import WebcamManager
from perception.object_detector import YOLOObjectDetector
from perception.object_tracker import ObjectTracker, TrackerConfig
from perception.camera_calibrator import CameraCalibrator
from perception.grid_mapper import GridMapper, GridMapperConfig

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
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="YOLO Non-Maximum Suppression (NMS) IoU threshold (default: 0.45)"
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="none",
        choices=["none", "clahe", "gamma"],
        help="Preprocessing contrast enhancement: 'none', 'clahe', or 'gamma' (default: 'none')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO model weights path (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--inflation",
        type=int,
        default=0,
        help="Perception obstacle safety inflation margin in cells (default: 0)"
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Explicitly open the webcam preview upon startup"
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Do not open the webcam preview at startup (standalone grid mode)"
    )
    parser.add_argument(
        "--mapping",
        action="store_true",
        help="Automatically activate perception-to-grid mapping upon startup"
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
    """Run the main simulation loop for Adaptive Path Planning."""
    import os
    args = parse_arguments()
    logger.info("Initializing Adaptive Path Planning & Autonomous Navigation Environment...")

    # Position Pygame simulation window on the left side of the display
    if "SDL_VIDEO_WINDOW_POS" not in os.environ:
        os.environ["SDL_VIDEO_WINDOW_POS"] = "10,40"

    # Initialize Pygame core
    pygame.init()
    pygame.display.set_caption(RoadEnvironment.WINDOW_TITLE)

    screen = pygame.display.set_mode((RoadEnvironment.WINDOW_WIDTH, RoadEnvironment.WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    fonts = initialize_fonts()

    # Query display resolution for side-by-side non-overlapping window placement
    screen_info = pygame.display.Info()
    disp_w = screen_info.current_w if screen_info.current_w > 0 else 1440
    disp_h = screen_info.current_h if screen_info.current_h > 0 else 900

    pygame_w = RoadEnvironment.WINDOW_WIDTH   # 1000
    pygame_h = RoadEnvironment.WINDOW_HEIGHT  # 700

    # Calculate non-overlapping side-by-side coordinates for the OpenCV sensor feed window
    if disp_w >= 1420:
        cv_x = pygame_w + 20
        cv_y = 40
        cv_w = min(400, disp_w - cv_x - 10)
        cv_h = int(cv_w * 0.75)
    else:
        cv_x = max(0, disp_w - 410)
        cv_y = 40
        cv_w = 400
        cv_h = 300

    # Instantiate Environment, Perception Manager, Detector, and Grid Mapper
    env = RoadEnvironment()
    webcam = WebcamManager(
        device_index=args.camera_index,
        window_pos=(cv_x, cv_y),
        window_size=(cv_w, cv_h)
    )
    logger.info("Webcam manager configured for device: %s (Index %d)", webcam.device_name, webcam.device_index)

    # Immediately render and flip the initial simulation frame so the grid is visible right away
    env.render(
        screen=screen,
        fonts=fonts,
        webcam_status=webcam.get_status(),
        model_status="Loading YOLO...",
        detection_count=0,
        inference_latency_ms=None
    )
    pygame.display.flip()

    detector = YOLOObjectDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        preprocess_mode=args.preprocess,
        auto_load=True
    )

    calibrator = CameraCalibrator()
    tracker = ObjectTracker(
        detector=detector,
        config=TrackerConfig(
            min_confidence=args.conf,
            track_activation_threshold=2,
            lost_timeout_sec=1.0,
            max_lost_frames=15,
            history_len=30,
            enable_box_smoothing=True
        )
    )

    mapper_config = GridMapperConfig(
        min_confidence=args.conf,
        grid_rows=env.rows,
        grid_cols=env.cols,
        inflation_radius=args.inflation,
        calibrator=calibrator,
        debounce_hits=1
    )
    mapper = GridMapper(config=mapper_config)

    # Throttling timer for vision inference (e.g. 100ms ~ 10 FPS)
    perception_timer_ms: int = 0
    perception_interval_ms: int = 100

    # Tracking & detection state cache for smooth rendering
    last_tracks = []
    last_telemetry = {}
    last_inf_ms = None

    # Simultaneous Dual-Window Operation:
    # If mapping is enabled, start camera and mapping
    if args.mapping:
        env.toggle_perception_mapping()
        webcam.start_preview()
        if not detector.is_loaded():
            detector.load_model()
        logger.info("Perception mapping auto-enabled from startup arguments.")
    elif not args.no_camera:
        # Default: auto-start camera preview so both windows run simultaneously
        opened = webcam.start_preview()
        if opened:
            logger.info("Simultaneous dual-window mode active: Pygame Grid (left) + Sensor Feed (right).")
        else:
            logger.info("Camera unavailable; running in standalone simulation grid mode.")
    elif args.camera:
        webcam.start_preview()
        logger.info("Webcam preview auto-started from startup arguments.")

    running = True
    logger.info("Simulation initialized successfully. Entering main tick loop.")

    try:
        while running:
            # 1. Cap frame rate (60 FPS) and retrieve elapsed delta time
            dt_ms = clock.tick(60)

            # 2. Event Handling: Gather events from Pygame and OpenCV windows
            events_to_process = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Exit requested via window close button.")
                    running = False
                elif event.type == pygame.KEYDOWN:
                    events_to_process.append(event.key)

            # Capture keypresses when the OpenCV preview window has active focus
            cv_key = webcam.pop_key()
            if cv_key is not None:
                if cv_key in (ord('m'), ord('M')):
                    events_to_process.append(pygame.K_m)
                elif cv_key in (ord('p'), ord('P')):
                    events_to_process.append(pygame.K_p)
                elif cv_key == ord(' '):
                    events_to_process.append(pygame.K_SPACE)
                elif cv_key in (ord('t'), ord('T')):
                    events_to_process.append(pygame.K_t)
                elif cv_key in (ord('r'), ord('R')):
                    events_to_process.append(pygame.K_r)
                elif cv_key in (ord('o'), ord('O')):
                    events_to_process.append(pygame.K_o)
                elif cv_key in (ord('c'), ord('C')):
                    events_to_process.append(pygame.K_c)
                elif cv_key in (ord('w'), ord('W')):
                    events_to_process.append(pygame.K_w)
                elif cv_key in (ord('s'), ord('S')):
                    events_to_process.append(pygame.K_s)
                elif cv_key in (ord('a'), ord('A')):
                    events_to_process.append(pygame.K_a)
                elif cv_key in (ord('d'), ord('D')):
                    events_to_process.append(pygame.K_d)
                elif cv_key == 27:
                    events_to_process.append(pygame.K_ESCAPE)

            for key in events_to_process:
                # Safe exit via ESC
                if key == pygame.K_ESCAPE:
                    logger.info("Exit requested via ESC key.")
                    running = False

                # Toggle Autonomous Navigation: Start / Pause / Resume
                elif key == pygame.K_SPACE:
                    env.toggle_autonomous_navigation()
                    logger.info("Autonomous mode toggled: state=%s", env.nav_mode.value)

                # Toggle Perception-to-Grid Mapping: M Key
                elif key == pygame.K_m:
                    is_enabled = env.toggle_perception_mapping()
                    logger.info("Perception mapping toggled: enabled=%s", is_enabled)
                    if is_enabled:
                        if not webcam.is_active():
                            webcam.start_preview()
                            logger.info("Auto-started camera feed for perception mapping.")
                        if not detector.is_loaded():
                            detector.load_model()
                    else:
                        mapper.clear()

                # Toggle YOLO Model weights (Pretrained yolo11n.pt <-> Custom models/custom_obstacle_detector.pt): T Key
                elif key == pygame.K_t:
                    custom_path = "models/custom_obstacle_detector.pt"
                    target_model = "yolo11n.pt" if detector.is_custom_model else custom_path
                    if target_model == custom_path and not os.path.exists(custom_path):
                        env.last_action_msg = f"Custom model '{custom_path}' not found. Train first with tools/train_custom_model.py"
                        logger.warning("Custom weights '%s' not found.", custom_path)
                    else:
                        switched = detector.switch_model(target_model)
                        if switched:
                            tracker.reset()
                            mapper.clear()
                            model_lbl = "Custom 3-Class" if detector.is_custom_model else "COCO 80-Class"
                            env.last_action_msg = f"YOLO model switched: {os.path.basename(detector.model_name)} ({model_lbl})"
                            logger.info("YOLO model switched: %s", detector.status_text)
                        else:
                            env.last_action_msg = f"Failed to switch YOLO model to {target_model}"

                # Dynamic obstacle insertion ahead of the vehicle
                elif key == pygame.K_o:
                    inserted = env.insert_dynamic_obstacle(ahead_steps=3)
                    if inserted:
                        logger.info("Dynamic hazard inserted at coordinate: %s", inserted)
                    else:
                        logger.warning("Could not insert dynamic hazard: no valid cell available.")

                # A* Path planning calculation
                elif key == pygame.K_p:
                    result = env.plan_path()
                    logger.info(
                        "A* planning triggered: success=%s, moves=%d, explored=%d, time=%.2f ms",
                        result.success,
                        result.path_length,
                        result.explored_nodes,
                        result.execution_time_ms
                    )

                # Manual vehicle navigation: W/A/S/D or Arrow keys (active only when autonomous mode is off)
                elif key in (pygame.K_w, pygame.K_UP):
                    env.move_vehicle(-1, 0)
                elif key in (pygame.K_s, pygame.K_DOWN):
                    env.move_vehicle(1, 0)
                elif key in (pygame.K_a, pygame.K_LEFT):
                    env.move_vehicle(0, -1)
                elif key in (pygame.K_d, pygame.K_RIGHT):
                    env.move_vehicle(0, 1)

                # Reset vehicle position, hazards, and clear route
                elif key == pygame.K_r:
                    env.reset_vehicle()
                    tracker.reset()
                    mapper.clear()
                    last_tracks = []
                    last_telemetry = {}
                    last_inf_ms = None
                    logger.info("Vehicle, dynamic hazards, perception cells, and route reset to initial state.")

                # Toggle OpenCV webcam preview
                elif key == pygame.K_c:
                    is_active = webcam.toggle_preview()
                    logger.info("Webcam toggled: active=%s (%s, state=%s)", is_active, webcam.device_name, webcam.get_state().value)

            # 3. Update Autonomous Navigation & Timing
            env.update(dt_ms)

            # 4. Update Perception Feed (Non-blocking) & Process Grid Mapping
            display_frame = None

            if webcam.is_active():
                raw_frame = webcam.get_current_frame()
                if raw_frame is not None:
                    perception_timer_ms += dt_ms
                    if perception_timer_ms >= perception_interval_ms:
                        perception_timer_ms = 0
                        last_tracks, last_telemetry = tracker.update(raw_frame)
                        last_inf_ms = last_telemetry.get("total_ms", 0.0)

                        if env.perception_enabled:
                            mapping_res = mapper.map_tracks(
                                tracks=last_tracks,
                                frame_shape=raw_frame.shape[:2],
                                current_time=time.time(),
                                vehicle_pos=env.vehicle_pos,
                                goal_pos=env.destination_pos
                            )
                            env.set_perception_obstacles(mapping_res.final_cells)
                            if len(mapping_res.final_cells) > 0:
                                mode_tag = "Homography" if mapping_res.mapping_mode.startswith("Planar") else "ROI"
                                env.last_action_msg = f"Perception [{mode_tag}]: {len(mapping_res.final_cells)} cell(s) mapped: {sorted(list(mapping_res.final_cells))[:2]}"
                                env.last_action_status = "success"
                            elif mapping_res.vehicle_conflicts:
                                env.last_action_msg = f"Perception: Candidate {list(mapping_res.vehicle_conflicts)[0]} overlaps vehicle (Protected by Safety Guard)."
                                env.last_action_status = "normal"

                    # Draw tracked obstacles, trajectory breadcrumbs, and calibration overlay
                    display_frame = tracker.draw_tracks(
                        frame=raw_frame,
                        tracks=last_tracks,
                        show_trails=True,
                        show_velocity=True,
                        telemetry=last_telemetry
                    )
                    display_frame = calibrator.draw_calibration_overlay(display_frame, show_fiducials=True)

            elif env.perception_enabled:
                # Camera inactive: prune any stale detections whose timeout elapsed
                active_cells = mapper.prune_stale_cells(time.time())
                if len(active_cells) != len(env.perception_obstacle_set):
                    env.set_perception_obstacles(active_cells)

            # Render frame in OpenCV window with annotated detections
            webcam.update(display_frame=display_frame)

            # 5. Render Graphics
            env.render(
                screen=screen,
                fonts=fonts,
                webcam_status=webcam.get_status(),
                model_status=detector.status_text,
                detection_count=tracker.active_track_count,
                inference_latency_ms=last_inf_ms
            )
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
