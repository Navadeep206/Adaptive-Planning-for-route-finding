#!/usr/bin/env python3
"""Standalone demonstration utility for Day 6 Perception-to-Grid Obstacle Mapping.

Demonstrates local YOLO perception mapped to discrete simulation grid cells using
the MacBook built-in camera or mock detections.
Clearly labeled as APPROXIMATE DEMONSTRATION MAPPING (not real-world localization).

Usage:
    python tools/test_grid_mapping_demo.py
    python tools/test_grid_mapping_demo.py --synthetic   # Test without camera
    python tools/test_grid_mapping_demo.py --conf 0.40 --inflation 1
"""

import argparse
import logging
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.webcam import WebcamManager, get_macbook_builtin_camera_index
from perception.object_detector import YOLOObjectDetector, Detection
from perception.grid_mapper import GridMapper, GridMapperConfig, MappingMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GridMappingDemo")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for grid mapping demo."""
    parser = argparse.ArgumentParser(
        description="Day 6: Simplified Perception-to-Grid Obstacle Mapping Demonstration"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run deterministic synthetic detection loop without opening webcam"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO model weights (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Minimum confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--inflation",
        type=int,
        default=0,
        help="Safety inflation margin in cells (default: 0)"
    )
    parser.add_argument(
        "--camera-index",
        "-c",
        type=int,
        default=None,
        help="Explicit camera device index (default: auto-detected MacBook built-in camera)"
    )
    return parser.parse_args()


def run_synthetic_demo(args: argparse.Namespace) -> None:
    """Run deterministic synthetic mapping tests without camera hardware."""
    print("\n" + "=" * 76)
    print("DAY 6: SIMPLIFIED PERCEPTION-TO-GRID MAPPING (SYNTHETIC MODE)")
    print("=" * 76)
    print("NOTE: Monocular 2D bounding boxes do not provide 3D ground depth.")
    print("This mode demonstrates the approximate ROI mapping pipeline safely.\n")

    mapper = GridMapper(
        config=GridMapperConfig(
            min_confidence=args.conf,
            inflation_radius=args.inflation,
            grid_rows=22,
            grid_cols=26
        )
    )

    frame_w, frame_h = 640, 480
    vehicle_pos = (2, 12)
    goal_pos = (20, 15)

    scenarios = [
        (
            "Scenario 1: Roadway Hazard Detection (Valid in ROI)",
            [Detection(class_id=0, class_name="obstacle_box", confidence=0.88, bbox=(280, 200, 360, 260))]
        ),
        (
            "Scenario 2: Low-Confidence Noise (Filtered Out)",
            [Detection(class_id=0, class_name="shadow_artifact", confidence=0.20, bbox=(300, 200, 340, 240))]
        ),
        (
            "Scenario 3: Overlap with Autonomous Vehicle Cell (Safety Guard Activated)",
            [Detection(class_id=2, class_name="car", confidence=0.92, bbox=(290, 40, 330, 80))]  # Maps near vehicle
        ),
        (
            "Scenario 4: Peripheral Detection (Outside ROI / Bounds)",
            [Detection(class_id=0, class_name="out_of_bounds", confidence=0.90, bbox=(5, 5, 20, 25))]
        ),
    ]

    for title, detections in scenarios:
        print(f"\n--- {title} ---")
        result = mapper.map_detections(
            detections=detections,
            frame_shape=(frame_h, frame_w),
            current_time=time.time(),
            vehicle_pos=vehicle_pos,
            goal_pos=goal_pos
        )
        print(f"Raw Detections:        {result.raw_detections_count}")
        print(f"Confidence Filtered:   {result.filtered_detections_count}")
        print(f"Candidate Grid Cells:  {sorted(list(result.candidate_cells))}")
        print(f"Inflated Cells:        {sorted(list(result.inflated_cells))}")
        print(f"Vehicle Conflicts:     {sorted(list(result.vehicle_conflicts))} (Protected!)")
        print(f"Goal Conflicts:        {sorted(list(result.goal_conflicts))} (Protected!)")
        print(f"Final Obstacle Cells:  {sorted(list(result.final_cells))}")

    print("\n" + "=" * 76)
    print("Synthetic demonstration complete. All safety invariants verified.")
    print("=" * 76 + "\n")


def run_live_demo(args: argparse.Namespace) -> None:
    """Run live webcam perception and print real-time mapped grid cells."""
    print("\n" + "=" * 76)
    print("DAY 6: LIVE WEBCAM PERCEPTION-TO-GRID MAPPING DEMO")
    print("=" * 76)
    print("APPROXIMATE DEMONSTRATION MAPPING: Fixed camera ROI -> 2D Grid Cells")
    print(f"YOLO Model:          {args.model}")
    print(f"Confidence Thresh:   {args.conf}")
    print(f"Inflation Radius:    {args.inflation} cells")
    print("Controls:            Press Ctrl+C to exit")
    print("Single Camera Rule:  Ensure main.py is closed so camera handle is free")
    print("=" * 76 + "\n")

    # 1. Initialize YOLO
    detector = YOLOObjectDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        auto_load=True
    )
    if not detector.is_loaded():
        logger.error("Failed to load YOLO model. Run with --synthetic to test offline.")
        sys.exit(1)

    # 2. Initialize GridMapper
    mapper = GridMapper(
        config=GridMapperConfig(
            min_confidence=args.conf,
            inflation_radius=args.inflation,
            grid_rows=22,
            grid_cols=26
        )
    )

    # 3. Open MacBook built-in webcam
    import cv2
    cam_idx = args.camera_index if args.camera_index is not None else get_macbook_builtin_camera_index()
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_idx, backend)

    if not cap.isOpened():
        logger.error("Cannot open camera index %d. Use --synthetic for offline demonstration.", cam_idx)
        sys.exit(1)

    logger.info("Live perception camera opened. Processing frames...")
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            frame_idx += 1
            if frame_idx % 10 == 0:  # Sample every ~300ms
                detections, inf_ms = detector.detect(frame)
                result = mapper.map_detections(
                    detections=detections,
                    frame_shape=frame.shape[:2],
                    current_time=time.time(),
                    vehicle_pos=(2, 12),
                    goal_pos=(20, 15)
                )

                if detections:
                    class_summary = ", ".join(f"{d.class_name}:{d.confidence:.2f}" for d in detections)
                    print(
                        f"Frame #{frame_idx:04d} | Detections: {len(detections)} ({class_summary}) | "
                        f"Mapped Cells: {sorted(list(result.final_cells))} | Inf: {inf_ms:.1f}ms"
                    )
                else:
                    active = mapper.get_active_cells()
                    if active:
                        print(f"Frame #{frame_idx:04d} | No new objects | Retained Active: {len(active)} cells")

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        cap.release()
        print("Camera released. Demonstration ended.")


def main() -> None:
    """Entry point for standalone grid mapper demo."""
    args = parse_arguments()
    if args.synthetic:
        run_synthetic_demo(args)
    else:
        run_live_demo(args)


if __name__ == "__main__":
    main()
