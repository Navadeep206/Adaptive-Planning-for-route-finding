#!/usr/bin/env python3
"""Standalone live YOLO object detection, tuning, and diagnostic verification utility.

Connects to the verified MacBook built-in FaceTime HD camera using the existing
perception.webcam configuration, runs local YOLO object detection, and renders
bounding boxes, class labels, confidence percentages, real-time inference latency,
processing throughput FPS, and comprehensive camera diagnostic telemetry.

Interactive Tuning Controls (while preview window is focused):
    [ / - : Decrease confidence threshold by 0.05
    ] / + : Increase confidence threshold by 0.05
    < / , : Decrease IoU (NMS) threshold by 0.05
    > / . : Increase IoU (NMS) threshold by 0.05
    p / P : Cycle preprocessing mode (None -> CLAHE -> Gamma -> None)
    v / V : Toggle view mode (Single Window <-> Side-by-Side Comparison)
    s / S : Toggle inference image size (640 <-> 800)
    d / D : Toggle camera diagnostic telemetry overlay
    m / M : Toggle inference device (CPU <-> MPS)
    t / T : Toggle model (yolo11n.pt <-> models/custom_obstacle_detector.pt)
    k / K : Toggle ByteTrack multi-object tracking (ON <-> OFF)
    y / Y : Toggle trajectory trails (ON <-> OFF)
    c / C : Capture diagnostic snapshot to assets/test_images/
    q/ESC : Clean shutdown

Usage:
    python tools/test_detection_live.py
    python tools/test_detection_live.py --conf 0.25 --iou 0.45 --imgsz 640 --preprocess none
    python tools/test_detection_live.py --track --trails
"""

import argparse
from collections import deque
from datetime import datetime
import logging
import os
import sys
import time
import cv2
import numpy as np
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.webcam import WebcamManager, get_macbook_builtin_camera_index
from perception.object_detector import YOLOObjectDetector
from perception.object_tracker import ObjectTracker, TrackerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DetectionLiveTest")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for live detection, tuning, and diagnostics."""
    parser = argparse.ArgumentParser(
        description="Standalone live YOLO object detection and tuning tool on MacBook camera."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO model identifier or weights path (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence detection threshold from 0.05 to 0.95 (default: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Non-Maximum Suppression (NMS) IoU overlap threshold (default: 0.45)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image input dimension: 640 or 800 (default: 640)"
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=50,
        help="Maximum detections per frame for autonomous robotics scene (default: 50)"
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="none",
        choices=["none", "clahe", "gamma"],
        help="Preprocessing contrast enhancement: 'none', 'clahe', or 'gamma' (default: 'none')"
    )
    parser.add_argument(
        "--view-mode",
        type=str,
        default="single",
        choices=["single", "side_by_side"],
        help="Display view mode: 'single' or 'side_by_side' comparison (default: 'single')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cpu', 'mps', or None for auto (default: None)"
    )
    parser.add_argument(
        "--camera-index",
        "-c",
        type=int,
        default=None,
        help="Explicit camera device index (default: auto-detected MacBook built-in camera)"
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        default=True,
        help="Enable camera diagnostic telemetry overlay (default: True)"
    )
    parser.add_argument(
        "--no-diagnostic",
        action="store_false",
        dest="diagnostic",
        help="Disable camera diagnostic telemetry overlay"
    )
    parser.add_argument(
        "--track",
        action="store_true",
        default=True,
        help="Enable ByteTrack multi-object tracking (default: True)"
    )
    parser.add_argument(
        "--no-track",
        action="store_false",
        dest="track",
        help="Disable multi-object tracking (raw detection only)"
    )
    parser.add_argument(
        "--trails",
        action="store_true",
        default=True,
        help="Render trajectory trails for moving objects (default: True)"
    )
    parser.add_argument(
        "--no-trails",
        action="store_false",
        dest="trails",
        help="Disable trajectory trails"
    )
    return parser.parse_args()


def run_live_detection_demo(args: argparse.Namespace) -> None:
    """Run live object detection on frames from the MacBook webcam with interactive tuning."""
    mps_available = torch.backends.mps.is_available()
    selected_device = args.device

    print("=" * 76)
    print("MACBOOK WEBCAM LOCAL YOLO DETECTION & TUNING STUDIO (PHASE 2)")
    print("=" * 76)
    print(f"YOLO Model:          {args.model}")
    print(f"Confidence Thresh:   {args.conf:.2f}")
    print(f"IoU (NMS) Thresh:    {args.iou:.2f}")
    print(f"Inference Image Sz:  {args.imgsz}x{args.imgsz}")
    print(f"Max Detections:      {args.max_det}")
    print(f"Preprocessing:       {args.preprocess.upper()}")
    print(f"View Mode:           {args.view_mode.upper()}")
    print(f"Inference Device:    {selected_device or 'auto'} (Apple Silicon MPS: {mps_available})")
    print(f"Tracking (ByteTrack):{'ACTIVE' if args.track else 'OFF'}")
    print(f"Trajectory Trails:   {'ACTIVE' if args.trails else 'OFF'}")
    print(f"Diagnostic Mode:     {'ACTIVE' if args.diagnostic else 'OFF'}")
    print("-" * 76)
    print("INTERACTIVE TUNING HOTKEYS (Change parameters without reloading model):")
    print("  '[' or '-' : Decrease confidence threshold (-0.05)")
    print("  ']' or '+' : Increase confidence threshold (+0.05)")
    print("  '<' or ',' : Decrease IoU (NMS) threshold (-0.05)")
    print("  '>' or '.' : Increase IoU (NMS) threshold (+0.05)")
    print("  'p' or 'P' : Cycle preprocessing mode (None -> CLAHE -> Gamma -> None)")
    print("  'v' or 'V' : Toggle view mode (Single Window <-> Side-by-Side)")
    print("  's' or 'S' : Toggle image size (640 <-> 800)")
    print("  'd' or 'D' : Toggle image quality diagnostic telemetry overlay")
    print("  'm' or 'M' : Toggle device (CPU <-> MPS)")
    print("  't' or 'T' : Toggle model (yolo11n.pt <-> models/custom_obstacle_detector.pt)")
    print("  'k' or 'K' : Toggle ByteTrack multi-object tracking (ON <-> OFF)")
    print("  'y' or 'Y' : Toggle trajectory trails (ON <-> OFF)")
    print("  'c' or 'C' : Save snapshot frame to assets/test_images/")
    print("  'q' or ESC : Quit cleanly")
    print("=" * 76)

    # 1. Initialize YOLO Detector with Phase 2 parameters
    detector = YOLOObjectDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        preprocess_mode=args.preprocess,
        device=selected_device,
        auto_load=True
    )

    if not detector.is_loaded():
        logger.error(
            "Failed to load YOLO model. Please verify ultralytics is installed "
            "('pip install ultralytics') and model weights are accessible."
        )
        sys.exit(1)

    # 2. Initialize Real-Time Multi-Object Tracker
    tracker = ObjectTracker(
        detector=detector,
        config=TrackerConfig(
            min_confidence=args.conf,
            track_activation_threshold=2,
            lost_timeout_sec=1.0,
            max_lost_frames=15,
            history_len=30,
            enable_box_smoothing=True,
            smoothing_alpha=0.70
        )
    )
    tracking_active = args.track
    trails_active = args.trails

    # 3. Configure window dimensions: 960x540 for single view, 1280x400 for side-by-side
    initial_window_size = (1280, 420) if args.view_mode == "side_by_side" else (960, 540)
    window_title = f"Perception Tuning Studio: YOLO Detection ({args.model})"
    webcam = WebcamManager(
        device_index=args.camera_index,
        window_title=window_title,
        window_size=initial_window_size
    )

    logger.info("Connecting to %s at index %d...", webcam.device_name, webcam.device_index)
    if not webcam.start_preview():
        logger.error(
            "Cannot open %s (Index %d)! If another application (like main.py) is "
            "currently running and owning the camera, please close it first.",
            webcam.device_name,
            webcam.device_index
        )
        sys.exit(1)

    logger.info("Camera opened successfully. Starting live object detection stream...")

    last_detections = []
    last_inf_ms = None
    frame_count = 0
    consecutive_empty = 0
    diagnostic_active = args.diagnostic
    view_mode = args.view_mode

    # Rolling window for true loop throughput FPS
    loop_times = deque(maxlen=20)
    last_loop_time = time.perf_counter()

    try:
        while webcam.is_active():
            now = time.perf_counter()
            dt = now - last_loop_time
            last_loop_time = now
            if dt > 0:
                loop_times.append(dt)
            loop_fps = len(loop_times) / sum(loop_times) if loop_times else 0.0

            raw_frame = webcam.get_current_frame()
            display_frame = None

            if raw_frame is not None:
                frame_count += 1

                # Frame processing: Object Tracking (ByteTrack) or Raw Detection
                if tracking_active:
                    tracks, telemetry = tracker.update(raw_frame)
                    annotated_frame = tracker.draw_tracks(
                        frame=raw_frame,
                        tracks=tracks,
                        show_trails=trails_active,
                        show_velocity=True,
                        telemetry=telemetry,
                        fps=loop_fps
                    )
                else:
                    last_detections, last_inf_ms = detector.detect(raw_frame)
                    annotated_frame = detector.draw_detections(
                        frame=raw_frame,
                        detections=last_detections,
                        inference_time_ms=last_inf_ms,
                        fps=loop_fps,
                        diagnostic_mode=diagnostic_active
                    )

                # Assemble display frame according to view mode
                if view_mode == "side_by_side":
                    # Resize both to 640x360 for crisp side-by-side display
                    orig_thumb = cv2.resize(raw_frame, (640, 360), interpolation=cv2.INTER_AREA)
                    annotated_thumb = cv2.resize(annotated_frame, (640, 360), interpolation=cv2.INTER_AREA)

                    # Add label header to original preview
                    cv2.putText(
                        orig_thumb,
                        "ORIGINAL CAMERA FEED (Raw Sensor)",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 200, 255),
                        1,
                        cv2.LINE_AA
                    )

                    # Add label header to processed preview
                    prep_title = f"PROCESSED FEED ({detector.preprocess_mode.upper()})"
                    cv2.putText(
                        annotated_thumb,
                        prep_title,
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 160),
                        1,
                        cv2.LINE_AA
                    )

                    display_frame = np.hstack([orig_thumb, annotated_thumb])
                else:
                    display_frame = annotated_frame

            # Update preview window and poll GUI key events
            webcam.update(display_frame=display_frame)

            # Process interactive hotkeys from the OpenCV window
            key = webcam.pop_key()
            if key is not None:
                if key in (ord('q'), ord('Q'), 27):
                    logger.info("Exit requested via keypress.")
                    break

                # Adjust confidence threshold
                elif key in (ord('['), ord('-')):
                    detector.set_thresholds(conf=round(detector.conf_threshold - 0.05, 2))
                    tracker.config.min_confidence = detector.conf_threshold
                    logger.info("Confidence threshold adjusted: %.2f", detector.conf_threshold)

                elif key in (ord(']'), ord('+'), ord('=')):
                    detector.set_thresholds(conf=round(detector.conf_threshold + 0.05, 2))
                    tracker.config.min_confidence = detector.conf_threshold
                    logger.info("Confidence threshold adjusted: %.2f", detector.conf_threshold)

                # Adjust IoU threshold
                elif key in (ord('<'), ord(',')):
                    detector.set_thresholds(iou=round(detector.iou_threshold - 0.05, 2))
                    logger.info("IoU (NMS) threshold adjusted: %.2f", detector.iou_threshold)

                elif key in (ord('>'), ord('.')):
                    detector.set_thresholds(iou=round(detector.iou_threshold + 0.05, 2))
                    logger.info("IoU (NMS) threshold adjusted: %.2f", detector.iou_threshold)

                # Cycle preprocessing mode (None -> CLAHE -> Gamma -> None)
                elif key in (ord('p'), ord('P')):
                    modes = ["none", "clahe", "gamma"]
                    cur_idx = modes.index(detector.preprocess_mode) if detector.preprocess_mode in modes else 0
                    next_mode = modes[(cur_idx + 1) % len(modes)]
                    detector.set_preprocessing(next_mode)
                    logger.info("Preprocessing mode toggled to: '%s'", next_mode.upper())

                # Toggle view mode (single <-> side_by_side)
                elif key in (ord('v'), ord('V')):
                    view_mode = "side_by_side" if view_mode == "single" else "single"
                    new_sz = (1280, 420) if view_mode == "side_by_side" else (960, 540)
                    cv2.resizeWindow(window_title, new_sz[0], new_sz[1])
                    logger.info("View mode toggled to: '%s' (Window resized to %dx%d)", view_mode.upper(), new_sz[0], new_sz[1])

                # Toggle image size (640 <-> 800)
                elif key in (ord('s'), ord('S')):
                    detector.imgsz = 800 if detector.imgsz == 640 else 640
                    logger.info("Inference image size toggled to: %dx%d", detector.imgsz, detector.imgsz)

                # Toggle diagnostic telemetry overlay
                elif key in (ord('d'), ord('D')):
                    diagnostic_active = not diagnostic_active
                    logger.info("Diagnostic telemetry overlay toggled: %s", "ACTIVE" if diagnostic_active else "OFF")

                # Toggle device (CPU <-> MPS)
                elif key in (ord('m'), ord('M')) and mps_available:
                    cur_dev = detector.device or "cpu"
                    new_dev = "mps" if cur_dev == "cpu" else "cpu"
                    detector.device = new_dev
                    logger.info("Inference device switched to: %s", new_dev)

                # Toggle model weights (yolo11n.pt <-> models/custom_obstacle_detector.pt)
                elif key in (ord('t'), ord('T')):
                    custom_path = "models/custom_obstacle_detector.pt"
                    new_target = "yolo11n.pt" if detector.is_custom_model else custom_path
                    if new_target == custom_path and not os.path.exists(custom_path):
                        logger.warning("Custom weights '%s' not found. Train first with tools/train_custom_model.py", custom_path)
                    else:
                        switched = detector.switch_model(new_target)
                        if switched:
                            tracker.reset()
                            logger.info("Active model switched to: %s (%s)", detector.model_name, detector.status_text)
                        else:
                            logger.error("Failed to switch model to: %s", new_target)

                # Toggle multi-object tracking (ByteTrack ON <-> OFF)
                elif key in (ord('k'), ord('K')):
                    tracking_active = not tracking_active
                    logger.info("Multi-object tracking toggled: %s", "ACTIVE (ByteTrack)" if tracking_active else "OFF (Raw Detection)")

                # Toggle trajectory trails (ON <-> OFF)
                elif key in (ord('y'), ord('Y')):
                    trails_active = not trails_active
                    logger.info("Trajectory trails toggled: %s", "ACTIVE" if trails_active else "OFF")

                # Capture diagnostic snapshot
                elif key in (ord('c'), ord('C')) and raw_frame is not None:
                    os.makedirs("assets/test_images", exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_path = f"assets/test_images/live_capture_{ts}.jpg"
                    cv2.imwrite(snap_path, raw_frame)
                    logger.info("Captured diagnostic snapshot saved to: %s", snap_path)

            time.sleep(0.002)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        logger.info("Releasing camera and closing windows...")
        webcam.release()
        logger.info("Live detection and tuning utility completed cleanly.")


def main() -> None:
    """Entry point for standalone detection demo."""
    args = parse_arguments()
    run_live_detection_demo(args)


if __name__ == "__main__":
    main()
