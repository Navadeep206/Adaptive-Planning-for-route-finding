"""Standalone live YOLO object detection verification utility using MacBook webcam.

Connects to the verified MacBook built-in FaceTime HD camera using the existing
perception.webcam configuration, runs local YOLO object detection, and renders
bounding boxes, class labels, confidence scores, and real-time inference latency.

Usage:
    python tools/test_detection_live.py
    python tools/test_detection_live.py --conf 0.4 --model yolo11n.pt
"""

import argparse
import logging
import os
import sys
import time
import cv2

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.webcam import WebcamManager, get_macbook_builtin_camera_index
from perception.object_detector import YOLOObjectDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DetectionLiveTest")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for live detection demo."""
    parser = argparse.ArgumentParser(
        description="Standalone live YOLO object detection test on MacBook built-in camera."
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
        default=0.35,
        help="Confidence detection threshold from 0.0 to 1.0 (default: 0.35)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image input dimension (default: 640)"
    )
    parser.add_argument(
        "--camera-index",
        "-c",
        type=int,
        default=None,
        help="Explicit camera device index (default: auto-detected MacBook built-in camera)"
    )
    return parser.parse_args()


def run_live_detection_demo(args: argparse.Namespace) -> None:
    """Run live object detection on frames from the MacBook webcam."""
    print("=" * 72)
    print("MACBOOK WEBCAM LOCAL YOLO DETECTION DEMO")
    print("=" * 72)
    print(f"YOLO Model:          {args.model}")
    print(f"Confidence Thresh:   {args.conf}")
    print(f"Inference Image Sz:  {args.imgsz}")
    print("Controls:            Press 'ESC' or 'q' to exit")
    print("Single Camera Rule:  Ensure main.py is closed so camera handle is free")
    print("=" * 72)

    # 1. Initialize YOLO Detector
    detector = YOLOObjectDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        auto_load=True
    )

    if not detector.is_loaded():
        logger.error(
            "Failed to load YOLO model. Please verify ultralytics is installed "
            "('pip install ultralytics') and model weights are accessible."
        )
        sys.exit(1)

    # 2. Initialize Webcam Manager using existing verified camera index
    camera_idx = args.camera_index if args.camera_index is not None else get_macbook_builtin_camera_index()
    logger.info("Connecting to camera device at verified index %d...", camera_idx)

    # We use cv2.VideoCapture directly here to be a dedicated, single-owner live process
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_idx, backend)

    if not cap.isOpened():
        logger.error(
            "Cannot open camera index %d! If another application (like main.py) is "
            "currently running and owning the camera, please close it first.",
            camera_idx
        )
        sys.exit(1)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    window_title = f"Live Perception: YOLO Detection ({args.model})"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    fps = 0.0
    last_time = time.perf_counter()

    logger.info("Camera opened successfully. Starting live object detection stream...")

    try:
        while True:
            ret, frame = cap.read()
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            if dt > 0:
                fps = (fps * 0.9) + ((1.0 / dt) * 0.1)

            # Perform local YOLO detection
            detections, inf_time_ms = detector.detect(frame)

            # Annotate frame with detections and HUD
            annotated_frame = detector.draw_detections(frame, detections, inf_time_ms)

            # Add FPS telemetry at top-right
            h, w = annotated_frame.shape[:2]
            fps_text = f"Stream FPS: {fps:.1f}"
            cv2.putText(
                annotated_frame,
                fps_text,
                (w - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(window_title, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                logger.info("Exit requested via keypress.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        logger.info("Releasing camera and closing windows...")
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Live detection demo completed cleanly.")


def main() -> None:
    """Entry point for standalone detection demo."""
    args = parse_arguments()
    run_live_detection_demo(args)


if __name__ == "__main__":
    main()
