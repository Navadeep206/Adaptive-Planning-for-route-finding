"""Standalone manual test utility for verifying reliable live capture on the MacBook webcam.

Connects strictly to the configured MacBook built-in camera (default index 0, or explicit index).
Never cycles through random indices or automatically connects to an iPhone Continuity Camera.
Renders a live window with on-screen HUD (FPS, frame count, resolution, device index).
Exits cleanly when ESC or 'q' is pressed.

Usage:
    python tools/test_webcam_live.py
    python tools/test_webcam_live.py --camera-index 0
"""

import argparse
import logging
import os
import sys
import time
import cv2

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.webcam import get_macbook_builtin_camera_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WebcamLiveTest")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the live test utility."""
    parser = argparse.ArgumentParser(
        description="Standalone manual live test for MacBook built-in camera."
    )
    parser.add_argument(
        "--camera-index",
        "-c",
        type=int,
        default=None,
        help="Explicit camera device index (default: auto-detected MacBook camera, bypassing iPhone)"
    )
    return parser.parse_args()


def run_live_test(camera_index: int) -> None:
    """Open the camera at camera_index and stream frames with an on-screen HUD."""
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    backend_name = "AVFOUNDATION" if backend == cv2.CAP_AVFOUNDATION else "DEFAULT"

    print("=" * 70)
    print("MACBOOK WEBCAM LIVE VERIFICATION UTILITY")
    print("=" * 70)
    print(f"Target Camera Index: {camera_index}")
    print(f"macOS Backend:       {backend_name}")
    print("Controls:            Press 'ESC' or 'q' to exit cleanly")
    print("=" * 70)

    logger.info("Opening VideoCapture on index %d with %s...", camera_index, backend_name)
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        logger.error(
            "FAILED to open camera at index %d! Please verify permissions under macOS "
            "System Settings -> Privacy & Security -> Camera.",
            camera_index
        )
        sys.exit(1)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    window_name = f"MacBook Camera Live Verification (Index {camera_index})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    fps = 0.0
    last_time = time.perf_counter()
    warmup_attempts = 0
    first_frame_captured = False

    logger.info("Camera handle opened successfully. Beginning continuous capture loop...")

    try:
        while True:
            ret, frame = cap.read()
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            if not ret or frame is None:
                if not first_frame_captured:
                    warmup_attempts += 1
                    if warmup_attempts > 30:
                        logger.error("Sensor warmup timeout: no frames received after 3 seconds.")
                        break
                    time.sleep(0.1)
                    continue
                else:
                    logger.warning("Frame read returned False / None.")
                    time.sleep(0.02)
                    continue

            # First valid frame diagnostics
            if not first_frame_captured:
                h, w = frame.shape[:2]
                first_frame_captured = True
                logger.info("VERIFIED: First live frame received! Resolution: %dx%d", w, h)

            frame_count += 1
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = (fps * 0.9) + (instant_fps * 0.1)

            # Draw on-screen telemetry overlay
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, 10), (420, 120), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, 10), (420, 120), (0, 255, 128), 1)

            cv2.putText(
                frame,
                f"MacBook Built-in Camera (Index {camera_index})",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                f"Resolution: {w} x {h}  |  FPS: {fps:.1f}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                f"Frames Captured: {frame_count}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 255, 180),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                "Press 'ESC' or 'q' to exit",
                (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (160, 160, 160),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                logger.info("Exit requested via keypress (code %d).", key)
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")

    finally:
        logger.info("Cleaning up camera resources...")
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Live test completed successfully. Total frames captured: %d", frame_count)


def main() -> None:
    """Entry point for the live webcam test tool."""
    args = parse_arguments()
    if args.camera_index is not None:
        idx = args.camera_index
    else:
        idx = get_macbook_builtin_camera_index()
    run_live_test(idx)


if __name__ == "__main__":
    main()
