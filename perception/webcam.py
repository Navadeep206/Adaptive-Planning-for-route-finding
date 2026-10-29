"""Webcam capture and preview manager using OpenCV.

Exclusively targets and connects to the MacBook's built-in FaceTime HD Camera.
Streams continuously without timing out or closing when focus shifts to Pygame.
"""

from typing import Optional, Tuple
import json
import logging
import subprocess
import sys
import threading
import time
import cv2

logger = logging.getLogger(__name__)


def get_macbook_builtin_camera_index() -> int:
    """Determine the device index for the MacBook's built-in camera.

    Inspects macOS system profiler metadata without opening any OpenCV handles,
    ensuring the iPhone Continuity Camera is never triggered or connected.

    Returns:
        int: Camera device index for the MacBook built-in camera.
    """
    if sys.platform != "darwin":
        return 0

    try:
        raw = subprocess.check_output(
            ["system_profiler", "-json", "SPCameraDataType"],
            text=True,
            timeout=1.0
        )
        data = json.loads(raw)
        cams = data.get("SPCameraDataType", [])

        # Check if an iPhone / Continuity Camera is recognized by macOS
        has_iphone = any(
            "iphone" in c.get("spcamera_model-id", "").lower() or "iphone" in c.get("_name", "").lower()
            for c in cams
        )

        # On macOS AVFoundation, if an iPhone is present, AVFoundation assigns Index 0 to the iPhone
        # and Index 1 to the MacBook's physical built-in FaceTime HD Camera.
        # If no iPhone is present, the MacBook camera is at Index 0.
        if has_iphone:
            logger.info("macOS has an iPhone paired. Selecting MacBook built-in FaceTime HD camera at index 1.")
            return 1
        return 0

    except Exception as exc:
        logger.debug("Error checking system_profiler camera list: %s", exc)
        return 0


class WebcamManager:
    """Manages OpenCV capture for the MacBook's built-in webcam with non-blocking updates."""

    def __init__(self, device_index: Optional[int] = None, window_title: str = "Sensor Feed: MacBook Built-in Camera"):
        """Initialize the webcam manager.

        Args:
            device_index: Explicit index, or None to automatically select the MacBook built-in camera.
            window_title: Window title for the OpenCV preview window.
        """
        if device_index is not None:
            self.device_index: int = device_index
        else:
            self.device_index: int = get_macbook_builtin_camera_index()

        self.window_title: str = window_title
        self.device_name: str = "MacBook Built-in Camera"
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_preview_active: bool = False
        self.status_message: str = "Standby (Press 'C' to toggle)"

        # Threading and synchronization for non-blocking frame capture
        self._capture_thread: Optional[threading.Thread] = None
        self._thread_running: bool = False
        self._frame_lock: threading.Lock = threading.Lock()
        self._latest_frame: Optional[object] = None
        self._new_frame_available: bool = False
        self._stream_error: Optional[str] = None

        # Window state
        self._window_created: bool = False

    def is_active(self) -> bool:
        """Check if the live camera preview is currently streaming."""
        return self.is_preview_active and self.cap is not None and self.cap.isOpened()

    def get_status(self) -> str:
        """Return user-facing status string for the HUD."""
        return self.status_message

    def _get_backend(self) -> int:
        """Select AVFoundation on macOS, default backend on others."""
        if sys.platform == "darwin":
            return cv2.CAP_AVFOUNDATION
        return cv2.CAP_ANY

    def start_preview(self) -> bool:
        """Open the MacBook built-in camera and begin continuous preview.

        Returns:
            bool: True if camera successfully initialized, False otherwise.
        """
        if self.is_preview_active or self.cap is not None:
            logger.debug("Camera preview is already active; ignoring duplicate start request.")
            return True

        self.status_message = "Initializing MacBook Camera..."
        try:
            backend = self._get_backend()
            logger.info("Opening %s (index %d) with backend %s",
                        self.device_name, self.device_index,
                        "AVFOUNDATION" if backend == cv2.CAP_AVFOUNDATION else "DEFAULT")

            self.cap = cv2.VideoCapture(self.device_index, backend)

            if not self.cap.isOpened():
                self.status_message = "Unavailable (Check connection/permissions)"
                logger.warning("Failed to open MacBook camera at index %d.", self.device_index)
                self.release()
                return False

            # Configure capture buffer size to reduce latency
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Reset synchronization flags
            self._stream_error = None
            self._latest_frame = None
            self._new_frame_available = False
            self._thread_running = True

            # Launch background worker for non-blocking frame capture
            self._capture_thread = threading.Thread(
                target=self._capture_worker,
                name="MacBookWebcamWorker",
                daemon=True
            )
            self._capture_thread.start()

            self.is_preview_active = True
            self.status_message = "Active (MacBook Built-in Camera)"
            return True

        except Exception as exc:
            self.status_message = f"Error: {exc.__class__.__name__}"
            logger.error("Unexpected error opening MacBook camera: %s", exc)
            self.release()
            return False

    def _capture_worker(self) -> None:
        """Background thread worker to grab frames from the MacBook camera at stable intervals."""
        logger.debug("MacBook webcam capture worker started.")
        consecutive_errors = 0

        while self._thread_running and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    consecutive_errors = 0
                    with self._frame_lock:
                        self._latest_frame = frame
                        self._new_frame_available = True
                else:
                    consecutive_errors += 1
                    # Only flag disconnection after 5 full seconds of continuous hardware failure
                    if consecutive_errors > 150:
                        logger.warning("Sustained frame read failure (>5s) on MacBook camera.")
                        with self._frame_lock:
                            self._stream_error = "Disconnected (Frame read error)"
                        break

                # Sleep 15ms (~60 FPS) to pace the capture thread and avoid exhausting the AVFoundation queue
                time.sleep(0.015)

            except Exception as exc:
                logger.error("Exception in capture worker thread: %s", exc)
                time.sleep(0.05)

        logger.debug("MacBook webcam capture worker exiting.")

    def stop_preview(self) -> None:
        """Stop preview stream, release camera hardware handle, and destroy window."""
        self.is_preview_active = False
        self.status_message = "Standby (Press 'C' to toggle)"
        self.release()

    def toggle_preview(self) -> bool:
        """Toggle preview on or off.

        Returns:
            bool: True if now active, False if stopped or failed.
        """
        if self.is_preview_active or self.cap is not None:
            self.stop_preview()
            return False
        return self.start_preview()

    def update(self) -> Tuple[bool, Optional[object]]:
        """Fetch latest frame and render OpenCV preview window if active.

        Must be called on the main thread for Cocoa GUI compatibility.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (frame_ready, frame)
        """
        if not self.is_preview_active:
            return False, None

        # Check for stream errors reported by worker
        with self._frame_lock:
            if self._stream_error:
                err_msg = self._stream_error
                self.stop_preview()
                self.status_message = err_msg
                return False, None

            frame_to_display = None
            if self._new_frame_available and self._latest_frame is not None:
                frame_to_display = self._latest_frame
                self._new_frame_available = False

        # Render frame in OpenCV window
        if frame_to_display is not None:
            try:
                cv2.imshow(self.window_title, frame_to_display)
                self._window_created = True
            except Exception as exc:
                logger.error("Error during cv2.imshow: %s", exc)
                self.stop_preview()
                return False, None

        # Process Cocoa window events (1ms delay)
        # Does NOT automatically close when window loses focus to Pygame!
        if self._window_created:
            key = cv2.waitKey(1) & 0xFF
            # Allow closing preview via 'Q' key inside the OpenCV window
            if key in (ord('q'), ord('Q')):
                self.stop_preview()
                return False, None

        return (frame_to_display is not None), frame_to_display

    def release(self) -> None:
        """Release capture device and clean up all OpenCV windows safely."""
        self.is_preview_active = False
        self._thread_running = False

        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=0.25)
        self._capture_thread = None

        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("MacBook camera device %d released.", self.device_index)
            except Exception as exc:
                logger.debug("Error releasing VideoCapture: %s", exc)
            self.cap = None

        if self._window_created:
            try:
                cv2.destroyWindow(self.window_title)
            except Exception:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            self._window_created = False
