"""Webcam capture and preview manager using OpenCV and macOS AVFoundation.

Exclusively targets and connects to the MacBook's verified built-in FaceTime HD Camera.
Streams continuously without timing out or closing when focus shifts to Pygame.
Provides clear diagnostics, bounded thread-safe frame access, and robust error recovery.
"""

from enum import Enum
from typing import Optional, Tuple
import json
import logging
import os
import subprocess
import sys
import threading
import time
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraState(str, Enum):
    """Enumeration of camera operational states."""
    OFF = "OFF"
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    ERROR = "ERROR"


def get_macbook_builtin_camera_index() -> int:
    """Determine the device index for the MacBook's built-in FaceTime HD camera.

    On macOS with OpenCV AVFoundation:
    - When a paired iPhone (Continuity Camera) is present on the system, macOS AVFoundation
      assigns device index 0 to the iPhone (as the default virtual camera) and device index 1
      to the physical MacBook built-in FaceTime HD camera.
    - When no iPhone is paired, device index 0 is the physical MacBook built-in camera.

    This function inspects system_profiler SPCameraDataType to detect whether an iPhone
    is recognized and selects the MacBook built-in camera index, guaranteeing that the
    MacBook camera is chosen as the primary device and the iPhone is NEVER connected.

    Returns:
        int: Camera device index for the MacBook built-in camera (1 if iPhone paired, 0 otherwise).
    """
    # 1. Explicit environment override
    env_index = os.environ.get("MACBOOK_CAMERA_INDEX")
    if env_index is not None:
        try:
            parsed = int(env_index.strip())
            logger.info("Using explicit camera index %d from MACBOOK_CAMERA_INDEX environment variable.", parsed)
            return parsed
        except ValueError:
            logger.warning("Invalid MACBOOK_CAMERA_INDEX value '%s'; falling back to discovery.", env_index)

    if sys.platform != "darwin":
        return 0

    # 2. Query macOS system profiler for camera hardware order
    try:
        raw = subprocess.check_output(
            ["system_profiler", "-json", "SPCameraDataType"],
            text=True,
            timeout=2.0
        )
        data = json.loads(raw)
        cams = data.get("SPCameraDataType", [])

        # Check if an iPhone / Continuity Camera is recognized by macOS
        has_iphone = any(
            "iphone" in (c.get("spcamera_model-id") or "").lower()
            or "iphone" in (c.get("_name") or "").lower()
            for c in cams
        )

        if has_iphone:
            logger.info(
                "macOS has a paired iPhone (Continuity Camera). AVFoundation assigns index 0 to iPhone. "
                "Exclusively selecting MacBook built-in FaceTime HD camera at index 1 to bypass iPhone."
            )
            return 1

        logger.info("No paired iPhone detected. Selecting MacBook built-in camera at index 0.")
        return 0

    except Exception as exc:
        logger.debug("Could not query system_profiler for cameras (%s); defaulting to index 1.", exc)
        return 1


class WebcamManager:
    """Manages OpenCV capture for the MacBook's built-in webcam with non-blocking updates."""

    def __init__(
        self,
        device_index: Optional[int] = None,
        window_title: str = "Sensor Feed: MacBook Built-in Camera",
        window_pos: Optional[Tuple[int, int]] = (1020, 40),
        window_size: Tuple[int, int] = (400, 300)
    ):
        """Initialize the webcam manager.

        Args:
            device_index: Explicit device index, or None to automatically select
                          the MacBook built-in camera.
            window_title: Window title for the OpenCV preview window.
            window_pos: Optional screen (x, y) coordinates for side-by-side positioning.
            window_size: Optional (width, height) in pixels for the preview window.
        """
        if device_index is not None:
            self.device_index: int = device_index
            self.device_name: str = f"Configured Camera (Index {device_index})"
        else:
            self.device_index = get_macbook_builtin_camera_index()
            self.device_name = "MacBook Built-in Camera"

        self.window_title: str = window_title
        self.window_pos: Optional[Tuple[int, int]] = window_pos
        self.window_size: Tuple[int, int] = window_size
        self.cap: Optional[cv2.VideoCapture] = None
        self.state: CameraState = CameraState.OFF
        self.is_preview_active: bool = False
        self.status_message: str = "Standby (Press 'C' to toggle)"

        # Performance & Telemetry
        self.frame_count: int = 0
        self.fps: float = 0.0
        self.resolution: Tuple[int, int] = (0, 0)
        self._first_frame_captured: bool = False

        # Threading and synchronization for non-blocking frame capture
        self._capture_thread: Optional[threading.Thread] = None
        self._thread_running: bool = False
        self._frame_lock: threading.Lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._new_frame_available: bool = False
        self._stream_error: Optional[str] = None

        # GUI Window state
        self._window_created: bool = False
        self._move_attempts: int = 0
        self.last_key: Optional[int] = None

    def pop_key(self) -> Optional[int]:
        """Retrieve and reset the last key pressed within the OpenCV window."""
        k = self.last_key
        self.last_key = None
        return k

    def is_active(self) -> bool:
        """Check if the live camera preview is currently streaming and active."""
        return (
            self.is_preview_active
            and self.state == CameraState.ACTIVE
            and self.cap is not None
            and self.cap.isOpened()
        )

    def get_status(self) -> str:
        """Return user-facing status string for the HUD."""
        return self.status_message

    def get_state(self) -> CameraState:
        """Return the current CameraState enum."""
        return self.state

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest captured BGR frame in a thread-safe manner."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def get_current_frame_rgb(self) -> Optional[np.ndarray]:
        """Return the latest frame converted to RGB color format for display/processing."""
        frame = self.get_current_frame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def _get_backend(self) -> int:
        """Select AVFoundation on macOS, default backend on others."""
        if sys.platform == "darwin":
            return cv2.CAP_AVFOUNDATION
        return cv2.CAP_ANY

    def start_preview(self) -> bool:
        """Open the MacBook built-in camera and begin continuous live capture.

        Returns:
            bool: True if camera successfully initialized, False otherwise.
        """
        # Idempotent: avoid opening duplicate camera handles
        if self.is_preview_active or self.cap is not None:
            logger.debug("Camera preview is already active or initializing; ignoring duplicate start request.")
            return True

        backend = self._get_backend()
        backend_name = "AVFOUNDATION" if backend == cv2.CAP_AVFOUNDATION else "DEFAULT"
        logger.info(
            "Camera startup requested: device='%s', index=%d, backend=%s",
            self.device_name,
            self.device_index,
            backend_name
        )

        self.state = CameraState.STARTING
        self.status_message = "STARTING (Connecting to camera...)"

        try:
            self.cap = cv2.VideoCapture(self.device_index, backend)

            if not self.cap.isOpened():
                self.state = CameraState.ERROR
                self.status_message = "Unavailable (Check connection/permissions)"
                logger.warning(
                    "Camera initialization failure: failed to open %s at index %d via backend %s.",
                    self.device_name,
                    self.device_index,
                    backend_name
                )
                self.release()
                return False

            # Configure capture buffer size to 1 to eliminate frame latency
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            logger.info("Camera hardware handle successfully opened: index %d (%s).", self.device_index, self.device_name)

            # Reset synchronization flags and counters
            self._stream_error = None
            self._latest_frame = None
            self._new_frame_available = False
            self._first_frame_captured = False
            self.frame_count = 0
            self.fps = 0.0
            self._thread_running = True

            # Launch background worker for non-blocking frame capture
            self._capture_thread = threading.Thread(
                target=self._capture_worker,
                name="MacBookWebcamWorker",
                daemon=True
            )
            self._capture_thread.start()

            self.is_preview_active = True
            return True

        except Exception as exc:
            self.state = CameraState.ERROR
            self.status_message = f"Error: {exc.__class__.__name__}"
            logger.error("Camera initialization failure with unexpected exception: %s", exc, exc_info=True)
            self.release()
            return False

    def _capture_worker(self) -> None:
        """Background thread worker to grab frames from the camera continuously."""
        logger.debug("Webcam capture worker thread started for device index %d.", self.device_index)
        consecutive_errors = 0
        warmup_attempts = 0
        last_time = time.perf_counter()
        fps_smoothing = 0.9

        while self._thread_running and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                now = time.perf_counter()
                dt = now - last_time
                last_time = now

                if ret and frame is not None:
                    consecutive_errors = 0

                    # First valid frame capture diagnostic
                    if not self._first_frame_captured:
                        h, w = frame.shape[:2]
                        self.resolution = (w, h)
                        self._first_frame_captured = True
                        self.state = CameraState.ACTIVE
                        logger.info(
                            "Successful frame capture: live stream verified from %s (Index %d) at %dx%d resolution.",
                            self.device_name,
                            self.device_index,
                            w,
                            h
                        )

                    # Update rolling FPS
                    if dt > 0:
                        instant_fps = 1.0 / dt
                        self.fps = (self.fps * fps_smoothing) + (instant_fps * (1.0 - fps_smoothing))

                    with self._frame_lock:
                        self._latest_frame = frame
                        self._new_frame_available = True
                        self.frame_count += 1
                        self.status_message = (
                            f"ACTIVE ({self.fps:.1f} FPS • {self.resolution[0]}x{self.resolution[1]})"
                        )

                else:
                    # Sensor warmup phase (allow up to 25 empty frames before declaring error)
                    if not self._first_frame_captured:
                        warmup_attempts += 1
                        if warmup_attempts > 25:
                            logger.warning(
                                "Camera initialization failure: sensor warmup timeout (>2.5s) on index %d.",
                                self.device_index
                            )
                            with self._frame_lock:
                                self.state = CameraState.ERROR
                                self._stream_error = "Unavailable (Sensor warmup timeout)"
                            break
                        time.sleep(0.1)
                        continue

                    # Stream disruption after already capturing frames
                    consecutive_errors += 1
                    if consecutive_errors > 45:  # ~1.5 seconds of sustained read failures
                        logger.warning(
                            "Frame capture failure: sustained read errors (>1.5s) on %s (Index %d).",
                            self.device_name,
                            self.device_index
                        )
                        with self._frame_lock:
                            self.state = CameraState.ERROR
                            self._stream_error = "Error: Camera disconnected / frame read failure"
                            self.status_message = "Error: Camera disconnected / frame read failure"
                        break

                # Sleep 15ms (~60 FPS) to pace capture and avoid consuming excessive CPU
                time.sleep(0.015)

            except Exception as exc:
                logger.error("Exception in camera capture worker thread: %s", exc)
                time.sleep(0.05)

        logger.debug("Webcam capture worker thread exiting.")

    def stop_preview(self) -> None:
        """Stop preview stream, release camera hardware handle, and destroy preview window."""
        if not self.is_preview_active and self.cap is None and self.state == CameraState.OFF:
            return

        logger.info("Camera preview disabled: stopping stream for %s (Index %d).", self.device_name, self.device_index)
        self.is_preview_active = False
        self.state = CameraState.OFF
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

    def update(self, display_frame: Optional[np.ndarray] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """Fetch latest frame and render OpenCV preview window if active.

        Args:
            display_frame: Optional annotated frame (e.g. with YOLO bounding boxes and telemetry)
                           to render in the OpenCV preview window instead of the raw camera frame.

        Must be called on the main thread for Cocoa GUI compatibility.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (frame_ready, frame)
        """
        if not self.is_preview_active:
            return False, None

        # Check for stream errors reported by background worker
        with self._frame_lock:
            if self._stream_error:
                err_msg = self._stream_error
                self.stop_preview()
                self.state = CameraState.ERROR
                self.status_message = err_msg
                return False, None

            frame_to_display = None
            if self._new_frame_available and self._latest_frame is not None:
                frame_to_display = self._latest_frame
                self._new_frame_available = False

        # If the caller provided a pre-annotated frame, prioritize it for GUI preview
        frame_to_render = display_frame if display_frame is not None else frame_to_display

        # Render frame in OpenCV window if a frame is ready
        if frame_to_render is not None:
            try:
                if not self._window_created:
                    cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.window_title, self.window_size[0], self.window_size[1])
                    if self.window_pos is not None:
                        cv2.moveWindow(self.window_title, self.window_pos[0], self.window_pos[1])
                    self._window_created = True

                cv2.imshow(self.window_title, frame_to_render)

                if self._move_attempts < 5:
                    cv2.resizeWindow(self.window_title, self.window_size[0], self.window_size[1])
                    if self.window_pos is not None:
                        cv2.moveWindow(self.window_title, self.window_pos[0], self.window_pos[1])
                    self._move_attempts += 1
            except Exception as exc:
                logger.error("Error during cv2.imshow: %s", exc)
                self.stop_preview()
                self.state = CameraState.ERROR
                self.status_message = f"Error: GUI display failed ({exc.__class__.__name__})"
                return False, None

        # Process Cocoa window events (1ms delay)
        # Allows closing preview via 'Q' key or ESC inside the OpenCV window,
        # or capturing hotkeys (M, P, Space, etc.) forwarded to the application.
        if self._window_created:
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.last_key = key
                if key in (ord('q'), ord('Q'), 27):
                    logger.info("Preview disabled by user key event in OpenCV window.")
                    self.stop_preview()
                    return False, None
            else:
                self.last_key = None

        has_frame = (frame_to_display is not None or display_frame is not None)
        returned_frame = frame_to_display if frame_to_display is not None else display_frame
        return has_frame, returned_frame

    def release(self) -> None:
        """Safely release camera hardware handles and destroy OpenCV windows.

        This method is strictly idempotent and safe to call multiple times.
        """
        self.is_preview_active = False
        self._thread_running = False

        # Terminate capture thread cleanly
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=0.5)
        self._capture_thread = None

        # Release OpenCV VideoCapture handle
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("Resource cleanup: released VideoCapture hardware handle for index %d.", self.device_index)
            except Exception as exc:
                logger.debug("Error releasing VideoCapture handle: %s", exc)
            self.cap = None

        # Clean up OpenCV windows
        if self._window_created:
            try:
                cv2.destroyWindow(self.window_title)
            except Exception:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            self._window_created = False
