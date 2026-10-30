"""Perception package for camera streams, sensor inputs, and local object detection."""

from .webcam import WebcamManager, CameraState, get_macbook_builtin_camera_index
from .object_detector import Detection, YOLOObjectDetector
from .grid_mapper import GridMapper, GridMapperConfig, GridMappingResult, MappingMode

__all__ = [
    "WebcamManager",
    "CameraState",
    "get_macbook_builtin_camera_index",
    "Detection",
    "YOLOObjectDetector",
    "GridMapper",
    "GridMapperConfig",
    "GridMappingResult",
    "MappingMode",
]
