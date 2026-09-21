"""
Realtime Instance Segmentation Application

A modular multi-backend segmentation application supporting YOLO, SAM 2, and SAM 3
with auto-detection of hardware and FPS display.
"""

from .config import Config, detect_device
from .detectors import create_detector, BaseDetector, DetectionResult
from .camera import WebcamCapture
from .visualizer import Visualizer
from .app import SegmentationApp
from .image_processor import ImageProcessor, generate_colors

__all__ = [
    # Config
    "Config",
    "detect_device",
    # Detectors
    "create_detector",
    "BaseDetector",
    "DetectionResult",
    # Components
    "WebcamCapture",
    "Visualizer",
    "SegmentationApp",
    "ImageProcessor",
    "generate_colors",
]
