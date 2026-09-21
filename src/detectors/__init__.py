"""
Segmentation detector implementations.

Provides unified interface for YOLO, SAM 2, and SAM 3 backends.
"""

from .base import BaseDetector, DetectionResult
from .factory import create_detector
from .amg import AMGConfig

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "create_detector",
    "AMGConfig",
]
