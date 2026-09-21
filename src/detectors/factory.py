"""
Detector factory for creating segmentation backends.

Provides a unified interface to create YOLO, SAM 2, or SAM 3 detectors
based on configuration.
"""

from .base import BaseDetector


def create_detector(config) -> BaseDetector:
    """
    Create a detector based on the backend configuration.

    Args:
        config: Application configuration with 'backend' field

    Returns:
        BaseDetector instance (YOLODetector, SAM2Detector, or SAM3Detector)

    Raises:
        ValueError: If backend is not recognized
    """
    backend = getattr(config, "backend", "yolo")

    if backend == "sam3":
        from .sam3_detector import SAM3Detector

        return SAM3Detector(config)
    elif backend == "sam":
        from .sam2_detector import SAM2Detector

        return SAM2Detector(config)
    elif backend == "yolo":
        from .yolo_detector import YOLODetector

        return YOLODetector(config)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Supported backends: yolo, sam, sam3"
        )
