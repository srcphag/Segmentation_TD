"""
Abstract base class for segmentation detectors.

Provides a common interface for YOLO, SAM 2, and SAM 3 backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Callable
import numpy as np

# Progress callback type: (current, total, stage, message) -> None
ProgressCallback = Callable[[int, int, str, str], None]


@dataclass
class DetectionResult:
    """Structured result from segmentation inference."""

    frame: np.ndarray  # Original frame
    annotated_frame: np.ndarray  # Frame with visualizations
    masks: Optional[np.ndarray]  # Binary masks (N, H, W)
    boxes: Optional[np.ndarray]  # Bounding boxes (N, 4) in xyxy format
    class_ids: Optional[List[int]]  # Class IDs for each detection
    class_names: Optional[List[str]]  # Class names for each detection
    confidences: Optional[List[float]]  # Confidence scores
    inference_time_ms: float  # Inference time in milliseconds


class BaseDetector(ABC):
    """
    Abstract base class for segmentation detectors.

    All detector implementations (YOLO, SAM 2, SAM 3) must inherit from this
    and implement the required methods.
    """

    def __init__(self):
        """Initialize base detector with progress callback support."""
        self._progress_callback: Optional[ProgressCallback] = None

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """
        Set a callback for progress reporting.

        Args:
            callback: Function that receives (current, total, stage, message)
        """
        self._progress_callback = callback

    def _report_progress(self, current: int, total: int, stage: str, message: str) -> None:
        """
        Report progress to the callback if set.

        Args:
            current: Current progress value
            total: Total value for completion
            stage: Current processing stage (e.g., "loading", "amg", "nms")
            message: Human-readable progress message
        """
        if self._progress_callback:
            self._progress_callback(current, total, stage, message)

    @abstractmethod
    def load(self) -> None:
        """Load the model. Called once during initialization."""
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on a frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            DetectionResult with masks, boxes, and metadata
        """
        pass

    @property
    @abstractmethod
    def supports_realtime(self) -> bool:
        """
        Whether this detector supports realtime webcam mode.

        Returns:
            True if suitable for realtime inference, False otherwise.
            SAM models return False (too slow), YOLO returns True.
        """
        pass

    @property
    @abstractmethod
    def class_names(self) -> dict:
        """Get the class name mapping."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
