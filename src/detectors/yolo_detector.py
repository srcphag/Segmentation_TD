"""
YOLO11 segmentation detector.

Provides fast, realtime-capable instance segmentation using YOLO11-seg models.
"""

from typing import Optional
import numpy as np
from ultralytics import YOLO

from .base import BaseDetector, DetectionResult


class YOLODetector(BaseDetector):
    """
    YOLO11 segmentation model wrapper.

    Handles model loading, inference, and result extraction.
    Suitable for realtime webcam processing.
    """

    def __init__(self, config):
        """
        Initialize the detector with configuration.

        Args:
            config: Application configuration
        """
        super().__init__()  # Initialize progress callback support
        self.config = config
        self.model: Optional[YOLO] = None
        self._class_names: dict = {}

    def load(self) -> None:
        """Load the YOLO model."""
        print(f"Loading YOLO model: {self.config.model_path}")
        print(f"Device: {self.config.device}")

        self.model = YOLO(self.config.model_path)
        self._class_names = self.model.names

        print(f"Model loaded successfully. Classes: {len(self._class_names)}")

    def predict(self, frame: np.ndarray) -> DetectionResult:
        """
        Run segmentation inference on a frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            DetectionResult with masks, boxes, and metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Run inference
        results = self.model(
            frame,
            conf=self.config.confidence,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=False,
        )

        result = results[0]

        # Extract inference time
        speed = result.speed
        inference_time = speed.get("inference", 0)

        # Get annotated frame
        annotated = result.plot(
            masks=self.config.show_masks,
            boxes=self.config.show_boxes,
            labels=self.config.show_labels,
            conf=self.config.show_confidence,
        )

        # Extract masks
        masks = None
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()

        # Extract boxes and metadata
        boxes = None
        class_ids = None
        class_names = None
        confidences = None

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
            class_names = [self._class_names[cid] for cid in class_ids]
            confidences = result.boxes.conf.cpu().numpy().tolist()

        return DetectionResult(
            frame=frame,
            annotated_frame=annotated,
            masks=masks,
            boxes=boxes,
            class_ids=class_ids,
            class_names=class_names,
            confidences=confidences,
            inference_time_ms=inference_time,
        )

    @property
    def supports_realtime(self) -> bool:
        """YOLO supports realtime webcam processing."""
        return True

    @property
    def class_names(self) -> dict:
        """Get the class name mapping."""
        return self._class_names
