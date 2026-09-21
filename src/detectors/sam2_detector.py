"""
SAM 2 segmentation detector.

Provides zero-shot segmentation with visual prompts (points, boxes).
Uses automatic mask generation (AMG) for segment-everything mode.
"""

import time
from typing import Optional
import numpy as np
from ultralytics import SAM

from .base import BaseDetector, DetectionResult
from .amg import (
    AMGConfig,
    build_point_grid,
    scale_points_to_image,
    nms_boxes,
    create_mask_overlay,
)


class SAM2Detector(BaseDetector):
    """
    SAM 2 segmentation model wrapper.

    Supports:
    - Point prompts (click locations)
    - Bounding box prompts
    - Automatic mask generation (segment everything with point grid)

    NOT suitable for realtime webcam due to slow inference.
    """

    def __init__(self, config):
        """
        Initialize the detector with configuration.

        Args:
            config: Application configuration with SAM-specific fields
        """
        super().__init__()  # Initialize progress callback support
        self.config = config
        self.model: Optional[SAM] = None
        self._class_names: dict = {0: "segment"}  # SAM doesn't have class labels

        # AMG configuration from config or defaults
        self.amg_config = AMGConfig(
            points_per_side=getattr(config, "amg_points_per_side", 16),
            box_nms_thresh=getattr(config, "amg_nms_thresh", 0.7),
        )

    def load(self) -> None:
        """Load the SAM 2 model."""
        print(f"Loading SAM 2 model: {self.config.model_path}")
        print(f"Device: {self.config.device}")

        self.model = SAM(self.config.model_path)

        print("SAM 2 model loaded successfully.")

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

        # If prompts provided, use direct prompting
        if self.config.sam_points or self.config.sam_bbox:
            return self._predict_with_prompts(frame)

        # No prompts = automatic mask generation
        return self._predict_automatic(frame)

    def _predict_with_prompts(self, frame: np.ndarray) -> DetectionResult:
        """Run inference with user-provided prompts."""
        kwargs = {"verbose": False}

        # Add point prompts if provided
        if self.config.sam_points:
            kwargs["points"] = self.config.sam_points
            kwargs["labels"] = [1] * len(self.config.sam_points)

        # Add bounding box prompt if provided
        if self.config.sam_bbox:
            kwargs["bboxes"] = [self.config.sam_bbox]

        # Run inference
        results = self.model(frame, **kwargs)
        result = results[0]

        # Extract inference time
        speed = result.speed
        inference_time = speed.get("inference", 0)

        # Get annotated frame
        annotated = result.plot()

        # Extract masks
        masks = None
        num_masks = 0
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            num_masks = len(masks)

        # Extract boxes
        boxes = None
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()

        # SAM doesn't provide class information - use generic labels
        class_ids = [0] * num_masks
        class_names = ["segment"] * num_masks
        confidences = [1.0] * num_masks

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

    def _predict_automatic(self, frame: np.ndarray) -> DetectionResult:
        """
        Generate masks using automatic point grid approach.

        This implements a simplified version of Meta's SamAutomaticMaskGenerator:
        1. Generate a grid of points across the image
        2. Run SAM prediction for each point
        3. Filter by mask area
        4. Remove duplicates using NMS
        """
        h, w = frame.shape[:2]
        start_time = time.time()

        # Generate point grid
        points_per_side = self.amg_config.points_per_side
        points_normalized = build_point_grid(points_per_side)
        points = scale_points_to_image(points_normalized, h, w)

        total_points = len(points)
        print(f"AMG: Processing {total_points} points ({points_per_side}x{points_per_side} grid)...")

        # Report initial progress
        self._report_progress(0, total_points, "amg", f"Starting AMG ({total_points} points)")

        all_masks = []
        all_boxes = []
        all_scores = []

        # Process each point
        for idx, pt in enumerate(points):
            # Report progress every 10 points
            if idx % 10 == 0:
                self._report_progress(idx, total_points, "amg", f"Point {idx}/{total_points}")

            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  Point {idx + 1}/{total_points}...")

            try:
                results = self.model(
                    frame,
                    points=[[pt[0], pt[1]]],
                    labels=[1],
                    verbose=False,
                )
                result = results[0]

                if result.masks is not None and len(result.masks) > 0:
                    # Take the first (best) mask for this point
                    mask = result.masks.data[0].cpu().numpy()
                    box = result.boxes.xyxy[0].cpu().numpy()

                    # Filter by minimum area
                    mask_area = mask.sum()
                    if mask_area > self.amg_config.min_mask_region_area:
                        all_masks.append(mask)
                        all_boxes.append(box)
                        all_scores.append(float(mask_area))

            except Exception as e:
                # Skip failed points
                continue

        print(f"AMG: Found {len(all_masks)} candidate masks")

        # Report NMS stage
        self._report_progress(total_points, total_points, "nms", "Filtering masks...")

        # Apply NMS to remove duplicates
        masks = None
        boxes = None
        num_masks = 0

        if all_masks:
            boxes_arr = np.array(all_boxes)
            scores_arr = np.array(all_scores)

            keep_indices = nms_boxes(
                boxes_arr, scores_arr, self.amg_config.box_nms_thresh
            )

            masks = np.array([all_masks[i] for i in keep_indices])
            boxes = boxes_arr[keep_indices]
            num_masks = len(masks)

            print(f"AMG: {num_masks} masks after NMS")

        # Calculate total inference time
        inference_time = (time.time() - start_time) * 1000  # ms

        # Create annotated frame with mask overlays
        annotated = create_mask_overlay(frame, masks, alpha=0.5)

        # SAM doesn't provide class information - use generic labels
        class_ids = [0] * num_masks
        class_names = ["segment"] * num_masks
        confidences = [1.0] * num_masks

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
        """SAM 2 is too slow for realtime webcam processing."""
        return False

    @property
    def class_names(self) -> dict:
        """Get the class name mapping (SAM uses generic 'segment')."""
        return self._class_names
