"""
SAM 3 segmentation detector.

Provides concept-based segmentation with text prompts and image exemplars.
Can find ALL instances of a concept (e.g., "all people", "all cars").
Uses automatic mask generation (AMG) for segment-everything mode.
"""

import time
from pathlib import Path
from typing import Optional, List
import numpy as np

from .base import BaseDetector, DetectionResult
from .amg import (
    AMGConfig,
    build_point_grid,
    scale_points_to_image,
    nms_boxes,
    create_mask_overlay,
)


class SAM3Detector(BaseDetector):
    """
    SAM 3 segmentation model wrapper.

    Supports:
    - Text prompts (noun phrases like "person", "car", "yellow school bus")
    - Image exemplars (bounding boxes as examples to find similar objects)
    - Segment everything (fallback to SAM interface)

    NOT suitable for realtime webcam due to model size (3.4GB) and inference time.
    """

    def __init__(self, config):
        """
        Initialize the detector with configuration.

        Args:
            config: Application configuration with SAM-specific fields
        """
        super().__init__()  # Initialize progress callback support
        self.config = config
        self.semantic_predictor = None  # For text/exemplar prompts
        self.model = None  # For segment-everything fallback
        self._class_names: dict = {}
        self._use_semantic = False

        # AMG configuration from config or defaults
        self.amg_config = AMGConfig(
            points_per_side=getattr(config, "amg_points_per_side", 16),
            box_nms_thresh=getattr(config, "amg_nms_thresh", 0.7),
        )

    def load(self) -> None:
        """Load the SAM 3 model."""
        print(f"Loading SAM 3 model: {self.config.model_path}")
        print(f"Device: {self.config.device}")

        # Check if model exists and provide guidance
        model_path = Path(self.config.model_path)
        if not model_path.exists() and not model_path.name.startswith("sam"):
            # Not an absolute path, will try to download
            pass

        print("Attempting to load SAM 3 model...")
        print("Note: SAM 3 (3.4GB) may require HuggingFace access approval.")
        print("If download fails, visit: https://huggingface.co/facebook/sam3")

        # Determine which interface to use based on prompts
        self._use_semantic = bool(self.config.sam_text) or (
            self.config.sam_bbox and self.config.backend == "sam3"
        )

        if self._use_semantic:
            # Use SAM3SemanticPredictor for text/exemplar prompts
            from ultralytics.models.sam import SAM3SemanticPredictor

            overrides = dict(
                conf=self.config.confidence,
                task="segment",
                mode="predict",
                model=self.config.model_path,
                half=True,  # FP16 for faster inference
            )
            self.semantic_predictor = SAM3SemanticPredictor(overrides=overrides)
            print("SAM 3 loaded with semantic predictor (text/exemplar prompts)")
        else:
            # Use standard SAM interface for segment-everything
            from ultralytics import SAM

            self.model = SAM(self.config.model_path)
            print("SAM 3 loaded with standard interface (segment everything)")

        # Build class names from text prompts if available
        if self.config.sam_text:
            self._class_names = {i: name for i, name in enumerate(self.config.sam_text)}
        else:
            self._class_names = {0: "segment"}

    def predict(self, frame: np.ndarray) -> DetectionResult:
        """
        Run segmentation inference on a frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            DetectionResult with masks, boxes, and metadata
        """
        import time

        start_time = time.time()

        if self._use_semantic:
            result = self._predict_semantic(frame)
        else:
            result = self._predict_segment_all(frame)

        # Calculate total inference time if not provided
        if result.inference_time_ms == 0:
            result = DetectionResult(
                frame=result.frame,
                annotated_frame=result.annotated_frame,
                masks=result.masks,
                boxes=result.boxes,
                class_ids=result.class_ids,
                class_names=result.class_names,
                confidences=result.confidences,
                inference_time_ms=(time.time() - start_time) * 1000,
            )

        return result

    def _predict_semantic(self, frame: np.ndarray) -> DetectionResult:
        """Run prediction with semantic predictor (text/exemplar prompts)."""
        if self.semantic_predictor is None:
            raise RuntimeError("Semantic predictor not loaded.")

        # Set image once for efficient multiple queries
        self.semantic_predictor.set_image(frame)

        # Run inference based on prompt type
        if self.config.sam_text:
            # Text-based concept segmentation
            results = self.semantic_predictor(text=self.config.sam_text)
        elif self.config.sam_bbox:
            # Image exemplar (bbox as example to find similar)
            results = self.semantic_predictor(bboxes=[self.config.sam_bbox])
        else:
            raise ValueError("SAM 3 semantic mode requires --text or --bbox")

        result = results[0]

        # Extract results
        return self._extract_result(frame, result)

    def _predict_segment_all(self, frame: np.ndarray) -> DetectionResult:
        """
        Run prediction in segment-everything mode using AMG.

        Uses automatic mask generation with point grid approach.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

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

    def _extract_result(self, frame: np.ndarray, result) -> DetectionResult:
        """Extract DetectionResult from ultralytics result object."""
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

        # Build class information
        class_ids: List[int] = []
        class_names: List[str] = []
        confidences: List[float] = []

        if result.boxes is not None and len(result.boxes) > 0:
            # If boxes have class info (from text prompts)
            if hasattr(result.boxes, "cls") and result.boxes.cls is not None:
                raw_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
                class_ids = raw_ids
                # Map to text prompt names if available
                if self.config.sam_text:
                    class_names = [
                        self.config.sam_text[cid] if cid < len(self.config.sam_text) else "segment"
                        for cid in raw_ids
                    ]
                else:
                    class_names = ["segment"] * len(raw_ids)
            else:
                class_ids = [0] * num_masks
                class_names = ["segment"] * num_masks

            if hasattr(result.boxes, "conf") and result.boxes.conf is not None:
                confidences = result.boxes.conf.cpu().numpy().tolist()
            else:
                confidences = [1.0] * num_masks
        else:
            # No boxes, just masks
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
        """SAM 3 is too slow for realtime webcam processing."""
        return False

    @property
    def class_names(self) -> dict:
        """Get the class name mapping."""
        return self._class_names
