"""
Image segmentation processor.

Handles single-image segmentation with mask extraction and file output.
Supports TouchDesigner integration via UUID subdirectories and done.json markers.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
import cv2
import numpy as np

from .config import Config
from .detectors import create_detector, DetectionResult


def generate_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Generate visually distinct colors using HSV color space.

    Args:
        num_colors: Number of distinct colors needed

    Returns:
        List of BGR color tuples
    """
    if num_colors == 0:
        return []

    colors = []
    for i in range(num_colors):
        # Distribute hues evenly across the spectrum
        hue = int(180 * i / num_colors)  # OpenCV uses 0-180 for hue
        saturation = 255
        value = 255

        color_hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color_bgr)))

    return colors


class ImageProcessor:
    """
    Single-image segmentation processor.

    Processes an image and generates annotated output plus individual masks.
    """

    def __init__(self, config: Config):
        """
        Initialize the image processor.

        Args:
            config: Application configuration
        """
        self.config = config
        self.detector = create_detector(config)  # Use factory for multi-backend support
        self.base_output_dir = Path(config.output_dir)
        self.run_id: Optional[str] = None
        self.output_dir: Optional[Path] = None

        # Set up progress callback for TouchDesigner integration
        self.detector.set_progress_callback(self._write_progress)

    def initialize(self) -> bool:
        """
        Initialize the processor (load model).

        Returns:
            True if successful, False otherwise
        """
        try:
            self.detector.load()
            self.base_output_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error initializing: {e}")
            return False

    def _write_progress(self, current: int, total: int, stage: str, message: str) -> None:
        """
        Write progress.json for TouchDesigner polling.

        This file is written to the output directory and updated during processing
        so TouchDesigner can poll it to show progress.

        Args:
            current: Current progress value (e.g., current point number)
            total: Total value for completion (e.g., total points)
            stage: Current processing stage ("loading", "amg", "nms", "saving")
            message: Human-readable progress message
        """
        if self.output_dir is None:
            return  # Output directory not created yet

        progress_data = {
            "status": "processing",
            "stage": stage,
            "current_point": current,
            "total_points": total,
            "percent": int((current / total) * 100) if total > 0 else 0,
            "message": message,
        }

        progress_path = self.output_dir / "progress.json"
        with open(progress_path, "w") as f:
            json.dump(progress_data, f)

    def process(self, image_path: str) -> Optional[str]:
        """
        Process a single image.

        Args:
            image_path: Path to input image

        Returns:
            Path to output directory (with UUID), or None on error
        """
        # Generate unique run ID (or use provided one) and create output subdirectory
        self.run_id = self.config.run_id or str(uuid.uuid4())[:8]
        self.output_dir = self.base_output_dir / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Run ID: {self.run_id}")
        print(f"Output: {self.output_dir}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image: {image_path}")
            return None

        print(f"Processing: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")

        # Run inference
        result = self.detector.predict(image)

        # Generate output files
        output_files = self._save_outputs(result, image_path)

        # Write completion marker for TouchDesigner
        self._write_completion_marker(output_files)

        return str(self.output_dir)

    def _save_outputs(
        self,
        result: DetectionResult,
        source_path: str
    ) -> Dict[str, Any]:
        """
        Save all output files.

        Args:
            result: Detection result from model
            source_path: Original image path (for naming)

        Returns:
            Dict with paths to generated files
        """
        stem = Path(source_path).stem
        output_files = {"source": source_path, "outputs": []}

        # 1. Save annotated image
        annotated_path = self.output_dir / f"{stem}_annotated.jpg"
        cv2.imwrite(str(annotated_path), result.annotated_frame)
        output_files["annotated"] = str(annotated_path)
        print(f"  Saved: {annotated_path}")

        # 2. Save individual masks and composite
        if result.masks is not None and len(result.masks) > 0:
            num_objects = len(result.masks)
            colors = generate_colors(num_objects)

            # Create composite overlay
            composite = result.frame.copy()

            for i, mask in enumerate(result.masks):
                class_name = result.class_names[i] if result.class_names else "unknown"
                confidence = result.confidences[i] if result.confidences else 0.0

                # Resize mask to original image size if needed
                if mask.shape[:2] != result.frame.shape[:2]:
                    mask = cv2.resize(
                        mask,
                        (result.frame.shape[1], result.frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                # Convert to binary mask (0 or 255)
                binary_mask = (mask > 0.5).astype(np.uint8) * 255

                # Save individual mask
                if self.config.save_individual_masks:
                    mask_path = self.output_dir / f"{stem}_mask_{i}_{class_name}.png"
                    cv2.imwrite(str(mask_path), binary_mask)
                    output_files["outputs"].append({
                        "type": "mask",
                        "path": str(mask_path),
                        "index": i,
                        "class": class_name,
                        "confidence": float(confidence),
                    })
                    print(f"  Saved: {mask_path}")

                # Add to composite with color
                if self.config.save_composite:
                    color = colors[i]
                    colored_mask = np.zeros_like(result.frame)
                    colored_mask[binary_mask > 0] = color

                    # Blend with composite (alpha = 0.5)
                    mask_bool = binary_mask > 0
                    composite[mask_bool] = cv2.addWeighted(
                        composite[mask_bool], 0.5,
                        colored_mask[mask_bool], 0.5, 0
                    )

            # Save composite
            if self.config.save_composite:
                composite_path = self.output_dir / f"{stem}_composite.png"
                cv2.imwrite(str(composite_path), composite)
                output_files["composite"] = str(composite_path)
                print(f"  Saved: {composite_path}")

        # 3. Save metadata
        if self.config.save_metadata:
            metadata = {
                "source": source_path,
                "backend": self.config.backend,
                "model": self.config.model_path,
                "inference_time_ms": result.inference_time_ms,
                "num_objects": len(result.class_ids) if result.class_ids else 0,
                "detections": [],
            }

            # Add prompt info for SAM backends
            if self.config.sam_text:
                metadata["prompts"] = {"type": "text", "values": self.config.sam_text}
            elif self.config.sam_bbox:
                metadata["prompts"] = {"type": "bbox", "values": self.config.sam_bbox}
            elif self.config.sam_points:
                metadata["prompts"] = {"type": "points", "values": self.config.sam_points}

            if result.class_ids:
                for i in range(len(result.class_ids)):
                    detection = {
                        "index": i,
                        "class_id": int(result.class_ids[i]),
                        "class_name": result.class_names[i],
                        "confidence": float(result.confidences[i]),
                        "bbox_xyxy": result.boxes[i].tolist() if result.boxes is not None else None,
                    }
                    metadata["detections"].append(detection)

            metadata_path = self.output_dir / f"{stem}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            output_files["metadata"] = str(metadata_path)
            print(f"  Saved: {metadata_path}")

        return output_files

    def _write_completion_marker(self, output_files: Dict[str, Any]) -> None:
        """
        Write done.json completion marker for TouchDesigner integration.

        This file signals to TouchDesigner that processing is complete and
        provides a manifest of generated files.

        Args:
            output_files: Dict of generated file paths from _save_outputs
        """
        # Collect mask files
        mask_files = []
        for output in output_files.get("outputs", []):
            if output.get("type") == "mask":
                mask_files.append(Path(output["path"]).name)

        # Get source stem for composite/metadata filenames
        source_stem = Path(output_files.get("source", "unknown")).stem

        done_data = {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "num_masks": len(mask_files),
            "masks": sorted(mask_files),
            "composite": f"{source_stem}_composite.png" if output_files.get("composite") else None,
            "annotated": f"{source_stem}_annotated.jpg" if output_files.get("annotated") else None,
            "metadata_file": f"{source_stem}_metadata.json" if output_files.get("metadata") else None,
        }

        done_path = self.output_dir / "done.json"
        with open(done_path, "w") as f:
            json.dump(done_data, f, indent=2)

        print(f"  Saved: {done_path}")
