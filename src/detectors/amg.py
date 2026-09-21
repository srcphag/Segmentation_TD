"""
Automatic Mask Generator utilities for SAM models.

Implements point grid generation and NMS filtering to enable
"segment everything" mode similar to Meta's SamAutomaticMaskGenerator.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AMGConfig:
    """Configuration for automatic mask generation."""

    points_per_side: int = 16  # 16x16 = 256 points (balanced default)
    # Options: 8 (fast), 16 (balanced), 32 (thorough)
    pred_iou_thresh: float = 0.88  # Filter by predicted IoU
    box_nms_thresh: float = 0.7  # NMS overlap threshold
    min_mask_region_area: int = 100  # Minimum mask area in pixels


def build_point_grid(points_per_side: int) -> np.ndarray:
    """
    Generate a grid of points normalized to [0, 1].

    Args:
        points_per_side: Number of points along each side of the grid

    Returns:
        Array of shape (points_per_side^2, 2) with normalized (x, y) coordinates
    """
    offset = 1 / (2 * points_per_side)
    points_one_side = np.linspace(offset, 1 - offset, points_per_side)
    points_x = np.tile(points_one_side[None, :], (points_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, points_per_side))
    return np.stack([points_x.flatten(), points_y.flatten()], axis=-1)


def scale_points_to_image(
    points: np.ndarray, height: int, width: int
) -> np.ndarray:
    """
    Scale normalized points [0,1] to image dimensions.

    Args:
        points: Array of shape (N, 2) with normalized coordinates
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Array of shape (N, 2) with pixel coordinates
    """
    return points * np.array([width, height])


def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in xyxy format.

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        Intersection over Union value
    """
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def nms_boxes(
    boxes: np.ndarray, scores: np.ndarray, iou_thresh: float
) -> List[int]:
    """
    Non-maximum suppression on boxes.

    Args:
        boxes: Array of shape (N, 4) with boxes in xyxy format
        scores: Array of shape (N,) with scores for each box
        iou_thresh: IoU threshold for suppression

    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []

    # Sort by score (descending)
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        # Keep the highest scoring box
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # Compute IoU with remaining boxes
        remaining = order[1:]
        ious = np.array([compute_box_iou(boxes[i], boxes[j]) for j in remaining])

        # Keep boxes with IoU below threshold
        mask = ious < iou_thresh
        order = remaining[mask]

    return keep


def create_mask_overlay(
    frame: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.5,
    colors: Optional[List[tuple]] = None,
) -> np.ndarray:
    """
    Create an annotated frame with colored mask overlays.

    Args:
        frame: Original BGR image
        masks: Array of shape (N, H, W) with binary masks
        alpha: Overlay transparency (0-1)
        colors: Optional list of BGR colors for each mask

    Returns:
        Annotated frame with mask overlays
    """
    if masks is None or len(masks) == 0:
        return frame.copy()

    overlay = frame.copy()
    output = frame.copy()

    # Generate colors if not provided
    if colors is None:
        np.random.seed(42)  # Consistent colors
        colors = [
            tuple(int(c) for c in np.random.randint(0, 255, 3))
            for _ in range(len(masks))
        ]

    for mask, color in zip(masks, colors):
        # Resize mask to frame size if needed
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Apply color overlay where mask is True
        mask_bool = mask > 0.5
        overlay[mask_bool] = color

    # Blend overlay with original
    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

    return output
