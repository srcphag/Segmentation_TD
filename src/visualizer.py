"""
Visualization utilities for the segmentation application.

Handles FPS display and overlay rendering.
"""

from collections import deque
from typing import Optional
import time
import cv2
import numpy as np

from .config import Config
from .detectors import DetectionResult


class FPSCounter:
    """
    Rolling average FPS counter.

    Calculates FPS based on a sliding window of frame timestamps.
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.

        Args:
            window_size: Number of frames for rolling average
        """
        self._timestamps = deque(maxlen=window_size)
        self._window_size = window_size

    def update(self) -> None:
        """Record a new frame timestamp."""
        self._timestamps.append(time.time())

    @property
    def fps(self) -> float:
        """
        Calculate current FPS.

        Returns:
            Frames per second (0 if insufficient data)
        """
        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Reset the FPS counter."""
        self._timestamps.clear()


class Visualizer:
    """
    Visualization manager for the segmentation application.

    Handles FPS overlay and display formatting.
    """

    def __init__(self, config: Config):
        """
        Initialize visualizer with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.fps_counter = FPSCounter(config.fps_window)

        # Display styling
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.7
        self._font_thickness = 2
        self._fps_color = (0, 255, 0)  # Green
        self._info_color = (255, 255, 255)  # White
        self._bg_color = (0, 0, 0)  # Black background

    def draw(
        self,
        result: DetectionResult,
        extra_info: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Draw overlays on the annotated frame.

        Args:
            result: Detection result containing annotated frame
            extra_info: Optional dict of additional info to display

        Returns:
            Frame with FPS and info overlays
        """
        # Update FPS counter
        self.fps_counter.update()

        # Start with the annotated frame from YOLO
        frame = result.annotated_frame.copy()

        y_offset = 30

        # Draw FPS
        if self.config.show_fps:
            fps = self.fps_counter.fps
            fps_text = f"FPS: {fps:.1f}"
            frame = self._draw_text_with_background(
                frame, fps_text, (10, y_offset), self._fps_color
            )
            y_offset += 30

        # Draw inference time
        if self.config.show_fps:
            inference_text = f"Inference: {result.inference_time_ms:.1f}ms"
            frame = self._draw_text_with_background(
                frame, inference_text, (10, y_offset), self._info_color
            )
            y_offset += 30

        # Draw detection count
        if result.class_ids is not None:
            count = len(result.class_ids)
            count_text = f"Objects: {count}"
            frame = self._draw_text_with_background(
                frame, count_text, (10, y_offset), self._info_color
            )
            y_offset += 30

        # Draw extra info if provided
        if extra_info:
            for key, value in extra_info.items():
                info_text = f"{key}: {value}"
                frame = self._draw_text_with_background(
                    frame, info_text, (10, y_offset), self._info_color
                )
                y_offset += 30

        # Draw controls hint at bottom
        frame = self._draw_controls_hint(frame)

        return frame

    def _draw_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple,
        color: tuple,
    ) -> np.ndarray:
        """
        Draw text with a semi-transparent background.

        Args:
            frame: Image to draw on
            text: Text to display
            position: (x, y) position
            color: Text color (BGR)

        Returns:
            Frame with text drawn
        """
        x, y = position

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self._font, self._font_scale, self._font_thickness
        )

        # Draw background rectangle
        padding = 5
        cv2.rectangle(
            frame,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            self._bg_color,
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y),
            self._font,
            self._font_scale,
            color,
            self._font_thickness,
        )

        return frame

    def _draw_controls_hint(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw controls hint at the bottom of the frame.

        Args:
            frame: Image to draw on

        Returns:
            Frame with controls hint
        """
        height = frame.shape[0]
        hint_text = "Press 'q' to quit"

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(hint_text, self._font, 0.5, 1)

        # Draw at bottom center
        x = 10
        y = height - 10

        cv2.putText(
            frame,
            hint_text,
            (x, y),
            self._font,
            0.5,
            (128, 128, 128),
            1,
        )

        return frame

    def reset_fps(self) -> None:
        """Reset the FPS counter."""
        self.fps_counter.reset()
