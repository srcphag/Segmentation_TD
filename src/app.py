"""
Main application class for realtime segmentation.

Orchestrates camera, detector, and visualizer components.
"""

import cv2

from .config import Config
from .detectors import create_detector
from .camera import WebcamCapture
from .visualizer import Visualizer


class SegmentationApp:
    """
    Main application for realtime instance segmentation.

    Coordinates all components and runs the main processing loop.
    """

    def __init__(self, config: Config):
        """
        Initialize the application with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.detector = create_detector(config)
        self.camera = WebcamCapture(config)
        self.visualizer = Visualizer(config)
        self._running = False

    def initialize(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if initialization successful, False otherwise
        """
        print("=" * 50)
        print("YOLO11 Realtime Instance Segmentation")
        print("=" * 50)
        print()
        print(self.config)
        print()

        # Load model
        try:
            self.detector.load()
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

        # Open camera
        if not self.camera.open():
            print("Error: Failed to open camera")
            return False

        print()
        print("Initialization complete!")
        print()

        return True

    def run(self) -> None:
        """
        Run the main application loop.

        Captures frames, runs inference, and displays results.
        Press 'q' to quit.
        """
        if not self.camera.is_opened:
            print("Error: Camera not initialized. Call initialize() first.")
            return

        self._running = True
        print("Starting segmentation loop... Press 'q' to quit.")
        print()

        try:
            while self._running:
                # Capture frame
                success, frame = self.camera.read()

                if not success or frame is None:
                    print("Warning: Failed to read frame")
                    continue

                # Run inference
                result = self.detector.predict(frame)

                # Draw visualizations
                display_frame = self.visualizer.draw(result)

                # Show frame
                cv2.imshow(self.config.window_name, display_frame)

                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quit requested")
                    self._running = False

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Clean up resources."""
        self._running = False
        self.camera.close()
        cv2.destroyAllWindows()
        print("Application shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        if self.initialize():
            return self
        raise RuntimeError("Failed to initialize application")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
