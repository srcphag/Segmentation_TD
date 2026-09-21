"""
Webcam capture management.

Provides a clean interface for webcam access with error handling.
"""

from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np
import subprocess

from .config import Config


def list_available_cameras(max_cameras: int = 10) -> List[Dict]:
    """
    List available cameras with their indices and names.

    On macOS, uses system_profiler to get camera names.
    Falls back to probing indices if names unavailable.

    Args:
        max_cameras: Maximum number of camera indices to probe

    Returns:
        List of dicts with 'index', 'name', and 'available' keys
    """
    cameras = []
    camera_names = _get_macos_camera_names()

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to verify it actually works
            ret, _ = cap.read()
            cap.release()

            name = camera_names.get(i, f"Camera {i}")
            cameras.append({
                "index": i,
                "name": name,
                "available": ret,
            })

    return cameras


def _get_macos_camera_names() -> Dict[int, str]:
    """
    Get camera names on macOS using system_profiler.

    Returns:
        Dict mapping camera index to name
    """
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        # Parse the output to extract camera names
        names = {}
        lines = result.stdout.split("\n")
        camera_idx = 0

        for line in lines:
            # Camera names appear as headers with colons
            stripped = line.strip()
            if stripped and not stripped.startswith("Camera") and ":" in stripped:
                # Skip metadata lines
                if any(x in stripped.lower() for x in ["model id", "unique id", "location"]):
                    continue
                # This is likely a camera name line
                if stripped.endswith(":"):
                    name = stripped[:-1]  # Remove trailing colon
                    # Skip "Cameras:" header
                    if name.lower() != "cameras":
                        names[camera_idx] = name
                        camera_idx += 1

        return names
    except Exception:
        return {}


def select_camera_interactive(cameras: List[Dict]) -> Optional[int]:
    """
    Interactively prompt user to select a camera.

    Args:
        cameras: List of camera dicts from list_available_cameras()

    Returns:
        Selected camera index, or None if cancelled
    """
    if not cameras:
        print("No cameras found!")
        return None

    # Filter to only working cameras
    working = [c for c in cameras if c["available"]]

    if not working:
        print("No working cameras found!")
        return None

    if len(working) == 1:
        print(f"Using only available camera: {working[0]['name']} (index {working[0]['index']})")
        return working[0]["index"]

    print("\nAvailable cameras:")
    print("-" * 50)

    for i, cam in enumerate(working):
        continuity_marker = ""
        name_lower = cam["name"].lower()
        if "continuity" in name_lower or "iphone" in name_lower or "ipad" in name_lower:
            continuity_marker = " [Continuity Camera]"
        print(f"  [{i + 1}] {cam['name']} (index {cam['index']}){continuity_marker}")

    print("-" * 50)

    while True:
        try:
            choice = input(f"Select camera [1-{len(working)}] (or 'q' to quit): ").strip()

            if choice.lower() == 'q':
                return None

            choice_num = int(choice)
            if 1 <= choice_num <= len(working):
                selected = working[choice_num - 1]
                print(f"Selected: {selected['name']}")
                return selected["index"]
            else:
                print(f"Please enter a number between 1 and {len(working)}")
        except ValueError:
            print("Invalid input. Enter a number or 'q' to quit.")


class WebcamCapture:
    """
    Webcam capture wrapper with configuration and error handling.

    Supports context manager usage for automatic resource cleanup.
    """

    def __init__(self, config: Config):
        """
        Initialize webcam capture with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False

    def open(self) -> bool:
        """
        Open the webcam.

        Returns:
            True if webcam opened successfully, False otherwise
        """
        print(f"Opening camera {self.config.camera_id}...")

        self._cap = cv2.VideoCapture(self.config.camera_id)

        if not self._cap.isOpened():
            print(f"Error: Could not open camera {self.config.camera_id}")
            return False

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

        # Verify actual resolution
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera opened: {actual_width}x{actual_height}")

        self._is_opened = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the webcam.

        Returns:
            Tuple of (success, frame) where frame is BGR numpy array or None
        """
        if self._cap is None or not self._is_opened:
            return False, None

        ret, frame = self._cap.read()

        if not ret:
            return False, None

        return True, frame

    def close(self) -> None:
        """Release the webcam."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._is_opened = False
            print("Camera released")

    @property
    def is_opened(self) -> bool:
        """Check if webcam is currently open."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get current resolution (width, height)."""
        if self._cap is None:
            return 0, 0
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    @property
    def fps(self) -> float:
        """Get camera FPS capability."""
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
