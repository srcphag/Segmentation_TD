"""
Configuration management for the segmentation application.

Handles device detection, model selection, and application settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List
import torch

# Models directory (relative to project root)
MODELS_DIR = Path(__file__).parent.parent / "models"


def detect_device() -> Tuple[str, str]:
    """
    Auto-detect the best available device and recommend an appropriate model.

    Returns:
        Tuple of (device_string, recommended_model)

    Device priority:
        1. CUDA (NVIDIA GPU) - use yolo11s-seg for better accuracy
        2. MPS (Apple Silicon) - use yolo11n-seg for speed
        3. CPU - use yolo11n-seg (smallest/fastest)
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {device_name}")
        return "cuda:0", "yolo11s-seg.pt"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple Silicon (MPS) detected")
        return "mps", "yolo11n-seg.pt"
    else:
        print("No GPU detected, using CPU")
        return "cpu", "yolo11n-seg.pt"


@dataclass
class Config:
    """Application configuration settings."""

    # Backend selection
    backend: str = "yolo"  # "yolo", "sam", or "sam3"

    # Model settings
    model_path: Optional[str] = None  # None = auto-select based on device/backend
    device: Optional[str] = None  # None = auto-detect
    confidence: float = 0.25
    iou_threshold: float = 0.7
    image_size: int = 640

    # SAM-specific prompt settings
    sam_text: Optional[List[str]] = None  # SAM 3 text prompts (e.g., ["person", "car"])
    sam_points: Optional[List[List[int]]] = None  # Point prompts [[x1,y1], [x2,y2]]
    sam_bbox: Optional[List[int]] = None  # Bounding box prompt [x1, y1, x2, y2]

    # AMG (Automatic Mask Generation) settings for SAM segment-everything mode
    amg_points_per_side: int = 16  # 8=fast, 16=balanced, 32=thorough
    amg_nms_thresh: float = 0.7  # NMS threshold for duplicate removal

    # Camera settings
    camera_id: int = 0
    camera_width: int = 1280
    camera_height: int = 720

    # Display settings
    window_name: str = "Realtime Segmentation"
    show_fps: bool = True
    show_masks: bool = True
    show_boxes: bool = False
    show_labels: bool = False
    show_confidence: bool = True

    # FPS counter settings
    fps_window: int = 30  # Rolling average window

    # Image mode settings
    image_path: Optional[str] = None  # Path to input image (None = webcam mode)
    output_dir: str = "output"  # Output directory for image mode
    save_individual_masks: bool = True  # Save separate mask files
    save_composite: bool = True  # Save composite colored overlay
    save_metadata: bool = True  # Save metadata JSON

    # TouchDesigner integration
    run_id: Optional[str] = None  # Override run ID (auto-generated if None)

    # Auto-detected values (populated at runtime)
    _detected_device: str = field(default="", init=False)
    _detected_model: str = field(default="", init=False)

    def __post_init__(self):
        """Auto-detect device and model if not specified."""
        # Ensure models directory exists
        MODELS_DIR.mkdir(exist_ok=True)

        # Auto-detect device
        if self.device is None:
            self._detected_device, self._detected_model = detect_device()
            self.device = self._detected_device
        else:
            self._detected_device = self.device
            self._detected_model = "yolo11n-seg.pt"

        # Auto-select model based on backend if not specified
        if self.model_path is None:
            self.model_path = self._get_default_model()

        # Resolve model path to models directory (for all backends)
        self.model_path = self._resolve_model_path(self.model_path)

        # Create output directory if in image mode
        if self.image_path is not None:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_default_model(self) -> str:
        """Get default model based on backend and device."""
        if self.backend == "sam3":
            return "sam3.pt"
        elif self.backend == "sam":
            # Use SAM 2 Large (best SAM 2 quality) as default
            return "sam2_l.pt"
        else:
            # YOLO model selection (existing logic)
            return self._detected_model

    def _resolve_model_path(self, model_name: str) -> str:
        """
        Resolve model name to full path in models directory.

        If given just a model name (e.g., 'yolo11n-seg.pt'), resolves to models/yolo11n-seg.pt.
        If given an absolute path or path with directory, uses as-is.
        """
        model_path = Path(model_name)

        # If it's just a filename, put it in models directory
        if model_path.parent == Path(".") or str(model_path.parent) == "":
            return str(MODELS_DIR / model_name)

        # Otherwise use as-is (absolute path or relative with directory)
        return model_name

    @property
    def effective_device(self) -> str:
        """Get the device that will be used."""
        return self.device or self._detected_device

    @property
    def effective_model(self) -> str:
        """Get the model that will be used."""
        return self.model_path or self._detected_model

    def __str__(self) -> str:
        if self.image_path:
            # Image mode
            prompt_info = ""
            if self.sam_text:
                prompt_info = f"  prompts={self.sam_text},\n"
            elif self.sam_bbox:
                prompt_info = f"  bbox={self.sam_bbox},\n"
            elif self.sam_points:
                prompt_info = f"  points={self.sam_points},\n"
            elif self.backend in ("sam", "sam3"):
                # SAM/SAM3 with no prompts = AMG mode
                prompt_info = f"  amg_grid={self.amg_points_per_side}x{self.amg_points_per_side},\n"

            return (
                f"Config(\n"
                f"  mode=image,\n"
                f"  backend={self.backend},\n"
                f"  input={self.image_path},\n"
                f"  output={self.output_dir},\n"
                f"  model={self.effective_model},\n"
                f"  device={self.effective_device},\n"
                f"  confidence={self.confidence},\n"
                f"{prompt_info}"
                f")"
            )
        else:
            return (
                f"Config(\n"
                f"  mode=webcam,\n"
                f"  backend={self.backend},\n"
                f"  model={self.effective_model},\n"
                f"  device={self.effective_device},\n"
                f"  confidence={self.confidence},\n"
                f"  camera={self.camera_id} @ {self.camera_width}x{self.camera_height}\n"
                f")"
            )
