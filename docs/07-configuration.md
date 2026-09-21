# 07 - Configuration

Configuration lives in the `Config` dataclass in `src/config.py`. It is used by the CLI (`main.py`), the subprocess wrapper (`td_segment.py`), and its device/model auto-detection logic. The Script TOP (`td_script_top.py`) does **not** use `Config` - it reads its own parameters directly.

## Device Auto-Detection

`Config.detect_device()` picks the best device:

| Priority | Device | Recommended Model |
|----------|--------|-------------------|
| 1 | CUDA (NVIDIA GPU) | `yolo11s-seg.pt` (accuracy) |
| 2 | MPS (Apple Silicon) | `yolo11n-seg.pt` (speed) |
| 3 | CPU | `yolo11n-seg.pt` (smallest) |

If `device` is unset it is detected at runtime. If `model_path` is unset it defaults per backend:

- `sam3` → `sam3.pt`
- `sam` → `sam2_l.pt`
- `yolo` → the device-recommended model

Bare model names resolve into `models/`; absolute paths are used as-is.

## Field Reference

| Field | Default | Purpose |
|-------|---------|---------|
| `backend` | `"yolo"` | `yolo`, `sam`, `sam3` |
| `model_path` | `None` | Auto-selected if unset |
| `device` | `None` | Auto-detected (`cuda:0`, `mps`, `cpu`) |
| `confidence` | `0.25` | Detection threshold |
| `iou_threshold` | `0.7` | NMS IoU |
| `image_size` | `640` | Input size |
| `sam_text` | `None` | SAM 3 text prompts |
| `sam_points` | `None` | SAM 2 point prompts |
| `sam_bbox` | `None` | SAM 2/3 box prompt |
| `amg_points_per_side` | `16` | AMG grid density |
| `amg_nms_thresh` | `0.7` | AMG duplicate removal |
| `camera_id` | `0` | Webcam index |
| `camera_width` / `camera_height` | 1280 / 720 | Capture resolution |
| `show_fps` | `True` | FPS overlay (CLI) |
| `show_masks` | `True` | Mask overlay (CLI) |
| `show_boxes` | `False` | Bounding boxes |
| `show_labels` | `False` | Class labels |
| `image_path` | `None` | Set → image mode |
| `output_dir` | `"output"` | Base output directory |
| `save_individual_masks` | `True` | Per-object mask PNGs |
| `save_composite` | `True` | Composite overlay |
| `save_metadata` | `True` | Metadata JSON |
| `run_id` | `None` | Force output subdir name |

## Constructing a Config

```python
from src.config import Config

cfg = Config(
    backend="sam",
    image_path="photo.jpg",
    output_dir="results",
    amg_points_per_side=32,
)
```

`__post_init__` ensures `models/` exists, creates `output_dir` in image mode, and fills in `effective_device` / `effective_model`.

## Detector Architecture

`src/detectors/factory.py` maps a config to a detector instance:

| backend | Class | Notes |
|---------|-------|-------|
| `yolo` | `YOLODetector` | realtime-capable, class labels |
| `sam` | `SAM2Detector` | points/bbox or AMG |
| `sam3` | `SAM3Detector` | text/bbox semantic or AMG |

All inherit `BaseDetector` (`src/detectors/base.py`), which defines `load()`, `predict(frame)`, `supports_realtime`, `class_names`, and an optional progress callback. `DetectionResult` carries `frame`, `annotated_frame`, `masks`, `boxes`, `class_ids`, `class_names`, `confidences`, `inference_time_ms`.

The Script TOP re-implements this logic inline (numpy-only AMG, no `cv2` dependency for resizing) so it can run inside TD without importing `src`.