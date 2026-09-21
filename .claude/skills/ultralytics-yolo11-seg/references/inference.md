# Inference Reference

Complete reference for YOLO11 segmentation inference parameters and input sources.

## Input Sources

### Images

```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

# Single image
results = model("image.jpg")

# Multiple images
results = model(["img1.jpg", "img2.jpg"])

# URL
results = model("https://example.com/image.jpg")

# PIL Image
from PIL import Image
img = Image.open("image.jpg")
results = model(img)

# OpenCV/numpy array (BGR)
import cv2
frame = cv2.imread("image.jpg")
results = model(frame)

# PyTorch tensor (BCHW, RGB, float32 0-1)
import torch
tensor = torch.rand(1, 3, 640, 640)
results = model(tensor)
```

### Video Files

```python
# Video file with streaming (memory efficient)
for result in model("video.mp4", stream=True):
    annotated = result.plot()

# Process specific frames
for result in model("video.mp4", stream=True, vid_stride=2):
    # Process every 2nd frame
    pass
```

### Webcam

```python
# Webcam by index
for result in model(0, stream=True):  # First webcam
    pass

for result in model(1, stream=True):  # Second webcam
    pass
```

### Network Streams

```python
# RTSP
results = model("rtsp://admin:password@192.168.1.100:554/stream", stream=True)

# RTMP
results = model("rtmp://server.com/live/stream", stream=True)

# HTTP/TCP
results = model("tcp://192.168.1.100:8080", stream=True)

# YouTube
results = model("https://youtu.be/VIDEO_ID", stream=True)
```

### Multiple Streams

Create a `.streams` file with one URL per line:

```
# streams.txt
rtsp://camera1.local/stream
rtsp://camera2.local/stream
rtsp://camera3.local/stream
```

```python
# Process multiple streams with batch inference
results = model("streams.txt", stream=True)
# Batch size automatically set to number of streams
```

### Directory

```python
# All images in directory
results = model("path/to/images/", stream=True)

# Recursive glob
results = model("path/**/*.jpg", stream=True)
```

## Inference Arguments

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `source` | str/int | required | Input source (path, URL, webcam index) |
| `stream` | bool | False | Return generator (required for video/webcam) |
| `conf` | float | 0.25 | Confidence threshold |
| `iou` | float | 0.7 | NMS IoU threshold |
| `imgsz` | int/tuple | 640 | Input size (640 or (640, 480)) |
| `device` | str | auto | Device ("cuda:0", "cpu", "mps") |

### Performance Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `half` | bool | False | FP16 inference (GPU only) |
| `batch` | int | 1 | Batch size for directory/video |
| `vid_stride` | int | 1 | Video frame stride |
| `stream_buffer` | bool | False | Buffer frames (True) or drop (False) |
| `max_det` | int | 300 | Max detections per frame |

### Filtering Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `classes` | list | None | Filter by class IDs [0, 1, 2] |
| `agnostic_nms` | bool | False | Class-agnostic NMS |

### Output Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `save` | bool | False | Save annotated images/video |
| `save_txt` | bool | False | Save results as .txt |
| `save_crop` | bool | False | Save cropped detections |
| `show` | bool | False | Display results in window |
| `project` | str | None | Save directory |
| `name` | str | None | Subdirectory name |

### Mask-Specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `retina_masks` | bool | False | High-res masks (original image size) |

## Common Configurations

### Realtime Webcam (Low Latency)

```python
results = model(
    0,                    # Webcam
    stream=True,          # Generator mode
    stream_buffer=False,  # Drop frames if slow
    vid_stride=1,         # Every frame
    half=True,            # FP16 on GPU
    imgsz=640,            # Standard size
    conf=0.3              # Slightly higher confidence
)
```

### High Accuracy

```python
results = model(
    source,
    stream=True,
    conf=0.1,             # Lower confidence = more detections
    iou=0.5,              # Lower IoU = fewer merged boxes
    imgsz=1280,           # Larger input
    retina_masks=True,    # High-res masks
    augment=True          # Test-time augmentation
)
```

### Multi-stream Processing

```python
# Process 4 cameras at once
results = model(
    "cameras.streams",    # File with 4 URLs
    stream=True,
    batch=4,              # Auto-set from stream count
    vid_stride=2,         # Process every 2nd frame per stream
    half=True
)
```

### Batch Image Processing

```python
results = model(
    "images/*.jpg",
    stream=True,
    batch=8,              # Process 8 images at once
    save=True,            # Save results
    project="runs/segment"
)
```

## Streaming Mode Details

### When to Use `stream=True`

- **Videos**: Always use streaming to avoid OOM
- **Webcams**: Always streaming
- **Large image sets**: Recommended
- **Single image**: Not needed

### Stream Buffer Behavior

```python
# stream_buffer=False (default) - Realtime mode
# Old frames dropped if inference is slower than input
for result in model(0, stream=True, stream_buffer=False):
    # Always processes most recent frame
    pass

# stream_buffer=True - Process all frames
# Frames queued, may cause latency buildup
for result in model(0, stream=True, stream_buffer=True):
    # Processes every frame in order
    pass
```

### Memory Efficiency

```python
# Without streaming - loads all results into memory
results = model("long_video.mp4")  # OOM risk!

# With streaming - processes one frame at a time
for result in model("long_video.mp4", stream=True):
    # Memory efficient
    process_frame(result)
```

## Device Configuration

```python
# Auto-select best available
model = YOLO("yolo11n-seg.pt")
results = model(source)  # Uses CUDA if available

# Specific GPU
results = model(source, device="cuda:0")
results = model(source, device="cuda:1")  # Second GPU

# CPU only
results = model(source, device="cpu")

# Apple Silicon
results = model(source, device="mps")

# Multiple GPUs (export to TensorRT first)
model.export(format="engine", device=[0, 1])
```

## Further Reading

For complete inference documentation, see:
- `ultralytics-docs/modes/predict.md` - Full predict mode reference
- `ultralytics-docs/modes/track.md` - Object tracking
- `ultralytics-docs/usage/cfg.md` - All configuration options
