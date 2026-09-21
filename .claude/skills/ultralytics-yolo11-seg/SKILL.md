---
name: ultralytics-yolo11-seg
description: YOLO11 instance segmentation for realtime video analysis. Use when implementing object segmentation on video streams, webcams, or images. Covers model selection (yolo11n-seg to yolo11x-seg), streaming inference with stream=True, mask extraction, and object tracking with persist=True.
---

# YOLO11 Instance Segmentation

Ultralytics YOLO11 provides state-of-the-art instance segmentation for realtime applications. This skill covers model selection, streaming inference, mask extraction, and object tracking.

## Quick Start

### Basic Segmentation

```python
from ultralytics import YOLO

# Load a pretrained segmentation model
model = YOLO("yolo11n-seg.pt")

# Run inference on an image
results = model("image.jpg")

# Access segmentation masks
for result in results:
    masks = result.masks  # Masks object

    # Polygon format (list of xy coordinates per object)
    polygons = masks.xy

    # Normalized polygon format
    polygons_normalized = masks.xyn

    # Binary mask tensor (num_objects x H x W)
    mask_data = masks.data
```

### Realtime Video Segmentation

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-seg.pt")
cap = cv2.VideoCapture(0)  # Webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run segmentation
    results = model(frame)

    # Visualize results
    annotated = results[0].plot()
    cv2.imshow("Segmentation", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Streaming Mode (Memory Efficient)

```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

# Use stream=True for memory-efficient video processing
for result in model("video.mp4", stream=True):
    masks = result.masks
    # Process each frame without loading all into memory
```

## Model Selection

YOLO11-seg models come in 5 sizes, trading off speed vs accuracy:

| Model | Size | mAP (mask) | Speed GPU (ms) | Speed CPU (ms) | Use Case |
|-------|------|------------|----------------|----------------|----------|
| yolo11n-seg | 2.9M | 32.0 | 1.8 | 65.9 | Edge devices, mobile |
| yolo11s-seg | 10.1M | 37.8 | 2.9 | 117.6 | Balanced speed/accuracy |
| yolo11m-seg | 22.4M | 41.5 | 6.3 | 281.6 | Standard production |
| yolo11l-seg | 27.6M | 42.9 | 7.8 | 344.2 | High accuracy |
| yolo11x-seg | 62.1M | 43.8 | 15.8 | 664.5 | Maximum accuracy |

**Recommendations:**
- **Realtime webcam**: `yolo11n-seg` or `yolo11s-seg`
- **Edge devices (Jetson)**: `yolo11n-seg`
- **Cloud/GPU processing**: `yolo11m-seg` or higher
- **Accuracy-critical**: `yolo11x-seg`

## Realtime Inference

### Input Sources

```python
# Webcam
results = model(0, stream=True)

# Video file
results = model("video.mp4", stream=True)

# RTSP/RTMP stream
results = model("rtsp://192.168.1.100:554/stream", stream=True)

# YouTube
results = model("https://youtu.be/VIDEO_ID", stream=True)

# Multiple streams (batch processing)
results = model("streams.txt", stream=True)  # File with one URL per line
```

### Key Inference Arguments

```python
results = model(
    source,
    stream=True,        # Memory-efficient generator (REQUIRED for video)
    conf=0.25,          # Confidence threshold
    iou=0.7,            # NMS IoU threshold
    imgsz=640,          # Input image size
    device="cuda:0",    # GPU device
    half=True,          # FP16 inference (faster on GPU)
    max_det=300,        # Max detections per frame
    vid_stride=1,       # Frame stride (skip frames)
    stream_buffer=False # False=drop old frames (realtime), True=queue all
)
```

### Optimizing for Realtime

```python
# For lowest latency realtime applications:
model = YOLO("yolo11n-seg.pt")
results = model(
    0,  # webcam
    stream=True,
    stream_buffer=False,  # Drop frames if processing is slow
    vid_stride=2,         # Process every 2nd frame
    half=True,            # FP16 on GPU
    imgsz=320             # Smaller input for speed
)
```

## Working with Masks

### Mask Formats

```python
result = results[0]
masks = result.masks

# Polygon coordinates (pixel values)
for polygon in masks.xy:
    # polygon is ndarray of shape (N, 2) - x,y coordinates
    pass

# Normalized polygon coordinates (0-1)
for polygon in masks.xyn:
    pass

# Binary mask tensor
mask_tensor = masks.data  # Shape: (num_objects, H, W)

# Access individual mask
for i, mask in enumerate(mask_tensor):
    binary_mask = mask.cpu().numpy()  # Shape: (H, W)
```

### Visualizing Masks

```python
# Built-in visualization
annotated_frame = result.plot()

# With options
annotated_frame = result.plot(
    masks=True,       # Show masks
    boxes=True,       # Show bounding boxes
    labels=True,      # Show class labels
    conf=True         # Show confidence
)
```

### Extracting Mask for Specific Class

```python
# Get masks for specific class (e.g., class 0 = person)
for i, (mask, box) in enumerate(zip(masks.data, result.boxes)):
    class_id = int(box.cls)
    if class_id == 0:  # person
        binary_mask = mask.cpu().numpy()
```

## Object Tracking

Combine segmentation with tracking to maintain object IDs across frames:

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-seg.pt")
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use track() instead of predict()
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        # Access track IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()
        masks = results[0].masks

        for track_id, mask in zip(track_ids, masks.data):
            print(f"Object {track_id}: mask shape {mask.shape}")

    annotated = results[0].plot()
    cv2.imshow("Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### Tracker Configuration

```python
# BoT-SORT (default) - better accuracy
results = model.track(frame, persist=True, tracker="botsort.yaml")

# ByteTrack - faster
results = model.track(frame, persist=True, tracker="bytetrack.yaml")
```

## Export for Deployment

Export models for faster inference on specific hardware:

```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

# ONNX (cross-platform)
model.export(format="onnx")

# TensorRT (NVIDIA GPU - fastest)
model.export(format="engine", half=True)

# CoreML (Apple devices)
model.export(format="coreml")

# Use exported model
model = YOLO("yolo11n-seg.engine")  # TensorRT
results = model(frame)
```

## Documentation References

For detailed documentation, see the markdown files in `ultralytics-docs/`:

- **Instance Segmentation**: `tasks/segment.md`
- **YOLO11 Models**: `models/yolo11.md`
- **Prediction Mode**: `modes/predict.md`
- **Object Tracking**: `modes/track.md`
- **Live Inference**: `guides/streamlit-live-inference.md`
- **Export Formats**: `modes/export.md`

## Additional Resources

- [references/models.md](references/models.md) - Detailed model comparison
- [references/inference.md](references/inference.md) - Full inference parameter reference
- [references/results.md](references/results.md) - Working with Results and Masks objects
- [examples/realtime-workflows.md](examples/realtime-workflows.md) - Complete workflow examples
