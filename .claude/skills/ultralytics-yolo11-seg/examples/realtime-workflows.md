# Realtime Segmentation Workflows

Complete workflow examples for YOLO11 instance segmentation in realtime applications.

## 1. Basic Webcam Segmentation

```python
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n-seg.pt")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run segmentation
    results = model(frame)

    # Visualize
    annotated = results[0].plot()

    # Show FPS
    fps = 1000 / results[0].speed['inference']
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Segmentation", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 2. Video File Processing with Streaming

```python
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

# Input/output paths
input_video = "input.mp4"
output_video = "output.mp4"

# Get video properties
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Process with streaming (memory efficient)
for result in model(input_video, stream=True):
    annotated = result.plot()
    out.write(annotated)

out.release()
print(f"Saved to {output_video}")
```

## 3. RTSP Stream Processing

```python
import cv2
from ultralytics import YOLO
import time

model = YOLO("yolo11n-seg.pt")

# RTSP stream URL
rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream"

# Open stream with retry logic
def open_stream(url, max_retries=5):
    for i in range(max_retries):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            return cap
        print(f"Retry {i+1}/{max_retries}...")
        time.sleep(2)
    raise ConnectionError("Cannot connect to stream")

cap = open_stream(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream lost, reconnecting...")
        cap.release()
        time.sleep(1)
        cap = open_stream(rtsp_url)
        continue

    # Process frame
    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("RTSP Stream", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 4. Segmentation with Object Tracking

Track objects across frames while maintaining segmentation masks.

```python
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

model = YOLO("yolo11n-seg.pt")
cap = cv2.VideoCapture(0)

# Store track history
track_history = defaultdict(lambda: [])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use track() for persistent IDs
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes
        masks = results[0].masks

        track_ids = boxes.id.int().cpu().tolist()
        centers = boxes.xywh.cpu().numpy()[:, :2]  # x, y centers

        # Draw tracking trails
        annotated = results[0].plot()

        for track_id, center in zip(track_ids, centers):
            track = track_history[track_id]
            track.append(center)
            if len(track) > 30:
                track.pop(0)

            # Draw trail
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [points], False, (0, 255, 0), 2)
    else:
        annotated = results[0].plot()

    cv2.imshow("Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 5. Multi-Stream Processing

Process multiple camera feeds simultaneously.

```python
import cv2
from ultralytics import YOLO
import threading
from queue import Queue

model = YOLO("yolo11n-seg.pt")

# Stream sources
sources = [
    0,  # Webcam
    "rtsp://192.168.1.101:554/stream",
    "rtsp://192.168.1.102:554/stream",
]

# Thread-safe results
results_queue = Queue()

def process_stream(source_id, source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()
        results_queue.put((source_id, annotated))

    cap.release()

# Start threads
threads = []
for i, source in enumerate(sources):
    t = threading.Thread(target=process_stream, args=(i, source))
    t.daemon = True
    t.start()
    threads.append(t)

# Display results
windows = [f"Stream {i}" for i in range(len(sources))]
for name in windows:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

while True:
    try:
        source_id, frame = results_queue.get(timeout=1)
        cv2.imshow(windows[source_id], frame)
    except:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

## 6. Extract and Save Segmented Objects

```python
import cv2
import numpy as np
from ultralytics import YOLO
import os

model = YOLO("yolo11n-seg.pt")

# Create output directory
os.makedirs("extracted_objects", exist_ok=True)

cap = cv2.VideoCapture(0)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, retina_masks=True)  # High-res masks

    if results[0].masks is not None:
        for i, (box, mask) in enumerate(zip(results[0].boxes, results[0].masks.data)):
            # Get class info
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            conf = float(box.conf[0])

            # Get mask at original resolution
            mask_np = mask.cpu().numpy()

            # Create RGBA image with transparency
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = (mask_np * 255).astype(np.uint8)

            # Get bounding box for cropping
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop to object
            cropped = rgba[y1:y2, x1:x2]

            # Save
            filename = f"extracted_objects/{class_name}_{frame_count}_{i}.png"
            cv2.imwrite(filename, cropped)

    # Visualize
    annotated = results[0].plot()
    cv2.imshow("Segmentation", annotated)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 7. Background Replacement

Replace background using segmentation masks.

```python
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

# Load background image
background = cv2.imread("background.jpg")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize background to match frame
    bg = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    # Run segmentation
    results = model(frame, retina_masks=True, classes=[0])  # Only person

    if results[0].masks is not None:
        # Combine all person masks
        all_masks = results[0].masks.data.cpu().numpy()
        combined_mask = np.any(all_masks > 0.5, axis=0).astype(np.float32)

        # Smooth mask edges
        combined_mask = cv2.GaussianBlur(combined_mask, (7, 7), 0)

        # Expand mask to 3 channels
        mask_3ch = np.stack([combined_mask] * 3, axis=-1)

        # Composite: foreground where mask, background elsewhere
        output = (frame * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
    else:
        output = frame

    cv2.imshow("Background Replacement", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 8. Streamlit Live Inference

```python
# Save as app.py, run with: streamlit run app.py

from ultralytics import solutions

# Create inference interface
inf = solutions.Inference(
    model="yolo11n-seg.pt",
)

# Launch Streamlit app
inf.inference()
```

Or use CLI:

```bash
yolo solutions inference model=yolo11n-seg.pt
```

## 9. Optimized TensorRT Inference

```python
import cv2
from ultralytics import YOLO

# First, export to TensorRT (one-time)
model = YOLO("yolo11n-seg.pt")
model.export(format="engine", half=True, imgsz=640)

# Use exported model for inference
model = YOLO("yolo11n-seg.engine")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference is 2-5x faster with TensorRT
    results = model(frame)

    annotated = results[0].plot()
    fps = 1000 / results[0].speed['inference']
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("TensorRT Segmentation", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 10. Class-Specific Segmentation

Filter for specific object classes only.

```python
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

# COCO class IDs
# 0: person, 1: bicycle, 2: car, 3: motorcycle, ...
# Full list: https://docs.ultralytics.com/datasets/detect/coco/

# Only segment people and cars
target_classes = [0, 2]  # person, car

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Filter by class
    results = model(frame, classes=target_classes)

    annotated = results[0].plot()
    cv2.imshow("People & Cars", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 11. Mask Area Analysis

Analyze object coverage in frame.

```python
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    if results[0].masks is not None:
        total_pixels = frame.shape[0] * frame.shape[1]

        for i, (box, mask) in enumerate(zip(results[0].boxes, results[0].masks.data)):
            class_name = results[0].names[int(box.cls[0])]
            mask_np = mask.cpu().numpy()

            # Calculate area
            area_pixels = np.sum(mask_np > 0.5)
            area_pct = area_pixels / total_pixels * 100

            # Display info
            x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
            cv2.putText(annotated, f"{area_pct:.1f}%", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Area Analysis", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Tips for Realtime Performance

1. **Use smallest model that meets accuracy needs**: yolo11n-seg for most realtime use
2. **Enable FP16**: `half=True` on NVIDIA GPUs
3. **Use TensorRT export**: 2-5x speedup
4. **Reduce input size**: `imgsz=320` for fastest inference
5. **Skip frames**: `vid_stride=2` processes every 2nd frame
6. **Disable buffer**: `stream_buffer=False` for lowest latency
7. **Filter classes**: Only detect needed classes with `classes=[0,1]`

## Further Reading

For complete workflow documentation, see:
- `ultralytics-docs/modes/predict.md` - Prediction details
- `ultralytics-docs/modes/track.md` - Object tracking
- `ultralytics-docs/guides/streamlit-live-inference.md` - Streamlit setup
- `ultralytics-docs/guides/instance-segmentation-and-tracking.md` - Advanced tracking
