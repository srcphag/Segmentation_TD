# Results and Masks Reference

Complete reference for working with YOLO11 segmentation results and mask objects.

## Results Object

Each inference returns a list of `Results` objects (one per image/frame).

### Basic Access

```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
results = model("image.jpg")

result = results[0]  # First image result

# Key attributes
result.orig_img      # Original image (numpy array)
result.orig_shape    # Original image shape (H, W)
result.boxes         # Bounding boxes
result.masks         # Segmentation masks
result.names         # Class names dict {0: 'person', 1: 'bicycle', ...}
result.path          # Image path
result.speed         # Inference timing dict
```

### Results Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `orig_img` | ndarray | Original image (H, W, C) BGR |
| `orig_shape` | tuple | Original (height, width) |
| `boxes` | Boxes | Bounding box detections |
| `masks` | Masks | Segmentation masks |
| `names` | dict | Class ID to name mapping |
| `path` | str | Source path |
| `speed` | dict | Timing: preprocess, inference, postprocess (ms) |
| `save_dir` | str | Output save directory |

### Results Methods

```python
# Visualization
annotated = result.plot()           # Returns annotated image (numpy)
result.show()                       # Display in window
result.save(filename="result.jpg")  # Save to file

# Export
result.to_json()    # JSON string
result.to_csv()     # CSV string

# Device transfer
result.cpu()        # Move tensors to CPU
result.cuda()       # Move tensors to GPU
result.numpy()      # Convert to numpy
```

## Boxes Object

Bounding boxes for detected objects.

### Accessing Boxes

```python
boxes = result.boxes

# Box coordinates
boxes.xyxy      # (N, 4) x1, y1, x2, y2 format
boxes.xywh      # (N, 4) x_center, y_center, width, height
boxes.xyxyn     # Normalized xyxy (0-1)
boxes.xywhn     # Normalized xywh (0-1)

# Class and confidence
boxes.cls       # (N,) class indices
boxes.conf      # (N,) confidence scores

# Track IDs (if tracking)
boxes.id        # (N,) track IDs or None
boxes.is_track  # True if tracking enabled
```

### Iterating Over Boxes

```python
for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    confidence = box.conf[0].item()
    class_id = int(box.cls[0])
    class_name = result.names[class_id]

    print(f"{class_name}: {confidence:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

## Masks Object

Segmentation masks for detected objects.

### Mask Formats

```python
masks = result.masks

# Polygon format - list of (N, 2) arrays
# Each array contains x, y coordinates for one object's contour
masks.xy        # Pixel coordinates
masks.xyn       # Normalized coordinates (0-1)

# Tensor format - (num_objects, H, W)
masks.data      # Binary mask tensor

# Shape info
masks.shape     # (num_objects, mask_H, mask_W)
masks.orig_shape  # Original image shape
```

### Working with Polygon Masks

```python
# Get polygon for each detected object
for i, polygon in enumerate(masks.xy):
    # polygon is numpy array of shape (N, 2)
    # Contains x, y coordinates of mask contour

    # Draw on image with OpenCV
    import cv2
    import numpy as np

    pts = polygon.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Fill polygon
    cv2.fillPoly(image, [pts], color=(0, 255, 0, 128))
```

### Working with Binary Masks

```python
# Get binary mask tensor
mask_tensor = masks.data  # Shape: (num_objects, H, W)

# Process each object's mask
for i, mask in enumerate(mask_tensor):
    binary_mask = mask.cpu().numpy()  # Shape: (H, W)

    # Mask is at inference resolution, not original
    # Use retina_masks=True for original resolution

    # Find pixels belonging to this object
    object_pixels = np.where(binary_mask > 0.5)

    # Apply mask to extract object region
    masked_region = image.copy()
    masked_region[binary_mask < 0.5] = 0
```

### High-Resolution Masks

```python
# Default: masks at inference resolution (e.g., 640x640)
results = model("image.jpg")
masks.data.shape  # (N, 160, 160) or similar

# High-res: masks at original image resolution
results = model("image.jpg", retina_masks=True)
masks.data.shape  # (N, H, W) matching original image
```

### Resizing Masks to Original Size

```python
import cv2
import numpy as np

# Get mask at inference resolution
mask = masks.data[0].cpu().numpy()

# Resize to original image size
orig_h, orig_w = result.orig_shape
mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

# Threshold to binary
mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
```

## Common Operations

### Extract Object with Mask

```python
import cv2
import numpy as np

result = results[0]
orig_img = result.orig_img.copy()

for i, (box, mask) in enumerate(zip(result.boxes, result.masks.data)):
    # Get mask at original resolution
    mask_np = mask.cpu().numpy()
    mask_resized = cv2.resize(mask_np, (orig_img.shape[1], orig_img.shape[0]))
    mask_binary = (mask_resized > 0.5).astype(np.uint8)

    # Create RGBA image with transparency
    rgba = cv2.cvtColor(orig_img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask_binary * 255

    # Or extract just the object pixels
    object_only = orig_img.copy()
    object_only[mask_binary == 0] = 0

    cv2.imwrite(f"object_{i}.png", object_only)
```

### Get Mask for Specific Class

```python
# Extract masks for specific class (e.g., person = 0)
target_class = 0

for i, (box, mask) in enumerate(zip(result.boxes, result.masks.data)):
    class_id = int(box.cls[0])
    if class_id == target_class:
        person_mask = mask.cpu().numpy()
        # Process person mask
```

### Combine All Masks

```python
import numpy as np

# Combine all object masks into one
if result.masks is not None:
    all_masks = result.masks.data.cpu().numpy()  # (N, H, W)
    combined = np.any(all_masks > 0.5, axis=0)   # (H, W)
```

### Calculate Mask Area

```python
for mask in result.masks.data:
    mask_np = mask.cpu().numpy()
    area_pixels = np.sum(mask_np > 0.5)
    total_pixels = mask_np.shape[0] * mask_np.shape[1]
    area_percentage = area_pixels / total_pixels * 100
    print(f"Object covers {area_percentage:.1f}% of frame")
```

### Find Mask Centroid

```python
import numpy as np

for mask in result.masks.data:
    mask_np = mask.cpu().numpy()
    y_indices, x_indices = np.where(mask_np > 0.5)
    if len(x_indices) > 0:
        centroid_x = np.mean(x_indices)
        centroid_y = np.mean(y_indices)
        print(f"Centroid: ({centroid_x:.0f}, {centroid_y:.0f})")
```

## Visualization Options

### plot() Method Arguments

```python
annotated = result.plot(
    conf=True,          # Show confidence scores
    line_width=2,       # Box line thickness
    font_size=12,       # Label font size
    labels=True,        # Show class labels
    boxes=True,         # Draw bounding boxes
    masks=True,         # Draw segmentation masks
    probs=False,        # Show classification probs (not for seg)
    show=False,         # Display window
    save=False,         # Save to file
    filename=None       # Save filename
)
```

### Custom Visualization

```python
import cv2
import numpy as np

image = result.orig_img.copy()

# Draw masks with transparency
if result.masks is not None:
    for mask in result.masks.data:
        mask_np = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))

        # Create colored overlay
        color = np.random.randint(0, 255, 3).tolist()
        overlay = image.copy()
        overlay[mask_resized > 0.5] = color

        # Blend with original
        alpha = 0.4
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# Draw boxes
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

## Further Reading

For complete results documentation, see:
- `ultralytics-docs/reference/engine/results.md` - Full Results API
- `ultralytics-docs/modes/predict.md` - Predict mode details
