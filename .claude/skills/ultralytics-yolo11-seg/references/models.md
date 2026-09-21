# YOLO11 Segmentation Model Reference

Comprehensive guide to YOLO11 segmentation model selection and performance characteristics.

## Model Variants

All YOLO11-seg models are pretrained on COCO dataset with 80 classes.

### Performance Comparison

| Model | Parameters | FLOPs | mAP (box) | mAP (mask) | GPU (ms) | CPU (ms) |
|-------|------------|-------|-----------|------------|----------|----------|
| yolo11n-seg | 2.9M | 9.7B | 38.9 | 32.0 | 1.8 | 65.9 |
| yolo11s-seg | 10.1M | 33.0B | 46.6 | 37.8 | 2.9 | 117.6 |
| yolo11m-seg | 22.4M | 113.2B | 51.5 | 41.5 | 6.3 | 281.6 |
| yolo11l-seg | 27.6M | 132.2B | 53.4 | 42.9 | 7.8 | 344.2 |
| yolo11x-seg | 62.1M | 296.4B | 54.7 | 43.8 | 15.8 | 664.5 |

**Notes:**
- GPU speed measured on NVIDIA T4 with TensorRT10
- CPU speed measured with ONNX Runtime
- mAP values on COCO val2017 at 640px input

## Model Selection Guide

### By Use Case

| Use Case | Recommended Model | Rationale |
|----------|-------------------|-----------|
| Realtime webcam (30 FPS) | yolo11n-seg | 1.8ms GPU = 550+ FPS headroom |
| Edge devices (Jetson Nano) | yolo11n-seg | Low memory, fast inference |
| Edge devices (Jetson Orin) | yolo11s-seg | Good balance |
| Smartphone inference | yolo11n-seg | Minimal compute |
| Cloud GPU processing | yolo11m-seg | Best balance |
| Accuracy-critical | yolo11x-seg | Highest mAP |
| Multi-stream processing | yolo11n-seg | Allows more concurrent streams |

### Speed vs Accuracy Tradeoff

```
Accuracy (mAP)
    ^
44% |                           x-seg
43% |                     l-seg
42% |               m-seg
38% |         s-seg
32% |   n-seg
    +---------------------------------> Speed (ms)
        2    4    6    8   10   12   14   16
```

## Loading Models

### Pretrained Models

```python
from ultralytics import YOLO

# Load pretrained (downloads automatically)
model = YOLO("yolo11n-seg.pt")
model = YOLO("yolo11s-seg.pt")
model = YOLO("yolo11m-seg.pt")
model = YOLO("yolo11l-seg.pt")
model = YOLO("yolo11x-seg.pt")
```

### Custom Trained Models

```python
# Load custom trained model
model = YOLO("path/to/best.pt")
```

### Exported Models

```python
# Load exported formats (faster inference)
model = YOLO("yolo11n-seg.onnx")        # ONNX
model = YOLO("yolo11n-seg.engine")      # TensorRT
model = YOLO("yolo11n-seg.mlpackage")   # CoreML
```

## Model Export

Export for optimized inference on specific hardware:

### ONNX (Cross-platform)

```python
model = YOLO("yolo11n-seg.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

### TensorRT (NVIDIA GPU)

```python
model = YOLO("yolo11n-seg.pt")
model.export(
    format="engine",
    half=True,          # FP16 for faster inference
    imgsz=640,
    workspace=4,        # GB of GPU memory for optimization
    dynamic=True        # Dynamic batch size
)
```

### CoreML (Apple)

```python
model = YOLO("yolo11n-seg.pt")
model.export(format="coreml", imgsz=640)
```

### OpenVINO (Intel)

```python
model = YOLO("yolo11n-seg.pt")
model.export(format="openvino", half=True)
```

### TFLite (Mobile/Edge)

```python
model = YOLO("yolo11n-seg.pt")
model.export(format="tflite", imgsz=320, int8=True)  # Quantized
```

## Export Formats Summary

| Format | Argument | Best For | Speed Gain |
|--------|----------|----------|------------|
| PyTorch | - | Development | Baseline |
| ONNX | `onnx` | Cross-platform | 1.5-2x |
| TensorRT | `engine` | NVIDIA GPU | 2-5x |
| CoreML | `coreml` | Apple devices | 2-3x |
| OpenVINO | `openvino` | Intel CPU/GPU | 2-3x |
| TFLite | `tflite` | Mobile/Edge | 2-4x |
| NCNN | `ncnn` | Mobile (Android) | 2-4x |

## Benchmarking

Compare models on your hardware:

```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
metrics = model.benchmark(imgsz=640, half=True, device="cuda:0")
```

## Memory Requirements

Approximate GPU memory usage during inference:

| Model | FP32 (MB) | FP16 (MB) |
|-------|-----------|-----------|
| yolo11n-seg | ~150 | ~100 |
| yolo11s-seg | ~300 | ~180 |
| yolo11m-seg | ~600 | ~350 |
| yolo11l-seg | ~750 | ~420 |
| yolo11x-seg | ~1500 | ~850 |

## Further Reading

For complete model documentation, see:
- `ultralytics-docs/models/yolo11.md` - Full YOLO11 documentation
- `ultralytics-docs/tasks/segment.md` - Segmentation task details
- `ultralytics-docs/modes/export.md` - Export format details
