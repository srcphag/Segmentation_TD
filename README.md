# Segment Everything TD

A multi-backend segmentation toolkit designed for creative applications and TouchDesigner integration. Supports YOLO11 for realtime performance, Meta SAM 2 for zero-shot segmentation, and SAM 3 for text-based concept segmentation.

## Overview

**Segment Everything TD** bridges the gap between state-of-the-art segmentation models and creative tools like TouchDesigner. Whether you need realtime webcam segmentation, automatic mask generation, or text-prompted object detection, this toolkit provides a unified interface with seamless TD integration.

### Key Features

- **TouchDesigner Integration** - Custom COMP with subprocess-based processing, progress reporting, and dynamic mask loading
- **Multi-Backend Architecture** - Switch between YOLO11, SAM 2, and SAM 3 based on your needs
- **Automatic Mask Generation** - Segment everything in an image without prompts
- **Text Prompts** - Find all instances of a concept using natural language (SAM 3)
- **Visual Prompts** - Point and box prompts for precise segmentation (SAM 2)
- **Realtime Performance** - 30-100+ FPS webcam segmentation (YOLO11)
- **Progress Reporting** - Live progress updates during SAM processing

---

## Supported Models

| Model | Backend | Speed | Realtime | Prompts | Best For |
|-------|---------|-------|----------|---------|----------|
| **YOLO11** | `yolo` | ~30ms | Yes | None (auto-detect) | Webcam, fast batch processing |
| **SAM 2** | `sam` | 2-10 min | No | Points, boxes | Zero-shot with visual prompts |
| **SAM 3** | `sam3` | ~30ms GPU | No | Text, exemplars | Concept segmentation |

### YOLO11 Models

| Model | Size | GPU FPS | MPS FPS | CPU FPS |
|-------|------|---------|---------|---------|
| yolo11n-seg | 5.9 MB | 200+ | 30-60 | 10-15 |
| yolo11s-seg | 23 MB | 150+ | 25-40 | 8-12 |
| yolo11m-seg | 83 MB | 100+ | 15-25 | 5-8 |

### SAM 2 Models

| Model | Size | Use Case |
|-------|------|----------|
| sam2_t.pt | 78 MB | Fastest, edge devices |
| sam2_s.pt | ~100 MB | Balanced |
| sam2_b.pt | 162 MB | Standard |
| sam2_l.pt | ~200 MB | Highest accuracy |
| sam2.1_* | Same | Updated versions |

### SAM 3

| Model | Size | Features |
|-------|------|----------|
| sam3.pt | 3.4 GB | Text prompts, image exemplars, concept detection |

---

## Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/ehfazrezwan/segment-everything-td.git
cd segment-everything-td
```

### Step 2: Create Python Environment

```bash
# Create virtual environment
python -m venv env

# Activate it
source env/bin/activate      # macOS/Linux
# or: env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Note Your Python Path

You'll need the full path to the Python executable for TouchDesigner:

```bash
# macOS/Linux - run this to get the path:
which python
# Example output: /Users/yourname/segment-everything-td/env/bin/python

# Windows:
where python
# Example output: C:\Users\yourname\segment-everything-td\env\Scripts\python.exe
```

### Step 4: Verify Installation

```bash
# Test with webcam (YOLO)
python main.py --list-cameras

# Test with an image
python main.py --image path/to/test.jpg --output output/
```

---

## Quick Start (Command Line)

```bash
# Realtime webcam segmentation (YOLO)
python main.py

# Image segmentation with YOLO
python main.py --image photo.jpg

# Segment everything with SAM 2 (Automatic Mask Generation)
python main.py --backend sam --image photo.jpg

# Text-based segmentation with SAM 3
python main.py --backend sam3 --image photo.jpg --text "person,car,dog"
```

---

## TouchDesigner Setup

### Step 1: Load the SegmentationCOMP

Drag `SegmentationCOMP.tox` into your TouchDesigner project.

### Step 2: Configure Parameters

| Parameter | Description |
|-----------|-------------|
| **Input TOP** | The TOP to segment (drag any TOP here) |
| **Python Executable** | Path to your env's Python (from Setup Step 3) |
| **Project Folder** | Path to your segment-everything-td directory |
| **Backend** | `YOLO` (fast), `SAM2` (visual prompts), or `SAM3` (text prompts) |
| **Points Per Side** | AMG grid size for SAM (8=fast, 16=balanced, 32=thorough) |
| **Confidence** | Detection threshold (0.0-1.0) |

### Step 3: Run Segmentation

1. Connect any TOP to **Input TOP**
2. Click **Segment** to process
3. Watch the Textport for progress
4. Masks appear as OUT TOPs when complete
5. Click **Clear Masks** to remove generated TOPs

### Features

- **Non-blocking** - TD stays responsive during processing
- **Progress reporting** - See "Point 128/256 (50%)" during SAM AMG
- **Dynamic mask loading** - MovieFileIn + OUT TOPs created automatically
- **Clear Masks** - Remove all generated mask TOPs with one button

---

## Command Reference

### Webcam Mode

```bash
python main.py [options]

# Camera
--list-cameras          # Show available cameras
--camera N              # Use camera N directly

# Model
--model MODEL           # Model file (e.g., yolo11s-seg.pt)
--device DEVICE         # Force device (cuda:0, mps, cpu)
--conf THRESHOLD        # Confidence threshold (default: 0.25)

# Display
--boxes                 # Show bounding boxes
--labels                # Show class labels
--no-masks              # Hide segmentation masks
--no-fps                # Hide FPS overlay
```

### Image Mode

```bash
python main.py --image PATH [options]

# Backend selection
--backend yolo          # YOLO11 (default, fast)
--backend sam           # SAM 2 (zero-shot, visual prompts)
--backend sam3          # SAM 3 (text prompts)

# Output
--output DIR            # Output directory (default: output/)
--no-individual-masks   # Skip individual mask files
--no-composite          # Skip composite overlay
--no-metadata           # Skip metadata JSON

# SAM prompts
--text "a,b,c"          # Text prompts (SAM 3 only)
--points X,Y            # Point prompt
--bbox X1,Y1,X2,Y2      # Box prompt

# Automatic Mask Generation
--points-per-side N     # Grid size: 8 (fast), 16 (balanced), 32 (thorough)
--nms-thresh T          # NMS threshold for duplicate removal
```

### TouchDesigner CLI

```bash
python td_segment.py --image PATH --output DIR --run-id ID [options]

--backend BACKEND       # yolo, sam, or sam3
--points-per-side N     # AMG grid size
--conf THRESHOLD        # Confidence threshold
```

---

## Output Structure

Each segmentation run produces:

```
output/
└── {run_id}/
    ├── done.json                 # Completion marker (for TD polling)
    ├── progress.json             # Progress updates (for TD display)
    ├── input_annotated.jpg       # Annotated visualization
    ├── input_composite.png       # All masks composited
    ├── input_mask_0_person.png   # Individual binary masks
    ├── input_mask_1_car.png
    └── input_metadata.json       # Detection details
```

---

## Project Structure

```
segment-everything-td/
├── main.py                 # Main entry point
├── td_segment.py           # TouchDesigner CLI wrapper
├── SegmentationCOMP.tox    # TouchDesigner component
├── requirements.txt        # Python dependencies
├── models/                 # Downloaded model files
├── output/                 # Segmentation output
├── temp/                   # TD temp files
├── src/
│   ├── config.py           # Configuration & device detection
│   ├── detectors/
│   │   ├── base.py         # Abstract detector + progress callbacks
│   │   ├── yolo_detector.py
│   │   ├── sam2_detector.py
│   │   ├── sam3_detector.py
│   │   ├── amg.py          # Automatic Mask Generation
│   │   └── factory.py      # Detector factory
│   ├── camera.py           # Webcam handling
│   ├── visualizer.py       # Display utilities
│   ├── app.py              # Webcam application
│   └── image_processor.py  # Image mode + TD integration
└── docs/
    ├── 01-overview.md
    ├── 02-installation.md
    ├── 03-configuration.md
    ├── 04-running.md
    ├── 05-customization.md
    ├── 06-image-mode.md
    ├── 07-meta-sam-integration.md
    ├── 08-automatic-mask-generation.md
    └── 09-touchdesigner-integration.md
```

---

## Requirements

- Python 3.9+
- Webcam (for realtime mode)
- TouchDesigner 2023.10000+ (for TD integration)
- Optional: NVIDIA GPU with CUDA for faster inference
- Optional: Apple Silicon for MPS acceleration

---

## Roadmap

- [x] YOLO11 instance segmentation
- [x] Image segmentation with mask extraction
- [x] Meta SAM 2 integration (visual prompts)
- [x] Meta SAM 3 integration (text prompts)
- [x] Automatic Mask Generation (segment everything)
- [x] TouchDesigner integration with progress reporting
- [ ] Interactive point selection in TD (visual prompts for SAM 2)
- [ ] Text prompt input in TD (concept segmentation with SAM 3)
- [ ] Background replacement/removal
- [ ] NDI/Spout/Syphon direct output
- [ ] Multi-camera support
- [ ] Batch processing mode

---

## License

This project uses:
- **Ultralytics YOLO11** - AGPL-3.0
- **Meta SAM 2/3** - Apache 2.0

---

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO11 and SAM integration
- [Meta AI](https://ai.meta.com/) for SAM 2 and SAM 3 models
- [Derivative](https://derivative.ca/) for TouchDesigner
