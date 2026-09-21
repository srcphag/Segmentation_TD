# 01 - Overview

**Segment Everything TD** runs YOLO11 / SAM 2 / SAM 3 segmentation inference **directly inside TouchDesigner** using a single Script TOP. There is no subprocess, no external window, and no file round-trip: a TOP goes in, an annotated TOP comes out, every frame.

## Primary Workflow: TouchDesigner Script TOP

The heart of the project is `td_script_top.py`, a TouchDesigner Script TOP callback module. Drop it onto a Script TOP, press **Setup Parameters**, wire any TOP (VideoDeviceIn, MovieFileIn, Render, ...) into the input, and the TOP cooks the segmentation live:

- **YOLO11** - runs every frame (realtime, 30-100+ FPS with the nano model)
- **SAM 2 / SAM 3** - slow, so they run on demand via a **Run SAM** pulse (or per-frame if **Auto Run** is on); the last result is cached between runs

The Script TOP exposes all controls as custom parameters on a **Segmentation** page: backend, model file, device, confidence, IoU, image size, mask alpha, show boxes/labels, mask-only output, text prompts, points, bbox, and AMG grid density.

## Multi-Backend Design

All three backends share one interface (`BaseDetector` / the `_infer` dispatcher in the Script TOP):

| Backend | Library | Realtime in TOP | Prompts | Best For |
|---------|---------|-----------------|---------|----------|
| **YOLO11** | Ultralytics | Yes (every cook) | None (auto-detect) | Live video, quick looks |
| **SAM 2** | Ultralytics SAM | No - on demand | Points, boxes, AMG grid | Zero-shot segmentation |
| **SAM 3** | Ultralytics SAM | No - on demand | Text, bbox exemplars, AMG grid | Concept segmentation |

## Modes in the TOP

1. **YOLO realtime** - default backend; runs on every cook.
2. **SAM 2 prompted** - point / bbox prompts via the `Points` and `BBox` parameters.
3. **SAM 3 semantic** - text prompts (`Text Prompts` parameter) or a bbox exemplar.
4. **AMG (segment everything)** - SAM 2 / SAM 3 with no prompts runs a point grid + NMS to extract every object (see [08](08-automatic-mask-generation.md)).
5. **Mask Only** - white-on-black mask output instead of the annotated overlay.

## Supporting CLI

A Python CLI (`main.py`) and a TouchDesigner subprocess wrapper (`td_segment.py`) remain available for batch / offline use, but they are **secondary** - the project's focus is the Script TOP. See [06](06-command-line.md).

## Layout

```
td_script_top.py          Script TOP callbacks (primary entry point)
SegmentationTD.toe        Ready-made TD project with the Script TOP wired up
src/                      Python package backing both the TOP and the CLI
├── config.py             Device/model auto-detection
├── detectors/            YOLO / SAM 2 / SAM 3 + AMG implementations
│   ├── base.py           BaseDetector ABC + DetectionResult
│   ├── yolo_detector.py
│   ├── sam2_detector.py
│   ├── sam3_detector.py
│   ├── amg.py            Point grid + NMS utilities
│   └── factory.py        create_detector(config)
├── app.py                Realtime webcam app (CLI)
├── camera.py             Webcam handling (CLI)
├── visualizer.py         FPS/overlay (CLI)
└── image_processor.py    Single-image processor (CLI)
main.py                   CLI entry point
td_segment.py             Subprocess wrapper for TD (legacy path)
models/                   Downloaded model files (gitignored)
```

## Documentation Index

| Doc | Topic |
|-----|-------|
| [README](../README.md) | Quick start |
| [01-overview](01-overview.md) | You are here |
| [02-installation](02-installation.md) | Environment setup |
| [03-touchdesigner-setup](03-touchdesigner-setup.md) | Script TOP setup & parameters |
| [04-script-top-inference](04-script-top-inference.md) | How inference works in the TOP |
| [05-backends](05-backends.md) | YOLO / SAM 2 / SAM 3 modes |
| [06-command-line](06-command-line.md) | CLI & subprocess wrapper |
| [07-configuration](07-configuration.md) | Config & auto-detection |
| [08-automatic-mask-generation](08-automatic-mask-generation.md) | Segment-everything mode |
| [09-troubleshooting](09-troubleshooting.md) | Common issues |