# 06 - Command Line

The CLI is a **secondary** interface for batch / offline / testing work. The primary workflow is the Script TOP (see [03](03-touchdesigner-setup.md)).

## `main.py`

Entry point for webcam (realtime, YOLO only) and single-image modes.

### Image mode

```bash
python main.py --image photo.jpg                    # YOLO
python main.py --backend sam --image photo.jpg      # SAM 2 AMG (segment everything)
python main.py --backend sam3 --image photo.jpg --text "person,car"
python main.py --backend sam --image photo.jpg --points 200,300
python main.py --backend sam --image photo.jpg --bbox 100,100,300,300
python main.py --image photo.jpg --output results/
```

### Webcam mode (YOLO only)

```bash
python main.py                    # interactive camera selection
python main.py --camera 0         # specific camera
python main.py --list-cameras
python main.py --model yolo11s-seg.pt --conf 0.5 --boxes --labels
```

SAM 2 / SAM 3 are rejected in webcam mode (`~1000x slower than YOLO`); the CLI prints image-mode hints instead.

### Common flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--backend` | `yolo` | `yolo`, `sam`, `sam3` |
| `--model` | auto | Model path |
| `--device` | auto | `cuda:0`, `mps`, `cpu` |
| `--conf` | 0.25 | Confidence threshold |
| `--iou` | 0.7 | NMS IoU |
| `--imgsz` | 640 | Input size |
| `--image` | - | Enable image mode |
| `--output` | `output` | Output directory |
| `--no-individual-masks` | off | Skip per-object mask PNGs |
| `--no-composite` | off | Skip composite overlay |
| `--no-metadata` | off | Skip metadata JSON |
| `--text` | - | SAM 3 text prompts |
| `--points` | - | SAM 2 point prompts |
| `--bbox` | - | Box prompt |
| `--points-per-side` | 16 | AMG grid density |
| `--nms-thresh` | 0.7 | AMG duplicate removal |

Exit codes: `0` success, `1` error.

## `td_segment.py`

Legacy subprocess wrapper: TouchDesigner (or any process) calls this script, which prints a JSON status line to stdout. This is the **old** integration path superseded by the Script TOP, kept for compatibility.

```bash
python td_segment.py --image input.jpg --output output/ --backend yolo --run-id 1234abcd
```

On success:

```json
{"status": "complete", "output_dir": "output/1234abcd", "run_id": "1234abcd"}
```

On failure:

```json
{"status": "error", "message": "Image not found: input.jpg"}
```

## Image Output Structure

Image mode writes into a UUID subdirectory under `--output`:

```
output/<run_id>/
├── done.json                 # Completion marker + manifest
├── progress.json             # Progress updates (written during SAM)
├── <stem>_annotated.jpg      # Annotated visualization
├── <stem>_composite.png      # All masks composited
├── <stem>_mask_0_<class>.png # Individual binary masks
└── <stem>_metadata.json      # Detections, prompts, timing
```