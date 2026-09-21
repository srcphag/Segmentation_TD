# Segment Everything TD

Run **YOLO11, SAM 2, and SAM 3** segmentation inference directly inside **TouchDesigner** with a single Script TOP - no subprocess, no external window, no file round-trip.

Wire any TOP (camera, movie, render) into the input and get an annotated output TOP with per-object masks, in realtime for YOLO and on-demand for SAM 2 / SAM 3.

## Key Features

- **Script TOP inference** - segmentation runs on TD's main thread via `td_script_top.py`, parameterized on a custom **Segmentation** page
- **Multi-backend** - switch `yolo` / `sam` / `sam3` from a menu parameter
- **YOLO11 realtime** - runs every frame (30-100+ FPS with the nano model)
- **SAM 2 prompts** - point and box prompts via parameters
- **SAM 3 text** - concept segmentation ("person, car", "person with red shirt")
- **Segment everything (AMG)** - point-grid + NMS automatic mask generation
- **Mask Only** - white-on-black mask output for downstream compositing
- **On-demand SAM** - slow backends run via a **Run SAM** pulse, cached between runs
- **CLI included** - `main.py` for webcam/image batch work; `td_segment.py` legacy subprocess wrapper

## Quick Start

1. **Install** into a Python 3.11 venv (matches TD 2023+):
   ```bash
   python -m venv .env
   .env\Scripts\activate            # Windows
   pip install -r requirements.txt
   ```
2. **Point TD at the venv** - Edit > Preferences > Python > *Python 64-bit Module Path* → `.env\Lib\site-packages`, or apply the bundled `TDPyEnvManagerContext.yaml` context. (See docs/02.)
3. **Open `SegmentationTD.toe`**, or create a Script TOP and paste in `td_script_top.py`.
4. **Setup Parameters** on the Script TOP → pick a **Backend** → **Reload Model**.
5. **Wire any TOP** into the Script TOP input. YOLO runs immediately; for SAM pulse **Run SAM** (leave **Auto Run** off).

Full docs: **[docs/README.md](docs/README.md)** - installation, Script TOP setup, backends, AMG, troubleshooting.

## Backends

| Backend | Realtime | Prompts | Best For |
|---------|----------|---------|----------|
| **YOLO11** | Yes (every cook) | None (auto-detect) | Live video |
| **SAM 2** | On demand | Points, box, AMG | Zero-shot masks |
| **SAM 3** | On demand | Text, box exemplar, AMG | Concept segmentation |

## Script TOP Parameters

| Parameter | Purpose |
|-----------|---------|
| `Backend` | `yolo` / `sam` / `sam3` |
| `Model` | `.pt` file (empty = auto per backend) |
| `Device` | `auto` / `cpu` / `cuda:0` |
| `Confidence` / `Iou` / `Imgsize` | Inference tuning |
| `Maskalpha` | Overlay opacity (SAM/AMG) |
| `Showboxes` / `Showlabels` | YOLO annotations |
| `Maskonly` | White-on-black mask output |
| `Autorun` | SAM runs every frame when on |
| `Textprompts` | SAM 3 concepts ("person, car") |
| `Points` / `Bbox` | SAM 2/3 visual prompts |
| `Amgpoints` | AMG grid density (8/16/32) |
| `Reloadmodel` | Force model reload |
| `Runsam` | Run SAM/AMG now |

## Command Line (secondary)

```bash
# Single image - YOLO
python main.py --image photo.jpg

# SAM 2 segment-everything
python main.py --backend sam --image photo.jpg

# SAM 3 text prompts
python main.py --backend sam3 --image photo.jpg --text "person,car"

# Realtime webcam (YOLO only)
python main.py --camera 0
```

## Requirements

- TouchDesigner 2023.10000+ (Python 3.11)
- Python 3.9+
- Optional: NVIDIA GPU with CUDA
- Models download automatically to `models/` (SAM 3 ~3.4 GB, may need Hugging Face approval)

## Project Structure

```
├── td_script_top.py          # Script TOP callbacks (primary)
├── SegmentationTD.toe        # Ready-made TD project
├── main.py                   # CLI entry point
├── td_segment.py             # Legacy subprocess wrapper
├── TDPyEnvManagerContext.yaml# TD Python Environment Manager context
├── src/                      # Python package (TOP + CLI)
│   ├── config.py             # Config & device auto-detection
│   ├── detectors/            # YOLO / SAM 2 / SAM 3 + AMG
│   └── ...
└── docs/                     # Documentation
```

## Documentation

- [01 - Overview](docs/01-overview.md)
- [02 - Installation](docs/02-installation.md)
- [03 - TouchDesigner Setup](docs/03-touchdesigner-setup.md)
- [04 - Script TOP Inference](docs/04-script-top-inference.md)
- [05 - Backends](docs/05-backends.md)
- [06 - Command Line](docs/06-command-line.md)
- [07 - Configuration](docs/07-configuration.md)
- [08 - Automatic Mask Generation](docs/08-automatic-mask-generation.md)
- [09 - Troubleshooting](docs/09-troubleshooting.md)

## License

- **Ultralytics YOLO11** - AGPL-3.0
- **Meta SAM 2 / SAM 3** - Apache 2.0