# Documentation

**Segment Everything TD** - multi-backend segmentation (YOLO11, SAM 2, SAM 3) that runs inference directly inside TouchDesigner via a Script TOP.

## Index

| Doc | Topic |
|-----|-------|
| [01 - Overview](01-overview.md) | Project intro, architecture, modes |
| [02 - Installation](02-installation.md) | venv, TD Python module path, models |
| [03 - TouchDesigner Setup](03-touchdesigner-setup.md) | Script TOP wiring & parameters |
| [04 - Script TOP Inference](04-script-top-inference.md) | How the TOP works internally |
| [05 - Backends](05-backends.md) | YOLO / SAM 2 / SAM 3 modes |
| [06 - Command Line](06-command-line.md) | `main.py` and `td_segment.py` |
| [07 - Configuration](07-configuration.md) | `Config` dataclass & auto-detection |
| [08 - Automatic Mask Generation](08-automatic-mask-generation.md) | Segment-everything mode |
| [09 - Troubleshooting](09-troubleshooting.md) | Common issues & fixes |

## Quick Start

1. Install dependencies into a Python 3.11 venv (`pip install -r requirements.txt`).
2. Point TouchDesigner at the venv's site-packages (or apply `TDPyEnvManagerContext.yaml`).
3. Open `SegmentationTD.toe`, or drop a Script TOP and paste `td_script_top.py`.
4. **Setup Parameters** → pick a **Backend** → **Reload Model** → wire a TOP into the input.

See [03 - TouchDesigner Setup](03-touchdesigner-setup.md) for details.