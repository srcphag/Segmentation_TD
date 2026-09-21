# 03 - TouchDesigner Setup

The project's primary interface is a **Script TOP** driven by `td_script_top.py`. There is no `.tox` component and no subprocess: inference runs inside TD on the Script TOP's main thread.

## Quick Start

1. Open `SegmentationTD.toe` (the ready-made project), or create a new project.
2. Create a **Script TOP** (`Top COMP` palette → Script TOP).
3. Open its Python editor and paste the contents of `td_script_top.py`, or set the DAT reference so the callbacks resolve to this module.
4. Press **Setup Parameters** on the Script TOP - this calls `onSetupParameters` and creates the **Segmentation** page.
5. Wire any TOP (VideoDeviceIn, MovieFileIn, Render, Noise, ...) into the Script TOP's input.
6. Select a **Backend**, then pulse **Reload Model**.
7. The Script TOP's output now shows the annotated result.

> **Important:** Before anything works, TD must be able to import `ultralytics`/`torch`/`numpy`. Follow [02-installation](02-installation.md) Step 4. The first model load blocks the UI (torch import + model load) - this is normal.

## Script TOP Setup Details

`td_script_top.py` hardcodes a few project paths at the top - edit them if your checkout is elsewhere:

```python
PROJECT_DIR = r"D:\Dev\segment-everything-td"
ENV_SITE_PACKAGES = PROJECT_DIR + r"\.env\Lib\site-packages"
MODELS_DIR = PROJECT_DIR + r"\models"
```

- `ENV_SITE_PACKAGES` is appended to `sys.path` so TD can import the venv's packages even without the Python Module Path preference.
- Bare model filenames are resolved against `MODELS_DIR` (e.g. `yolo11n-seg.pt` → `models/yolo11n-seg.pt`).

## Segmentation Page Parameters

`onSetupParameters` creates the **Segmentation** custom page with:

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `Backend` | Menu | `yolo` | `yolo`, `sam`, or `sam3` |
| `Model` | File | empty | `.pt` file; empty = auto per backend/device |
| `Device` | Menu | `auto` | `auto`, `cpu`, `cuda:0` |
| `Confidence` | Float | 0.25 | Detection confidence |
| `Iou` | Float | 0.7 | NMS IoU threshold |
| `Imgsize` | Int | 640 | Inference image size |
| `Maskalpha` | Float | 0.5 | Overlay alpha (SAM / AMG) |
| `Showboxes` | Toggle | off | YOLO bounding boxes |
| `Showlabels` | Toggle | off | YOLO class labels |
| `Maskonly` | Toggle | off | White-on-black mask output |
| `Autorun` | Toggle | on | SAM/AMG runs every frame if on |
| `Textprompts` | Str | `person` | SAM 3 text prompts (`person, car`) |
| `Points` | Str | empty | SAM 2 points (`x1,y1;x2,y2`) |
| `Bbox` | Str | empty | SAM 2/3 box (`x1,y1,x2,y2`) |
| `Amgpoints` | Int | 16 | AMG grid points per side |
| `Reloadmodel` | Pulse | - | Reload the model (e.g. after param change) |
| `Runsam` | Pulse | - | Run SAM/AMG now (when Auto Run is off) |

> Note: `Backend`, `Device`, and `Showboxes`/`Showlabels` use menu names directly as labels, so `par.Backend.eval()` always returns `yolo` / `sam` / `sam3` (see `_append_menu` in `td_script_top.py`).

## Output Resolution

Set the Script TOP's **Common** page > **Output Resolution** to follow the input (default). `onCook` copies a numpy array sized to the input TOP, so a mismatch will misalign the result.

## Behavior Summary

- **YOLO**: runs on every cook (realtime).
- **SAM 2 / SAM 3**: with **Auto Run** ON they run every cook (slow). With Auto Run OFF they run only when **Run SAM** is pulsed; the last annotated result is re-output from cache otherwise.
- **Reload Model** forces a fresh model load even if parameters didn't change (e.g. after downloading a new model file).
- If the model fails to load, the input passes through and a Script Error is flagged on the OP.