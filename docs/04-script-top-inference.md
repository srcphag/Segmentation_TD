# 04 - Script TOP Inference

This document explains how `td_script_top.py` works internally so you can reason about behavior and extend it.

## Callbacks

The module implements the standard Script TOP callbacks:

| Callback | Role |
|----------|------|
| `onSetupParameters` | Creates the **Segmentation** page (menu/float/int/toggle/str/pulse params) |
| `onPulse` | Handles `Reloadmodel` and `Runsam` pulses |
| `onCook` | Main per-frame logic: read input TOP → infer → write output TOP |
| `onGetCookLevel` | Returns `CookLevel.AUTOMATIC` |

## Model Lifecycle

Models are **not** stored in `scriptOp.storage` because TD pickles storage on project save, and model objects (thread locks) cannot be pickled. Instead they live in a module-level registry keyed by the OP's path:

```python
_MODELS = {}   # key: op path -> {model, kind, key}

def _model_key(scriptOp):
    return getattr(scriptOp, "path", None) or id(scriptOp)
```

`_ensure_model` reloads a model only when its config "key" changes (backend, model, device, text, bbox, conf). `Reload Model` sets `storage["force_reload"]` to force a reload regardless.

The `kind` discriminates the actual interface:

| Backend | Prompts | kind | Object |
|---------|---------|------|--------|
| `yolo` | - | `yolo` | `ultralytics.YOLO` |
| `sam` | points/bbox | `sam` | `ultralytics.SAM` |
| `sam` | none | `sam` | `ultralytics.SAM` (→ AMG) |
| `sam3` | text/bbox | `sam3_semantic` | `SAM3SemanticPredictor` |
| `sam3` | none | `sam3_amg` | `ultralytics.SAM` (→ AMG) |

## Per-Cook Flow (`onCook`)

1. Read parameters into a plain dict (`_cfg`).
2. Resolve device (`auto` → `cuda:0` if available, else `cpu`).
3. Ensure the model is loaded/cached (`_ensure_model`).
4. Read the input TOP as a BGR numpy array (`_read_input_bgr`), flipping from TD's bottom/left origin to OpenCV's top/left, and splitting the alpha channel.
5. Decide SAM behavior:
   - If SAM backend and Auto Run is **off** and `run_sam` was not pulsed → re-output the cached result (or pass input through if none yet), then return.
6. Run inference (`_infer`), time it, and cache `annotated` + `masks`.
7. Optionally render **Mask Only** (white on black) from `masks`.
8. Write the output back (`_write_output`): BGR + alpha → RGBA, flipped back to TD origin, via `scriptOp.copyNumpyArray`.
9. On error: flag `scriptOp.addScriptError`, pass the input through, and clear the error on the next successful cook.

## Image Conversion

TD TOPs are RGBA with bottom/left origin; OpenCV/ultralytics expect BGR with top/left origin.

```python
_read_input_bgr(top):  numpyArray() -> flip vertically -> split alpha -> BGR
_write_output(scriptOp, bgr, alpha):  BGR -> RGBA -> flip vertically -> copyNumpyArray
```

The vertical flip on read is undone on write, so orientation stays consistent.

## Inference Paths

`_infer` dispatches on `kind`:

- `yolo` → `_infer_yolo`: `model(frame, conf, iou, imgsz, device)`, plot with boxes/labels per params.
- `sam` + prompts → `_infer_sam_prompted`: point prompts (`points`+`labels=[1]*n`) and/or `bboxes`.
- `sam` / `sam3_amg` no prompts → `_infer_sam_amg`: AMG point grid + NMS (see [08](08-automatic-mask-generation.md)).
- `sam3_semantic` → `_infer_sam3_semantic`: `set_image(frame)` then `predictor(text=...)` or `predictor(bboxes=[...])`.

## Storage Keys Used

| Key | Purpose |
|-----|---------|
| `storage["key"]` | Config hash used to detect model-reload-worthy changes |
| `storage["force_reload"]` | Set by Reload Model pulse |
| `storage["run_sam"]` | Set by Run SAM pulse |
| `storage["cached"]` | Last annotated frame (manual SAM mode) |
| `storage["cached_masks"]` | Last masks (for Mask Only in manual mode) |
| `storage["inference_ms"]` | Last inference time |
| `storage["error"]` | Pending error string |

## Performance Notes

- Inference runs on TD's main thread; heavy models drop framerate.
- For smooth realtime use `yolo11n-seg.pt` and a small input resolution.
- torch CUDA works inside TD but shares GPU/VRAM with TouchDesigner.
- SAM 2/SAM 3 AMG is slow (many grid points) - use Auto Run **off** + **Run SAM** pulses for interactive work.