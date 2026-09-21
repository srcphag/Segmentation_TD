r"""
Segment Everything - Script TOP Callbacks

Runs the project's YOLO11 / SAM 2 / SAM 3 segmentation inference directly
inside TouchDesigner (single Script TOP, no subprocess).

me - this DAT
scriptOp - the OP which is cooking

SETUP
-----
1. TouchDesigner must be able to import ultralytics/torch/numpy. Either:
     - Edit > Preferences > Python > "Python 64-bit Module Path"
       -> D:\Dev\segment-everything-td\.env\Lib\site-packages
     - Or use a TD Python Environment Manager context (TDPyEnvManagerContext.yaml)
       pointing at this project's .env venv.
   or keep the ENV_SITE_PACKAGES fallback below (edit if your path differs).
   TD 2023+ uses Python 3.11, which matches this project's venv.

2. Wire any TOP (VideoDeviceIn, MovieFileIn, Render, ...) into the input.

3. Press 'Setup Parameters' on the Script TOP to create the parameters,
   pick a Backend, then pulse 'Reload Model'. First load takes a while
   (torch import + model load) and blocks the UI - this is normal.

4. YOLO runs every frame (realtime). SAM 2 / SAM 3 are slow, so they only
   run when you pulse 'Run SAM'; the last result is kept between runs.

NOTES
-----
- Inference runs on TD's main thread: heavy models will drop the framerate.
  For smooth realtime use the nano YOLO model (yolo11n-seg.pt) and/or a
  small input resolution.
- Set the Script TOP's Common page > Output Resolution to follow the input
  (default) so the copied numpy array matches the TOP resolution.
- torch CUDA works inside TD but shares the GPU/VRAM with TouchDesigner.
"""

from __future__ import annotations

import sys
import time
import numpy as np

# --- Path to this project / its venv (edit if needed) -----------------------
PROJECT_DIR = r"D:\Dev\segment-everything-td"
ENV_SITE_PACKAGES = PROJECT_DIR + r"\.env\Lib\site-packages"
MODELS_DIR = PROJECT_DIR + r"\models"

if ENV_SITE_PACKAGES not in sys.path:
    sys.path.append(ENV_SITE_PACKAGES)

# TouchDesigner builtins (debug, absTime) - provide fallbacks so this module
# can also be imported/tested in a plain Python interpreter.
try:
    debug  # noqa: F821
except NameError:
    def debug(msg):
        print(msg)


try:
    absTime  # noqa: F821
except NameError:
    class _AbsTime:
        frame = 0

    absTime = _AbsTime()

# Model registry: model objects are NOT picklable (they hold thread locks), so
# they must NOT go into scriptOp.storage - TD pickles storage on project save
# which raises "cannot pickle '_thread.lock' object". Keep them in module-level
# globals keyed by the Script TOP's path instead.
_MODELS = {}


def _model_key(scriptOp):
    """Stable key for the model registry (TD op path, falling back to id)."""
    return getattr(scriptOp, "path", None) or id(scriptOp)

# =============================================================================
# Parameter helpers
# =============================================================================


def _cfg(scriptOp):
    """Read all custom parameters into a plain dict."""
    p = scriptOp.par
    return {
        "backend": p.Backend.eval() if hasattr(p, "Backend") else "yolo",
        "model": str(p.Model.eval()) if hasattr(p, "Model") else "",
        "device": p.Device.eval() if hasattr(p, "Device") else "auto",
        "conf": float(p.Confidence.eval()) if hasattr(p, "Confidence") else 0.25,
        "iou": float(p.Iou.eval()) if hasattr(p, "Iou") else 0.7,
        "imgsz": int(p.Imgsize.eval()) if hasattr(p, "Imgsize") else 640,
        "alpha": float(p.Maskalpha.eval()) if hasattr(p, "Maskalpha") else 0.5,
        "show_boxes": bool(p.Showboxes.eval()) if hasattr(p, "Showboxes") else False,
        "show_labels": bool(p.Showlabels.eval()) if hasattr(p, "Showlabels") else False,
        "mask_only": bool(p.Maskonly.eval()) if hasattr(p, "Maskonly") else False,
        "autorun": bool(p.Autorun.eval()) if hasattr(p, "Autorun") else True,
        "text": _parse_texts(p.Textprompts.eval()) if hasattr(p, "Textprompts") else [],
        "points": _parse_points(p.Points.eval()) if hasattr(p, "Points") else [],
        "bbox": _parse_bbox(p.Bbox.eval()) if hasattr(p, "Bbox") else [],
        "amg_pps": int(p.Amgpoints.eval()) if hasattr(p, "Amgpoints") else 16,
    }


def _parse_texts(s):
    """'person, car' -> ['person', 'car']"""
    return [t.strip() for t in str(s).split(",") if t.strip()]


def _parse_points(s):
    """'100,200;300,400' -> [[100, 200], [300, 400]] (pixel coords)"""
    pts = []
    for chunk in str(s).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        xy = chunk.split(",")
        if len(xy) == 2:
            try:
                pts.append([float(xy[0]), float(xy[1])])
            except ValueError:
                pass
    return pts


def _parse_bbox(s):
    """'100,100,300,300' -> [100, 100, 300, 300] (x1, y1, x2, y2)"""
    parts = [v.strip() for v in str(s).split(",") if v.strip()]
    if len(parts) != 4:
        return []
    try:
        return [float(v) for v in parts]
    except ValueError:
        return []


def _resolve_device(device):
    if device == "auto":
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_model(cfg):
    """Resolve the model filename to an absolute path (defaults per backend)."""
    import os

    name = cfg["model"].strip()
    if not name:
        if cfg["backend"] == "sam3":
            name = "sam3.pt"
        elif cfg["backend"] == "sam":
            name = "sam2_l.pt"
        else:
            name = "yolo11n-seg.pt" if cfg["device"] == "cpu" else "yolo11s-seg.pt"
    if os.path.isabs(name):
        return name
    # Bare filename -> look in the project's models dir.
    if os.path.dirname(name) == "":
        return os.path.join(MODELS_DIR, name)
    # Relative path with a directory component -> use as given (project-relative).
    return os.path.abspath(os.path.join(PROJECT_DIR, name))


def _cfg_key(cfg):
    """Hashable key to detect parameter changes that require a model reload."""
    return (
        cfg["backend"],
        cfg["model"],
        cfg["device"],
        tuple(cfg["text"]),
        tuple(cfg["bbox"]),
        cfg["conf"],
    )


# =============================================================================
# Model management (cached in scriptOp.storage so it loads only once)
# =============================================================================


def _load_model(scriptOp, cfg):
    """Load (and cache) the model for the current parameter set."""
    model_path = _resolve_model(cfg)
    backend = cfg["backend"]

    if backend == "yolo":
        from ultralytics import YOLO

        model = YOLO(model_path)
        kind = "yolo"
    elif backend == "sam":
        from ultralytics import SAM

        model = SAM(model_path)
        kind = "sam"
    elif backend == "sam3":
        if cfg["text"] or cfg["bbox"]:
            from ultralytics.models.sam import SAM3SemanticPredictor

            overrides = dict(
                conf=cfg["conf"],
                task="segment",
                mode="predict",
                model=model_path,
                imgsz=cfg["imgsz"],
                half=cfg["device"].startswith("cuda"),
            )
            model = SAM3SemanticPredictor(overrides=overrides)
            kind = "sam3_semantic"
        else:
            from ultralytics import SAM

            model = SAM(model_path)
            kind = "sam3_amg"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    scriptOp.storage["key"] = _cfg_key(cfg)
    scriptOp.storage["error"] = ""
    _MODELS[_model_key(scriptOp)] = {
        "model": model,
        "kind": kind,
        "key": _cfg_key(cfg),
    }
    debug(f"[SegmentEverything] Loaded {backend} ({kind}): {model_path}")
    return model


def _ensure_model(scriptOp, cfg):
    """Return the cached model, loading/reloading if parameters changed."""
    key = _cfg_key(cfg)
    entry = _MODELS.get(_model_key(scriptOp))
    if (
        entry is None
        or entry.get("key") != key
        or scriptOp.storage.get("force_reload")
    ):
        scriptOp.storage["force_reload"] = False
        try:
            return _load_model(scriptOp, cfg)
        except Exception as e:
            _MODELS.pop(_model_key(scriptOp), None)
            scriptOp.storage["error"] = str(e)
            scriptOp.addScriptError(f"Model load failed: {e}")
            return None
    return entry["model"]


# =============================================================================
# AMG helpers (translated from src/detectors/amg.py - numpy only)
# =============================================================================


def _build_point_grid(points_per_side):
    """Grid of points normalized to [0, 1], shape (N, 2)."""
    offset = 1 / (2 * points_per_side)
    one_side = np.linspace(offset, 1 - offset, points_per_side)
    xs = np.tile(one_side[None, :], (points_per_side, 1))
    ys = np.tile(one_side[:, None], (1, points_per_side))
    return np.stack([xs.flatten(), ys.flatten()], axis=-1)


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_boxes(boxes, scores, iou_thresh):
    """Non-maximum suppression; returns list of indices to keep."""
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([_box_iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious < iou_thresh]
    return keep


def _resize_mask_nearest(mask, h, w):
    """Nearest-neighbor resize without cv2."""
    if mask.shape == (h, w):
        return mask
    yi = np.linspace(0, mask.shape[0] - 1, h).astype(np.int64)
    xi = np.linspace(0, mask.shape[1] - 1, w).astype(np.int64)
    return mask[np.ix_(yi, xi)]


def _mask_overlay(frame, masks, alpha=0.5):
    """Colored mask overlay (BGR frame). Same behavior as create_mask_overlay."""
    if masks is None or len(masks) == 0:
        return frame.copy()
    rng = np.random.RandomState(42)
    colors = rng.randint(0, 255, (len(masks), 3))
    h, w = frame.shape[:2]
    out = frame.astype(np.float32)
    for mask, color in zip(masks, colors):
        m = _resize_mask_nearest(mask, h, w) > 0.5
        out[m] = out[m] * (1 - alpha) + color.astype(np.float32) * alpha
    return out.astype(np.uint8)


def _mask_only_image(masks, h, w):
    """White mask over black background (BGR, 3 channels).

    Returns a (h, w, 3) uint8 image where mask pixels are white (255) and
    everything else is black (0). If no masks, returns an all-black frame.
    """
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if masks is not None and len(masks) > 0:
        for mask in masks:
            m = _resize_mask_nearest(mask, h, w) > 0.5
            out[m] = (255, 255, 255)
    return out


# =============================================================================
# Inference (translated from src/detectors/*.py)
# =============================================================================


def _infer_yolo(model, frame, cfg):
    """YOLO11 segmentation - realtime path (src/detectors/yolo_detector.py)."""
    results = model(
        frame,
        conf=cfg["conf"],
        iou=cfg["iou"],
        imgsz=cfg["imgsz"],
        device=cfg["device"],
        verbose=False,
    )
    result = results[0]
    annotated = result.plot(
        masks=True,
        boxes=cfg["show_boxes"],
        labels=cfg["show_labels"],
        conf=True,
        color_mode="instance",
    )
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    return annotated, masks


def _infer_sam_prompted(model, frame, cfg):
    """SAM 2 with point/box prompts (src/detectors/sam2_detector.py)."""
    kwargs = {"verbose": False}
    if cfg["points"]:
        kwargs["points"] = cfg["points"]
        kwargs["labels"] = [1] * len(cfg["points"])
    if cfg["bbox"]:
        kwargs["bboxes"] = [cfg["bbox"]]
    results = model(frame, **kwargs)
    result = results[0]
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    return result.plot(), masks


def _infer_sam_amg(model, frame, cfg, scriptOp=None):
    """SAM 2 / SAM 3 segment-everything via point grid + NMS (AMG)."""
    h, w = frame.shape[:2]
    points = _build_point_grid(cfg["amg_pps"]) * np.array([w, h])
    total = len(points)
    min_area = 100

    all_masks, all_boxes, all_scores = [], [], []
    for idx, pt in enumerate(points):
        try:
            results = model(frame, points=[[pt[0], pt[1]]], labels=[1], verbose=False)
            r = results[0]
            if r.masks is not None and len(r.masks) > 0:
                mask = r.masks.data[0].cpu().numpy()
                box = r.boxes.xyxy[0].cpu().numpy()
                area = int(mask.sum())
                if area > min_area:
                    all_masks.append(mask)
                    all_boxes.append(box)
                    all_scores.append(float(area))
        except Exception:
            continue

    masks = None
    if all_masks:
        boxes_arr = np.array(all_boxes)
        scores_arr = np.array(all_scores)
        keep = _nms_boxes(boxes_arr, scores_arr, 0.7)
        masks = np.array([all_masks[i] for i in keep])
        debug(f"[SegmentEverything] AMG: {len(masks)} masks after NMS ({total} points)")

    return _mask_overlay(frame, masks, alpha=cfg["alpha"]), masks


def _infer_sam3_semantic(predictor, frame, cfg):
    """SAM 3 text/exemplar prompts (src/detectors/sam3_detector.py)."""
    predictor.set_image(frame)
    if cfg["text"]:
        results = predictor(text=cfg["text"])
    elif cfg["bbox"]:
        results = predictor(bboxes=[cfg["bbox"]])
    else:
        raise ValueError("SAM 3 semantic mode requires text prompts or a bbox")
    result = results[0]
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    return result.plot(), masks


def _infer(scriptOp, model, frame, cfg):
    """Run inference, returning (annotated_frame, masks)."""
    entry = _MODELS.get(_model_key(scriptOp)) or {}
    kind = entry.get("kind")
    if kind == "yolo":
        return _infer_yolo(model, frame, cfg)
    if kind == "sam":
        if cfg["points"] or cfg["bbox"]:
            return _infer_sam_prompted(model, frame, cfg)
        return _infer_sam_amg(model, frame, cfg, scriptOp)
    if kind == "sam3_semantic":
        return _infer_sam3_semantic(model, frame, cfg)
    if kind == "sam3_amg":
        return _infer_sam_amg(model, frame, cfg, scriptOp)
    raise RuntimeError(f"Unknown model kind: {kind}")


# =============================================================================
# Image conversion (TD TOPs are RGBA; ultralytics expects BGR)
# =============================================================================


def _read_input_bgr(top):
    """Input TOP -> BGR uint8 (H, W, 3). Also returns the alpha channel.

    TD textures are bottom/left origin; OpenCV/ultralytics are top/left, so
    the array is flipped vertically on read (and flipped back on write).
    """
    arr = top.numpyArray()  # (H, W, 4) - TD bottom/left origin
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr[::-1])  # flip to top/left (OpenCV space)
    alpha = arr[:, :, 3] if arr.shape[2] >= 4 else np.full(arr.shape[:2], 255, np.uint8)
    bgr = arr[:, :, :3][:, :, ::-1].copy()
    return bgr, alpha


def _write_output(scriptOp, bgr, alpha):
    """BGR uint8 -> RGBA copyNumpyArray.

    Flipped back vertically (OpenCV top/left -> TD bottom/left) to undo the
    flip applied in _read_input_bgr, keeping orientation consistent.
    """
    rgba = np.dstack([bgr[:, :, ::-1], alpha]).astype(np.uint8)
    rgba = np.ascontiguousarray(rgba[::-1])  # flip back to TD space
    scriptOp.copyNumpyArray(rgba)


# =============================================================================
# Script TOP callbacks
# =============================================================================


def _append_menu(page, name, label, names, default):
    """Create a menu parameter and populate its dropdown items.

    appendMenu() itself does NOT take menu items: it only creates an empty
    menu (returned as a ParGroup). The items must be assigned afterwards via
    par.menuNames / par.menuLabels. `names` are also used as the labels so
    par.eval() always returns one of `names`.
    """
    result = page.appendMenu(name, label=label)
    # appendMenu returns a ParGroup; grab the first (only) Par.
    try:
        p = result[0]
    except TypeError:
        p = result

    names = [str(n) for n in names]
    p.menuNames = names
    p.menuLabels = names
    p.default = default
    p.val = default
    return p


# press 'Setup Parameters' in the OP to call this function to re-create
# the parameters.
def onSetupParameters(scriptOp: scriptTOP):
    """
    Called to setup custom parameters for the Script TOP.
    """
    page = scriptOp.appendCustomPage("Segmentation")

    # NOTE: TD builds differ widely in appendMenu keyword support, so menus
    # are created through a tolerant helper. Names are used as labels so
    # par.eval() always returns 'yolo' / 'sam' / 'sam3'.
    _append_menu(
        page, "Backend", "Backend",
        ["yolo", "sam", "sam3"],
        "yolo",
    )

    p = page.appendFile("Model", label="Model File (.pt, empty = default)")

    _append_menu(page, "Device", "Device", ["auto", "cpu", "cuda:0"], "auto")

    p = page.appendFloat("Confidence", label="Confidence")[0]
    p.default, p.normMin, p.normMax = 0.25, 0.0, 1.0

    p = page.appendFloat("Iou", label="IOU Threshold")[0]
    p.default, p.normMin, p.normMax = 0.7, 0.0, 1.0

    p = page.appendInt("Imgsize", label="Image Size")[0]
    p.default, p.normMin, p.normMax = 640, 64, 2048

    p = page.appendFloat("Maskalpha", label="Mask Alpha (SAM/AMG overlay)")[0]
    p.default, p.normMin, p.normMax = 0.5, 0.0, 1.0

    page.appendToggle("Showboxes", label="Show Boxes (YOLO)")[0].default = False
    page.appendToggle("Showlabels", label="Show Labels (YOLO)")[0].default = False
    page.appendToggle("Maskonly", label="Mask Only (White on Black)")[0].default = False
    page.appendToggle("Autorun", label="Auto Run (SAM per frame)")[0].default = True

    p = page.appendStr("Textprompts", label="Text Prompts (SAM 3: 'person, car')")[0]
    p.default = "person"
    p.val = "person"

    page.appendStr("Points", label="Points (SAM 2: 'x1,y1;x2,y2')")
    page.appendStr("Bbox", label="BBox (SAM 2/3: 'x1,y1,x2,y2')")

    p = page.appendInt("Amgpoints", label="AMG Points Per Side")[0]
    p.default, p.normMin, p.normMax = 16, 4, 64

    page.appendPulse("Reloadmodel", label="Reload Model")
    page.appendPulse("Runsam", label="Run SAM (slow, on-demand)")
    return


def onPulse(par: Par):
    """
    Called when a custom pulse parameter is pushed.

    Args:
        par: The parameter that was pulsed
    """
    scriptOp = par.owner
    if par.name == "Reloadmodel":
        scriptOp.storage["force_reload"] = True
        scriptOp.storage["cached"] = None
        scriptOp.cook(force=True)
    elif par.name == "Runsam":
        scriptOp.storage["run_sam"] = True
        scriptOp.cook(force=True)
    return


def onCook(scriptOp: scriptTOP):
    """
    Called when the Script TOP needs to cook.

    All backends run inference every cook (realtime). For SAM/SAM3, if the
    'Auto Run' toggle is OFF, inference runs only when the 'Run SAM' pulse is
    hit and the last cached result is re-output otherwise.
    """
    if not scriptOp.inputs:
        return

    cfg = _cfg(scriptOp)
    cfg["device"] = _resolve_device(cfg["device"])

    model = _ensure_model(scriptOp, cfg)
    bgr, alpha = _read_input_bgr(scriptOp.inputs[0])

    if model is None:
        # Model failed to load: pass input through, error already flagged.
        _write_output(scriptOp, bgr, alpha)
        return

    is_sam = cfg["backend"] in ("sam", "sam3")

    if is_sam and not cfg["autorun"] and not scriptOp.storage.get("run_sam"):
        # Manual mode: only run on demand. Re-use cached result if we have one.
        cached = scriptOp.storage.get("cached")
        if cached is not None and cached.shape[:2] == bgr.shape[:2]:
            if cfg["mask_only"]:
                out = _mask_only_image(
                    scriptOp.storage.get("cached_masks"), bgr.shape[0], bgr.shape[1]
                )
            else:
                out = cached
        else:
            out = bgr
        _write_output(scriptOp, out, alpha)
        return

    try:
        t0 = time.time()
        annotated, masks = _infer(scriptOp, model, bgr, cfg)
        ms = (time.time() - t0) * 1000
        scriptOp.storage["cached"] = annotated
        scriptOp.storage["cached_masks"] = masks
        scriptOp.storage["inference_ms"] = ms

        # One-time error clear / timing feedback
        if scriptOp.storage.get("error"):
            scriptOp.clearScriptErrors()
            scriptOp.storage["error"] = ""
        if absTime.frame % 60 == 0 or is_sam:
            debug(f"[SegmentEverything] {cfg['backend']} inference: {ms:.1f} ms")

        if cfg["mask_only"]:
            annotated = _mask_only_image(masks, bgr.shape[0], bgr.shape[1])

        _write_output(scriptOp, annotated, alpha)
    except Exception as e:
        scriptOp.storage["error"] = str(e)
        scriptOp.addScriptError(f"Inference failed: {e}")
        _write_output(scriptOp, bgr, alpha)
    finally:
        scriptOp.storage["run_sam"] = False
    return


def onGetCookLevel(scriptOp: scriptTOP) -> CookLevel:
    """
    Sets the scriptOp's cook level, the conditions necessary to cause a cook.

    Return one of the following:
        CookLevel.AUTOMATIC - inputs changed and output being used.
                             TD default behavior.
        CookLevel.ON_CHANGE - inputs changed, output used or not.
        CookLevel.WHEN_USED - every frame when output is being used
        CookLevel.ALWAYS - every frame
    """

    return CookLevel.AUTOMATIC
