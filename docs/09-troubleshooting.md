# 09 - Troubleshooting

## TD can't import ultralytics / torch

**Symptom:** Script TOP flags `ModuleNotFoundError: No module named 'ultralytics'` (or `torch`).

**Fix:** Point TD at the venv's site-packages (see [02-installation](02-installation.md) Step 4):
- Edit > Preferences > Python > **Python 64-bit Module Path** → `D:\Dev\segment-everything-td\.env\Lib\site-packages`, or
- Apply the `TDPyEnvManagerContext.yaml` Python Environment Manager context.
- Verify `PROJECT_DIR` / `ENV_SITE_PACKAGES` at the top of `td_script_top.py` match your checkout path.

## "cannot pickle '_thread.lock' object" on project save

**Cause:** A model object got into `scriptOp.storage`, which TD pickles on save.

**Fix:** Don't put models in storage. The Script TOP keeps them in the module-level `_MODELS` registry (see [04](04-script-top-inference.md)). If you added your own code, store only plain data (paths, params, frames) in `storage`.

## First load blocks / freezes TouchDesigner

**Cause:** torch import + model load on the main thread.

**Fix:** This is normal. Pre-download the model (`python -c "from ultralytics import SAM; SAM('sam3.pt')"`) and use the smallest model that fits your task. Avoid loading SAM 3 interactively if you don't need it.

## Output is misaligned / wrong orientation

**Cause:** Script TOP **Output Resolution** doesn't match the input TOP.

**Fix:** Set Common page > Output Resolution to **Follow Input**. The TOP reads/writes arrays sized to the input and flips vertically for OpenCV/BGR (see [04](04-script-top-inference.md)).

## Low framerate in realtime YOLO mode

- Use the nano model (`yolo11n-seg.pt`) and a smaller `Imgsize`.
- Reduce the input resolution feeding the Script TOP.
- Inference runs on TD's main thread - heavy models and large inputs will always cost frames. Consider running on a GPU (`Device` → `cuda:0`).

## SAM 3 fails to download

**Symptom:** Error mentioning Hugging Face / approval / `facebook/sam3`.

**Fix:** SAM 3 is gated. Accept the terms at https://huggingface.co/facebook/sam3, then re-run, or pre-download with:
```bash
python -c "from ultralytics import SAM; SAM('sam3.pt')"
```

## SAM/AMG is extremely slow in the TOP

**Cause:** AMG runs a grid of prompts (256 by default) - see [08](08-automatic-mask-generation.md).

**Fix:** Leave **Auto Run** off and pulse **Run SAM**; lower **AMG Points Per Side** to 8 for a quick pass.

## Model reload doesn't take effect

**Fix:** Pulse **Reload Model**. `_ensure_model` only reloads automatically when backend/model/device/text/bbox/conf change - the pulse forces a reload after downloading a new `.pt` file.

## Model load failure flags an error, input passes through

The Script TOP passes the input TOP through and raises `scriptOp.addScriptError`. Check the Textport for the underlying exception (model path wrong, download failed, OOM). The error clears automatically on the next successful cook.

## CUDA out of memory

torch shares VRAM with TouchDesigner. Close unused TOPs, reduce `Imgsize`, use CPU for YOLO, or use a lighter SAM model (`sam2_t.pt`).

## CLI-specific

- `Error: Image not found` - check the path; image mode requires `--image`.
- `SAM 2 is not suitable for realtime webcam mode` - SAM backends are blocked in webcam mode by design (see [06](06-command-line.md)); use `--image` instead.