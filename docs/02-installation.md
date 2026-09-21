# 02 - Installation

## Requirements

- **TouchDesigner 2023.10000+** (Python 3.11 built in) - for the Script TOP
- **Python 3.9+** - for the venv used by TD and the CLI
- **NVIDIA GPU with CUDA** - optional, for fast YOLO/SAM inference
- **Apple Silicon** - optional, for MPS acceleration

## Step 1: Clone the Repository

```bash
git clone https://github.com/srcphag/Segmentation_TD.git
cd Segmentation_TD
```

## Step 2: Create the Virtual Environment

The project ships a TD Python Environment Manager context (`TDPyEnvManagerContext.yaml`) that points at a `.env` venv with Python 3.11 (matching TD 2023+). Create it:

```bash
# Windows
python -m venv .env
.env\Scripts\activate

# macOS / Linux
python -m venv .env
source .env/bin/activate
```

> The venv **must** be Python 3.11 for TD's `td_script_top.py` to import it. Verify with `python --version`.

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` installs:

- `ultralytics>=8.0.0` - YOLO11 and the SAM 2 / SAM 3 interfaces
- `opencv-python>=4.8.0` - image IO and mask processing
- `torch>=2.0.0` - inference engine
- `timm>=1.0.0` - required by SAM 3

### CUDA / GPU Notes

The default PyPI `torch` may be CPU-only. For GPU acceleration install a CUDA build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Step 4: Point TouchDesigner at the venv

TD must be able to import `ultralytics`, `torch`, and `numpy`. Either:

1. **Edit > Preferences > Python > "Python 64-bit Module Path"** - add the venv's `site-packages`, e.g. `D:\Dev\segment-everything-td\.env\Lib\site-packages`, **or**
2. **Python Environment Manager** - apply the bundled `TDPyEnvManagerContext.yaml` context (mode `Python vEnv`, env `.env`, install path `.`), which makes TD activate the project venv automatically.

`td_script_top.py` also has a fallback: it appends `.env\Lib\site-packages` to `sys.path` at import time using the `PROJECT_DIR` constant at the top of the file (edit `PROJECT_DIR` / `ENV_SITE_PACKAGES` if your checkout lives elsewhere).

## Step 5: Model Downloads

Models download automatically on first load and are cached in `models/` (gitignored):

- **YOLO11** - `yolo11n-seg.pt` (CPU/fallback) or `yolo11s-seg.pt` (GPU) by default
- **SAM 2** - `sam2_l.pt` by default (or `sam2_t.pt` for faster)
- **SAM 3** - `sam3.pt` (~3.4 GB, may require Hugging Face access)

Pre-download SAM 3 in the terminal (keeps the first TD load fast):

```bash
python -c "from ultralytics import SAM; SAM('sam3.pt')"
```

## Step 6: Verify

```bash
# Quick YOLO sanity check (downloads the model on first run)
python main.py --image path/to/test.jpg --output output/
```

Then open `SegmentationTD.toe` in TouchDesigner and confirm the Script TOP cooks (see [03-touchdesigner-setup](03-touchdesigner-setup.md)).