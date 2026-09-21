# 05 - Backends

The Script TOP (and the CLI) supports three backends via the `Backend` parameter / `--backend` flag: `yolo`, `sam`, `sam3`.

## YOLO11 (`yolo`)

Ultralytics YOLO11-seg. Real-time capable - runs on every Script TOP cook.

- **Model**: `yolo11n-seg.pt` (CPU/fallback) or `yolo11s-seg.pt` (GPU) by default.
- **Prompts**: none (auto-detects objects).
- **Parameters**: `Confidence`, `Iou`, `Imgsize`, `Showboxes`, `Showlabels`.
- **Output**: annotated frame with instance-colored masks, optional boxes/labels.

```bash
# CLI
python main.py                                  # webcam
python main.py --image photo.jpg                # single image
```

## SAM 2 (`sam`)

Ultralytics SAM interface. Zero-shot segmentation; **not** realtime (on-demand in the TOP).

Modes (selected by which prompt parameters are filled):

| Prompts | Mode |
|---------|------|
| `Points` filled | Point prompting (`x1,y1;x2,y2`) |
| `Bbox` filled | Box prompting (`x1,y1,x2,y2`) |
| neither | Automatic Mask Generation (segment everything, see [08](08-automatic-mask-generation.md)) |

- **Model**: `sam2_l.pt` by default (heaviest/best; `sam2_t.pt` is faster).
- **Output**: colored mask overlay on the input frame (SAM has no class labels).

```bash
# CLI
python main.py --backend sam --image photo.jpg --points 200,300
python main.py --backend sam --image photo.jpg --bbox 100,100,300,300
python main.py --backend sam --image photo.jpg            # AMG
```

## SAM 3 (`sam3`)

Concept / text-driven segmentation via `SAM3SemanticPredictor`, or AMG via the plain SAM interface.

Modes:

| Prompts | Mode |
|---------|------|
| `Textprompts` filled (`person, car`) | Text concept segmentation - finds **all** instances |
| `Bbox` filled | Exemplar-based (find similar objects to the boxed one) |
| neither | Automatic Mask Generation (AMG) |

- **Model**: `sam3.pt` (~3.4 GB, may require Hugging Face approval to download).
- **Parameters**: `Textprompts`, `Confidence`, `Imgsize`.

```bash
# CLI
python main.py --backend sam3 --image photo.jpg --text "person,car"
python main.py --backend sam3 --image photo.jpg --text "person with red shirt"
```

## Selecting the Right Backend

| You want... | Use |
|-------------|-----|
| Live / realtime segmentation in TD | `yolo` |
| Precise zero-shot mask on a still image | `sam` with points/bbox |
| "Everything in this image" masks | `sam` or `sam3` (AMG) |
| "Find all people / cars / red things" | `sam3` with text |
| A still, boxed object as an exemplar | `sam3` with bbox |

## Model Auto-Selection

When the `Model` parameter is empty, defaults are chosen per backend and device:

- `sam3` → `sam3.pt`
- `sam` → `sam2_l.pt`
- `yolo` → `yolo11s-seg.pt` on GPU, `yolo11n-seg.pt` otherwise

Bare filenames resolve into `models/`. Absolute paths are used as-is.