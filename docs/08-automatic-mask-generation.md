# 08 - Automatic Mask Generation

Automatic Mask Generation (**AMG**) is the "segment everything" mode. When SAM 2 or SAM 3 runs with **no** prompts, it walks a grid of points over the image and keeps every distinct object mask.

Works in both the Script TOP and the CLI.

## Algorithm

1. **Build a point grid** - `points_per_side` x `points_per_side` points spread evenly across the image (normalized [0,1], offset by half a cell).
2. **Prompt once per point** - SAM predicts a mask for each grid point as a positive click.
3. **Filter by area** - masks smaller than a minimum area (100 px) are discarded.
4. **NMS** - overlapping masks are deduplicated via non-maximum suppression on their boxes (IoU threshold `0.7`).
5. **Overlay** - surviving masks are composited onto the frame at `alpha` opacity.

## Tuning

The Script TOP exposes **AMG Points Per Side** (`Amgpoints`); the CLI exposes `--points-per-side` and `--nms-thresh`.

| Points/side | Grid points | Speed | Quality |
|-------------|-------------|-------|---------|
| 8 | 64 | Fastest | Coarse, fewer masks |
| 16 | 256 | Balanced (default) | Good coverage |
| 32 | 1024 | Slow | Thorough, many masks |

> **In the Script TOP:** AMG is expensive. With `yolo` it's not used (auto-detect). For SAM backends leave **Auto Run** off and pulse **Run SAM** - the result is cached, so subsequent cooks are free until you pulse again or change the frame.

## Output

- **TOP mode**: the annotated overlay is written to the Script TOP output. With **Mask Only** enabled, only the white-on-black masks are output (no background), useful for downstream TD compositing.
- **CLI mode**: individual binary masks, a composite, metadata, and `done.json` are written per run (see [06-command-line](06-command-line.md)).

## Implementation Notes

- The CLI uses `src/detectors/amg.py` (`build_point_grid`, `nms_boxes`, `create_mask_overlay`).
- The Script TOP inlines its own numpy-only copies (`_build_point_grid`, `_nms_boxes`, `_mask_overlay`, `_mask_only_image`) so it has no `cv2` dependency for resizing (`_resize_mask_nearest`). Keep both in sync when tuning.
- Mask area is used as the NMS score so larger, more confident segments win.
- The SAM 3 variant of AMG uses the plain `ultralytics.SAM` interface (`kind = sam3_amg`), since `SAM3SemanticPredictor` is only for text/bbox prompts.