#!/usr/bin/env python3
"""
TouchDesigner-friendly segmentation wrapper.

This script provides a simple interface for TouchDesigner to call the
segmentation system via subprocess. It outputs JSON status messages
for easy parsing in TD.

Usage:
    python td_segment.py --image input.jpg --output output/ --backend yolo
    python td_segment.py --image input.jpg --output output/ --backend sam --points-per-side 16
    python td_segment.py --image input.jpg --output output/ --backend sam3 --text "person,car"

Output:
    Prints JSON to stdout with either:
    - {"status": "complete", "output_dir": "output/a1b2c3d4", "run_id": "a1b2c3d4"}
    - {"status": "error", "message": "Error description"}
"""

import argparse
import sys
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Segmentation for TouchDesigner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--image",
        required=True,
        help="Input image path",
    )

    # Output settings
    parser.add_argument(
        "--output",
        default="output",
        help="Base output directory (UUID subdirectory will be created)",
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        default="yolo",
        choices=["yolo", "sam", "sam3"],
        help="Segmentation backend",
    )

    # Model settings
    parser.add_argument(
        "--model",
        default=None,
        help="Model path (auto-detected if not specified)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )

    # SAM-specific settings
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=16,
        help="AMG grid size (8=fast, 16=balanced, 32=thorough)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="SAM 3 text prompts (comma-separated)",
    )
    parser.add_argument(
        "--points",
        default=None,
        help="Point prompts as x,y pairs (comma-separated)",
    )
    parser.add_argument(
        "--bbox",
        default=None,
        help="Bounding box prompt as x1,y1,x2,y2",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID for output directory (auto-generated if not specified)",
    )

    args = parser.parse_args()

    # Validate image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(json.dumps({
            "status": "error",
            "message": f"Image not found: {args.image}"
        }))
        sys.exit(1)

    # Parse SAM prompts
    sam_text = None
    sam_points = None
    sam_bbox = None

    if args.text:
        sam_text = [t.strip() for t in args.text.split(",") if t.strip()]

    if args.points:
        coords = [int(x.strip()) for x in args.points.split(",")]
        sam_points = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]

    if args.bbox:
        sam_bbox = [int(x.strip()) for x in args.bbox.split(",")]

    try:
        # Import here to avoid slow startup when just checking help
        from src.config import Config
        from src.image_processor import ImageProcessor

        # Create configuration
        config = Config(
            backend=args.backend,
            model_path=args.model,
            confidence=args.conf,
            image_path=str(image_path),
            output_dir=args.output,
            amg_points_per_side=args.points_per_side,
            sam_text=sam_text,
            sam_points=sam_points,
            sam_bbox=sam_bbox,
            run_id=args.run_id,
        )

        # Initialize processor
        processor = ImageProcessor(config)
        if not processor.initialize():
            print(json.dumps({
                "status": "error",
                "message": "Failed to initialize processor"
            }))
            sys.exit(1)

        # Process image
        result_dir = processor.process(str(image_path))

        if result_dir:
            print(json.dumps({
                "status": "complete",
                "output_dir": result_dir,
                "run_id": processor.run_id,
            }))
        else:
            print(json.dumps({
                "status": "error",
                "message": "Processing failed"
            }))
            sys.exit(1)

    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": str(e)
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
