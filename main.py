#!/usr/bin/env python3
"""
Realtime Instance Segmentation Application

Entry point for multi-backend segmentation application (YOLO, SAM 2, SAM 3).

Usage:
    # Webcam mode (YOLO only - SAM is too slow for realtime)
    python main.py                    # Auto-detect device, shows camera selection
    python main.py --list-cameras     # List available cameras and exit
    python main.py --camera 0         # Use specific camera index (skip selection)
    python main.py --model yolo11n-seg.pt  # Specify model
    python main.py --conf 0.5         # Set confidence threshold

    # Image mode with YOLO
    python main.py --image photo.jpg              # Process single image
    python main.py --image photo.jpg --output results/    # Custom output dir

    # Image mode with SAM 3 (text prompts)
    python main.py --backend sam3 --image photo.jpg --text "person,car"
    python main.py --backend sam3 --image photo.jpg --text "person with red shirt"

    # Image mode with SAM 2 (visual prompts)
    python main.py --backend sam --image photo.jpg --bbox 100,100,300,300
    python main.py --backend sam --image photo.jpg --points 200,300

    # Image mode with SAM 2 automatic mask generation (segment everything)
    python main.py --backend sam --image photo.jpg                        # Default 16x16 grid
    python main.py --backend sam --image photo.jpg --points-per-side 32   # Thorough (slower)
    python main.py --backend sam --image photo.jpg --points-per-side 8    # Fast (fewer masks)
"""

import argparse
import sys

from pathlib import Path

from src.config import Config
from src.app import SegmentationApp
from src.camera import list_available_cameras, select_camera_interactive
from src.image_processor import ImageProcessor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-backend Instance Segmentation (YOLO, SAM 2, SAM 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        choices=["yolo", "sam", "sam3"],
        default="yolo",
        help="Segmentation backend: yolo (realtime), sam (SAM 2), sam3 (SAM 3 with text prompts).",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path. Auto-detected based on backend if not specified.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda:0, mps, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size.",
    )

    # Camera settings
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index to use. If not specified, shows camera selection menu.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras and exit.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera capture height.",
    )

    # Display settings
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Disable detection (show camera feed only).",
    )
    parser.add_argument(
        "--no-fps",
        action="store_true",
        help="Disable FPS display.",
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Disable mask display.",
    )
    parser.add_argument(
        "--boxes",
        action="store_true",
        help="Show bounding boxes (off by default).",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Show class labels (off by default).",
    )

    # Image mode settings
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image for single-image segmentation mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for image mode results.",
    )
    parser.add_argument(
        "--no-individual-masks",
        action="store_true",
        help="Skip saving individual binary mask files.",
    )
    parser.add_argument(
        "--no-composite",
        action="store_true",
        help="Skip saving composite overlay image.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip saving metadata JSON file.",
    )

    # SAM-specific prompt settings
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="SAM 3 text prompts (comma-separated). E.g., 'person,car' or 'person with red shirt'.",
    )
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help="Point prompts as x,y pairs (comma-separated). E.g., '200,300' or '200,300,400,500'.",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Bounding box prompt as x1,y1,x2,y2. E.g., '100,100,300,300'.",
    )

    # AMG (Automatic Mask Generation) settings for SAM segment-everything mode
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=16,
        help="Points per side for auto-mask generation grid (8=fast, 16=balanced, 32=thorough).",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.7,
        help="NMS threshold for duplicate mask removal in auto-mask generation.",
    )

    return parser.parse_args()


def parse_text_prompts(text_arg: str) -> list:
    """Parse comma-separated text prompts."""
    if not text_arg:
        return None
    return [t.strip() for t in text_arg.split(",") if t.strip()]


def parse_points(points_arg: str) -> list:
    """Parse point prompts from comma-separated string."""
    if not points_arg:
        return None
    coords = [int(x.strip()) for x in points_arg.split(",")]
    # Convert flat list to list of [x, y] pairs
    points = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
    return points


def parse_bbox(bbox_arg: str) -> list:
    """Parse bounding box from comma-separated string."""
    if not bbox_arg:
        return None
    return [int(x.strip()) for x in bbox_arg.split(",")]


def run_image_mode(args: argparse.Namespace) -> int:
    """
    Run image segmentation mode.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Validate image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    if not image_path.is_file():
        print(f"Error: Not a file: {args.image}")
        return 1

    # Parse SAM prompts
    sam_text = parse_text_prompts(args.text)
    sam_points = parse_points(args.points)
    sam_bbox = parse_bbox(args.bbox)

    # Create configuration
    config = Config(
        backend=args.backend,
        model_path=args.model,
        device=args.device,
        confidence=args.conf,
        iou_threshold=args.iou,
        image_size=args.imgsz,
        image_path=str(image_path),
        output_dir=args.output,
        save_individual_masks=not args.no_individual_masks,
        save_composite=not args.no_composite,
        save_metadata=not args.no_metadata,
        show_masks=not args.no_masks,
        show_boxes=args.boxes,
        show_labels=args.labels,
        sam_text=sam_text,
        sam_points=sam_points,
        sam_bbox=sam_bbox,
        amg_points_per_side=args.points_per_side,
        amg_nms_thresh=args.nms_thresh,
    )

    # Print header based on backend
    backend_names = {"yolo": "YOLO11", "sam": "SAM 2", "sam3": "SAM 3"}
    backend_name = backend_names.get(args.backend, args.backend.upper())

    print("=" * 50)
    print(f"{backend_name} Image Segmentation")
    print("=" * 50)
    print()
    print(config)
    print()

    # Create and run processor
    processor = ImageProcessor(config)

    if not processor.initialize():
        print("Error: Failed to initialize processor")
        return 1

    try:
        result = processor.process(str(image_path))

        if result is None:
            print("Error: Processing failed")
            return 1

        print()
        print("Processing complete!")
        print(f"Output files saved to: {args.output}/")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args()

    # Handle image mode
    if args.image is not None:
        return run_image_mode(args)

    # Block realtime mode for SAM backends (too slow for webcam)
    if args.backend in ["sam", "sam3"]:
        backend_names = {"sam": "SAM 2", "sam3": "SAM 3"}
        backend_name = backend_names.get(args.backend, args.backend)
        print(f"Error: {backend_name} is not suitable for realtime webcam mode.")
        print(f"       {backend_name} inference is ~1000x slower than YOLO.")
        print()
        print("Use image mode instead:")
        print(f"  python main.py --backend {args.backend} --image photo.jpg")
        if args.backend == "sam3":
            print(f"  python main.py --backend sam3 --image photo.jpg --text 'person,car'")
        return 1

    # Handle --list-cameras
    if args.list_cameras:
        print("Scanning for available cameras...")
        cameras = list_available_cameras()
        if not cameras:
            print("No cameras found.")
            return 1
        print("\nAvailable cameras:")
        print("-" * 50)
        for cam in cameras:
            status = "OK" if cam["available"] else "NOT RESPONDING"
            continuity = ""
            name_lower = cam["name"].lower()
            if "continuity" in name_lower or "iphone" in name_lower or "ipad" in name_lower:
                continuity = " [Continuity]"
            print(f"  Index {cam['index']}: {cam['name']}{continuity} [{status}]")
        print("-" * 50)
        return 0

    # Handle camera selection
    camera_id = args.camera
    if camera_id is None:
        print("Scanning for available cameras...")
        cameras = list_available_cameras()
        camera_id = select_camera_interactive(cameras)
        if camera_id is None:
            print("No camera selected. Exiting.")
            return 1
        print()

    # Create configuration from arguments
    config = Config(
        backend=args.backend,  # Will be "yolo" here (SAM blocked above)
        model_path=args.model,
        device=args.device,
        confidence=args.conf,
        iou_threshold=args.iou,
        image_size=args.imgsz,
        camera_id=camera_id,
        camera_width=args.width,
        camera_height=args.height,
        show_fps=not args.no_fps,
        show_masks=not args.no_masks,
        show_boxes=args.boxes,
        show_labels=args.labels,
    )

    # Create and run application
    app = SegmentationApp(config)

    if not app.initialize():
        print("Error: Failed to initialize application")
        return 1

    try:
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        app.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
