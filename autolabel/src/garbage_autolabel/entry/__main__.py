"""Entry points for autolabel module."""

import argparse
import sys
from pathlib import Path

from garbage_autolabel.application.usecases import AutoLabelDatasetUseCase


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-label dataset with garbage detection models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, required=True, choices=["yolo", "faster_rcnn", "qwen_vl"], help="Model type to use")
    parser.add_argument("--input", type=str, required=True, help="Input directory with images")
    parser.add_argument("--out", type=str, required=True, help="Output directory for labels")
    parser.add_argument("--viz", type=str, default=None, help="Output directory for visualizations")
    parser.add_argument("--format", type=str, default="bbox", choices=["bbox", "coco", "yolo"], help="Output format")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (0-1)")
    return parser.parse_args()


def main():
    args = parse_args()
    use_case = AutoLabelDatasetUseCase()

    result = use_case.execute(
        config_path=Path(args.config),
        model_type=args.model,
        input_dir=Path(args.input),
        output_dir=Path(args.out),
        viz_dir=Path(args.viz) if args.viz else None,
        output_format=args.format,
        confidence_threshold=args.confidence,
    )

    if result["success"]:
        print(f"Auto-labeling completed: {result['stats']}")
        sys.exit(0)
    else:
        print(f"Auto-labeling failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
