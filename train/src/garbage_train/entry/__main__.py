"""Entry points for train module."""

import argparse
import sys
from pathlib import Path

from garbage_train.application.usecases import TrainAndExportModelUseCase


def parse_args():
    parser = argparse.ArgumentParser(description="Train garbage detection models")
    parser.add_argument("--train-id", type=str, required=True, help="Training profile ID")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    use_case = TrainAndExportModelUseCase()

    result = use_case.execute(
        train_id=args.train_id,
        output_dir=args.out,
    )

    if result["success"]:
        print("Training completed successfully")
        print(f"Manifest: {result['manifest_path']}")
        sys.exit(0)
    else:
        print(f"Training failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
