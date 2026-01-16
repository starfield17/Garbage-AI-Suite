"""Storage and dataset utilities."""

from pathlib import Path
from typing import List

from garbage_autolabel.application.ports import DatasetScannerPort
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class DatasetScanner(DatasetScannerPort):
    def __init__(self, supported_extensions=None):
        self.supported_extensions = supported_extensions or [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
        ]

    def list_images(self, input_dir: Path) -> List[Path]:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        images = []
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                images.append(file_path)

        log.info("Found images", count=len(images), directory=str(input_dir))
        return sorted(images)
