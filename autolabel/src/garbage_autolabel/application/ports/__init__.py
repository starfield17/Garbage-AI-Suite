"""Ports/interfaces for autolabel module."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict

import numpy as np


class ModelAdapterPort(ABC):
    """Port for model prediction adapters."""

    @abstractmethod
    def predict(self, image_path: Path) -> List[Dict]:
        """Predict detections for an image."""
        pass

    @abstractmethod
    def get_class_map(self) -> Dict[str, int]:
        """Get class name to ID mapping."""
        pass

    @abstractmethod
    def get_image_size(self, image_path: Path) -> tuple[int, int]:
        """Get image dimensions (width, height)."""
        pass


class LabelWriterPort(ABC):
    """Port for writing label outputs."""

    @abstractmethod
    def write(self, image_path: Path, label_data) -> None:
        """Write label data to disk."""
        pass


class DatasetScannerPort(ABC):
    """Port for scanning datasets."""

    @abstractmethod
    def list_images(self, input_dir: Path) -> List[Path]:
        """List all images in directory."""
        pass


class ConverterPort(ABC):
    """Port for label format conversion."""

    @abstractmethod
    def convert_bbox_to_coco(self, bbox_data, image_size) -> Dict:
        """Convert bbox format to COCO."""
        pass

    @abstractmethod
    def convert_bbox_to_yolo(self, bbox_data, image_size) -> List[str]:
        """Convert bbox format to YOLO."""
        pass
