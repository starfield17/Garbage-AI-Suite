"""YOLO model adapter."""

from pathlib import Path
from typing import List, Dict
import os

import cv2
import numpy as np
from ultralytics import YOLO

from garbage_autolabel.application.ports import ModelAdapterPort
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class YOLOAdapter(ModelAdapterPort):
    def __init__(self, config: dict):
        self.model_path = config.get("model_path", "")
        self.device = config.get("device", "cpu")
        self._model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path or not Path(self.model_path).exists():
            raise ValueError(f"YOLO model not found: {self.model_path}")

        log.info("Loading YOLO model", path=self.model_path)
        self._model = YOLO(self.model_path)

    def predict(self, image_path: Path) -> List[Dict]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        results = self._model(image, verbose=False)
        detections = []

        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.cpu().numpy()
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i]
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])

                    detections.append(
                        {
                            "class_id": cls_id,
                            "class_name": self._model.names[cls_id] if self._model.names else str(cls_id),
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                            },
                            "confidence": conf,
                        }
                    )

        return detections

    def get_class_map(self) -> Dict[str, int]:
        if self._model and self._model.names:
            return {name: idx for idx, name in self._model.names.items()}
        return {}

    def get_image_size(self, image_path: Path) -> tuple[int, int]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        return width, height
