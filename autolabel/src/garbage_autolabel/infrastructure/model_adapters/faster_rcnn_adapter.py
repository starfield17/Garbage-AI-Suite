"""Faster R-CNN model adapter."""

from pathlib import Path
from typing import List, Dict
import os

import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

from garbage_autolabel.application.ports import ModelAdapterPort
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class FasterRCNNAdapter(ModelAdapterPort):
    def __init__(self, config: dict):
        self.model_path = config.get("model_path", "")
        self.device = config.get("device", "cpu")
        self.model_type = config.get("model_type", "resnet50_fpn")
        self._model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path or not Path(self.model_path).exists():
            raise ValueError(f"Faster R-CNN model not found: {self.model_path}")

        log.info("Loading Faster R-CNN model", path=self.model_path, type=self.model_type)

        if self.model_type == "resnet50_fpn":
            self._model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=4)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self._model.load_state_dict(checkpoint["model"])
        self._model.to(self.device)
        self._model.eval()

    def predict(self, image_path: Path) -> List[Dict]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self._model(image_tensor)

        boxes = prediction[0]["boxes"].cpu().numpy()
        labels = prediction[0]["labels"].cpu().numpy()
        scores = prediction[0]["scores"].cpu().numpy()

        detections = []
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if score > 0.5:
                x1, y1, x2, y2 = box.tolist()
                detections.append(
                    {
                        "class_id": int(label),
                        "class_name": str(label),
                        "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                        "confidence": float(score),
                    }
                )

        return detections

    def get_class_map(self) -> Dict[str, int]:
        return {
            "0": "Kitchen_waste",
            "1": "Recyclable_waste",
            "2": "Hazardous_waste",
            "3": "Other_waste",
        }

    def get_image_size(self, image_path: Path) -> tuple[int, int]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        return width, height
