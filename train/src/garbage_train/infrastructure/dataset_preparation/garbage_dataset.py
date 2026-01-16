"""Dataset preparation utilities."""

import json
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np

from garbage_shared.observability import get_logger

log = get_logger(__name__)


@dataclass
class ImageLabel:
    path: Path
    class_id: int
    bbox: List[float]


class GarbageDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.images: List[Path] = []
        self.labels: List[ImageLabel] = []
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._load_data()

    def _load_data(self):
        log.info("Loading dataset", path=str(self.data_path))

        for img_path in self.data_path.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                json_path = img_path.with_suffix(".json")

                if json_path.exists():
                    with open(json_path, "r", encoding="utf-8") as f:
                        label_data = json.load(f)

                    if "labels" in label_data:
                        self.images.append(img_path)
                        self.labels.append(
                            ImageLabel(
                                path=img_path,
                                class_id=self._map_class_name(label_data["labels"][0]["name"]),
                                bbox=label_data["labels"][0],
                            )
                        )

        if len(self.images) == 0:
            raise ValueError("No valid image/label pairs found")

        log.info("Loaded dataset", images=len(self.images), labels=len(self.labels))

    def _map_class_name(self, class_name: str) -> int:
        class_mapping = {
            "Kitchen_waste": 0,
            "Recyclable_waste": 1,
            "Hazardous_waste": 2,
            "Other_waste": 3,
            "potato": 0,
            "daikon": 0,
            "carrot": 0,
            "bottle": 1,
            "can": 1,
            "battery": 2,
            "drug": 2,
            "inner_packing": 2,
            "tile": 3,
            "stone": 3,
            "brick": 3,
        }
        return class_mapping.get(class_name, 0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        x1, y1, x2, y2 = label.bbox

        boxes = torch.tensor(
            [
                [
                    max(0, x1),
                    max(0, y1),
                    min(639, x2),
                    min(639, y2),
                ]
            ],
            dtype=torch.float32,
        )

        labels = torch.tensor([label.class_id], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target
