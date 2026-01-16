"""Label format writers."""

import json
from pathlib import Path

from garbage_autolabel.application.ports import LabelWriterPort
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class BBoxWriter(LabelWriterPort):
    def write(self, image_path: Path, label_data) -> None:
        output_path = image_path.with_suffix(".json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(label_data.model_dump(), f, indent=2)


class COCOWriter(LabelWriterPort):
    def __init__(self):
        self.coco_data = {"images": [], "annotations": [], "categories": []}
        self.image_id = 1
        self.annotation_id = 1

    def write(self, image_path: Path, label_data) -> None:
        import cv2
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        self.coco_data["images"].append(
            {"id": self.image_id, "file_name": image_path.name, "width": width, "height": height}
        )

        for label in label_data.labels:
            self.coco_data["annotations"].append(
                {
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": self._get_or_create_category(label.name),
                    "bbox": [
                        label.x1,
                        label.y1,
                        label.x2 - label.x1,
                        label.y2 - label.y1,
                    ],
                    "area": (label.x2 - label.x1) * (label.y2 - label.y1),
                    "iscrowd": 0,
                }
            )
            self.annotation_id += 1

        self.image_id += 1

    def _get_or_create_category(self, name: str) -> int:
        for cat in self.coco_data["categories"]:
            if cat["name"] == name:
                return cat["id"]

        new_id = len(self.coco_data["categories"]) + 1
        self.coco_data["categories"].append({"id": new_id, "name": name})
        return new_id

    def finalize(self, output_path: Path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.coco_data, f, indent=2)


class YOLOWriter(LabelWriterPort):
    def __init__(self, class_map: dict):
        self.class_map = class_map

    def write(self, image_path: Path, label_data) -> None:
        output_path = image_path.with_suffix(".txt")
        import cv2

        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        lines = []
        for label in label_data.labels:
            x_center = (label.x1 + label.x2) / 2 / width
            y_center = (label.y1 + label.y2) / 2 / height
            bbox_width = (label.x2 - label.x1) / width
            bbox_height = (label.y2 - label.y1) / height

            class_id = self.class_map.get(label.name, 0)
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
