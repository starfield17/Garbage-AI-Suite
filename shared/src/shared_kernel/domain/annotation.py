"""统一标注 Schema - 所有引擎输出都落到此格式"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import uuid4

from .taxonomy import WasteCategory


@dataclass(frozen=True)
class BoundingBox:
    """边界框值对象（归一化坐标 0-1）"""
    x_center: float     # 中心点 X（0-1）
    y_center: float     # 中心点 Y（0-1）
    width: float        # 宽度（0-1）
    height: float       # 高度（0-1）
    
    def __post_init__(self):
        for name, value in [
            ("x_center", self.x_center),
            ("y_center", self.y_center),
            ("width", self.width),
            ("height", self.height)
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
    
    def to_xyxy(self, img_width: int, img_height: int) -> tuple:
        """转换为像素坐标 (x1, y1, x2, y2)"""
        x1 = int((self.x_center - self.width / 2) * img_width)
        y1 = int((self.y_center - self.height / 2) * img_height)
        x2 = int((self.x_center + self.width / 2) * img_width)
        y2 = int((self.y_center + self.height / 2) * img_height)
        return (x1, y1, x2, y2)
    
    def to_xywh(self, img_width: int, img_height: int) -> tuple:
        """转换为 COCO 格式 (x, y, w, h)"""
        x = int((self.x_center - self.width / 2) * img_width)
        y = int((self.y_center - self.height / 2) * img_height)
        w = int(self.width * img_width)
        h = int(self.height * img_height)
        return (x, y, w, h)
    
    @classmethod
    def from_xyxy(
        cls, x1: int, y1: int, x2: int, y2: int,
        img_width: int, img_height: int
    ) -> "BoundingBox":
        """从像素坐标创建"""
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return cls(x_center, y_center, width, height)


@dataclass(frozen=True)
class Confidence:
    """置信度值对象"""
    value: float
    
    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.value}")
    
    def is_above_threshold(self, threshold: float) -> bool:
        return self.value >= threshold


class DetectionSource(Enum):
    """检测来源"""
    YOLO = "yolo"
    FASTER_RCNN = "faster_rcnn"
    VLM = "vlm"
    ENSEMBLE = "ensemble"
    MANUAL = "manual"


@dataclass
class Detection:
    """单个检测结果"""
    detection_id: str
    category: WasteCategory
    confidence: Confidence
    bounding_box: BoundingBox
    source: DetectionSource
    raw_label: Optional[str] = None     # 原始标签（如 VLM 返回的细粒度标签）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        category: WasteCategory,
        confidence: float,
        bbox: BoundingBox,
        source: DetectionSource,
        raw_label: Optional[str] = None
    ) -> "Detection":
        return cls(
            detection_id=str(uuid4()),
            category=category,
            confidence=Confidence(confidence),
            bounding_box=bbox,
            source=source,
            raw_label=raw_label
        )


@dataclass
class LabelFile:
    """标注文件（对应单张图片）"""
    file_id: str
    image_path: str
    image_width: int
    image_height: int
    detections: List[Detection]
    created_at: datetime = field(default_factory=datetime.utcnow)
    schema_version: str = "1.0.0"
    
    def add_detection(self, detection: Detection) -> None:
        """添加检测结果"""
        self.detections.append(detection)
    
    def filter_by_confidence(self, threshold: float) -> List[Detection]:
        """按置信度过滤"""
        return [d for d in self.detections if d.confidence.is_above_threshold(threshold)]
    
    def to_yolo_format(self) -> List[str]:
        """转换为 YOLO 标注格式"""
        lines = []
        # 创建分类到编号的映射
        category_to_id = {
            WasteCategory.KITCHEN_WASTE: 0,
            WasteCategory.RECYCLABLE_WASTE: 1,
            WasteCategory.HAZARDOUS_WASTE: 2,
            WasteCategory.OTHER_WASTE: 3,
        }
        for det in self.detections:
            class_id = category_to_id[det.category]
            bbox = det.bounding_box
            line = f"{class_id} {bbox.x_center:.6f} {bbox.y_center:.6f} {bbox.width:.6f} {bbox.height:.6f}"
            lines.append(line)
        return lines
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "file_id": self.file_id,
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat(),
            "detections": [
                {
                    "detection_id": d.detection_id,
                    "category": d.category.value,
                    "confidence": d.confidence.value,
                    "bounding_box": {
                        "x_center": d.bounding_box.x_center,
                        "y_center": d.bounding_box.y_center,
                        "width": d.bounding_box.width,
                        "height": d.bounding_box.height,
                    },
                    "source": d.source.value,
                    "raw_label": d.raw_label,
                    "metadata": d.metadata,
                }
                for d in self.detections
            ]
        }
