"""标签结果实体"""

from dataclasses import dataclass, field
from typing import Dict
from datetime import datetime
from uuid import uuid4

from shared_kernel.domain.annotation import Detection

from .image_item import ImageItem


@dataclass
class LabelResult:
    """标签结果实体
    
    封装单张图片的标注结果
    """
    result_id: str
    image_item: ImageItem
    detections: list[Detection]
    status: str
    error_message: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def success(cls, image_item: ImageItem, detections: list[Detection]) -> "LabelResult":
        """创建成功的结果"""
        return cls(
            result_id=str(uuid4()),
            image_item=image_item,
            detections=detections,
            status="success"
        )
    
    @classmethod
    def skipped(cls, image_item: ImageItem) -> "LabelResult":
        """创建跳过的结果"""
        return cls(
            result_id=str(uuid4()),
            image_item=image_item,
            detections=[],
            status="skipped"
        )
    
    @classmethod
    def failed(cls, image_item: ImageItem, error_message: str) -> "LabelResult":
        """创建失败的结果"""
        return cls(
            result_id=str(uuid4()),
            image_item=image_item,
            detections=[],
            status="failed",
            error_message=error_message
        )
    
    @property
    def is_success(self) -> bool:
        return self.status == "success"
    
    @property
    def is_skipped(self) -> bool:
        return self.status == "skipped"
    
    @property
    def is_failed(self) -> bool:
        return self.status == "failed"
    
    @property
    def detection_count(self) -> int:
        return len(self.detections)
    
    @property
    def category_counts(self) -> Dict[str, int]:
        """按类别统计检测数量"""
        counts: Dict[str, int] = {}
        for det in self.detections:
            category = det.category.value
            counts[category] = counts.get(category, 0) + 1
        return counts
