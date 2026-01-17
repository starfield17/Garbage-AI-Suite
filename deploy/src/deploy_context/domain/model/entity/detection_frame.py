"""检测帧实体 - 封装单帧检测结果"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

from shared_kernel.domain.base import Entity
from shared_kernel.domain.taxonomy import WasteCategory


@dataclass
class DetectionFrame(Entity):
    """检测帧实体
    
    职责:
    - 封装单帧的检测结果
    - 包含检测框、类别、置信度信息
    - 管理检测元数据
    """
    frame_id: str
    image_width: int
    image_height: int
    detected_category: Optional[WasteCategory] = None
    confidence: Optional[float] = None
    x_normalized: Optional[float] = None  # 归一化中心X坐标
    y_normalized: Optional[float] = None  # 归一化中心Y坐标
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_processed: bool = False
    
    @property
    def id(self) -> str:
        return self.frame_id
    
    @property
    def has_detection(self) -> bool:
        return self.detected_category is not None
    
    def to_serial_coordinates(self) -> tuple:
        """转换为串口坐标 (0-255)"""
        if self.x_normalized is None or self.y_normalized is None:
            return (0, 0)
        return (
            min(255, max(0, int(self.x_normalized * 255))),
            min(255, max(0, int(self.y_normalized * 255)))
        )
    
    def mark_processed(self) -> None:
        """标记为已处理"""
        self.is_processed = True
    
    def with_updated_detection(
        self,
        category: WasteCategory,
        confidence: float,
        x: float,
        y: float
    ) -> "DetectionFrame":
        """创建新的检测帧（不可变操作）"""
        return DetectionFrame(
            frame_id=self.frame_id,
            image_width=self.image_width,
            image_height=self.image_height,
            detected_category=category,
            confidence=confidence,
            x_normalized=x,
            y_normalized=y,
            timestamp=self.timestamp,
            metadata=self.metadata.copy(),
            is_processed=False
        )
