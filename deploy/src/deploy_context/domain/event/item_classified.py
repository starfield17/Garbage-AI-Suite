"""物品分类领域事件"""

from dataclasses import dataclass, field
from datetime import datetime

from shared_kernel.domain.base import DomainEvent


@dataclass
class ItemClassified(DomainEvent):
    """物品分类完成事件
    
    当分类会话完成物品分类时发布此事件
    """
    session_id: str = ""
    category_id: int = 0
    x: float = 0.0
    y: float = 0.0
    
    def __post_init__(self):
        if not 0.0 <= self.x <= 1.0:
            raise ValueError(f"x must be between 0 and 1, got {self.x}")
        if not 0.0 <= self.y <= 1.0:
            raise ValueError(f"y must be between 0 and 1, got {self.y}")
