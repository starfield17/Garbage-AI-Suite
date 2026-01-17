"""计数实体 - 管理分类统计"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

from shared_kernel.domain.base import Entity
from shared_kernel.domain.taxonomy import WasteCategory


@dataclass
class Counter(Entity):
    """分类计数实体
    
    职责:
    - 管理各类别的检测计数
    - 跟踪计数历史
    - 提供统计查询
    """
    counter_id: str
    counts: Dict[WasteCategory, int] = field(default_factory=dict)
    total_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.counts:
            object.__setattr__(self, 'counts', {
                WasteCategory.KITCHEN_WASTE: 0,
                WasteCategory.RECYCLABLE_WASTE: 0,
                WasteCategory.HAZARDOUS_WASTE: 0,
                WasteCategory.OTHER_WASTE: 0,
            })
    
    @property
    def id(self) -> str:
        return self.counter_id
    
    def increment(self, category: WasteCategory) -> None:
        """增加指定类别的计数"""
        self.counts[category] = self.counts.get(category, 0) + 1
        self.total_count += 1
        self.last_updated = datetime.utcnow()
    
    def get_count(self, category: WasteCategory) -> int:
        """获取指定类别的计数"""
        return self.counts.get(category, 0)
    
    def get_counts_by_protocol(self, protocol_map: Dict[int, int]) -> Dict[int, int]:
        """获取按协议编号的计数
        
        Args:
            protocol_map: 分类到协议编号的映射
        """
        result = {}
        for category, count in self.counts.items():
            if count > 0 and category in protocol_map:
                protocol_id = protocol_map[category]
                result[protocol_id] = result.get(protocol_id, 0) + count
        return result
    
    def reset(self) -> None:
        """重置所有计数"""
        for category in self.counts:
            self.counts[category] = 0
        self.total_count = 0
        self.last_updated = datetime.utcnow()
    
    def get_distribution(self) -> Dict[WasteCategory, float]:
        """获取各类别的分布比例"""
        if self.total_count == 0:
            return {cat: 0.0 for cat in self.counts}
        return {
            cat: count / self.total_count
            for cat, count in self.counts.items()
        }
