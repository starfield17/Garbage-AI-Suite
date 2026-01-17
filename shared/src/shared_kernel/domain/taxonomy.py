"""垃圾分类领域核心值对象"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class WasteCategory(Enum):
    """垃圾主分类（领域语言）
    
    这是整个系统的核心领域概念，所有模块共享此定义
    """
    KITCHEN_WASTE = "Kitchen_waste"         # 厨余垃圾
    RECYCLABLE_WASTE = "Recyclable_waste"   # 可回收垃圾
    HAZARDOUS_WASTE = "Hazardous_waste"     # 有害垃圾
    OTHER_WASTE = "Other_waste"             # 其他垃圾
    
    @classmethod
    def from_string(cls, value: str) -> "WasteCategory":
        """从字符串解析分类"""
        normalized = value.strip()
        for category in cls:
            if category.value.lower() == normalized.lower():
                return category
        raise ValueError(f"Unknown waste category: {value}")


@dataclass(frozen=True)
class LabelAlias:
    """标签别名值对象
    
    用于将细粒度标签（如 potato, bottle）映射到主分类
    """
    alias: str                      # 细粒度标签，如 "potato"
    category: WasteCategory         # 对应的主分类
    confidence_boost: float = 0.0   # 置信度加成（可选）
    
    def matches(self, label: str) -> bool:
        """判断是否匹配此别名"""
        return self.alias.lower() == label.lower()


@dataclass(frozen=True)
class TaxonomyVersion:
    """分类体系版本"""
    major: int
    minor: int
    patch: int
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def is_compatible_with(self, other: "TaxonomyVersion") -> bool:
        """检查版本兼容性（主版本相同即兼容）"""
        return self.major == other.major
