"""检测引擎接口"""

from abc import ABC, abstractmethod
from typing import List

from autolabel_context.domain.model.value_object.engine_type import EngineType


class IDetectionEngine(ABC):
    """检测引擎接口"""
    
    @property
    @abstractmethod
    def engine_type(self) -> EngineType:
        pass
    
    @abstractmethod
    def detect(self, image_path: str) -> List[dict]:
        """检测图像中的对象
        
        Returns:
            List[dict]: 检测结果列表，每个dict包含:
                - category: 分类名称
                - confidence: 置信度
                - x1, y1, x2, y2: 边界框坐标
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """验证引擎是否可用"""
        pass
