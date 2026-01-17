"""引擎仓储接口"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from shared_kernel.config.loader import ConfigLoader

from ..model.value_object.engine_type import EngineType


class IEngineRepository(ABC):
    """引擎仓储接口（定义在 Domain 层）"""
    
    @abstractmethod
    def get_engine(self, engine_type: EngineType) -> "IDetectionEngine":
        """根据类型获取检测引擎"""
        pass
    
    @abstractmethod
    def validate_engine_availability(self, engine_type: EngineType) -> bool:
        """验证引擎是否可用"""
        pass


class IDetectionEngine(ABC):
    """检测引擎接口"""
    
    @property
    @abstractmethod
    def engine_type(self) -> EngineType:
        """返回引擎类型"""
        pass
    
    @abstractmethod
    def detect(self, image_path: str) -> list:
        """检测图像中的对象
        
        Returns:
            list: 检测结果列表
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """验证引擎是否可用"""
        pass
