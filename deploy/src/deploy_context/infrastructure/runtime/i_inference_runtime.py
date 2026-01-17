"""推理运行时接口"""

from abc import ABC, abstractmethod
from typing import Optional, List, Any

from shared_kernel.domain.annotation import Detection, LabelFile


class IInferenceRuntime(ABC):
    """推理运行时接口
    
    定义统一的推理接口，所有运行时实现必须遵守
    """
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """加载模型"""
        pass
    
    @abstractmethod
    def infer(self, image) -> List[Detection]:
        """执行推理
        
        Args:
            image: 输入图像
            
        Returns:
            Detection 列表
        """
        pass
    
    @abstractmethod
    def detect(self, image_path: str) -> LabelFile:
        """检测图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            LabelFile: 标注结果
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """卸载模型"""
        pass
