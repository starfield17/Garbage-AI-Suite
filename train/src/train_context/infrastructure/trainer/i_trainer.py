# train/src/train_context/infrastructure/trainer/i_trainer.py
"""训练器接口"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class ITrainer(ABC):
    """训练器接口
    
    定义训练器的通用接口
    """
    
    @abstractmethod
    def train(
        self,
        model_path: str,
        dataset_path: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        output_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """执行训练
        
        Args:
            model_path: 预训练模型路径
            dataset_path: 数据集路径
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            output_dir: 输出目录
            **kwargs: 其他参数
        
        Returns:
            训练结果字典
        """
        pass
    
    @abstractmethod
    def validate(self, model_path: str, dataset_path: str) -> Dict[str, Any]:
        """验证模型
        
        Args:
            model_path: 模型路径
            dataset_path: 验证数据集路径
        
        Returns:
            验证指标
        """
        pass
    
    @abstractmethod
    def export(self, model_path: str, output_path: str, format: str) -> str:
        """导出模型
        
        Args:
            model_path: 模型路径
            output_path: 输出路径
            format: 导出格式
        
        Returns:
            导出后的模型路径
        """
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> list:
        """支持的导出格式"""
        pass
