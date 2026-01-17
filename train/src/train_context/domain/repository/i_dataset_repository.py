# train/src/train_context/domain/repository/i_dataset_repository.py
"""数据集仓储接口"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class IDatasetRepository(ABC):
    """数据集仓储接口"""
    
    @abstractmethod
    def list_datasets(self) -> List[str]:
        """列出可用数据集"""
        pass
    
    @abstractmethod
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        pass
    
    @abstractmethod
    def validate_dataset(self, dataset_path: str) -> bool:
        """验证数据集完整性"""
        pass
    
    @abstractmethod
    def get_dataset_path(self, dataset_name: str) -> Optional[str]:
        """获取数据集路径"""
        pass
