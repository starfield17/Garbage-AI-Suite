# train/src/train_context/domain/repository/i_artifact_store.py
"""产物存储仓储接口"""

from abc import ABC, abstractmethod
from typing import Optional


class IArtifactStore(ABC):
    """训练产物存储仓储接口"""
    
    @abstractmethod
    def save_artifact(self, run_id: str, artifact_path: str) -> str:
        """保存训练产物
        
        Args:
            run_id: 运行 ID
            artifact_path: 产物路径
        
        Returns:
            保存后的产物路径
        """
        pass
    
    @abstractmethod
    def get_artifact(self, run_id: str) -> Optional[str]:
        """获取训练产物路径
        
        Args:
            run_id: 运行 ID
        
        Returns:
            产物路径，如果不存在则返回 None
        """
        pass
    
    @abstractmethod
    def artifact_exists(self, run_id: str) -> bool:
        """检查产物是否存在"""
        pass
    
    @abstractmethod
    def delete_artifact(self, run_id: str) -> bool:
        """删除产物"""
        pass
