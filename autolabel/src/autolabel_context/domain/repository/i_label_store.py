"""标签存储接口"""

from abc import ABC, abstractmethod
from typing import Protocol

from ..model.aggregate.autolabel_job import AutoLabelJob


class ILabelStore(ABC):
    """标签存储接口（定义在 Domain 层）"""
    
    @abstractmethod
    def save_job(self, job: AutoLabelJob) -> None:
        """保存任务"""
        pass
    
    @abstractmethod
    def find_by_id(self, job_id: str) -> AutoLabelJob | None:
        """根据 ID 查找任务"""
        pass
