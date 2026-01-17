"""任务ID值对象"""

from dataclasses import dataclass
from typing import Self
from uuid import uuid4


@dataclass(frozen=True)
class JobId:
    """自动标注任务ID值对象"""
    value: str
    
    @classmethod
    def generate(cls) -> Self:
        """生成新的任务ID"""
        return cls(value=str(uuid4()))
    
    @classmethod
    def from_string(cls, value: str) -> Self:
        """从字符串创建任务ID"""
        if not value:
            raise ValueError("JobId cannot be empty")
        return cls(value=value)
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"JobId({self.value[:8]}...)"
