# train/src/train_context/domain/model/value_object/hyper_params.py
"""超参数值对象"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class HyperParams:
    """训练超参数
    
    封装所有训练相关的超参数配置
    """
    epochs: int
    batch_size: int
    learning_rate: float
    
    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }
    
    @classmethod
    def default_yolo(cls) -> "HyperParams":
        return cls(epochs=100, batch_size=16, learning_rate=0.01)
    
    @classmethod
    def default_faster_rcnn(cls) -> "HyperParams":
        return cls(epochs=50, batch_size=8, learning_rate=0.002)
