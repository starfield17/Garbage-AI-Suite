# train/src/train_context/domain/model/entity/model_spec.py
"""模型规格实体"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelSpec:
    """模型规格实体
    
    封装模型的族系和变体信息
    """
    family: str
    variant: str
    pretrained: bool = True
    pretrained_path: Optional[str] = None
    
    def __post_init__(self):
        if not self.family:
            raise ValueError("Model family cannot be empty")
        if not self.variant:
            raise ValueError("Model variant cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "variant": self.variant,
            "pretrained": self.pretrained,
            "pretrained_path": self.pretrained_path,
        }
    
    @classmethod
    def yolov8(cls, variant: str = "n") -> "ModelSpec":
        return cls(family="yolo", variant=f"yolov8_{variant}")
    
    @classmethod
    def yolov12(cls, variant: str = "n") -> "ModelSpec":
        return cls(family="yolo", variant=f"yolov12_{variant}")
    
    @classmethod
    def faster_rcnn(cls, variant: str = "resnet50") -> "ModelSpec":
        return cls(family="faster_rcnn", variant=variant)
    
    @property
    def full_name(self) -> str:
        return f"{self.family}_{self.variant}"
