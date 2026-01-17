# train/src/train_context/domain/model/value_object/class_mapping_id.py
"""类别映射 ID 值对象"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassMappingId:
    """类别映射 ID
    
    用于引用 config/mappings/train_class_map.yaml 中的映射配置
    
    示例:
    - "default": 默认四分类
    - "yolo_v12": YOLO v12 特定映射
    - "custom_10class": 自定义十分类
    """
    value: str
    
    def __post_init__(self):
        if not self.value:
            raise ValueError("Class mapping ID cannot be empty")
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def default(cls) -> "ClassMappingId":
        return cls("default")
