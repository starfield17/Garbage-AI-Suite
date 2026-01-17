"""值对象定义"""

from enum import Enum


class EngineType(Enum):
    """检测引擎类型
    
    支持的引擎:
    - YOLO: 基于 ultralytics 的 YOLO 系列
    - FASTER_RCNN: 基于 torchvision 的 Faster R-CNN
    - VLM: 视觉语言模型（如 Qwen-VL）
    - ENSEMBLE: 组合引擎（YOLO 出框 + VLM 定类）
    """
    YOLO = "yolo"
    FASTER_RCNN = "faster_rcnn"
    VLM = "vlm"
    ENSEMBLE = "ensemble"
    
    @classmethod
    def from_string(cls, value: str) -> "EngineType":
        """从字符串解析"""
        normalized = value.lower().strip()
        for engine in cls:
            if engine.value == normalized:
                return engine
        raise ValueError(f"Unknown engine type: {value}. Valid: {[e.value for e in cls]}")
    
    def supports_bounding_box(self) -> bool:
        """是否支持边界框输出"""
        return self in (EngineType.YOLO, EngineType.FASTER_RCNN, EngineType.ENSEMBLE)
    
    def requires_api_key(self) -> bool:
        """是否需要 API Key"""
        return self in (EngineType.VLM, EngineType.ENSEMBLE)
