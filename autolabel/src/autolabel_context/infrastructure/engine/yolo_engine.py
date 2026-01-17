"""YOLO检测引擎"""

import cv2
from pathlib import Path
from typing import List, Dict, Any

from ultralytics import YOLO

from shared_kernel.config.loader import ConfigLoader
from shared_kernel.domain.taxonomy import WasteCategory

from .i_detection_engine import IDetectionEngine
from autolabel_context.domain.model.value_object.engine_type import EngineType


class YoloEngine(IDetectionEngine):
    """YOLO检测引擎实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化YOLO引擎
        
        Args:
            config: 引擎配置，包含model_path, confidence_threshold等
        """
        self._config = config
        self._model_path = config.get("model_path")
        self._confidence_threshold = config.get("confidence_threshold", 0.5)
        self._model = None
        self._category_mapping = self._build_category_mapping()
    
    def _build_category_mapping(self) -> Dict[int, str]:
        """构建类别映射"""
        return {
            0: "Kitchen_waste",
            1: "Recyclable_waste",
            2: "Hazardous_waste",
            3: "Other_waste"
        }
    
    @property
    def engine_type(self) -> EngineType:
        return EngineType.YOLO
    
    def _load_model(self) -> None:
        """加载模型"""
        if self._model is None:
            self._model = YOLO(self._model_path)
    
    def validate(self) -> bool:
        """验证模型是否可用"""
        try:
            model_path = Path(self._model_path)
            if not model_path.exists():
                return False
            self._load_model()
            return True
        except Exception:
            return False
    
    def detect(self, image_path: str) -> List[dict]:
        """检测图像中的对象
        
        Args:
            image_path: 图像路径
        
        Returns:
            检测结果列表
        """
        self._load_model()
        
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        results = self._model(img, conf=self._confidence_threshold, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                category_name = self._category_mapping.get(cls)
                if category_name:
                    detections.append({
                        "name": category_name,
                        "confidence": conf,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })
        
        return detections
