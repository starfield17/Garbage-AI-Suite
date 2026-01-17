"""YOLO 推理运行时实现"""

import cv2
import numpy as np
from typing import Optional, List, Any, Dict
from pathlib import Path

from shared_kernel.domain.annotation import (
    Detection, LabelFile, BoundingBox, Confidence, DetectionSource
)
from shared_kernel.domain.taxonomy import WasteCategory

from ..domain.repository import IInferenceRuntime


class YoloRuntime(IInferenceRuntime):
    """YOLO 推理运行时
    
    使用 ultralytics YOLO 进行实时推理
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: str = "cuda"
    ):
        self._model = None
        self._model_path: Optional[str] = None
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._class_mapping: Dict[int, WasteCategory] = {
            0: WasteCategory.KITCHEN_WASTE,
            1: WasteCategory.RECYCLABLE_WASTE,
            2: WasteCategory.HAZARDOUS_WASTE,
            3: WasteCategory.OTHER_WASTE,
        }
    
    def load_model(self, model_path: str) -> None:
        """加载 YOLO 模型"""
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._model.fuse()
            self._model_path = model_path
        except ImportError:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
    
    def infer(self, image) -> List[Detection]:
        """执行推理"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        results = self._model(image, conf=self._confidence_threshold, verbose=False)
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    xywh = box.xywh[0].cpu().numpy()
                    x_center, y_center, width, height = xywh
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    category = self._class_mapping.get(cls)
                    if category is None:
                        continue
                    
                    detection = Detection.create(
                        category=category,
                        confidence=float(conf),
                        bounding_box=BoundingBox(
                            x_center=float(x_center) / image.shape[1],
                            y_center=float(y_center) / image.shape[0],
                            width=float(width) / image.shape[1],
                            height=float(height) / image.shape[0]
                        ),
                        source=DetectionSource.YOLO
                    )
                    detections.append(detection)
        
        return detections
    
    def detect(self, image_path: str) -> LabelFile:
        """检测图像"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        h, w = image.shape[:2]
        detections = self.infer(image)
        
        return LabelFile(
            file_id=str(Path(image_path).stem),
            image_path=image_path,
            image_width=w,
            image_height=h,
            detections=detections
        )
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model is not None
    
    def unload(self) -> None:
        """卸载模型"""
        self._model = None
        self._model_path = None
    
    def set_class_mapping(self, mapping: Dict[int, WasteCategory]) -> None:
        """设置分类映射"""
        self._class_mapping = mapping
