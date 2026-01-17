"""RKNN 推理运行时实现"""

import cv2
import numpy as np
from typing import Optional, List, Any, Dict
from pathlib import Path

from shared_kernel.domain.annotation import (
    Detection, LabelFile, BoundingBox, Confidence, DetectionSource
)
from shared_kernel.domain.taxonomy import WasteCategory

from ..domain.repository import IInferenceRuntime


class RknnRuntime(IInferenceRuntime):
    """RKNN 推理运行时
    
    使用瑞芯微 NPU 进行高效推理
    适用于 RK3588 等设备
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
    ):
        self._model = None
        self._model_path: Optional[str] = None
        self._confidence_threshold = confidence_threshold
        self._rknn = None
        self._class_mapping: Dict[int, WasteCategory] = {
            0: WasteCategory.KITCHEN_WASTE,
            1: WasteCategory.RECYCLABLE_WASTE,
            2: WasteCategory.HAZARDOUS_WASTE,
            3: WasteCategory.OTHER_WASTE,
        }
    
    def load_model(self, model_path: str) -> None:
        """加载 RKNN 模型"""
        try:
            from rknn.api import RKNN
            self._rknn = RKNN()
            self._rknn.load_rknn(model_path)
            self._model_path = model_path
        except ImportError:
            raise RuntimeError("rknn-toolkit not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to load RKNN model: {e}")
    
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
        
        # 预处理
        input_tensor = self._preprocess(image)
        
        # 推理
        outputs = self._rknn.inference(inputs=[input_tensor])
        
        # 后处理
        detections = self._postprocess(outputs, image.shape)
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 调整大小、归一化等
        resized = cv2.resize(image, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0)
    
    def _postprocess(self, outputs: List[np.ndarray], image_shape: tuple) -> List[Detection]:
        """后处理输出"""
        detections = []
        h, w = image_shape[:2]
        
        if not outputs or len(outputs) == 0:
            return detections
        
        output = outputs[0]
        
        if len(output.shape) == 3:
            output = output[0]
        
        for detection in output:
            conf = float(detection[4])
            if conf < self._confidence_threshold:
                continue
            
            cls = int(detection[5])
            category = self._class_mapping.get(cls)
            if category is None:
                continue
            
            bbox = detection[0:4]
            x_center = (bbox[0] + bbox[2] / 2) / 640
            y_center = (bbox[1] + bbox[3] / 2) / 640
            width = bbox[2] / 640
            height = bbox[3] / 640
            
            detection_obj = Detection.create(
                category=category,
                confidence=conf,
                bounding_box=BoundingBox(
                    x_center=max(0, min(1, x_center)),
                    y_center=max(0, min(1, y_center)),
                    width=max(0, min(1, width)),
                    height=max(0, min(1, height))
                ),
                source=DetectionSource.YOLO
            )
            detections.append(detection_obj)
        
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
        return self._rknn is not None
    
    def unload(self) -> None:
        """卸载模型"""
        if self._rknn:
            self._rknn.release()
        self._rknn = None
        self._model_path = None
    
    def set_class_mapping(self, mapping: Dict[int, WasteCategory]) -> None:
        """设置分类映射"""
        self._class_mapping = mapping
