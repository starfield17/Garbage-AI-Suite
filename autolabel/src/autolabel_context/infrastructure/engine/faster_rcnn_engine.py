"""Faster R-CNN检测引擎"""

import cv2
import torch
from pathlib import Path
from typing import List, Dict, Any

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T

from .i_detection_engine import IDetectionEngine
from autolabel_context.domain.model.value_object.engine_type import EngineType


class FasterRcnnEngine(IDetectionEngine):
    """Faster R-CNN检测引擎实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化Faster R-CNN引擎
        
        Args:
            config: 引擎配置，包含model_path, confidence_threshold等
        """
        self._config = config
        self._model_path = config.get("model_path")
        self._confidence_threshold = config.get("confidence_threshold", 0.5)
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._category_names = {
            0: "Kitchen_waste",
            1: "Recyclable_waste",
            2: "Hazardous_waste",
            3: "Other_waste"
        }
        
        self._transforms = T.Compose([
            T.ToTensor(),
        ])
    
    @property
    def engine_type(self) -> EngineType:
        return EngineType.FASTER_RCNN
    
    def _load_model(self) -> None:
        """加载模型"""
        if self._model is None:
            num_classes = len(self._category_names) + 1  # +1 for background
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            state_dict = torch.load(self._model_path, map_location=self._device)
            model.load_state_dict(state_dict)
            
            model.to(self._device)
            model.eval()
            self._model = model
    
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
        """检测图像中的对象"""
        self._load_model()
        
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        img_height, img_width = img.shape[:2]
        img_tensor = self._transforms(img)
        img_tensor = img_tensor.to(self._device)
        
        with torch.no_grad():
            predictions = self._model([img_tensor])
        
        detections = []
        boxes = predictions[0]["boxes"]
        labels = predictions[0]["labels"]
        scores = predictions[0]["scores"]
        
        for box, label, score in zip(boxes, labels, scores):
            if score < self._confidence_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, box.tolist())
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            class_id = label.item() - 1  # 减去背景类
            if class_id in self._category_names:
                detections.append({
                    "name": self._category_names[class_id],
                    "confidence": score.item(),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })
        
        return detections
