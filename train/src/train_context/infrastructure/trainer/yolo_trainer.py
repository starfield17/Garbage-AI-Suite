# train/src/train_context/infrastructure/trainer/yolo_trainer.py
"""YOLO 训练器实现"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from .i_trainer import ITrainer


class YoloTrainer(ITrainer):
    """YOLO 训练器
    
    支持 YOLO 系列模型的训练和导出
    """
    
    def __init__(self, device: str = "auto"):
        """初始化 YOLO 训练器
        
        Args:
            device: 训练设备 (auto/cpu/cuda)
        """
        self._device = device
        self._supported_formats = ["pt", "onnx", "rknn"]
    
    @property
    def supported_formats(self) -> list:
        return self._supported_formats
    
    def train(
        self,
        model_path: str,
        dataset_path: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        output_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """执行 YOLO 训练"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Please install with: pip install ultralytics")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model = YOLO(model_path)
        
        results = model.train(
            data=self._create_data_yaml(dataset_path),
            epochs=epochs,
            batch=batch_size,
            lr0=learning_rate,
            device=self._device,
            project=str(output_path),
            name="train_results",
            exist_ok=True,
            **kwargs
        )
        
        return {
            "success": True,
            "final_epoch": epochs,
            "final_metrics": results.metrics.__dict__,
            "best_model": str(results.save_dir / "weights" / "best.pt"),
            "last_model": str(results.save_dir / "weights" / "last.pt"),
        }
    
    def validate(self, model_path: str, dataset_path: str) -> Dict[str, Any]:
        """验证 YOLO 模型"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed")
        
        model = YOLO(model_path)
        results = model.val(
            data=self._create_data_yaml(dataset_path),
            device=self._device
        )
        
        return {
            "mAP50": results.metrics.box.map50,
            "mAP50_95": results.metrics.box.map,
            "precision": results.metrics.box.mp,
            "recall": results.metrics.box.mr,
            "f1": results.metrics.box.f1,
        }
    
    def export(self, model_path: str, output_path: str, format: str) -> str:
        """导出 YOLO 模型"""
        if format not in self._supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self._supported_formats}")
        
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed")
        
        model = YOLO(model_path)
        
        exported_path = model.export(
            format=format,
            imgsz=640,
            device=self._device
        )
        
        if output_path != model_path:
            import shutil
            shutil.move(exported_path, output_path)
            return output_path
        
        return str(exported_path)
    
    def _create_data_yaml(self, dataset_path: str) -> str:
        """创建 YOLO 数据配置文件"""
        data_yaml = {
            "path": dataset_path,
            "train": "images",
            "val": "images",
            "names": {
                0: "Kitchen_waste",
                1: "Recyclable_waste",
                2: "Hazardous_waste",
                3: "Other_waste"
            }
        }
        
        yaml_path = Path(dataset_path) / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f)
        
        return str(yaml_path)
