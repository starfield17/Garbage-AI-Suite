# train/src/train_context/domain/service/model_selector.py
"""模型选择服务"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class ModelInfo:
    """模型信息"""
    family: str
    variant: str
    parameters: int
    speed: str
    accuracy: str
    recommended_for: List[str]


class ModelSelector:
    """模型选择服务
    
    职责:
    - 根据场景推荐合适的模型
    - 提供模型比较信息
    """
    
    _MODEL_REGISTRY = {
        "yolo": {
            "n": ModelInfo(
                family="yolo",
                variant="yolov8n",
                parameters=3.2,
                speed="fastest",
                accuracy="lowest",
                recommended_for=["cpu", "edge devices", "raspberry pi"]
            ),
            "s": ModelInfo(
                family="yolo",
                variant="yolov8s",
                parameters=11.2,
                speed="fast",
                accuracy="low",
                recommended_for=["cpu", "轻量级部署"]
            ),
            "m": ModelInfo(
                family="yolo",
                variant="yolov8m",
                parameters=25.9,
                speed="medium",
                accuracy="medium",
                recommended_for=["gpu", "平衡场景"]
            ),
            "l": ModelInfo(
                family="yolo",
                variant="yolov8l",
                parameters=43.7,
                speed="slow",
                accuracy="high",
                recommended_for=["gpu", "高精度需求"]
            ),
            "x": ModelInfo(
                family="yolo",
                variant="yolov8x",
                parameters=68.2,
                speed="slowest",
                accuracy="highest",
                recommended_for=["gpu", "服务器部署"]
            ),
        },
        "faster_rcnn": {
            "resnet50": ModelInfo(
                family="faster_rcnn",
                variant="resnet50",
                parameters=41.5,
                speed="slow",
                accuracy="high",
                recommended_for=["gpu", "高精度检测"]
            ),
            "resnet101": ModelInfo(
                family="faster_rcnn",
                variant="resnet101",
                parameters=60.4,
                speed="slowest",
                accuracy="higher",
                recommended_for=["gpu", "高精度研究"]
            ),
        }
    }
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        models = []
        for family, variants in self._MODEL_REGISTRY.items():
            for variant in variants.keys():
                models.append(f"{family}_{variant}")
        return models
    
    def get_model_info(self, model_family: str, variant: str) -> Optional[ModelInfo]:
        """获取模型信息
        
        Args:
            model_family: 模型族
            variant: 变体
        
        Returns:
            模型信息
        """
        family_info = self._MODEL_REGISTRY.get(model_family.lower(), {})
        return family_info.get(variant.lower())
    
    def recommend_model(
        self,
        device: str = "cpu",
        priority: str = "balance",
        min_accuracy: float = 0.5
    ) -> ModelInfo:
        """推荐模型
        
        Args:
            device: 设备类型 (cpu, gpu, edge)
            priority: 优先级 (speed, accuracy, balance)
            min_accuracy: 最低精度要求
        
        Returns:
            推荐的模型信息
        """
        candidates = []
        
        for family, variants in self._MODEL_REGISTRY.items():
            for variant, info in variants.items():
                if device == "cpu" and info.speed in ["slow", "slowest"]:
                    continue
                
                candidates.append((family, variant, info))
        
        if not candidates:
            return self._MODEL_REGISTRY["yolo"]["n"]
        
        if priority == "speed":
            candidates.sort(key=lambda x: x[2].speed)
        elif priority == "accuracy":
            candidates.sort(key=lambda x: x[2].accuracy, reverse=True)
        else:
            candidates.sort(key=lambda x: x[2].parameters)
        
        return candidates[0][2]
    
    def compare_models(
        self,
        model1_family: str,
        model1_variant: str,
        model2_family: str,
        model2_variant: str
    ) -> Dict[str, Any]:
        """比较两个模型
        
        Returns:
            比较结果字典
        """
        info1 = self.get_model_info(model1_family, model1_variant)
        info2 = self.get_model_info(model2_family, model2_variant)
        
        if not info1 or not info2:
            return {"error": "Model not found"}
        
        return {
            "model1": {
                "name": f"{model1_family}_{model1_variant}",
                "parameters_millions": info1.parameters,
                "speed": info1.speed,
                "accuracy": info1.accuracy,
            },
            "model2": {
                "name": f"{model2_family}_{model2_variant}",
                "parameters_millions": info2.parameters,
                "speed": info2.speed,
                "accuracy": info2.accuracy,
            },
            "recommendation": self._get_comparison_recommendation(info1, info2)
        }
    
    def _get_comparison_recommendation(
        self,
        info1: ModelInfo,
        info2: ModelInfo
    ) -> str:
        """获取比较建议"""
        if info1.parameters < info2.parameters and info1.accuracy >= info2.accuracy:
            return f"{info1.variant} is more efficient"
        elif info2.parameters < info1.parameters and info2.accuracy >= info1.accuracy:
            return f"{info2.variant} is more efficient"
        else:
            return f"Choose based on {'speed' if info1.speed == 'fastest' else 'accuracy'} priority"
