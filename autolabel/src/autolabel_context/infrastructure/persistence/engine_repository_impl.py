"""引擎仓储实现"""

from typing import Dict, Any

from shared_kernel.config.loader import ConfigLoader

from autolabel_context.domain.repository.i_engine_repository import IEngineRepository, IDetectionEngine
from autolabel_context.domain.model.value_object.engine_type import EngineType

from ..engine.yolo_engine import YoloEngine
from ..engine.faster_rcnn_engine import FasterRcnnEngine
from ..engine.vlm_engine import VlmEngine


class EngineRepositoryImpl(IEngineRepository):
    """引擎仓储实现"""
    
    def __init__(self, config_loader: ConfigLoader):
        self._config_loader = config_loader
        self._engines: Dict[EngineType, IDetectionEngine] = {}
    
    def get_engine(self, engine_type: EngineType) -> IDetectionEngine:
        if engine_type not in self._engines:
            self._engines[engine_type] = self._create_engine(engine_type)
        return self._engines[engine_type]
    
    def _create_engine(self, engine_type: EngineType) -> IDetectionEngine:
        config = self._config_loader.get_model_config(engine_type.value)
        
        if engine_type == EngineType.YOLO:
            return YoloEngine(config)
        elif engine_type == EngineType.FASTER_RCNN:
            return FasterRcnnEngine(config)
        elif engine_type == EngineType.VLM:
            api_key = self._config_loader.get_env("VLM_API_KEY")
            config["api_key"] = api_key
            return VlmEngine(config)
        
        raise ValueError(f"Unknown engine type: {engine_type}")
    
    def validate_engine_availability(self, engine_type: EngineType) -> bool:
        try:
            engine = self.get_engine(engine_type)
            return engine.validate()
        except Exception:
            return False
