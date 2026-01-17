"""引擎选择器领域服务"""

from typing import Dict, Any, Optional

from shared_kernel.config.loader import ConfigLoader

from ..model.value_object.engine_type import EngineType


class EngineSelector:
    """引擎选择器
    
    职责:
    - 根据配置选择合适的引擎
    - 验证引擎可用性
    - 提供引擎配置
    """
    
    def __init__(self, config_loader: ConfigLoader):
        self._config_loader = config_loader
        self._registry = config_loader.get_model_registry()
    
    def select_engine(
        self,
        engine_type: EngineType,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """选择引擎并返回配置
        
        Args:
            engine_type: 引擎类型
            model_id: 具体模型 ID（可选）
        
        Returns:
            引擎配置字典
        """
        engine_config = self._config_loader.get_model_config(engine_type.value)
        
        if model_id:
            model_config = self._get_model_from_registry(model_id)
            if model_config:
                engine_config.update(model_config)
        
        return engine_config
    
    def _get_model_from_registry(self, model_id: str) -> Optional[Dict[str, Any]]:
        """从注册表获取模型配置"""
        models = self._registry.get("models", {})
        return models.get(model_id)
    
    def get_available_engines(self) -> list:
        """获取可用引擎列表"""
        return [e.value for e in EngineType]
    
    def validate_engine_availability(self, engine_type: EngineType) -> bool:
        """验证引擎是否可用"""
        try:
            config = self._config_loader.get_model_config(engine_type.value)
            if engine_type.requires_api_key():
                api_key = self._config_loader.get_env(f"{engine_type.value.upper()}_API_KEY")
                return api_key is not None
            return True
        except FileNotFoundError:
            return False
