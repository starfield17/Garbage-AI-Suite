"""统一配置加载器 - 禁止业务层直接读文件"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigLoader:
    """配置加载器
    
    约束:
    - 所有配置必须通过此类读取
    - 支持环境变量覆盖
    - 单例模式确保配置一致性
    """
    
    _instance: Optional["ConfigLoader"] = None
    
    def __init__(self):
        # 配置目录在 REFACTOR/config/
        # Path(__file__) = .../shared_kernel/config/loader.py
        # parent = .../shared_kernel/config
        # parent.parent = .../shared_kernel
        # parent.parent.parent = .../shared/src
        # parent.parent.parent.parent = .../shared
        # parent.parent.parent.parent.parent = .../REFACTOR
        # 因此需要向上 5 级到 REFACTOR，然后添加 config
        self._config_root: Path = Path(__file__).resolve().parents[4] / "config"
        self._cache: Dict[str, Any] = {}
    
    @classmethod
    def set_config_root(cls, path: Path) -> None:
        """设置配置根目录（用于测试）"""
        cls._config_root = path
        if cls._instance:
            cls._instance._cache.clear()
    
    def _load_yaml(self, relative_path: str) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        if relative_path in self._cache:
            return self._cache[relative_path]
        
        file_path = self._config_root / relative_path
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        self._cache[relative_path] = data
        return data
    
    def get_waste_categories(self) -> Dict[str, Any]:
        """获取垃圾分类定义"""
        return self._load_yaml("taxonomy/waste_categories.yaml")
    
    def get_label_aliases(self) -> Dict[str, Any]:
        """获取标签别名映射"""
        return self._load_yaml("taxonomy/label_alias.yaml")
    
    def get_train_class_map(self, model_family: str = "default") -> Dict[int, str]:
        """获取训练类别编号映射
        
        Args:
            model_family: 模型族（default/yolo/faster_rcnn）
        
        Returns:
            {class_id: category_name} 映射
        """
        data = self._load_yaml("mappings/train_class_map.yaml")
        if model_family in data:
            return data[model_family]
        return data.get("default", {})
    
    def get_deploy_class_map(self, protocol: str = "default") -> Dict[int, int]:
        """获取部署编号映射（分类ID -> 串口协议字节）
        
        Args:
            protocol: 协议类型（default/stm32/arduino）
        """
        data = self._load_yaml("mappings/deploy_class_map.yaml")
        if protocol in data:
            return data[protocol]
        return data.get("default", {})
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """获取模型配置
        
        Args:
            model_type: 模型类型（yolo/faster_rcnn/vlm_qwen）
        """
        return self._load_yaml(f"models/{model_type}.yaml")
    
    def get_model_registry(self) -> Dict[str, Any]:
        """获取模型注册表"""
        return self._load_yaml("models/registry.yaml")
    
    def get_device_profile(self, device: str) -> Dict[str, Any]:
        """获取设备配置
        
        Args:
            device: 设备类型（raspi/rk3588/jetson）
        """
        return self._load_yaml(f"profiles/device_{device}.yaml")
    
    def get_prompt_template(self, template_name: str) -> str:
        """获取 prompt 模板"""
        template_path = self._config_root / "prompts" / "vlm" / f"{template_name}.jinja2"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self._load_yaml("logging.yaml")
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """获取环境变量（优先级高于配置文件）"""
        return os.environ.get(key, default)
