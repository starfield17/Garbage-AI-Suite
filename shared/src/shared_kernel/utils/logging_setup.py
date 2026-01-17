"""日志设置"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .fs import ensure_dir


def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None
) -> logging.Logger:
    """设置日志配置
    
    Args:
        config: 日志配置字典
        config_path: 配置文件路径（如果 config 为空则从文件加载）
    
    Returns:
        配置好的根 logger
    """
    import yaml
    
    if config is None:
        if config_path is None:
            # 使用默认配置
            config = _get_default_config()
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
    
    # 确保日志目录存在
    log_file = config.get("handlers", {}).get("file", {}).get("filename")
    if log_file:
        ensure_dir(Path(log_file).parent)
    
    # 应用配置
    logging.config.dictConfig(config)
    
    return logging.getLogger()


def _get_default_config() -> Dict[str, Any]:
    """获取默认日志配置"""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"]
        }
    }
