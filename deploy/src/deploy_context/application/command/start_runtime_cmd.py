"""启动运行时命令"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StartRuntimeCmd:
    """启动运行时命令"""
    model_path: str
    camera_id: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    serial_port: Optional[str] = None
    serial_baudrate: int = 115200
    confidence_threshold: float = 0.5
    protocol: str = "default"
    device_profile: str = "rk3588"
