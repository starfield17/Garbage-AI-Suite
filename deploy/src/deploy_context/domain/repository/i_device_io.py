"""设备 IO 接口"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any


class ICamera(ABC):
    """相机接口"""
    
    @abstractmethod
    def open(self, camera_id: int = 0, width: int = 1280, height: int = 720) -> bool:
        """打开相机"""
        pass
    
    @abstractmethod
    def read(self) -> Optional[Any]:
        """读取帧"""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """检查是否打开"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭相机"""
        pass
    
    @abstractmethod
    def set_resolution(self, width: int, height: int) -> bool:
        """设置分辨率"""
        pass
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """获取分辨率"""
        pass


class ISerialDevice(ABC):
    """串口设备接口"""
    
    @abstractmethod
    def open(self, port: str, baudrate: int = 115200, timeout: float = 0.1) -> bool:
        """打开串口"""
        pass
    
    @abstractmethod
    def write(self, data: bytes) -> int:
        """写入数据"""
        pass
    
    @abstractmethod
    def read(self, size: int = 1) -> bytes:
        """读取数据"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """检查是否连接"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭串口"""
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """刷新缓冲区"""
        pass
