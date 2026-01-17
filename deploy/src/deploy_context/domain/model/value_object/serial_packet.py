"""串口数据包值对象 - 用于串口通信协议"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SerialPacket:
    """串口数据包值对象
    
    封装串口通信的数据格式:
    - class_id: 垃圾分类编号 (1-4 对应厨余/可回收/有害/其他)
    - x: X坐标 (0-255)
    - y: Y坐标 (0-255)
    
    使用归一化坐标(0-255)来适配串口传输限制
    """
    class_id: int
    x: int
    y: int
    
    def __post_init__(self):
        if not 0 <= self.class_id <= 4:
            raise ValueError(f"class_id must be between 0 and 4, got {self.class_id}")
        if not 0 <= self.x <= 255:
            raise ValueError(f"x must be between 0 and 255, got {self.x}")
        if not 0 <= self.y <= 255:
            raise ValueError(f"y must be between 0 and 255, got {self.y}")
    
    def to_bytes(self) -> bytes:
        """转换为串口发送的字节"""
        return bytes([self.class_id, self.x, self.y])
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """转换为元组"""
        return (self.class_id, self.x, self.y)
    
    @classmethod
    def empty(cls) -> "SerialPacket":
        """创建空数据包（无检测时发送）"""
        return cls(class_id=0, x=0, y=0)
    
    @classmethod
    def from_normalized(cls, class_id: int, x_normalized: float, y_normalized: float) -> "SerialPacket":
        """从归一化坐标创建数据包
        
        Args:
            class_id: 分类编号
            x_normalized: X坐标 (0.0-1.0)
            y_normalized: Y坐标 (0.0-1.0)
        """
        x = int(x_normalized * 255)
        y = int(y_normalized * 255)
        return cls(class_id=class_id, x=max(0, min(255, x)), y=max(0, min(255, y)))
