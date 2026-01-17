"""PySerial 串口实现"""

import serial
import threading
import time
from typing import Optional, Tuple, Any

from ...domain.repository import ISerialDevice
from ...domain.model.value_object import SerialPacket


class SerialPyserial(ISerialDevice):
    """PySerial 串口实现
    
    使用 pyserial 进行串口通信
    """
    
    def __init__(self):
        self._port: Optional[serial.Serial] = None
        self._is_connected: bool = False
        self._port_name: str = ""
        self._baudrate: int = 115200
        self._send_lock = threading.Lock()
    
    def open(self, port: str, baudrate: int = 115200, timeout: float = 0.1) -> bool:
        """打开串口"""
        try:
            self._port = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                write_timeout=timeout
            )
            self._port_name = port
            self._baudrate = baudrate
            self._is_connected = True
            return True
        except Exception:
            self._is_connected = False
            return False
    
    def write(self, data: bytes) -> int:
        """写入数据"""
        if not self._is_connected:
            return 0
        
        with self._send_lock:
            try:
                bytes_written = self._port.write(data)
                self._port.flush()
                return bytes_written
            except Exception:
                return 0
    
    def write_packet(self, packet: SerialPacket) -> bool:
        """写入数据包"""
        return self.write(packet.to_bytes()) == len(packet.to_bytes())
    
    def read(self, size: int = 1) -> bytes:
        """读取数据"""
        if not self._is_connected:
            return b""
        
        try:
            return self._port.read(size)
        except Exception:
            return b""
    
    def is_connected(self) -> bool:
        """检查是否连接"""
        return self._is_connected
    
    def close(self) -> None:
        """关闭串口"""
        if self._port:
            self._port.close()
        self._port = None
        self._is_connected = False
    
    def flush(self) -> None:
        """刷新缓冲区"""
        if self._port:
            self._port.flush()
    
    def get_port_info(self) -> Tuple[str, int]:
        """获取端口信息"""
        return (self._port_name, self._baudrate)
