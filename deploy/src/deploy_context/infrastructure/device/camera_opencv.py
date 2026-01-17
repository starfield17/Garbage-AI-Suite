"""OpenCV 相机实现"""

import cv2
from typing import Optional, Tuple, Any

from ...domain.repository import ICamera


class CameraOpencv(ICamera):
    """OpenCV 相机实现
    
    使用 OpenCV 进行相机捕获
    """
    
    def __init__(self):
        self._camera = None
        self._camera_id: int = 0
        self._width: int = 1280
        self._height: int = 720
        self._is_opened: bool = False
    
    def open(self, camera_id: int = 0, width: int = 1280, height: int = 720) -> bool:
        """打开相机"""
        self._camera_id = camera_id
        self._width = width
        self._height = height
        
        self._camera = cv2.VideoCapture(camera_id)
        
        if not self._camera.isOpened():
            return False
        
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._camera.set(cv2.CAP_PROP_FPS, 30)
        
        self._is_opened = True
        return True
    
    def read(self) -> Optional[Any]:
        """读取帧"""
        if not self._is_opened:
            return None
        
        ret, frame = self._camera.read()
        if ret:
            return frame
        return None
    
    def is_opened(self) -> bool:
        """检查是否打开"""
        return self._is_opened and self._camera is not None
    
    def close(self) -> None:
        """关闭相机"""
        if self._camera:
            self._camera.release()
        self._camera = None
        self._is_opened = False
    
    def set_resolution(self, width: int, height: int) -> bool:
        """设置分辨率"""
        if not self._is_opened:
            return False
        
        self._width = width
        self._height = height
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return True
    
    def get_resolution(self) -> Tuple[int, int]:
        """获取分辨率"""
        if self._camera:
            w = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (self._width, self._height)
