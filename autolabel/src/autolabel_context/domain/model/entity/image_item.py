"""图片项实体"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageItem:
    """图片项实体
    
    封装单张图片的信息和处理状态
    """
    path: str
    _processed: bool = False
    _processing_error: str | None = None
    
    @classmethod
    def create(cls, path: str) -> "ImageItem":
        """从路径创建图片项"""
        if not path:
            raise ValueError("Image path cannot be empty")
        return cls(path=str(Path(path).resolve()))
    
    @property
    def file_path(self) -> Path:
        """获取Path对象"""
        return Path(self.path)
    
    @property
    def exists(self) -> bool:
        """检查图片是否存在"""
        return self.file_path.exists()
    
    @property
    def is_processed(self) -> bool:
        return self._processed
    
    @property
    def processing_error(self) -> str | None:
        return self._processing_error
    
    def mark_processed(self) -> None:
        self._processed = True
    
    def mark_failed(self, error: str) -> None:
        self._processed = True
        self._processing_error = error
    
    def reset(self) -> None:
        self._processed = False
        self._processing_error = None
    
    @property
    def stem(self) -> str:
        return self.file_path.stem
