"""文件系统工具"""

import os
from pathlib import Path
from typing import Optional


def ensure_dir(path: Path | str) -> Path:
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_safe(path: Path | str, encoding: str = "utf-8") -> Optional[str]:
    """安全读取文本文件"""
    try:
        path = Path(path)
        return path.read_text(encoding=encoding)
    except (FileNotFoundError, OSError):
        return None


def write_text_safe(
    path: Path | str,
    content: str,
    encoding: str = "utf-8"
) -> bool:
    """安全写入文本文件"""
    try:
        path = Path(path)
        ensure_dir(path.parent)
        path.write_text(content, encoding=encoding)
        return True
    except (FileNotFoundError, OSError):
        return False
