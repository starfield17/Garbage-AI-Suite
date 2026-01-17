# train/src/train_context/application/command/export_artifact_cmd.py
"""导出产物命令"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExportArtifactCmd:
    """导出产物命令
    
    封装模型导出请求的参数
    """
    run_id: str
    format: str = "onnx"
    output_path: Optional[str] = None
    
    def __post_init__(self):
        valid_formats = ["pt", "onnx", "rknn"]
        if self.format not in valid_formats:
            raise ValueError(f"Invalid format: {self.format}. Valid: {valid_formats}")
