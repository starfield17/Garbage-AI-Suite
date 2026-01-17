# train/src/train_context/application/command/convert_dataset_cmd.py
"""转换数据集命令"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConvertDatasetCmd:
    """转换数据集命令
    
    封装数据集格式转换请求的参数
    """
    input_path: str
    output_path: str
    source_format: str
    target_format: str
    
    def __post_init__(self):
        valid_formats = ["yolo", "coco", "voc"]
        if self.source_format not in valid_formats:
            raise ValueError(f"Invalid source format: {self.source_format}")
        if self.target_format not in valid_formats:
            raise ValueError(f"Invalid target format: {self.target_format}")
