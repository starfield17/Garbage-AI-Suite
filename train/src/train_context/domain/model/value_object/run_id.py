# train/src/train_context/domain/model/value_object/run_id.py
"""训练运行 ID 值对象"""

from dataclasses import dataclass
from datetime import datetime
import re


@dataclass(frozen=True)
class RunId:
    """训练运行 ID
    
    格式: {model_family}_{date}_{sequence}
    示例: yolo_20260116_001
    
    这是自定义编号体系的核心载体
    """
    value: str
    
    def __post_init__(self):
        # 验证格式
        pattern = r"^[a-z]+_\d{8}_\d{3}$"
        if not re.match(pattern, self.value):
            raise ValueError(
                f"Invalid run ID format: {self.value}. "
                f"Expected: {{model}}_{{date}}_{{seq}}, e.g., yolo_20260116_001"
            )
    
    @classmethod
    def generate(cls, model_family: str = "run") -> "RunId":
        """生成新的运行 ID"""
        date_str = datetime.now().strftime("%Y%m%d")
        # 简化：使用时间戳作为序号
        seq = datetime.now().strftime("%H%M%S")[-3:]
        return cls(f"{model_family}_{date_str}_{seq}")
    
    @classmethod
    def from_string(cls, value: str) -> "RunId":
        """从字符串创建"""
        return cls(value)
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def model_family(self) -> str:
        """提取模型族"""
        return self.value.split("_")[0]
    
    @property
    def date(self) -> str:
        """提取日期"""
        return self.value.split("_")[1]
    
    @property
    def sequence(self) -> str:
        """提取序号"""
        return self.value.split("_")[2]
