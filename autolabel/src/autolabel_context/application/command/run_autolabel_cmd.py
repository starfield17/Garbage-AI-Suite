"""运行自动标注命令"""

from dataclasses import dataclass
from typing import List, Optional

from autolabel_context.domain.model.value_object.engine_type import EngineType


@dataclass
class RunAutoLabelCmd:
    """运行自动标注命令"""
    engine_type: EngineType
    image_paths: List[str]
    output_dir: str
    confidence_threshold: float = 0.5
    batch_size: int = 4
    model_id: Optional[str] = None
    
    @classmethod
    def create(
        cls,
        engine_type: str,
        input_dir: str,
        output_dir: str,
        confidence_threshold: float = 0.5,
        batch_size: int = 4,
        model_id: Optional[str] = None
    ) -> "RunAutoLabelCmd":
        """工厂方法：创建命令"""
        return cls(
            engine_type=EngineType.from_string(engine_type),
            image_paths=[input_dir],  # 简化：使用目录作为输入
            output_dir=output_dir,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,
            model_id=model_id
        )
