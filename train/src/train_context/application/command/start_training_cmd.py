# train/src/train_context/application/command/start_training_cmd.py
"""开始训练命令"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StartTrainingCmd:
    """开始训练命令
    
    封装训练请求的所有参数
    """
    model_family: str
    model_variant: str
    dataset_path: str
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    class_mapping_id: str = "default"
    output_dir: Optional[str] = None
    device: str = "auto"
    
    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
