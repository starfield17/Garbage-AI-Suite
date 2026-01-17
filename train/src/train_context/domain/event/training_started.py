# train/src/train_context/domain/event/training_started.py
"""训练开始事件"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class TrainingStarted:
    """训练开始事件"""
    run_id: str
    model_spec: Dict[str, Any]
    hyper_params: Dict[str, Any]
    started_at: datetime = field(default_factory=datetime.utcnow)
