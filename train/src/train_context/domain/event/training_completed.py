# train/src/train_context/domain/event/training_completed.py
"""训练完成事件"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class TrainingCompleted:
    """训练完成事件"""
    run_id: str
    artifact_path: str
    best_metrics: Optional[Dict[str, Any]] = None
    completed_at: datetime = field(default_factory=datetime.utcnow)
