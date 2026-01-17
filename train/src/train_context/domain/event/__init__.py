# train/src/train_context/domain/event/__init__.py
"""Train Context Domain Events"""

from .training_started import TrainingStarted
from .training_completed import TrainingCompleted

__all__ = ["TrainingStarted", "TrainingCompleted"]
