"""Trainer implementations for different model families."""

from garbage_train.infrastructure.trainers.yolo_trainer import (
    YOLOTrainer,
    FasterRCNNTrainer,
)

__all__ = ["YOLOTrainer", "FasterRCNNTrainer"]
