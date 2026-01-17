# train/src/train_context/domain/repository/__init__.py
"""Train Context Repository Interfaces"""

from .i_artifact_store import IArtifactStore
from .i_dataset_repository import IDatasetRepository

__all__ = ["IArtifactStore", "IDatasetRepository"]
