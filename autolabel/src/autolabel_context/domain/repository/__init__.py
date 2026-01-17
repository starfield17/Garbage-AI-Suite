"""仓储模块"""

from .i_engine_repository import IEngineRepository, IDetectionEngine
from .i_label_store import ILabelStore

__all__ = ["IEngineRepository", "IDetectionEngine", "ILabelStore"]
