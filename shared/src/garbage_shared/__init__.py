"""Garbage Shared - Shared utilities for garbage AI suite."""

__version__ = "0.1.0"

from garbage_shared.config_loader import ConfigLoader
from garbage_shared.workflow_engine import WorkflowEngine
from garbage_shared.observability import get_logger, setup_logging
from garbage_shared.contracts_models import (
    BBoxLabelDTO,
    ModelManifestDTO,
    InferenceRequestDTO,
    InferenceResponseDTO,
)

__all__ = [
    "ConfigLoader",
    "WorkflowEngine",
    "get_logger",
    "setup_logging",
    "BBoxLabelDTO",
    "ModelManifestDTO",
    "InferenceRequestDTO",
    "InferenceResponseDTO",
]
