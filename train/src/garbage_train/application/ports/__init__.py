"""Ports/interfaces for train module."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional


class TrainerPort(ABC):
    """Port for model training."""

    @abstractmethod
    def train(
        self,
        profile: Dict,
        progress_callback=None,
    ) -> Dict:
        """Train model with given profile."""
        pass

    @abstractmethod
    def finetune(
        self,
        base_model: str,
        profile: Dict,
        progress_callback=None,
    ) -> Dict:
        """Finetune pretrained model."""
        pass


class ExporterPort(ABC):
    """Port for model export."""

    @abstractmethod
    def export_to_onnx(
        self, model_path: Path, config: Dict
    ) -> Dict:
        """Export model to ONNX format."""
        pass

    @abstractmethod
    def export_to_rknn(
        self, model_path: Path, config: Dict
    ) -> Dict:
        """Export model to RKNN format."""
        pass

    @abstractmethod
    def export_to_torchscript(
        self, model_path: Path, config: Dict
    ) -> Dict:
        """Export model to TorchScript format."""
        pass


class ArtifactStorePort(ABC):
    """Port for artifact storage."""

    @abstractmethod
    def save(
        self,
        manifest: Dict,
        output_dir: Path,
    ) -> Path:
        """Save model manifest and artifacts."""
        pass


class DatasetPort(ABC):
    """Port for dataset preparation."""

    @abstractmethod
    def prepare_dataset(
        self, profile: Dict
    ) -> Dict:
        """Prepare and validate dataset."""
        pass
