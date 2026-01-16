"""Ports/interfaces for deploy module."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List
import numpy as np


class RuntimePort(ABC):
    """Port for model runtime and inference."""

    @abstractmethod
    def load_model(self, manifest_path: Path | None, device: str) -> None:
        pass

    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[dict]:
        """Run inference on a frame."""
        pass


class CameraPort(ABC):
    """Port for camera access."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Read a frame from camera."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        pass


class SerialPort(ABC):
    """Port for serial communication."""

    @abstractmethod
    def open(self, port: str, baud: int) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def send(self, data: bytes) -> None:
        pass


class ClockPort(ABC):
    """Port for time/clock abstraction (for testability)."""

    @abstractmethod
    def now(self) -> float:
        """Get current time in seconds since epoch."""
        pass
