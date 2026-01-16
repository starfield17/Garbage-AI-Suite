"""Contracts models module."""

from .bbox_label import BBoxLabelDTO, BBoxEntryDTO
from .model_manifest import ModelManifestDTO, ClassInfoDTO, ModelInputDTO, ModelFileDTO
from .inference import InferenceRequestDTO, InferenceResponseDTO, DetectionDTO, FrameMetaDTO

__all__ = [
    "BBoxLabelDTO",
    "BBoxEntryDTO",
    "ModelManifestDTO",
    "ClassInfoDTO",
    "ModelInputDTO",
    "ModelFileDTO",
    "InferenceRequestDTO",
    "InferenceResponseDTO",
    "DetectionDTO",
    "FrameMetaDTO",
]
