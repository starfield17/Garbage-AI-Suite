"""Pydantic models for model manifest."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ClassInfoDTO(BaseModel):
    """Model class information."""

    id: int = Field(..., ge=0, description="Class ID")
    name: str = Field(..., description="Class name")
    group_id: Optional[int] = Field(None, ge=0, description="Group ID for classification")
    group_name: Optional[str] = Field(None, description="Group name")


class ModelInputDTO(BaseModel):
    """Model input specification."""

    width: int = Field(..., gt=0, description="Input image width")
    height: int = Field(..., gt=0, description="Input image height")
    channels: int = Field(..., ge=1, le=3, description="Number of channels")


class ModelFileDTO(BaseModel):
    """Model artifact file information."""

    path: str = Field(..., description="File path")
    sha256: Optional[str] = Field(None, description="SHA256 hash")
    type: str = Field(..., description="File type (e.g., 'model', 'weights', 'config')")
    size_bytes: Optional[int] = Field(None, ge=0, description="File size in bytes")


class ModelManifestDTO(BaseModel):
    """Model manifest for train/deploy/autolabel handshake."""

    model_name: str = Field(..., description="Human-readable model name")
    model_family: str = Field(..., pattern="^(yolo|faster_rcnn|vlm)$")
    train_id: str = Field(..., description="Training profile ID")
    deploy_id: Optional[str] = Field(None, description="Deploy profile ID")
    git_sha: Optional[str] = Field(None, min_length=7, max_length=40)
    created_at: datetime = Field(..., description="Model creation timestamp")
    classes: list[ClassInfoDTO] = Field(..., min_length=1)
    input: ModelInputDTO = Field(...)
    export_targets: list[str] = Field(default_factory=list)
    metrics: Optional[dict[str, float]] = Field(None)
    files: list[ModelFileDTO] = Field(..., min_length=1)
