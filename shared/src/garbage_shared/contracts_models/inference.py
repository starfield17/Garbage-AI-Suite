"""Pydantic models for inference request/response."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DetectionDTO(BaseModel):
    """Single detection result."""

    class_id: int = Field(..., ge=0, description="Detected class ID")
    class_name: str = Field(..., description="Detected class name")
    bbox: dict[str, float] = Field(
        ...,
        description="Bounding box: {x1, y1, x2, y2}",
    )
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")


class FrameMetaDTO(BaseModel):
    """Frame metadata."""

    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    timestamp: datetime | str = Field(...)


class InferenceRequestDTO(BaseModel):
    """Inference request payload."""

    request_id: Optional[str] = Field(None, description="Unique request ID")
    image: str = Field(..., description="Image data (base64 or path)")
    image_encoding: str = Field(
        ...,
        pattern="^(base64|path|numpy_array)$",
    )
    preprocess_options: Optional[dict] = Field(None)
    detection_options: Optional[dict] = Field(None)


class InferenceResponseDTO(BaseModel):
    """Inference response payload."""

    request_id: Optional[str] = Field(None)
    success: bool = Field(...)
    error: Optional[str] = Field(None)
    detections: list[DetectionDTO] = Field(default_factory=list)
    frame_meta: Optional[FrameMetaDTO] = Field(None)
    inference_time_ms: Optional[float] = Field(None, ge=0)
