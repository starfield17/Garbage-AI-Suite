"""Pydantic models for bounding box labels."""

from pydantic import BaseModel, Field, field_validator


class BBoxEntryDTO(BaseModel):
    """Single bounding box entry."""

    name: str = Field(..., description="Object class name")
    x1: int = Field(..., ge=0, description="Top-left X coordinate")
    y1: int = Field(..., ge=0, description="Top-left Y coordinate")
    x2: int = Field(..., ge=0, description="Bottom-right X coordinate")
    y2: int = Field(..., ge=0, description="Bottom-right Y coordinate")
    confidence: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Detection confidence (0-1)",
    )

    @field_validator("x2", "y2")
    @classmethod
    def validate_bbox_coords(cls, v, info):
        if "x1" in info.data and v <= info.data["x1"]:
            raise ValueError("x2 must be greater than x1")
        if "y1" in info.data and v <= info.data["y1"]:
            raise ValueError("y2 must be greater than y1")
        return v


class BBoxLabelDTO(BaseModel):
    """Bounding box label file format."""

    labels: list[BBoxEntryDTO] = Field(default_factory=list)
