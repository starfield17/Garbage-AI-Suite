"""自动标注结果DTO"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class AutoLabelResultDTO:
    """自动标注结果DTO"""
    job_id: str
    status: str
    total_images: int
    processed_images: int
    skipped_images: int
    failed_images: int
    total_detections: int
    detections_by_category: Dict[str, int]
    success_rate: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoLabelResultDTO":
        return cls(
            job_id=data.get("job_id", ""),
            status=data.get("status", ""),
            total_images=data.get("total_images", 0),
            processed_images=data.get("processed_images", 0),
            skipped_images=data.get("skipped_images", 0),
            failed_images=data.get("failed_images", 0),
            total_detections=data.get("total_detections", 0),
            detections_by_category=data.get("detections_by_category", {}),
            success_rate=data.get("success_rate", 0.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "statistics": {
                "total_images": self.total_images,
                "processed_images": self.processed_images,
                "skipped_images": self.skipped_images,
                "failed_images": self.failed_images,
                "total_detections": self.total_detections,
                "detections_by_category": self.detections_by_category,
                "success_rate": self.success_rate
            }
        }
