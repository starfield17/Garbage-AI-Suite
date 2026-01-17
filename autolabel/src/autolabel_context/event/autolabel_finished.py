"""自动标注完成领域事件"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

from shared_kernel.domain.base import DomainEvent


@dataclass
class JobStatisticsData:
    """任务统计数据（用于领域事件，避免循环导入）"""
    total_images: int = 0
    processed_images: int = 0
    skipped_images: int = 0
    failed_images: int = 0
    total_detections: int = 0
    detections_by_category: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0


@dataclass
class AutoLabelFinished(DomainEvent):
    """自动标注完成领域事件"""
    job_id: str = ""
    statistics: JobStatisticsData = None  # type: ignore
    
    @classmethod
    def from_job_statistics(cls, job_id: str, stats: "JobStatistics") -> "AutoLabelFinished":
        """从JobStatistics创建领域事件"""
        statistics_data = JobStatisticsData(
            total_images=stats.total_images,
            processed_images=stats.processed_images,
            skipped_images=stats.skipped_images,
            failed_images=stats.failed_images,
            total_detections=stats.total_detections,
            detections_by_category=stats.detections_by_category,
            success_rate=stats.success_rate
        )
        return cls(job_id=job_id, statistics=statistics_data)
    
    def to_dict(self) -> dict:
        return {
            "event_type": self.__class__.__name__,
            "job_id": self.job_id,
            "statistics": {
                "total_images": self.statistics.total_images,
                "processed_images": self.statistics.processed_images,
                "skipped_images": self.statistics.skipped_images,
                "failed_images": self.statistics.failed_images,
                "total_detections": self.statistics.total_detections,
                "detections_by_category": self.statistics.detections_by_category,
                "success_rate": self.statistics.success_rate
            },
            "occurred_at": self.occurred_at.isoformat()
        }
