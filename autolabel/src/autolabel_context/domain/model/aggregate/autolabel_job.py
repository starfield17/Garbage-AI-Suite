"""自动标注任务聚合根"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from shared_kernel.domain.base import AggregateRoot

from ..value_object.engine_type import EngineType
from ..value_object.job_id import JobId
from ..value_object.confidence import Confidence
from ..entity.image_item import ImageItem
from ..entity.label_result import LabelResult
from ....event.autolabel_finished import AutoLabelFinished


class JobStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobStatistics:
    """任务统计（值对象）"""
    total_images: int = 0
    processed_images: int = 0
    skipped_images: int = 0
    failed_images: int = 0
    total_detections: int = 0
    detections_by_category: dict = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.processed_images / self.total_images


class AutoLabelJob(AggregateRoot):
    """自动标注任务聚合根
    
    职责:
    - 管理标注任务的生命周期
    - 维护任务内数据一致性
    - 聚合 ImageItem 和 LabelResult
    
    不变量:
    - 任务一旦完成或取消，不能再添加结果
    - 统计数据始终与实际结果一致
    """
    
    def __init__(
        self,
        job_id: JobId,
        engine_type: EngineType,
        image_items: List[ImageItem],
        confidence_threshold: float = 0.5
    ):
        super().__init__()
        self._job_id = job_id
        self._engine_type = engine_type
        self._image_items = image_items
        self._confidence_threshold = confidence_threshold
        self._status = JobStatus.PENDING
        self._results: List[LabelResult] = []
        self._statistics = JobStatistics(total_images=len(image_items))
        self._created_at = datetime.utcnow()
        self._completed_at: Optional[datetime] = None
        self._error_message: Optional[str] = None
    
    @property
    def id(self) -> JobId:
        return self._job_id
    
    @property
    def engine_type(self) -> EngineType:
        return self._engine_type
    
    @property
    def status(self) -> JobStatus:
        return self._status
    
    @property
    def statistics(self) -> JobStatistics:
        return self._statistics
    
    @property
    def image_paths(self) -> List[str]:
        return [item.path for item in self._image_items]
    
    @property
    def results(self) -> List[LabelResult]:
        return self._results.copy()
    
    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold
    
    @property
    def created_at(self) -> datetime:
        return self._created_at
    
    @property
    def completed_at(self) -> datetime | None:
        return self._completed_at
    
    @property
    def error_message(self) -> str | None:
        return self._error_message
    
    @classmethod
    def create(
        cls,
        engine_type: EngineType,
        image_paths: List[str],
        confidence_threshold: float = 0.5
    ) -> "AutoLabelJob":
        """工厂方法：创建新任务"""
        job_id = JobId.generate()
        image_items = [ImageItem.create(path) for path in image_paths]
        return cls(
            job_id=job_id,
            engine_type=engine_type,
            image_items=image_items,
            confidence_threshold=confidence_threshold
        )
    
    def start(self) -> None:
        """启动任务"""
        if self._status != JobStatus.PENDING:
            raise InvalidJobStateError(f"Cannot start job in {self._status} status")
        self._status = JobStatus.RUNNING
    
    def add_result(self, result: LabelResult) -> None:
        """添加标注结果
        
        业务规则:
        - 只有 RUNNING 状态可以添加结果
        - 自动更新统计数据
        """
        if self._status != JobStatus.RUNNING:
            raise InvalidJobStateError(f"Cannot add result in {self._status} status")
        
        self._results.append(result)
        self._update_statistics(result)
    
    def _update_statistics(self, result: LabelResult) -> None:
        """更新统计数据"""
        if result.is_success:
            self._statistics.processed_images += 1
            self._statistics.total_detections += result.detection_count
            for category, count in result.category_counts.items():
                current = self._statistics.detections_by_category.get(category, 0)
                self._statistics.detections_by_category[category] = current + count
        elif result.is_skipped:
            self._statistics.skipped_images += 1
        else:
            self._statistics.failed_images += 1
    
    def complete(self) -> None:
        """完成任务"""
        if self._status != JobStatus.RUNNING:
            raise InvalidJobStateError(f"Cannot complete job in {self._status} status")
        
        self._status = JobStatus.COMPLETED
        self._completed_at = datetime.utcnow()
        
        self.add_domain_event(AutoLabelFinished.from_job_statistics(
            job_id=str(self._job_id),
            stats=self._statistics
        ))
    
    def fail(self, error_message: str) -> None:
        """标记任务失败"""
        self._status = JobStatus.FAILED
        self._error_message = error_message
        self._completed_at = datetime.utcnow()
    
    def cancel(self) -> None:
        """取消任务"""
        if self._status in (JobStatus.COMPLETED, JobStatus.FAILED):
            raise InvalidJobStateError(f"Cannot cancel job in {self._status} status")
        self._status = JobStatus.CANCELLED
        self._completed_at = datetime.utcnow()


class InvalidJobStateError(Exception):
    """无效的任务状态错误"""
    pass
