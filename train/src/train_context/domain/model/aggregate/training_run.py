# train/src/train_context/domain/model/aggregate/training_run.py
"""训练运行聚合根"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List

from shared_kernel.domain.base import AggregateRoot

from ..value_object.run_id import RunId
from ..value_object.hyper_params import HyperParams
from ..value_object.class_mapping_id import ClassMappingId
from ..entity.dataset import Dataset
from ..entity.model_spec import ModelSpec
from ...event.training_started import TrainingStarted
from ...event.training_completed import TrainingCompleted


class RunStatus(Enum):
    """训练状态"""
    CREATED = "created"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingMetrics:
    """训练指标（值对象）"""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    
    def is_better_than(self, other: "TrainingMetrics") -> bool:
        """判断是否比另一个指标更好"""
        return self.mAP50_95 > other.mAP50_95


class TrainingRun(AggregateRoot):
    """训练运行聚合根
    
    职责:
    - 管理训练任务的生命周期
    - 维护训练配置和超参数
    - 追踪训练指标和产物
    
    不变量:
    - 训练完成后配置不可修改
    - 编号映射必须在训练前确定
    """
    
    def __init__(
        self,
        run_id: RunId,
        model_spec: ModelSpec,
        dataset: Dataset,
        hyper_params: HyperParams,
        class_mapping_id: ClassMappingId
    ):
        super().__init__()
        self._run_id = run_id
        self._model_spec = model_spec
        self._dataset = dataset
        self._hyper_params = hyper_params
        self._class_mapping_id = class_mapping_id
        self._status = RunStatus.CREATED
        self._metrics: Optional[TrainingMetrics] = None
        self._best_metrics: Optional[TrainingMetrics] = None
        self._artifact_path: Optional[str] = None
        self._created_at = datetime.utcnow()
        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None
        self._error_message: Optional[str] = None
        self._metric_history: List[TrainingMetrics] = []
    
    @property
    def id(self) -> RunId:
        return self._run_id
    
    @property
    def status(self) -> RunStatus:
        return self._status
    
    @property
    def model_spec(self) -> ModelSpec:
        return self._model_spec
    
    @property
    def hyper_params(self) -> HyperParams:
        return self._hyper_params
    
    @property
    def class_mapping_id(self) -> ClassMappingId:
        return self._class_mapping_id
    
    @property
    def artifact_path(self) -> Optional[str]:
        return self._artifact_path
    
    @property
    def metrics(self) -> Optional[TrainingMetrics]:
        return self._metrics
    
    @property
    def best_metrics(self) -> Optional[TrainingMetrics]:
        return self._best_metrics
    
    @classmethod
    def create(
        cls,
        model_family: str,
        model_variant: str,
        dataset_path: str,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 0.01,
        class_mapping_id: str = "default"
    ) -> "TrainingRun":
        """工厂方法：创建训练运行"""
        run_id = RunId.generate(model_family)
        model_spec = ModelSpec(family=model_family, variant=model_variant)
        dataset = Dataset.from_path(dataset_path)
        hyper_params = HyperParams(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        mapping_id = ClassMappingId(class_mapping_id)
        
        return cls(
            run_id=run_id,
            model_spec=model_spec,
            dataset=dataset,
            hyper_params=hyper_params,
            class_mapping_id=mapping_id
        )
    
    def start(self) -> None:
        """开始训练"""
        if self._status != RunStatus.CREATED:
            raise InvalidRunStateError(f"Cannot start run in {self._status} status")
        
        self._status = RunStatus.PREPARING
        self._started_at = datetime.utcnow()
        
        self.add_domain_event(TrainingStarted(
            run_id=str(self._run_id),
            model_spec=self._model_spec.to_dict(),
            hyper_params=self._hyper_params.to_dict()
        ))
    
    def begin_training(self) -> None:
        """开始实际训练"""
        if self._status != RunStatus.PREPARING:
            raise InvalidRunStateError(f"Cannot begin training in {self._status} status")
        self._status = RunStatus.TRAINING
    
    def update_metrics(self, metrics: TrainingMetrics) -> None:
        """更新训练指标"""
        if self._status not in (RunStatus.TRAINING, RunStatus.VALIDATING):
            raise InvalidRunStateError(f"Cannot update metrics in {self._status} status")
        
        self._metrics = metrics
        self._metric_history.append(metrics)
        
        if self._best_metrics is None or metrics.is_better_than(self._best_metrics):
            self._best_metrics = metrics
    
    def complete(self, artifact_path: str) -> None:
        """完成训练"""
        if self._status not in (RunStatus.TRAINING, RunStatus.VALIDATING):
            raise InvalidRunStateError(f"Cannot complete run in {self._status} status")
        
        self._status = RunStatus.COMPLETED
        self._artifact_path = artifact_path
        self._completed_at = datetime.utcnow()
        
        self.add_domain_event(TrainingCompleted(
            run_id=str(self._run_id),
            artifact_path=artifact_path,
            best_metrics=self._best_metrics.__dict__ if self._best_metrics else None
        ))
    
    def fail(self, error_message: str) -> None:
        """标记训练失败"""
        self._status = RunStatus.FAILED
        self._error_message = error_message
        self._completed_at = datetime.utcnow()
    
    def cancel(self) -> None:
        """取消训练"""
        if self._status in (RunStatus.COMPLETED, RunStatus.FAILED):
            raise InvalidRunStateError(f"Cannot cancel run in {self._status} status")
        self._status = RunStatus.CANCELLED
        self._completed_at = datetime.utcnow()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            "run_id": str(self._run_id),
            "status": self._status.value,
            "model": self._model_spec.to_dict(),
            "dataset": self._dataset.name,
            "hyper_params": self._hyper_params.to_dict(),
            "class_mapping": str(self._class_mapping_id),
            "best_metrics": self._best_metrics.__dict__ if self._best_metrics else None,
            "artifact_path": self._artifact_path,
            "created_at": self._created_at.isoformat(),
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "completed_at": self._completed_at.isoformat() if self._completed_at else None,
            "error_message": self._error_message,
        }


class InvalidRunStateError(Exception):
    """无效的运行状态错误"""
    pass
