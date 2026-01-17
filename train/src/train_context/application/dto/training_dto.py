# train/src/train_context/application/dto/training_dto.py
"""训练相关 DTO"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class TrainingResultDTO:
    """训练结果 DTO"""
    success: bool
    run_id: str
    status: str
    model_family: str
    model_variant: str
    epochs: int
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    best_model_path: Optional[str] = None
    error: Optional[str] = None
    
    @classmethod
    def from_training_run(cls, training_run, result: Dict[str, Any]) -> "TrainingResultDTO":
        """从训练运行创建 DTO"""
        return cls(
            success=training_run.status.value == "completed",
            run_id=str(training_run.id),
            status=training_run.status.value,
            model_family=training_run.model_spec.family,
            model_variant=training_run.model_spec.variant,
            epochs=training_run.hyper_params.epochs,
            final_metrics=result.get("final_metrics", {}),
            best_model_path=training_run.artifact_path,
            error=training_run._error_message
        )


@dataclass
class ExportResultDTO:
    """导出结果 DTO"""
    success: bool
    run_id: str
    original_path: Optional[str] = None
    exported_path: Optional[str] = None
    format: str = ""
    error: Optional[str] = None


@dataclass
class ConvertResultDTO:
    """转换结果 DTO"""
    success: bool
    input_path: str = ""
    output_path: str = ""
    source_format: str = ""
    target_format: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
