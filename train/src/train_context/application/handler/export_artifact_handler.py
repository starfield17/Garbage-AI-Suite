# train/src/train_context/application/handler/export_artifact_handler.py
"""导出产物处理器"""

from typing import Dict, Any, Optional

from train_context.infrastructure.persistence.local_artifact_store import LocalArtifactStore
from train_context.infrastructure.trainer.yolo_trainer import YoloTrainer
from train_context.infrastructure.trainer.faster_rcnn_trainer import FasterRcnnTrainer
from train_context.application.dto.training_dto import ExportResultDTO
from train_context.application.command.export_artifact_cmd import ExportArtifactCmd


class ExportArtifactHandler:
    """导出产物处理器
    
    处理模型导出请求
    """
    
    def __init__(self, artifact_store: LocalArtifactStore = None):
        """初始化处理器
        
        Args:
            artifact_store: 产物存储
        """
        self._artifact_store = artifact_store or LocalArtifactStore()
    
    def handle(self, command: ExportArtifactCmd) -> ExportResultDTO:
        """处理导出产物命令
        
        Args:
            command: 导出产物命令
        
        Returns:
            导出结果 DTO
        """
        model_path = self._artifact_store.get_artifact(command.run_id)
        
        if not model_path:
            return ExportResultDTO(
                success=False,
                run_id=command.run_id,
                error=f"Artifact not found for run_id: {command.run_id}"
            )
        
        trainer = self._detect_trainer(model_path)
        
        if not trainer:
            return ExportResultDTO(
                success=False,
                run_id=command.run_id,
                error="Could not detect model type"
            )
        
        output_path = command.output_path or f"exports/{command.run_id}_exported.{command.format}"
        
        try:
            exported_path = trainer.export(model_path, output_path, command.format)
            
            return ExportResultDTO(
                success=True,
                run_id=command.run_id,
                original_path=model_path,
                exported_path=exported_path,
                format=command.format
            )
        except Exception as e:
            return ExportResultDTO(
                success=False,
                run_id=command.run_id,
                error=str(e)
            )
    
    def _detect_trainer(self, model_path: str):
        """检测模型类型并返回对应训练器"""
        if model_path.endswith(".pt"):
            return YoloTrainer()
        return None
