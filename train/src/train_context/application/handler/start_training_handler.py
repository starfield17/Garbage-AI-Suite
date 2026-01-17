# train/src/train_context/application/handler/start_training_handler.py
"""开始训练处理器"""

from typing import Dict, Any

from train_context.domain.model.aggregate.training_run import TrainingRun, RunStatus
from train_context.domain.model.aggregate.training_run import InvalidRunStateError
from train_context.infrastructure.trainer.yolo_trainer import YoloTrainer
from train_context.infrastructure.trainer.faster_rcnn_trainer import FasterRcnnTrainer
from train_context.infrastructure.persistence.local_artifact_store import LocalArtifactStore
from train_context.infrastructure.persistence.manifest_repo import ManifestRepository
from train_context.application.dto.training_dto import TrainingResultDTO
from train_context.application.command.start_training_cmd import StartTrainingCmd


class StartTrainingHandler:
    """开始训练处理器
    
    处理训练请求，编排训练流程
    """
    
    def __init__(
        self,
        artifact_store: LocalArtifactStore = None,
        manifest_repo: ManifestRepository = None
    ):
        """初始化处理器
        
        Args:
            artifact_store: 产物存储
            manifest_repo: 清单仓储
        """
        self._artifact_store = artifact_store or LocalArtifactStore()
        self._manifest_repo = manifest_repo or ManifestRepository()
    
    def handle(self, command: StartTrainingCmd) -> TrainingResultDTO:
        """处理开始训练命令
        
        Args:
            command: 开始训练命令
        
        Returns:
            训练结果 DTO
        """
        training_run = TrainingRun.create(
            model_family=command.model_family,
            model_variant=command.model_variant,
            dataset_path=command.dataset_path,
            epochs=command.epochs,
            batch_size=command.batch_size,
            learning_rate=command.learning_rate,
            class_mapping_id=command.class_mapping_id
        )
        
        trainer = self._create_trainer(command.model_family, command.device)
        
        output_dir = command.output_dir or f"runs/{training_run.id}"
        
        try:
            training_run.start()
            
            training_run.begin_training()
            
            result = trainer.train(
                model_path=self._get_pretrained_path(command.model_family),
                dataset_path=command.dataset_path,
                epochs=command.epochs,
                batch_size=command.batch_size,
                learning_rate=command.learning_rate,
                output_dir=output_dir
            )
            
            if result["success"]:
                artifact_path = self._artifact_store.save_artifact(
                    str(training_run.id),
                    result["best_model"]
                )
                
                training_run.complete(artifact_path)
                
                self._manifest_repo.save_training_record(
                    str(training_run.id),
                    training_run.get_summary()
                )
                
                return TrainingResultDTO.from_training_run(training_run, result)
            else:
                training_run.fail(result.get("error", "Unknown error"))
                return TrainingResultDTO.from_training_run(training_run, result)
        
        except Exception as e:
            training_run.fail(str(e))
            return TrainingResultDTO.from_training_run(training_run, {"error": str(e)})
    
    def _create_trainer(self, model_family: str, device: str):
        """创建训练器"""
        if model_family.lower() == "yolo":
            return YoloTrainer(device=device)
        elif model_family.lower() == "faster_rcnn":
            return FasterRcnnTrainer(device=device)
        else:
            raise ValueError(f"Unknown model family: {model_family}")
    
    def _get_pretrained_path(self, model_family: str) -> str:
        """获取预训练模型路径"""
        pretrained_paths = {
            "yolo": "yolov8n.pt",
            "faster_rcnn": "resnet50"
        }
        return pretrained_paths.get(model_family.lower(), "")
