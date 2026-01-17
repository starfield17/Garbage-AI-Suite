# train/src/train_context/application/assembler/training_assembler.py
"""训练汇编器"""

from typing import Dict, Any, List

from train_context.application.dto.training_dto import TrainingResultDTO
from train_context.application.command.start_training_cmd import StartTrainingCmd
from train_context.domain.model.aggregate.training_run import TrainingRun


class TrainingAssembler:
    """训练汇编器
    
    负责 DTO 和 Domain 对象之间的转换
    """
    
    @staticmethod
    def cmd_to_run(command: StartTrainingCmd) -> TrainingRun:
        """将命令转换为聚合根
        
        Args:
            command: 开始训练命令
        
        Returns:
            训练运行聚合根
        """
        return TrainingRun.create(
            model_family=command.model_family,
            model_variant=command.model_variant,
            dataset_path=command.dataset_path,
            epochs=command.epochs,
            batch_size=command.batch_size,
            learning_rate=command.learning_rate,
            class_mapping_id=command.class_mapping_id
        )
    
    @staticmethod
    def run_to_dto(training_run: TrainingRun, result: Dict[str, Any]) -> TrainingResultDTO:
        """将聚合根转换为 DTO
        
        Args:
            training_run: 训练运行聚合根
            result: 训练结果
        
        Returns:
            训练结果 DTO
        """
        return TrainingResultDTO.from_training_run(training_run, result)
    
    @staticmethod
    def result_to_summary(result: Dict[str, Any]) -> Dict[str, Any]:
        """将结果转换为摘要
        
        Args:
            result: 训练结果
        
        Returns:
            结果摘要
        """
        return {
            "success": result.get("success", False),
            "epochs": result.get("final_epoch", 0),
            "metrics": result.get("final_metrics", {}),
            "model_path": result.get("best_model", ""),
        }
