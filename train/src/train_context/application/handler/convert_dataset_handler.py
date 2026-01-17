# train/src/train_context/application/handler/convert_dataset_handler.py
"""转换数据集处理器"""

from typing import Dict, Any

from train_context.infrastructure.adapters.label_converter import LabelConverter
from train_context.application.dto.training_dto import ConvertResultDTO
from train_context.application.command.convert_dataset_cmd import ConvertDatasetCmd


class ConvertDatasetHandler:
    """转换数据集处理器
    
    处理数据集格式转换请求
    """
    
    def __init__(self):
        """初始化处理器"""
        self._converter = LabelConverter()
    
    def handle(self, command: ConvertDatasetCmd) -> ConvertResultDTO:
        """处理转换数据集命令
        
        Args:
            command: 转换数据集命令
        
        Returns:
            转换结果 DTO
        """
        try:
            if command.source_format == "yolo" and command.target_format == "coco":
                result = self._converter.yolo_to_coco(
                    command.input_path,
                    command.output_path
                )
            elif command.source_format == "coco" and command.target_format == "yolo":
                self._converter.coco_to_yolo(
                    command.input_path,
                    command.output_path
                )
                result = {"status": "success"}
            else:
                return ConvertResultDTO(
                    success=False,
                    error=f"Unsupported conversion: {command.source_format} -> {command.target_format}"
                )
            
            return ConvertResultDTO(
                success=True,
                input_path=command.input_path,
                output_path=command.output_path,
                source_format=command.source_format,
                target_format=command.target_format,
                details=result
            )
        
        except Exception as e:
            return ConvertResultDTO(
                success=False,
                error=str(e)
            )
