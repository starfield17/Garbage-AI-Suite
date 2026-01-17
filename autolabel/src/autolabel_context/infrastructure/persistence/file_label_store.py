"""文件标签存储实现"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from autolabel_context.domain.model.aggregate.autolabel_job import AutoLabelJob
from autolabel_context.domain.repository.i_label_store import ILabelStore


class FileLabelStore(ILabelStore):
    """文件标签存储实现"""
    
    def __init__(self, output_dir: str):
        """初始化文件标签存储
        
        Args:
            output_dir: 输出目录路径
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_job(self, job: AutoLabelJob) -> None:
        """保存任务结果到文件"""
        for result in job.results:
            output_path = self._output_dir / f"{result.image_item.stem}.json"
            
            if result.is_success:
                self._save_detection_result(output_path, result)
            elif result.is_skipped:
                self._save_empty_json(output_path)
    
    def _save_detection_result(self, output_path: Path, result) -> None:
        """保存检测结果到JSON文件"""
        labels = []
        for det in result.detections:
            labels.append({
                "name": det.category.value,
                "x1": det.bounding_box.x1,
                "y1": det.bounding_box.y1,
                "x2": det.bounding_box.x2,
                "y2": det.bounding_box.y2,
                "confidence": det.confidence.value
            })
        
        data = {"labels": labels}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_empty_json(self, output_path: Path) -> None:
        """保存空JSON文件"""
        data = {"labels": []}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def find_by_id(self, job_id: str) -> Optional[AutoLabelJob]:
        """根据ID查找任务（文件存储不支持）"""
        return None
