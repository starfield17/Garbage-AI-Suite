"""标签组装器"""

from pathlib import Path
from typing import List

from autolabel_context.domain.model.entity.image_item import ImageItem
from autolabel_context.domain.model.entity.label_result import LabelResult
from autolabel_context.domain.model.value_object.confidence import Confidence
from autolabel_context.domain.model.aggregate.autolabel_job import AutoLabelJob

from shared_kernel.domain.annotation import Detection, BoundingBox, DetectionSource
from shared_kernel.domain.taxonomy import WasteCategory

from ..dto.autolabel_dto import AutoLabelResultDTO


class LabelAssembler:
    """标签组装器"""
    
    @staticmethod
    def image_items_from_paths(paths: List[str]) -> List[ImageItem]:
        """从路径列表创建图片项列表"""
        return [ImageItem.create(path) for path in paths]
    
    @staticmethod
    def detections_from_engine_result(engine_result: dict) -> List[Detection]:
        """从引擎结果创建检测列表"""
        detections = []
        category_map = {
            "Kitchen_waste": WasteCategory.KITCHEN_WASTE,
            "Recyclable_waste": WasteCategory.RECYCLABLE_WASTE,
            "Hazardous_waste": WasteCategory.HAZARDOUS_WASTE,
            "Other_waste": WasteCategory.OTHER_WASTE
        }
        
        for det in engine_result:
            category_name = det.get("name")
            category = category_map.get(category_name)
            
            if category:
                bbox = BoundingBox(
                    x_center=(det["x1"] + det["x2"]) / 2,
                    y_center=(det["y1"] + det["y2"]) / 2,
                    width=det["x2"] - det["x1"],
                    height=det["y2"] - det["y1"]
                )
                
                detection = Detection.create(
                    category=category,
                    confidence=det.get("confidence", 0.0),
                    bounding_box=bbox,
                    source=DetectionSource.MANUAL,
                    raw_label=category_name
                )
                detections.append(detection)
        
        return detections
    
    @staticmethod
    def label_result_from_detection(
        image_item: ImageItem,
        engine_result: dict,
        status: str,
        error_message: str | None = None
    ) -> LabelResult:
        """从检测结果创建标签结果"""
        detections = LabelAssembler.detections_from_engine_result(engine_result)
        
        if status == "success":
            return LabelResult.success(image_item, detections)
        elif status == "skipped":
            return LabelResult.skipped(image_item)
        else:
            return LabelResult.failed(image_item, error_message or "Unknown error")
    
    @staticmethod
    def result_dto_from_job(job: AutoLabelJob) -> AutoLabelResultDTO:
        """从任务创建结果DTO"""
        return AutoLabelResultDTO(
            job_id=str(job.id),
            status=job.status.value,
            total_images=job.statistics.total_images,
            processed_images=job.statistics.processed_images,
            skipped_images=job.statistics.skipped_images,
            failed_images=job.statistics.failed_images,
            total_detections=job.statistics.total_detections,
            detections_by_category=job.statistics.detections_by_category,
            success_rate=job.statistics.success_rate
        )
