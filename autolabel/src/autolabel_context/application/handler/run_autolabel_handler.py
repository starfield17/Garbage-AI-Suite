"""运行自动标注处理器"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from tqdm import tqdm

from shared_kernel.config.loader import ConfigLoader
from shared_kernel.domain.annotation import BoundingBox, Detection, DetectionSource
from shared_kernel.domain.taxonomy import WasteCategory

from autolabel_context.domain.model.value_object.engine_type import EngineType
from autolabel_context.domain.model.entity.image_item import ImageItem
from autolabel_context.domain.model.entity.label_result import LabelResult
from autolabel_context.domain.model.aggregate.autolabel_job import AutoLabelJob
from autolabel_context.domain.repository.i_engine_repository import IEngineRepository
from autolabel_context.domain.repository.i_label_store import ILabelStore

from ..command.run_autolabel_cmd import RunAutoLabelCmd
from ..dto.autolabel_dto import AutoLabelResultDTO
from ..assembler.label_assembler import LabelAssembler

logger = logging.getLogger(__name__)


class RunAutoLabelHandler:
    """运行自动标注处理器（应用服务）"""
    
    def __init__(
        self,
        engine_repo: IEngineRepository,
        label_store: ILabelStore,
        config_loader: ConfigLoader
    ):
        self._engine_repo = engine_repo
        self._label_store = label_store
        self._config_loader = config_loader
    
    def handle(self, command: RunAutoLabelCmd) -> AutoLabelResultDTO:
        """处理自动标注命令"""
        engine = self._engine_repo.get_engine(command.engine_type)
        
        job = AutoLabelJob.create(
            engine_type=command.engine_type,
            image_paths=command.image_paths,
            confidence_threshold=command.confidence_threshold
        )
        
        job.start()
        
        self._process_images(job, engine, command)
        
        job.complete()
        
        self._label_store.save_job(job)
        
        return LabelAssembler.result_dto_from_job(job)
    
    def _process_images(self, job: AutoLabelJob, engine, command: RunAutoLabelCmd) -> None:
        """处理图片"""
        image_items = [item for item in job._image_items if item.exists]
        
        def process_single(item: ImageItem) -> LabelResult:
            if not item.exists:
                return LabelResult.failed(item, "Image file not found")
            
            try:
                detections = engine.detect(item.path)
                
                engine_result = []
                category_map = {
                    "Kitchen_waste": WasteCategory.KITCHEN_WASTE,
                    "Recyclable_waste": WasteCategory.RECYCLABLE_WASTE,
                    "Hazardous_waste": WasteCategory.HAZARDOUS_WASTE,
                    "Other_waste": WasteCategory.OTHER_WASTE
                }
                
                for det in detections:
                    category_name = det.get("name")
                    category = category_map.get(category_name)
                    
                    if category:
                        img = item._file_path if hasattr(item, '_file_path') else None
                        if img is None:
                            continue
                        import cv2
                        img_data = cv2.imread(str(item.path))
                        if img_data is None:
                            continue
                        img_height, img_width = img_data.shape[:2]
                        
                        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                        x_center = ((x1 + x2) / 2) / img_width
                        y_center = ((y1 + y2) / 2) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        bbox = BoundingBox(
                            x_center=x_center,
                            y_center=y_center,
                            width=width,
                            height=height
                        )
                        
                        detection = Detection.create(
                            category=category,
                            confidence=det.get("confidence", 0.0),
                            bounding_box=bbox,
                            source=DetectionSource.MANUAL,
                            raw_label=category_name
                        )
                        engine_result.append(detection)
                
                return LabelResult.success(item, engine_result)
                
            except Exception as e:
                logger.exception(f"Error processing {item.path}")
                return LabelResult.failed(item, str(e))
        
        with ThreadPoolExecutor(max_workers=command.batch_size) as executor:
            futures = [executor.submit(process_single, item) for item in image_items]
            
            for future in tqdm(futures, desc="Processing images"):
                result = future.result()
                job.add_result(result)
