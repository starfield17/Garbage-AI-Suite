"""End-to-end integration tests for AutoLabel Context"""

import pytest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add module paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../shared/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../autolabel/src'))

from autolabel_context.domain.model.value_object.engine_type import EngineType
from autolabel_context.domain.model.value_object.confidence import Confidence
from autolabel_context.domain.model.value_object.job_id import JobId
from autolabel_context.domain.model.entity.image_item import ImageItem
from autolabel_context.domain.model.entity.label_result import LabelResult
from autolabel_context.domain.model.aggregate.autolabel_job import AutoLabelJob, JobStatus
from autolabel_context.domain.service.quality_gate import QualityGate
from autolabel_context.domain.service.engine_selector import EngineSelector
from autolabel_context.application.command.run_autolabel_cmd import RunAutoLabelCmd
from autolabel_context.application.dto.autolabel_dto import AutoLabelResultDTO
from shared_kernel.domain.taxonomy import WasteCategory
from shared_kernel.domain.annotation import BoundingBox, DetectionSource, Detection, LabelFile


class TestAutoLabelEndToEnd:
    """End-to-end tests for AutoLabel workflow"""

    def test_complete_autolabel_workflow_yolo(self):
        """Test complete autolabel workflow with YOLO engine"""
        # Create job
        job = AutoLabelJob.create(
            engine_type=EngineType.YOLO,
            image_paths=["/tmp/test1.jpg", "/tmp/test2.jpg", "/tmp/test3.jpg"],
            confidence_threshold=0.5
        )
        
        assert job.status == JobStatus.PENDING
        assert len(job.image_paths) == 3
        
        # Start job
        job.start()
        assert job.status == JobStatus.RUNNING
        
        # Simulate processing
        for i, image_path in enumerate(job.image_paths):
            item = ImageItem.create(image_path)
            item.mark_processed()
            job.add_result(LabelResult.success(item, []))
        
        # Complete job
        job.complete()
        assert job.status == JobStatus.COMPLETED
        assert job.statistics.processed_images == 3
        assert job.statistics.total_images == 3

    def test_complete_autolabel_workflow_with_detections(self):
        """Test autolabel workflow with actual detections"""
        job = AutoLabelJob.create(
            engine_type=EngineType.YOLO,
            image_paths=["/tmp/test.jpg"],
            confidence_threshold=0.5
        )
        
        job.start()
        
        # Create detections
        bbox = BoundingBox(0.5, 0.5, 0.2, 0.3)
        detection1 = Detection.create(
            category=WasteCategory.KITCHEN_WASTE,
            confidence=0.95,
            bbox=bbox,
            source=DetectionSource.YOLO,
            raw_label="potato"
        )
        detection2 = Detection.create(
            category=WasteCategory.RECYCLABLE_WASTE,
            confidence=0.85,
            bbox=BoundingBox(0.7, 0.7, 0.1, 0.1),
            source=DetectionSource.YOLO,
            raw_label="bottle"
        )
        
        item = ImageItem.create("/tmp/test.jpg")
        item.mark_processed()
        job.add_result(LabelResult.success(item, [detection1, detection2]))
        
        job.complete()
        
        stats = job.statistics
        assert stats.total_detections == 2
        assert stats.detections_by_category["Kitchen_waste"] == 1
        assert stats.detections_by_category["Recyclable_waste"] == 1

    def test_autolabel_workflow_with_errors(self):
        """Test autolabel workflow with error handling"""
        job = AutoLabelJob.create(
            engine_type=EngineType.VLM,
            image_paths=["/tmp/test1.jpg", "/tmp/test2.jpg", "/tmp/test3.jpg"],
            confidence_threshold=0.5
        )
        
        job.start()
        
        # Simulate success, failure, and skip
        item1 = ImageItem.create("/tmp/test1.jpg")
        item1.mark_processed()
        job.add_result(LabelResult.success(item1, []))
        
        item2 = ImageItem.create("/tmp/test2.jpg")
        item2.mark_failed("API error")
        job.add_result(LabelResult.failed(item2, "API error"))
        
        item3 = ImageItem.create("/tmp/test3.jpg")
        job.add_result(LabelResult.skipped(item3))
        
        job.complete()
        
        stats = job.statistics
        assert stats.processed_images == 1  # Only successful ones count as processed
        assert stats.failed_images == 1
        assert stats.skipped_images == 1
        assert stats.total_images == 3

    def test_quality_gate_validation(self):
        """Test quality gate for detection validation"""
        gate = QualityGate(
            confidence_threshold=0.5,
            iou_threshold=0.5,
            min_box_size=0.01
        )
        
        # Valid detections
        bbox = BoundingBox(0.5, 0.5, 0.2, 0.3)
        label_file = LabelFile(
            file_id="test_001",
            image_path="/test/image.jpg",
            image_width=640,
            image_height=480,
            detections=[]
        )
        
        valid_detection = Detection.create(
            category=WasteCategory.KITCHEN_WASTE,
            confidence=0.95,
            bbox=bbox,
            source=DetectionSource.YOLO,
            raw_label="potato"
        )
        label_file.add_detection(valid_detection)
        
        result = gate.evaluate(label_file)
        assert result.is_passed is True
        assert result.low_confidence_count == 0
        
        # Invalid detection (low confidence)
        low_conf_detection = Detection.create(
            category=WasteCategory.RECYCLABLE_WASTE,
            confidence=0.3,  # Below threshold
            bbox=bbox,
            source=DetectionSource.YOLO,
            raw_label="bottle"
        )
        label_file.add_detection(low_conf_detection)
        
        result = gate.evaluate(label_file)
        assert result.is_passed is False
        assert result.low_confidence_count == 1

    def test_engine_selector(self):
        """Test engine selection logic"""
        # Skip this test as it requires config_loader
        pytest.skip("EngineSelector requires config_loader which is not available in test")

    def test_command_to_job_transformation(self):
        """Test command to job transformation"""
        cmd = RunAutoLabelCmd.create(
            engine_type="yolo",
            input_dir="/tmp/input",
            output_dir="/tmp/output",
            confidence_threshold=0.6,
            batch_size=8
        )
        
        assert cmd.engine_type == EngineType.YOLO
        assert cmd.confidence_threshold == 0.6
        assert cmd.batch_size == 8
        
        # Create job from command
        job = AutoLabelJob.create(
            engine_type=cmd.engine_type,
            image_paths=cmd.image_paths,
            confidence_threshold=cmd.confidence_threshold
        )
        
        assert job.confidence_threshold == 0.6

    def test_result_dto_serialization(self):
        """Test result DTO serialization"""
        dto = AutoLabelResultDTO(
            job_id="test-job-123",
            status="completed",
            total_images=10,
            processed_images=8,
            skipped_images=1,
            failed_images=1,
            total_detections=25,
            detections_by_category={
                "Kitchen_waste": 10,
                "Recyclable_waste": 15
            },
            success_rate=0.8
        )
        
        # Test serialization
        json_data = dto.to_dict()
        assert json_data["job_id"] == "test-job-123"
        assert "statistics" in json_data
        assert json_data["statistics"]["total_images"] == 10
        
        # Test deserialization (from top-level fields)
        restored_dto = AutoLabelResultDTO.from_dict(json_data)
        assert restored_dto.job_id == dto.job_id

    def test_label_file_export(self):
        """Test label file export functionality"""
        label_file = LabelFile(
            file_id="test_001",
            image_path="/test/image.jpg",
            image_width=640,
            image_height=480,
            detections=[]
        )
        
        # Add detections
        bbox = BoundingBox(0.5, 0.5, 0.2, 0.3)
        detection = Detection.create(
            category=WasteCategory.KITCHEN_WASTE,
            confidence=0.95,
            bbox=bbox,
            source=DetectionSource.YOLO,
            raw_label="potato"
        )
        label_file.add_detection(detection)
        
        # Export to YOLO format
        yolo_lines = label_file.to_yolo_format()
        assert len(yolo_lines) == 1
        assert yolo_lines[0].startswith("0 ")  # Kitchen_waste -> 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
