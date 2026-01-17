"""AutoLabel Context 集成测试"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock

from autolabel_context.domain.model.value_object.engine_type import EngineType
from autolabel_context.domain.model.value_object.confidence import Confidence
from autolabel_context.domain.model.value_object.job_id import JobId
from autolabel_context.domain.model.entity.image_item import ImageItem
from autolabel_context.domain.model.entity.label_result import LabelResult
from autolabel_context.domain.model.aggregate.autolabel_job import AutoLabelJob, JobStatus
from autolabel_context.application.command.run_autolabel_cmd import RunAutoLabelCmd
from autolabel_context.application.dto.autolabel_dto import AutoLabelResultDTO


class TestEngineType:
    """测试引擎类型值对象"""
    
    def test_from_string_yolo(self):
        assert EngineType.from_string("yolo") == EngineType.YOLO
    
    def test_from_string_faster_rcnn(self):
        assert EngineType.from_string("faster_rcnn") == EngineType.FASTER_RCNN
    
    def test_from_string_vlm(self):
        assert EngineType.from_string("vlm") == EngineType.VLM
    
    def test_from_string_ensemble(self):
        assert EngineType.from_string("ensemble") == EngineType.ENSEMBLE
    
    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            EngineType.from_string("invalid")
    
    def test_supports_bounding_box(self):
        assert EngineType.YOLO.supports_bounding_box() is True
        assert EngineType.FASTER_RCNN.supports_bounding_box() is True
        assert EngineType.VLM.supports_bounding_box() is False
        assert EngineType.ENSEMBLE.supports_bounding_box() is True
    
    def test_requires_api_key(self):
        assert EngineType.YOLO.requires_api_key() is False
        assert EngineType.VLM.requires_api_key() is True
        assert EngineType.ENSEMBLE.requires_api_key() is True


class TestConfidence:
    """测试置信度值对象"""
    
    def test_valid_confidence(self):
        conf = Confidence(0.5)
        assert conf.value == 0.5
    
    def test_invalid_confidence_too_low(self):
        with pytest.raises(ValueError):
            Confidence(-0.1)
    
    def test_invalid_confidence_too_high(self):
        with pytest.raises(ValueError):
            Confidence(1.5)
    
    def test_is_above_threshold(self):
        conf = Confidence(0.7)
        assert conf.is_above_threshold(0.5) is True
        assert conf.is_above_threshold(0.8) is False


class TestJobId:
    """测试任务ID值对象"""
    
    def test_generate(self):
        job_id = JobId.generate()
        assert isinstance(job_id, JobId)
        assert len(job_id.value) > 0
    
    def test_from_string(self):
        job_id = JobId.from_string("test-id")
        assert job_id.value == "test-id"
    
    def test_from_string_empty(self):
        with pytest.raises(ValueError):
            JobId.from_string("")


class TestImageItem:
    """测试图片项实体"""
    
    def test_create_valid_path(self):
        item = ImageItem.create("/tmp/test.jpg")
        assert item.path == "/tmp/test.jpg"
        assert item.exists is False
    
    def test_create_empty_path(self):
        with pytest.raises(ValueError):
            ImageItem.create("")
    
    def test_mark_processed(self):
        item = ImageItem.create("/tmp/test.jpg")
        item.mark_processed()
        assert item.is_processed is True
    
    def test_mark_failed(self):
        item = ImageItem.create("/tmp/test.jpg")
        item.mark_failed("Error message")
        assert item.is_processed is True
        assert item.processing_error == "Error message"


class TestLabelResult:
    """测试标签结果实体"""
    
    def test_success_result(self):
        item = ImageItem.create("/tmp/test.jpg")
        result = LabelResult.success(item, [])
        assert result.is_success is True
        assert result.is_failed is False
        assert result.is_skipped is False
    
    def test_skipped_result(self):
        item = ImageItem.create("/tmp/test.jpg")
        result = LabelResult.skipped(item)
        assert result.is_skipped is True
    
    def test_failed_result(self):
        item = ImageItem.create("/tmp/test.jpg")
        result = LabelResult.failed(item, "Error")
        assert result.is_failed is True
        assert result.error_message == "Error"
    
    def test_detection_count(self):
        item = ImageItem.create("/tmp/test.jpg")
        mock_detection = Mock()
        mock_detection.category.value = "Kitchen_waste"
        result = LabelResult.success(item, [mock_detection, mock_detection])
        assert result.detection_count == 2


class TestAutoLabelJob:
    """测试自动标注任务聚合根"""
    
    def test_create_job(self):
        job = AutoLabelJob.create(
            engine_type=EngineType.YOLO,
            image_paths=["/tmp/test1.jpg", "/tmp/test2.jpg"],
            confidence_threshold=0.5
        )
        assert job.engine_type == EngineType.YOLO
        assert job.status == JobStatus.PENDING
        assert job.confidence_threshold == 0.5
        assert len(job.image_paths) == 2
    
    def test_start_job(self):
        job = AutoLabelJob.create(
            engine_type=EngineType.YOLO,
            image_paths=["/tmp/test.jpg"],
            confidence_threshold=0.5
        )
        job.start()
        assert job.status == JobStatus.RUNNING
    
    def test_start_already_running(self):
        job = AutoLabelJob.create(
            engine_type=EngineType.YOLO,
            image_paths=["/tmp/test.jpg"],
            confidence_threshold=0.5
        )
        job.start()
        with pytest.raises(Exception):
            job.start()
    
    def test_complete_job(self):
        job = AutoLabelJob.create(
            engine_type=EngineType.YOLO,
            image_paths=["/tmp/test.jpg"],
            confidence_threshold=0.5
        )
        job.start()
        job.complete()
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
    
    def test_cancel_job(self):
        job = AutoLabelJob.create(
            engine_type=EngineType.YOLO,
            image_paths=["/tmp/test.jpg"],
            confidence_threshold=0.5
        )
        job.start()
        job.cancel()
        assert job.status == JobStatus.CANCELLED
    
    def test_statistics(self):
        job = AutoLabelJob.create(
            engine_type=EngineType.YOLO,
            image_paths=["/tmp/test1.jpg", "/tmp/test2.jpg"],
            confidence_threshold=0.5
        )
        assert job.statistics.total_images == 2
        assert job.statistics.processed_images == 0


class TestRunAutoLabelCmd:
    """测试运行自动标注命令"""
    
    def test_create_command(self):
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


class TestAutoLabelResultDTO:
    """测试自动标注结果DTO"""
    
    def test_from_dict(self):
        data = {
            "job_id": "test-123",
            "status": "completed",
            "total_images": 10,
            "processed_images": 8,
            "skipped_images": 1,
            "failed_images": 1,
            "total_detections": 25,
            "detections_by_category": {"Kitchen_waste": 10, "Recyclable_waste": 15},
            "success_rate": 0.8
        }
        dto = AutoLabelResultDTO.from_dict(data)
        assert dto.job_id == "test-123"
        assert dto.total_images == 10
        assert dto.success_rate == 0.8
    
    def test_to_dict(self):
        dto = AutoLabelResultDTO(
            job_id="test-123",
            status="completed",
            total_images=10,
            processed_images=8,
            skipped_images=1,
            failed_images=1,
            total_detections=25,
            detections_by_category={"Kitchen_waste": 10},
            success_rate=0.8
        )
        result = dto.to_dict()
        assert result["job_id"] == "test-123"
        assert "statistics" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
