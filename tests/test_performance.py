"""Performance tests for critical paths"""

import pytest
import sys
import os
import time
from pathlib import Path

# Add module paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../shared/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../autolabel/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../train/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../deploy/src'))

from shared_kernel.domain.taxonomy import WasteCategory
from shared_kernel.domain.annotation import BoundingBox, Confidence, DetectionSource, Detection, LabelFile
from shared_kernel.domain.mapping import MappingSet


class TestPerformanceSharedKernel:
    """Performance tests for Shared Kernel"""

    def test_waste_category_lookup_performance(self):
        """Test WasteCategory lookup performance"""
        categories = [
            WasteCategory.KITCHEN_WASTE,
            WasteCategory.RECYCLABLE_WASTE,
            WasteCategory.HAZARDOUS_WASTE,
            WasteCategory.OTHER_WASTE
        ]
        
        start = time.perf_counter()
        for _ in range(10000):
            for category in categories:
                _ = category.value
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"WasteCategory lookup took {elapsed:.3f}s, expected < 1.0s"
        print(f"WasteCategory lookup: {elapsed:.4f}s for 40000 iterations")

    def test_bounding_box_calculation_performance(self):
        """Test BoundingBox calculation performance"""
        start = time.perf_counter()
        for _ in range(10000):
            bbox = BoundingBox(0.5, 0.5, 0.2, 0.3)
            _ = bbox.to_xyxy(640, 480)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"BoundingBox calculation took {elapsed:.3f}s, expected < 1.0s"
        print(f"BoundingBox calculation: {elapsed:.4f}s for 10000 iterations")

    def test_detection_creation_performance(self):
        """Test Detection creation performance"""
        bbox = BoundingBox(0.5, 0.5, 0.2, 0.3)
        conf = Confidence(0.95)
        
        start = time.perf_counter()
        for _ in range(10000):
            detection = Detection.create(
                category=WasteCategory.KITCHEN_WASTE,
                confidence=0.95,
                bbox=bbox,
                source=DetectionSource.YOLO,
                raw_label="potato"
            )
        elapsed = time.perf_counter() - start
        
        assert elapsed < 2.0, f"Detection creation took {elapsed:.3f}s, expected < 2.0s"
        print(f"Detection creation: {elapsed:.4f}s for 10000 iterations")

    def test_label_file_filtering_performance(self):
        """Test LabelFile filtering performance"""
        # Create a label file with many detections
        label_file = LabelFile(
            file_id="test",
            image_path="/test/image.jpg",
            image_width=640,
            image_height=480,
            detections=[]
        )
        
        for i in range(100):
            bbox = BoundingBox(0.5, 0.5, 0.2, 0.3)
            detection = Detection.create(
                category=WasteCategory.KITCHEN_WASTE,
                confidence=0.5 + i * 0.005,  # Varying confidence
                bbox=bbox,
                source=DetectionSource.YOLO,
                raw_label=f"label_{i}"
            )
            label_file.add_detection(detection)
        
        # Test filtering performance
        start = time.perf_counter()
        for _ in range(1000):
            _ = label_file.filter_by_confidence(0.7)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"LabelFile filtering took {elapsed:.3f}s, expected < 1.0s"
        print(f"LabelFile filtering: {elapsed:.4f}s for 1000 iterations")

    def test_mapping_set_lookup_performance(self):
        """Test MappingSet lookup performance"""
        mapping_set = MappingSet.create_default()
        
        start = time.perf_counter()
        for _ in range(10000):
            mapping = mapping_set.get_class_mapping("yolo")
            protocol = mapping_set.get_protocol_mapping("stm32")
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"MappingSet lookup took {elapsed:.3f}s, expected < 1.0s"
        print(f"MappingSet lookup: {elapsed:.4f}s for 10000 iterations")


class TestPerformanceAutoLabel:
    """Performance tests for AutoLabel Context"""

    def test_engine_type_parsing_performance(self):
        """Test EngineType parsing performance"""
        from autolabel_context.domain.model.value_object.engine_type import EngineType
        
        start = time.perf_counter()
        for _ in range(10000):
            _ = EngineType.from_string("yolo")
            _ = EngineType.from_string("faster_rcnn")
            _ = EngineType.from_string("vlm")
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"EngineType parsing took {elapsed:.3f}s, expected < 1.0s"
        print(f"EngineType parsing: {elapsed:.4f}s for 30000 iterations")

    def test_confidence_comparison_performance(self):
        """Test Confidence comparison performance"""
        from autolabel_context.domain.model.value_object.confidence import Confidence
        
        conf = Confidence(0.75)
        
        start = time.perf_counter()
        for _ in range(100000):
            _ = conf.is_above_threshold(0.5)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"Confidence comparison took {elapsed:.3f}s, expected < 1.0s"
        print(f"Confidence comparison: {elapsed:.4f}s for 100000 iterations")

    def test_job_id_generation_performance(self):
        """Test JobId generation performance"""
        from autolabel_context.domain.model.value_object.job_id import JobId
        
        start = time.perf_counter()
        for _ in range(10000):
            _ = JobId.generate()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"JobId generation took {elapsed:.3f}s, expected < 1.0s"
        print(f"JobId generation: {elapsed:.4f}s for 10000 iterations")

    def test_image_item_creation_performance(self):
        """Test ImageItem creation performance"""
        from autolabel_context.domain.model.entity.image_item import ImageItem
        
        start = time.perf_counter()
        for i in range(10000):
            item = ImageItem.create(f"/tmp/test_{i}.jpg")
            item.mark_processed()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 2.0, f"ImageItem creation took {elapsed:.3f}s, expected < 2.0s"
        print(f"ImageItem creation: {elapsed:.4f}s for 10000 iterations")


class TestPerformanceTrain:
    """Performance tests for Train Context"""

    def test_hyper_params_creation_performance(self):
        """Test HyperParams creation performance"""
        from train_context.domain.model.value_object.hyper_params import HyperParams
        
        start = time.perf_counter()
        for _ in range(10000):
            params = HyperParams(epochs=100, batch_size=16, learning_rate=0.01)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"HyperParams creation took {elapsed:.3f}s, expected < 1.0s"
        print(f"HyperParams creation: {elapsed:.4f}s for 10000 iterations")

    def test_run_id_generation_performance(self):
        """Test RunId generation performance"""
        from train_context.domain.model.value_object.run_id import RunId
        
        start = time.perf_counter()
        for _ in range(10000):
            _ = RunId.generate("yolo")
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"RunId generation took {elapsed:.3f}s, expected < 1.0s"
        print(f"RunId generation: {elapsed:.4f}s for 10000 iterations")

    def test_model_selector_query_performance(self):
        """Test ModelSelector query performance"""
        from train_context.domain.service.model_selector import ModelSelector
        
        selector = ModelSelector()
        
        start = time.perf_counter()
        for _ in range(1000):
            models = selector.get_available_models()
            info = selector.get_model_info("yolo", "n")
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"ModelSelector query took {elapsed:.3f}s, expected < 1.0s"
        print(f"ModelSelector query: {elapsed:.4f}s for 1000 iterations")


class TestPerformanceDeploy:
    """Performance tests for Deploy Context"""

    def test_serial_packet_creation_performance(self):
        """Test SerialPacket creation performance"""
        from deploy_context.domain.model.value_object.serial_packet import SerialPacket
        
        start = time.perf_counter()
        for i in range(10000):
            packet = SerialPacket(class_id=1, x=127, y=127)
            _ = packet.to_bytes()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"SerialPacket creation took {elapsed:.3f}s, expected < 1.0s"
        print(f"SerialPacket creation: {elapsed:.4f}s for 10000 iterations")

    def test_cooldown_policy_check_performance(self):
        """Test CooldownPolicy check performance"""
        from deploy_context.domain.model.value_object.cooldown_policy import CooldownPolicy
        from datetime import datetime, timedelta
        
        policy = CooldownPolicy()
        
        start = time.perf_counter()
        for _ in range(10000):
            _ = policy.should_send(None, 1)
            _ = policy.should_send(datetime.utcnow() - timedelta(milliseconds=150), 1)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"CooldownPolicy check took {elapsed:.3f}s, expected < 1.0s"
        print(f"CooldownPolicy check: {elapsed:.4f}s for 10000 iterations")

    def test_stability_policy_check_performance(self):
        """Test StabilityPolicy check performance"""
        from deploy_context.domain.model.value_object.stability_policy import StabilityPolicy
        
        policy = StabilityPolicy()
        
        start = time.perf_counter()
        for _ in range(10000):
            _ = policy.is_position_stable(0.5, 0.5, 0.51, 0.49)
            _ = policy.is_position_stable(0.5, 0.5, 0.6, 0.6)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"StabilityPolicy check took {elapsed:.3f}s, expected < 1.0s"
        print(f"StabilityPolicy check: {elapsed:.4f}s for 10000 iterations")

    def test_counter_increment_performance(self):
        """Test Counter increment performance"""
        from deploy_context.domain.model.entity.counter import Counter
        
        counter = Counter(counter_id="test")
        
        start = time.perf_counter()
        for i in range(10000):
            counter.increment(WasteCategory.KITCHEN_WASTE)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"Counter increment took {elapsed:.3f}s, expected < 1.0s"
        print(f"Counter increment: {elapsed:.4f}s for 10000 iterations")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
