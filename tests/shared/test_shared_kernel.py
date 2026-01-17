"""Shared Kernel 单元测试"""

import pytest
import sys
from pathlib import Path

# 添加 shared 模块到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "src"))

from shared_kernel.domain.base import (
    ValueObject, Entity, AggregateRoot, DomainEvent, IRepository
)
from shared_kernel.domain.taxonomy import WasteCategory, LabelAlias, TaxonomyVersion
from shared_kernel.domain.annotation import (
    BoundingBox, Confidence, DetectionSource, Detection, LabelFile
)
from shared_kernel.domain.mapping import ClassMapping, ProtocolMapping, MappingSet


class TestWasteCategory:
    """测试垃圾分类值对象"""
    
    def test_enum_values(self):
        """测试枚举值"""
        assert WasteCategory.KITCHEN_WASTE.value == "Kitchen_waste"
        assert WasteCategory.RECYCLABLE_WASTE.value == "Recyclable_waste"
        assert WasteCategory.HAZARDOUS_WASTE.value == "Hazardous_waste"
        assert WasteCategory.OTHER_WASTE.value == "Other_waste"
    
    def test_from_string_valid(self):
        """测试有效的字符串解析"""
        assert WasteCategory.from_string("Kitchen_waste") == WasteCategory.KITCHEN_WASTE
        assert WasteCategory.from_string("kitchen_waste") == WasteCategory.KITCHEN_WASTE
        assert WasteCategory.from_string("KITCHEN_WASTE") == WasteCategory.KITCHEN_WASTE
    
    def test_from_string_invalid(self):
        """测试无效的字符串解析"""
        with pytest.raises(ValueError):
            WasteCategory.from_string("invalid_category")


class TestLabelAlias:
    """测试标签别名值对象"""
    
    def test_matches(self):
        """测试匹配功能"""
        alias = LabelAlias("potato", WasteCategory.KITCHEN_WASTE)
        assert alias.matches("potato")
        assert alias.matches("POTATO")
        assert not alias.matches("bottle")
    
    def test_immutable(self):
        """测试不可变性"""
        alias = LabelAlias("test", WasteCategory.RECYCLABLE_WASTE)
        with pytest.raises(Exception):
            alias.alias = "new_value"


class TestBoundingBox:
    """测试边界框值对象"""
    
    def test_valid_coordinates(self):
        """测试有效坐标"""
        bbox = BoundingBox(0.5, 0.5, 0.2, 0.3)
        assert bbox.x_center == 0.5
        assert bbox.y_center == 0.5
        assert bbox.width == 0.2
        assert bbox.height == 0.3
    
    def test_invalid_coordinates(self):
        """测试无效坐标"""
        with pytest.raises(ValueError):
            BoundingBox(1.5, 0.5, 0.2, 0.3)  # x_center > 1
    
    def test_to_xyxy(self):
        """测试坐标转换"""
        bbox = BoundingBox(0.5, 0.5, 0.2, 0.3)
        x1, y1, x2, y2 = bbox.to_xyxy(100, 100)
        assert x1 == 40
        assert y1 == 35
        assert x2 == 60
        assert y2 == 65
    
    def test_from_xyxy(self):
        """测试从像素坐标创建"""
        bbox = BoundingBox.from_xyxy(40, 35, 60, 65, 100, 100)
        assert abs(bbox.x_center - 0.5) < 0.01
        assert abs(bbox.y_center - 0.5) < 0.01


class TestConfidence:
    """测试置信度值对象"""
    
    def test_valid_confidence(self):
        """测试有效置信度"""
        conf = Confidence(0.95)
        assert conf.value == 0.95
    
    def test_invalid_confidence(self):
        """测试无效置信度"""
        with pytest.raises(ValueError):
            Confidence(1.5)
        with pytest.raises(ValueError):
            Confidence(-0.1)
    
    def test_threshold_comparison(self):
        """测试阈值比较"""
        conf = Confidence(0.8)
        assert conf.is_above_threshold(0.5)
        assert not conf.is_above_threshold(0.9)


class TestDetection:
    """测试检测结果"""
    
    def test_create_detection(self):
        """测试创建检测结果"""
        bbox = BoundingBox(0.5, 0.5, 0.2, 0.3)
        detection = Detection.create(
            category=WasteCategory.KITCHEN_WASTE,
            confidence=0.95,
            bbox=bbox,
            source=DetectionSource.YOLO,
            raw_label="potato"
        )
        assert detection.category == WasteCategory.KITCHEN_WASTE
        assert detection.confidence.value == 0.95
        assert detection.source == DetectionSource.YOLO
        assert detection.raw_label == "potato"


class TestLabelFile:
    """测试标注文件"""
    
    def test_create_label_file(self):
        """测试创建标注文件"""
        label_file = LabelFile(
            file_id="test_001",
            image_path="/test/image.jpg",
            image_width=640,
            image_height=480,
            detections=[]
        )
        assert label_file.image_width == 640
        assert label_file.image_height == 480
        assert len(label_file.detections) == 0
    
    def test_add_detection(self):
        """测试添加检测结果"""
        label_file = LabelFile(
            file_id="test_001",
            image_path="/test/image.jpg",
            image_width=640,
            image_height=480,
            detections=[]
        )
        bbox = BoundingBox(0.5, 0.5, 0.2, 0.3)
        detection = Detection.create(
            category=WasteCategory.RECYCLABLE_WASTE,
            confidence=0.85,
            bbox=bbox,
            source=DetectionSource.VLM
        )
        label_file.add_detection(detection)
        assert len(label_file.detections) == 1
    
    def test_filter_by_confidence(self):
        """测试按置信度过滤"""
        label_file = LabelFile(
            file_id="test_001",
            image_path="/test/image.jpg",
            image_width=640,
            image_height=480,
            detections=[]
        )
        # 添加低置信度检测
        low_bbox = BoundingBox(0.3, 0.3, 0.1, 0.1)
        low_det = Detection.create(
            category=WasteCategory.KITCHEN_WASTE,
            confidence=0.3,
            bbox=low_bbox,
            source=DetectionSource.YOLO
        )
        # 添加高置信度检测
        high_bbox = BoundingBox(0.7, 0.7, 0.2, 0.2)
        high_det = Detection.create(
            category=WasteCategory.RECYCLABLE_WASTE,
            confidence=0.9,
            bbox=high_bbox,
            source=DetectionSource.YOLO
        )
        label_file.add_detection(low_det)
        label_file.add_detection(high_det)
        
        filtered = label_file.filter_by_confidence(0.5)
        assert len(filtered) == 1
        assert filtered[0].category == WasteCategory.RECYCLABLE_WASTE
    
    def test_to_yolo_format(self):
        """测试转换为 YOLO 格式"""
        label_file = LabelFile(
            file_id="test_001",
            image_path="/test/image.jpg",
            image_width=640,
            image_height=480,
            detections=[]
        )
        bbox = BoundingBox(0.5, 0.5, 0.2, 0.3)
        detection = Detection.create(
            category=WasteCategory.KITCHEN_WASTE,
            confidence=0.95,
            bbox=bbox,
            source=DetectionSource.YOLO
        )
        label_file.add_detection(detection)
        
        yolo_lines = label_file.to_yolo_format()
        assert len(yolo_lines) == 1
        assert yolo_lines[0].startswith("0 ")  # Kitchen_waste -> 0


class TestMappingSet:
    """测试映射集合"""
    
    def test_create_default(self):
        """测试创建默认映射集"""
        mapping_set = MappingSet.create_default()
        assert "default" in mapping_set.class_mappings
        assert "yolo" in mapping_set.class_mappings
        assert "default" in mapping_set.protocol_mappings
    
    def test_get_class_mapping(self):
        """测试获取类别映射"""
        mapping_set = MappingSet.create_default()
        yolo_mapping = mapping_set.get_class_mapping("yolo")
        assert yolo_mapping is not None
        assert yolo_mapping.get_category(0) == "Kitchen_waste"
    
    def test_get_protocol_mapping(self):
        """测试获取协议映射"""
        mapping_set = MappingSet.create_default()
        stm32_mapping = mapping_set.get_protocol_mapping("stm32")
        assert stm32_mapping is not None
        assert stm32_mapping.encode(0) == 1  # Kitchen_waste -> 0x01


class TestAggregateRoot:
    """测试聚合根"""
    
    def test_domain_events(self):
        """测试领域事件"""
        class TestAggregate(AggregateRoot):
            def __init__(self):
                super().__init__()
            
            @property
            def id(self):
                return "test_id"
        
        aggregate = TestAggregate()
        event = DomainEvent()
        aggregate.add_domain_event(event)
        
        events = aggregate.clear_domain_events()
        assert len(events) == 1
        assert len(aggregate.clear_domain_events()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
