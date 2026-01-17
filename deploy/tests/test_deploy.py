"""Deploy Context 测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy_context.domain.model.value_object import SerialPacket, CooldownPolicy, StabilityPolicy
from deploy_context.domain.model.entity import DetectionFrame, Counter
from deploy_context.domain.model.aggregate import SortingSession, SessionStatus
from deploy_context.domain.service import StabilityJudge, PacketEncoder
from shared_kernel.domain.taxonomy import WasteCategory


class TestSerialPacket:
    """测试串口数据包"""
    
    def test_create_valid_packet(self):
        """测试创建有效数据包"""
        packet = SerialPacket(class_id=1, x=128, y=255)
        assert packet.class_id == 1
        assert packet.x == 128
        assert packet.y == 255
    
    def test_create_empty_packet(self):
        """测试创建空数据包"""
        packet = SerialPacket.empty()
        assert packet.class_id == 0
        assert packet.x == 0
        assert packet.y == 0
    
    def test_from_normalized(self):
        """测试从归一化坐标创建"""
        packet = SerialPacket.from_normalized(class_id=2, x_normalized=0.5, y_normalized=0.75)
        assert packet.class_id == 2
        assert packet.x == 127  # 0.5 * 255 ≈ 127
        assert packet.y == 191  # 0.75 * 255 ≈ 191
    
    def test_to_bytes(self):
        """测试转换为字节"""
        packet = SerialPacket(class_id=3, x=100, y=200)
        data = packet.to_bytes()
        assert data == bytes([3, 100, 200])
    
    def test_invalid_class_id(self):
        """测试无效的 class_id"""
        with pytest.raises(ValueError):
            SerialPacket(class_id=5, x=0, y=0)
    
    def test_invalid_x(self):
        """测试无效的 x"""
        with pytest.raises(ValueError):
            SerialPacket(class_id=1, x=256, y=0)


class TestCooldownPolicy:
    """测试冷却策略"""
    
    def test_default_policy(self):
        """测试默认策略"""
        policy = CooldownPolicy()
        assert policy.min_interval_ms == 100
        assert policy.max_queue_size == 10
    
    def test_should_send_first_time(self):
        """测试首次发送"""
        policy = CooldownPolicy()
        assert policy.should_send(None, 1) is True
    
    def test_should_send_after_cooldown(self):
        """测试冷却后发送"""
        from datetime import datetime, timedelta
        policy = CooldownPolicy(min_interval_ms=100)
        last_time = datetime.utcnow() - timedelta(milliseconds=150)
        assert policy.should_send(last_time, 1) is True


class TestStabilityPolicy:
    """测试稳定策略"""
    
    def test_default_policy(self):
        """测试默认策略"""
        policy = StabilityPolicy()
        assert policy.stability_threshold_ms == 1000
        assert policy.min_detection_count == 2
    
    def test_is_position_stable(self):
        """测试位置稳定性"""
        policy = StabilityPolicy(position_tolerance=0.05)
        assert policy.is_position_stable(0.5, 0.5, 0.52, 0.48) is True
        assert policy.is_position_stable(0.5, 0.5, 0.6, 0.6) is False


class TestDetectionFrame:
    """测试检测帧"""
    
    def test_create_frame(self):
        """测试创建检测帧"""
        frame = DetectionFrame(
            frame_id="test_001",
            image_width=1280,
            image_height=720,
            detected_category=WasteCategory.KITCHEN_WASTE,
            confidence=0.95,
            x_normalized=0.5,
            y_normalized=0.5
        )
        assert frame.has_detection is True
        assert frame.id == "test_001"
    
    def test_to_serial_coordinates(self):
        """测试转换为串口坐标"""
        frame = DetectionFrame(
            frame_id="test_001",
            image_width=1280,
            image_height=720,
            x_normalized=0.5,
            y_normalized=0.5
        )
        x, y = frame.to_serial_coordinates()
        assert x == 127
        assert y == 127


class TestCounter:
    """测试计数"""
    
    def test_increment(self):
        """测试计数增加"""
        counter = Counter(counter_id="test_counter")
        counter.increment(WasteCategory.KITCHEN_WASTE)
        counter.increment(WasteCategory.KITCHEN_WASTE)
        assert counter.get_count(WasteCategory.KITCHEN_WASTE) == 2
        assert counter.total_count == 2
    
    def test_reset(self):
        """测试重置"""
        counter = Counter(counter_id="test_counter")
        counter.increment(WasteCategory.RECYCLABLE_WASTE)
        counter.reset()
        assert counter.total_count == 0


class TestSortingSession:
    """测试分拣会话"""
    
    def test_create_session(self):
        """测试创建会话"""
        session = SortingSession.create(
            class_mapping={0: 1, 1: 2, 2: 3, 3: 4}
        )
        assert session.status == SessionStatus.IDLE
        assert session.is_running is False
    
    def test_lifecycle(self):
        """测试生命周期"""
        session = SortingSession.create()
        session.initialize(1280, 720)
        assert session.status == SessionStatus.INITIALIZING
        session.start()
        assert session.status == SessionStatus.RUNNING
        session.stop()
        assert session.status == SessionStatus.STOPPED
    
    def test_process_frame_with_detection(self):
        """测试处理有检测的帧"""
        session = SortingSession.create(
            class_mapping={0: 1}
        )
        session.initialize(1280, 720)
        session.start()
        
        frame = DetectionFrame(
            frame_id="frame_001",
            image_width=1280,
            image_height=720,
            detected_category=WasteCategory.KITCHEN_WASTE,
            confidence=0.95,
            x_normalized=0.5,
            y_normalized=0.5
        )
        
        # 处理第一帧（不稳定）
        packet = session.process_frame(frame)
        assert session.statistics.total_frames == 1


class TestStabilityJudge:
    """测试稳定性判断"""
    
    def test_evaluate_stable_detection(self):
        """测试稳定检测评估"""
        judge = StabilityJudge()
        frame = DetectionFrame(
            frame_id="frame_001",
            image_width=1280,
            image_height=720,
            detected_category=WasteCategory.KITCHEN_WASTE,
            confidence=0.95,
            x_normalized=0.5,
            y_normalized=0.5
        )
        
        report = judge.evaluate(frame)
        assert report.consecutive_matches == 1


class TestPacketEncoder:
    """测试包编码器"""
    
    def test_encode(self):
        """测试编码"""
        encoder = PacketEncoder()
        encoder.load_protocol_mapping("default")
        
        packet = encoder.encode(category_id=0, x_normalized=0.5, y_normalized=0.5)
        assert packet.class_id == 1  # 0 -> 1 (default mapping)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
