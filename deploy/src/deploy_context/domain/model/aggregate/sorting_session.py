"""分拣会话聚合根 - 管理部署运行时的核心业务逻辑"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import uuid4

from shared_kernel.domain.base import AggregateRoot

from ..value_object import SerialPacket, CooldownPolicy, StabilityPolicy
from ..entity import DetectionFrame, Counter
from ...event.item_classified import ItemClassified


class SessionStatus(Enum):
    """会话状态"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class TrackingState(Enum):
    """跟踪状态"""
    INITIAL = "initial"
    TRACKING = "tracking"
    STABLE = "stable"
    LOST = "lost"


@dataclass
class TrackedObject:
    """跟踪对象"""
    category_id: int
    first_x: float
    first_y: float
    last_x: float
    last_y: float
    first_seen: datetime
    last_updated: datetime
    detection_count: int = 0
    is_stable: bool = False
    is_counted: bool = False


@dataclass
class SessionStatistics:
    """会话统计"""
    total_frames: int = 0
    total_detections: int = 0
    stable_detections: int = 0
    serial_packets_sent: int = 0
    error_count: int = 0


class SortingSession(AggregateRoot):
    """分拣会话聚合根
    
    职责:
    - 管理部署运行时的完整生命周期
    - 协调相机、推理引擎、串口通信
    - 维护对象跟踪和稳定性判断
    - 发布领域事件
    
    不变量:
    - 只有 RUNNING 状态可以处理检测
    - 统计数据始终与实际结果一致
    - 串口发送遵守冷却策略
    """
    
    def __init__(
        self,
        session_id: str,
        cooldown_policy: Optional[CooldownPolicy] = None,
        stability_policy: Optional[StabilityPolicy] = None,
    ):
        super().__init__()
        self._session_id = session_id
        self._status = SessionStatus.IDLE
        self._cooldown_policy = cooldown_policy or CooldownPolicy()
        self._stability_policy = stability_policy or StabilityPolicy()
        
        # 运行时状态
        self._camera_width: Optional[int] = None
        self._camera_height: Optional[int] = None
        self._class_mapping: Dict[int, int] = {}  # 分类到协议字节的映射
        
        # 跟踪状态
        self._tracked_objects: Dict[str, TrackedObject] = {}
        self._last_serial_time: Optional[datetime] = None
        self._last_detected_category: Optional[int] = None
        self._detection_reset_timer: float = 0.0
        
        # 统计
        self._statistics = SessionStatistics()
        self._counter = Counter(counter_id=f"{session_id}_counter")
        
        # 时间戳
        self._created_at = datetime.utcnow()
        self._started_at: Optional[datetime] = None
        self._stopped_at: Optional[datetime] = None
    
    @property
    def id(self) -> str:
        return self._session_id
    
    @property
    def status(self) -> SessionStatus:
        return self._status
    
    @property
    def statistics(self) -> SessionStatistics:
        return self._statistics
    
    @property
    def counter(self) -> Counter:
        return self._counter
    
    @property
    def is_running(self) -> bool:
        return self._status == SessionStatus.RUNNING
    
    @classmethod
    def create(
        cls,
        session_id: Optional[str] = None,
        class_mapping: Optional[Dict[int, int]] = None,
        cooldown_policy: Optional[CooldownPolicy] = None,
        stability_policy: Optional[StabilityPolicy] = None,
    ) -> "SortingSession":
        """工厂方法：创建新会话"""
        session_id = session_id or str(uuid4())
        session = cls(
            session_id=session_id,
            cooldown_policy=cooldown_policy,
            stability_policy=stability_policy,
        )
        session._class_mapping = class_mapping or {}
        return session
    
    def initialize(self, camera_width: int, camera_height: int) -> None:
        """初始化会话"""
        if self._status != SessionStatus.IDLE:
            raise InvalidSessionStateError(f"Cannot initialize session in {self._status} status")
        
        self._status = SessionStatus.INITIALIZING
        self._camera_width = camera_width
        self._camera_height = camera_height
    
    def start(self) -> None:
        """启动会话"""
        if self._status not in (SessionStatus.IDLE, SessionStatus.INITIALIZING):
            raise InvalidSessionStateError(f"Cannot start session in {self._status} status")
        
        self._status = SessionStatus.RUNNING
        self._started_at = datetime.utcnow()
    
    def stop(self) -> None:
        """停止会话"""
        if self._status == SessionStatus.RUNNING:
            self._status = SessionStatus.STOPPED
            self._stopped_at = datetime.utcnow()
    
    def pause(self) -> None:
        """暂停会话"""
        if self._status != SessionStatus.RUNNING:
            raise InvalidSessionStateError(f"Cannot pause session in {self._status} status")
        self._status = SessionStatus.PAUSED
    
    def resume(self) -> None:
        """恢复会话"""
        if self._status != SessionStatus.PAUSED:
            raise InvalidSessionStateError(f"Cannot resume session in {self._status} status")
        self._status = SessionStatus.RUNNING
    
    def process_frame(self, frame: DetectionFrame) -> Optional[SerialPacket]:
        """处理检测帧
        
        Returns:
            SerialPacket: 如果需要发送串口数据，返回数据包；否则返回 None
        """
        if not self.is_running:
            raise InvalidSessionStateError(f"Cannot process frame in {self._status} status")
        
        self._statistics.total_frames += 1
        
        if not frame.has_detection:
            self._handle_no_detection()
            return None
        
        self._statistics.total_detections += 1
        
        # 更新或创建跟踪对象
        tracked = self._update_tracking(frame)
        if tracked is None:
            return None
        
        # 检查稳定性
        if self._check_stability(tracked):
            tracked.is_stable = True
            self._statistics.stable_detections += 1
            
            # 检查是否应该计数
            if self._stability_policy.should_count(
                tracked.detection_count,
                tracked.is_stable,
                tracked.is_counted
            ):
                return self._create_serial_packet(tracked)
        
        return None
    
    def _handle_no_detection(self) -> None:
        """处理无检测情况"""
        if self._last_detected_category is not None:
            self._detection_reset_timer += 1
            if self._detection_reset_timer * 0.033 > self._stability_policy.detection_reset_ms / 1000.0:
                self._last_detected_category = None
                self._detection_reset_timer = 0.0
    
    def _update_tracking(self, frame: DetectionFrame) -> Optional[TrackedObject]:
        """更新对象跟踪"""
        category_id = self._get_protocol_class_id(frame.detected_category)
        
        if category_id is None:
            return None
        
        x, y = frame.x_normalized, frame.y_normalized
        
        # 查找现有跟踪对象
        existing = None
        for obj in self._tracked_objects.values():
            if obj.category_id == category_id:
                if self._stability_policy.is_position_stable(x, y, obj.last_x, obj.last_y):
                    existing = obj
                    break
        
        if existing:
            # 更新现有对象
            existing.last_x = x
            existing.last_y = y
            existing.last_updated = datetime.utcnow()
            existing.detection_count += 1
            return existing
        else:
            # 创建新跟踪对象
            obj = TrackedObject(
                category_id=category_id,
                first_x=x,
                first_y=y,
                last_x=x,
                last_y=y,
                first_seen=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                detection_count=1,
                is_stable=False,
                is_counted=False
            )
            self._tracked_objects[str(uuid4())] = obj
            return obj
    
    def _check_stability(self, tracked: TrackedObject) -> bool:
        """检查稳定性"""
        if tracked.detection_count < self._stability_policy.min_detection_count:
            return False
        
        elapsed = (datetime.utcnow() - tracked.first_seen).total_seconds() * 1000
        return elapsed >= self._stability_policy.stability_threshold_ms
    
    def _create_serial_packet(self, tracked: TrackedObject) -> Optional[SerialPacket]:
        """创建串口数据包"""
        # 检查冷却
        if not self._cooldown_policy.should_send(
            self._last_serial_time,
            tracked.category_id
        ):
            return None
        
        # 检查是否重复发送相同类别
        if tracked.category_id == self._last_detected_category:
            return None
        
        # 创建数据包
        packet = SerialPacket.from_normalized(
            class_id=tracked.category_id,
            x_normalized=tracked.last_x,
            y_normalized=tracked.last_y
        )
        
        # 更新状态
        self._last_serial_time = datetime.utcnow()
        self._last_detected_category = tracked.category_id
        tracked.is_counted = True
        self._statistics.serial_packets_sent += 1
        
        # 更新计数
        self._counter.increment(self._get_waste_category(tracked.category_id))
        
        # 发布领域事件
        self.add_domain_event(ItemClassified(
            session_id=self._session_id,
            category_id=tracked.category_id,
            x=tracked.last_x,
            y=tracked.last_y
        ))
        
        return packet
    
    def _get_protocol_class_id(self, category) -> Optional[int]:
        """获取协议类别编号"""
        if category is None:
            return None
        return self._class_mapping.get(category.value, category.value)
    
    def _get_waste_category(self, protocol_id: int) -> Optional[int]:
        """从协议编号获取垃圾分类（返回协议中的分类值）"""
        for protocol, cat_value in self._class_mapping.items():
            if cat_value == protocol_id:
                return protocol
        return None
    
    def get_tracked_objects_info(self) -> List[Dict[str, Any]]:
        """获取跟踪对象信息"""
        return [
            {
                "category_id": obj.category_id,
                "first_position": (obj.first_x, obj.first_y),
                "last_position": (obj.last_x, obj.last_y),
                "detection_count": obj.detection_count,
                "is_stable": obj.is_stable,
                "is_counted": obj.is_counted,
            }
            for obj in self._tracked_objects.values()
        ]


class InvalidSessionStateError(Exception):
    """无效的会话状态错误"""
    pass
