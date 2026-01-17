"""稳定策略值对象 - 控制识别稳定性判断"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StabilityPolicy:
    """稳定策略值对象
    
    职责:
    - 定义识别稳定性的判断条件
    - 控制防重计数机制
    - 管理对象跟踪状态
    """
    stability_threshold_ms: int = 1000  # 稳定性判定时间（毫秒）
    detection_reset_ms: int = 500       # 检测重置时间（毫秒）
    position_tolerance: float = 0.05    # 位置容差（归一化坐标）
    min_detection_count: int = 2        # 最小检测次数
    max_retry_count: int = 3            # 最大重试次数
    
    def __post_init__(self):
        if not 0.0 <= self.position_tolerance <= 1.0:
            raise ValueError(f"position_tolerance must be between 0 and 1, got {self.position_tolerance}")
    
    def is_position_stable(
        self,
        current_x: float,
        current_y: float,
        previous_x: Optional[float],
        previous_y: Optional[float]
    ) -> bool:
        """判断位置是否稳定"""
        if previous_x is None or previous_y is None:
            return False
        
        x_diff = abs(current_x - previous_x)
        y_diff = abs(current_y - previous_y)
        
        return x_diff <= self.position_tolerance and y_diff <= self.position_tolerance
    
    def should_count(
        self,
        detection_count: int,
        is_stable: bool,
        is_already_counted: bool
    ) -> bool:
        """判断是否应该计数"""
        if is_already_counted:
            return False
        if not is_stable:
            return False
        if detection_count < self.min_detection_count:
            return False
        return True
