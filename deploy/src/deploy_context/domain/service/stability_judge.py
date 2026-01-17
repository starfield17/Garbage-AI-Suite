"""稳定性判断领域服务"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from shared_kernel.domain.annotation import Detection

from ..model.value_object import StabilityPolicy
from ..model.entity import DetectionFrame


@dataclass
class StabilityReport:
    """稳定性报告"""
    is_stable: bool
    tracking_duration_ms: float
    confidence_score: float
    position_delta: float
    consecutive_matches: int
    should_classify: bool


class StabilityJudge:
    """稳定性判断服务
    
    职责:
    - 判断检测结果是否稳定
    - 决定是否触发分类
    - 提供稳定性详细信息
    """
    
    def __init__(self, policy: Optional[StabilityPolicy] = None):
        self._policy = policy or StabilityPolicy()
        self._history: Dict[str, List[DetectionFrame]] = {}
    
    def evaluate(
        self,
        current_frame: DetectionFrame,
        history: Optional[List[DetectionFrame]] = None
    ) -> StabilityReport:
        """评估当前帧的稳定性
        
        Args:
            current_frame: 当前检测帧
            history: 历史检测帧列表
            
        Returns:
            StabilityReport: 稳定性评估报告
        """
        if not current_frame.has_detection:
            return StabilityReport(
                is_stable=False,
                tracking_duration_ms=0.0,
                confidence_score=0.0,
                position_delta=0.0,
                consecutive_matches=0,
                should_classify=False
            )
        
        history = history or []
        frames = history + [current_frame]
        
        # 计算连续匹配次数
        consecutive = self._count_consecutive_matches(frames)
        
        # 计算位置变化
        if len(frames) >= 2:
            first = frames[0]
            last = frames[-1]
            position_delta = self._calculate_position_delta(first, last)
        else:
            position_delta = 0.0
        
        # 计算跟踪时长
        tracking_duration = (current_frame.timestamp - frames[0].timestamp).total_seconds() * 1000
        
        # 计算置信度分数
        confidence_score = self._calculate_confidence_score(frames)
        
        # 判断是否稳定
        is_stable = (
            consecutive >= self._policy.min_detection_count and
            tracking_duration >= self._policy.stability_threshold_ms and
            position_delta <= self._policy.position_tolerance * 2
        )
        
        # 判断是否应该分类
        should_classify = (
            is_stable and
            current_frame.confidence is not None and
            current_frame.confidence > 0.5
        )
        
        return StabilityReport(
            is_stable=is_stable,
            tracking_duration_ms=tracking_duration,
            confidence_score=confidence_score,
            position_delta=position_delta,
            consecutive_matches=consecutive,
            should_classify=should_classify
        )
    
    def _count_consecutive_matches(self, frames: List[DetectionFrame]) -> int:
        """计算连续匹配次数"""
        if not frames:
            return 0
        
        consecutive = 0
        last_category = frames[0].detected_category
        
        for frame in frames:
            if frame.detected_category == last_category:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def _calculate_position_delta(self, frame1: DetectionFrame, frame2: DetectionFrame) -> float:
        """计算位置变化"""
        if frame1.x_normalized is None or frame1.y_normalized is None:
            return 1.0
        if frame2.x_normalized is None or frame2.y_normalized is None:
            return 1.0
        
        dx = frame2.x_normalized - frame1.x_normalized
        dy = frame2.y_normalized - frame1.y_normalized
        return (dx ** 2 + dy ** 2) ** 0.5
    
    def _calculate_confidence_score(self, frames: List[DetectionFrame]) -> float:
        """计算置信度分数"""
        if not frames:
            return 0.0
        
        confidences = [
            f.confidence for f in frames
            if f.confidence is not None
        ]
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)
    
    def reset(self) -> None:
        """重置历史记录"""
        self._history.clear()
