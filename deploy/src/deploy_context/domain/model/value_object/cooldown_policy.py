"""冷却策略值对象 - 控制发送频率防止过载"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass(frozen=True)
class CooldownPolicy:
    """冷却策略值对象
    
    职责:
    - 控制串口发送的最小间隔
    - 防止设备过载
    - 支持按类别设置不同的冷却时间
    """
    min_interval_ms: int = 100  # 最小发送间隔（毫秒）
    max_queue_size: int = 10    # 最大队列长度
    category_cooldowns: dict = None  # 按类别的特殊冷却时间
    
    def __post_init__(self):
        if self.category_cooldowns is None:
            object.__setattr__(self, 'category_cooldowns', {})
    
    def get_interval_for_category(self, category_id: int) -> int:
        """获取指定类别的冷却时间"""
        return self.category_cooldowns.get(category_id, self.min_interval_ms)
    
    def should_send(self, last_send_time: Optional[datetime], category_id: int) -> bool:
        """判断是否可以发送"""
        if last_send_time is None:
            return True
        
        interval = self.get_interval_for_category(category_id)
        elapsed = datetime.utcnow() - last_send_time
        return elapsed >= timedelta(milliseconds=interval)
    
    def get_next_send_time(self, last_send_time: datetime, category_id: int) -> datetime:
        """获取下次可发送时间"""
        interval = self.get_interval_for_category(category_id)
        return last_send_time + timedelta(milliseconds=interval)
