"""DDD 基础类型定义"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeVar, Generic, List
from uuid import uuid4


class ValueObject(ABC):
    """值对象基类
    
    约束:
    - 必须不可变（使用 @dataclass(frozen=True)）
    - 通过属性相等判断对象相等
    - 修改操作返回新对象
    """
    pass


class Entity(ABC):
    """实体基类
    
    约束:
    - 具有唯一标识
    - 通过标识判断对象相等
    - 包含业务行为方法
    """
    
    @property
    @abstractmethod
    def id(self):
        """返回实体唯一标识"""
        pass
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


class AggregateRoot(Entity):
    """聚合根基类
    
    约束:
    - 是聚合的入口点
    - 负责维护聚合内的一致性
    - 只有聚合根可以有 Repository
    """
    
    def __init__(self):
        self._domain_events: List["DomainEvent"] = []
    
    def add_domain_event(self, event: "DomainEvent") -> None:
        """添加领域事件"""
        self._domain_events.append(event)
    
    def clear_domain_events(self) -> List["DomainEvent"]:
        """清空并返回所有领域事件"""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events


@dataclass
class DomainEvent:
    """领域事件基类"""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=datetime.utcnow)


T = TypeVar("T", bound=AggregateRoot)


class IRepository(ABC, Generic[T]):
    """仓储接口基类"""
    
    @abstractmethod
    def save(self, aggregate: T) -> None:
        """保存聚合根"""
        pass
    
    @abstractmethod
    def find_by_id(self, id) -> T | None:
        """根据 ID 查找聚合根"""
        pass
