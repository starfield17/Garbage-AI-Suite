"""时间工具"""

from datetime import datetime
from typing import Union


def timestamp_to_datetime(ts: Union[int, float]) -> datetime:
    """时间戳转 datetime"""
    return datetime.fromtimestamp(ts)


def datetime_to_timestamp(dt: datetime) -> float:
    """datetime 转时间戳"""
    return dt.timestamp()


def now_utc() -> datetime:
    """获取当前 UTC 时间"""
    return datetime.utcnow()


def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """格式化 datetime"""
    return dt.strftime(fmt)
