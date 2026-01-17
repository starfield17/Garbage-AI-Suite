"""部署 DTO"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class DeployStatusDTO:
    """部署状态 DTO"""
    session_id: str
    status: str
    is_running: bool
    model_loaded: bool
    camera_opened: bool
    serial_connected: bool
    total_frames: int = 0
    total_detections: int = 0
    serial_packets_sent: int = 0
    counter: Dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DetectionResultDTO:
    """检测结果 DTO"""
    frame_id: str
    has_detection: bool
    category: Optional[str] = None
    confidence: Optional[float] = None
    x_normalized: Optional[float] = None
    y_normalized: Optional[float] = None
    serial_packet_sent: bool = False
    packet_data: Optional[List[int]] = None
