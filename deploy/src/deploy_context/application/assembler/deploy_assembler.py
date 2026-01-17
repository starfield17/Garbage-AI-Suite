"""部署汇编器"""

from ...domain.model import SortingSession
from .dto import DeployStatusDTO, DetectionResultDTO


class DeployAssembler:
    """部署汇编器
    
    负责聚合根与 DTO 之间的转换
    """
    
    @staticmethod
    def to_status_dto(session: SortingSession) -> DeployStatusDTO:
        """将会话转换为状态 DTO"""
        return DeployStatusDTO(
            session_id=session.id,
            status=session.status.value,
            is_running=session.is_running,
            model_loaded=True,
            camera_opened=True,
            serial_connected=True,
            total_frames=session.statistics.total_frames,
            total_detections=session.statistics.total_detections,
            serial_packets_sent=session.statistics.serial_packets_sent,
            counter={str(k): v for k, v in session.counter.counts.items()}
        )
