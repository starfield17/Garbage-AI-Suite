"""启动运行时处理器"""

from typing import Optional
import threading
import time
import cv2

from shared_kernel.config.loader import ConfigLoader

from ...domain.model import SortingSession, SessionStatus, CooldownPolicy, StabilityPolicy
from ...domain.model.entity import DetectionFrame
from ...domain.service import PacketEncoder
from ...infrastructure import YoloRuntime, CameraOpencv, SerialPyserial

from .dto import DeployStatusDTO, DetectionResultDTO
from .command import StartRuntimeCmd


class StartRuntimeHandler:
    """启动运行时处理器
    
    应用服务：编排部署运行时的启动流程
    """
    
    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
    ):
        self._config_loader = config_loader or ConfigLoader()
        self._runtime: Optional[YoloRuntime] = None
        self._camera: Optional[CameraOpencv] = None
        self._serial: Optional[SerialPyserial] = None
        self._session: Optional[SortingSession] = None
        self._packet_encoder: Optional[PacketEncoder] = None
        self._is_running = False
        self._processing_thread: Optional[threading.Thread] = None
    
    def handle(self, command: StartRuntimeCmd) -> DeployStatusDTO:
        """处理启动运行时命令"""
        # 加载协议映射
        self._packet_encoder = PacketEncoder(self._config_loader)
        self._packet_encoder.load_protocol_mapping(command.protocol)
        
        # 创建会话
        class_mapping = self._config_loader.get_deploy_class_map(command.protocol)
        self._session = SortingSession.create(
            class_mapping=class_mapping,
            cooldown_policy=CooldownPolicy(),
            stability_policy=StabilityPolicy()
        )
        
        # 初始化运行时
        self._runtime = YoloRuntime(confidence_threshold=command.confidence_threshold)
        self._runtime.load_model(command.model_path)
        
        # 打开相机
        self._camera = CameraOpencv()
        if not self._camera.open(command.camera_id, command.camera_width, command.camera_height):
            return DeployStatusDTO(
                session_id=self._session.id,
                status="error",
                is_running=False,
                model_loaded=True,
                camera_opened=False,
                serial_connected=False,
                error="Failed to open camera"
            )
        
        # 打开串口
        if command.serial_port:
            self._serial = SerialPyserial()
            if not self._serial.open(command.serial_port, command.serial_baudrate):
                return DeployStatusDTO(
                    session_id=self._session.id,
                    status="error",
                    is_running=False,
                    model_loaded=True,
                    camera_opened=True,
                    serial_connected=False,
                    error="Failed to open serial port"
                )
        
        # 初始化会话
        width, height = self._camera.get_resolution()
        self._session.initialize(width, height)
        self._session.start()
        self._is_running = True
        
        # 启动处理线程
        self._processing_thread = threading.Thread(target=self._process_frames)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        return self._get_status()
    
    def _process_frames(self) -> None:
        """处理相机帧"""
        while self._is_running and self._camera.is_opened():
            frame = self._camera.read()
            if frame is None:
                continue
            
            # 执行推理
            detections = self._runtime.infer(frame)
            
            # 创建检测帧
            if detections:
                detection = detections[0]
                detection_frame = DetectionFrame(
                    frame_id=str(time.time()),
                    image_width=frame.shape[1],
                    image_height=frame.shape[0],
                    detected_category=detection.category,
                    confidence=detection.confidence.value,
                    x_normalized=detection.bounding_box.x_center,
                    y_normalized=detection.bounding_box.y_center
                )
            else:
                detection_frame = DetectionFrame(
                    frame_id=str(time.time()),
                    image_width=frame.shape[1],
                    image_height=frame.shape[0]
                )
            
            # 处理帧
            packet = self._session.process_frame(detection_frame)
            
            # 发送串口数据
            if packet and self._serial:
                self._serial.write_packet(packet)
    
    def _get_status(self) -> DeployStatusDTO:
        """获取当前状态"""
        return DeployStatusDTO(
            session_id=self._session.id if self._session else "",
            status=self._session.status.value if self._session else "idle",
            is_running=self._is_running,
            model_loaded=self._runtime.is_loaded() if self._runtime else False,
            camera_opened=self._camera.is_opened() if self._camera else False,
            serial_connected=self._serial.is_connected() if self._serial else False,
            total_frames=self._session.statistics.total_frames if self._session else 0,
            total_detections=self._session.statistics.total_detections if self._session else 0,
            serial_packets_sent=self._session.statistics.serial_packets_sent if self._session else 0,
            counter={str(k): v for k, v in self._session.counter.counts.items()} if self._session else {}
        )
    
    def stop(self) -> None:
        """停止运行时"""
        self._is_running = False
        
        if self._session:
            self._session.stop()
        
        if self._camera:
            self._camera.close()
        
        if self._serial:
            self._serial.close()
        
        if self._runtime:
            self._runtime.unload()
    
    def get_session(self) -> Optional[SortingSession]:
        """获取会话"""
        return self._session
