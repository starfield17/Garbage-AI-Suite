"""部署 CLI 入口"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from application.command import StartRuntimeCmd
from application.handler import StartRuntimeHandler


def main():
    """CLI 主入口"""
    parser = argparse.ArgumentParser(description="Deploy Context - 垃圾分类部署系统")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="模型路径 (.pt 或 .onnx)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="相机 ID (默认: 0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="相机宽度 (默认: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="相机高度 (默认: 720)"
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="串口设备路径 (如 /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="串口波特率 (默认: 115200)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="置信度阈值 (默认: 0.5)"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="default",
        help="协议类型 (default/stm32/arduino)"
    )
    
    args = parser.parse_args()
    
    # 创建命令
    cmd = StartRuntimeCmd(
        model_path=args.model,
        camera_id=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        serial_port=args.serial,
        serial_baudrate=args.baudrate,
        confidence_threshold=args.threshold,
        protocol=args.protocol
    )
    
    # 处理命令
    handler = StartRuntimeHandler()
    result = handler.handle(cmd)
    
    print(f"Session ID: {result.session_id}")
    print(f"Status: {result.status}")
    print(f"Running: {result.is_running}")
    print(f"Model Loaded: {result.model_loaded}")
    print(f"Camera Opened: {result.camera_opened}")
    print(f"Serial Connected: {result.serial_connected}")
    print(f"Total Frames: {result.total_frames}")
    print(f"Total Detections: {result.total_detections}")
    print(f"Serial Packets Sent: {result.serial_packets_sent}")
    
    try:
        input("\n按 Enter 停止运行时...")
    except KeyboardInterrupt:
        pass
    finally:
        handler.stop()
        print("\n运行时已停止")


if __name__ == "__main__":
    main()
