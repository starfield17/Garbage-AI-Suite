"""Model adapters module."""

from .yolo_adapter import YOLOAdapter
from .faster_rcnn_adapter import FasterRCNNAdapter
from .qwen_vl_adapter import QwenVLAdapter

__all__ = ["YOLOAdapter", "FasterRCNNAdapter", "QwenVLAdapter"]
