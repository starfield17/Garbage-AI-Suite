"""Runtime adapter implementations."""

import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO
from pydantic import BaseModel

from garbage_deploy.application.ports import RuntimePort
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class ModelManifestDTO(BaseModel):
    model_path: str
    model_family: str
    input_width: int
    input_height: int
    classes: List[dict]


class TorchRuntimeAdapter(RuntimePort):
    def __init__(self, config: dict):
        self.model_path = config.get("model_path", "")
        self.device = config.get("device", "cpu")
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self._model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path or not Path(self.model_path).exists():
            raise ValueError(f"Model not found: {self.model_path}")

        log.info("Loading model", path=self.model_path, device=self.device)
        self._model = YOLO(self.model_path)
        self._model.to(self.device)

    def load_model(self, manifest_path: Path | None, device: str) -> None:
        self.device = device or self.device
        if manifest_path:
            log.info("Loading from manifest", path=str(manifest_path))
        self._load_model()

    def infer(self, frame: np.ndarray) -> List[dict]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        results = self._model(frame, verbose=False)
        detections = []

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()

                nms_keep_idx = self._nms(boxes.xyxy.cpu().numpy(), confidences, self.iou_threshold)

                for idx in nms_keep_idx:
                    if confidences[idx] >= self.confidence_threshold:
                        x1, y1, x2, y2 = boxes.xyxy[idx]
                        detections.append(
                            {
                                    "class_id": int(class_ids[idx]),
                                    "confidence": float(confidences[idx]),
                                    "bbox": {
                                        "x1": float(x1),
                                        "y1": float(y1),
                                        "x2": float(x2),
                                        "y2": float(y2),
                                    },
                                }
                        )

        return detections

    def _nms(self, boxes, confidences, iou_threshold):
        indices = []
        if len(boxes) == 0:
            return indices

        for i in range(len(boxes)):
            keep = True
            for j in range(len(boxes)):
                if i != j and j in indices:
                    iou = self._calculate_iou(boxes[i], boxes[j])
                    if iou > iou_threshold:
                        keep = False
                        break
            if keep:
                indices.append(i)
        return indices

    @staticmethod
    def _calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter_area = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


class RKNNRuntimeAdapter(RuntimePort):
    def __init__(self, config: dict):
        self.model_path = config.get("model_path", "")
        self.device = config.get("device", "rk3588")
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self._rknn_runtime = None
        self._load_model()

    def _load_model(self):
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise RuntimeError("RKNN runtime not available")

        if not self.model_path or not Path(self.model_path).exists():
            raise ValueError(f"RKNN model not found: {self.model_path}")

        log.info("Loading RKNN model", path=self.model_path)

        ret = RKNNLite(verbose=True)
        ret = ret.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=self.device,
            quantized_dtype="asymmetric_quantized-u8",
            optimization_level=3,
        )
        ret = ret.load_rknn(self.model_path)

        if ret != 0:
            raise RuntimeError(f"RKNN load failed: {ret}")

        self._rknn_runtime = ret

    def load_model(self, manifest_path: Path | None, device: str) -> None:
        self.device = device or self.device
        self._load_model()

    def infer(self, frame: np.ndarray) -> List[dict]:
        if self._rknn_runtime is None:
            raise RuntimeError("RKNN model not loaded")

        results = self._rknn_runtime.inference(inputs=[frame], data_format="nhwc")
        outputs = results[0]

        detections = []
        for output in outputs:
            if len(output) >= 6:
                x1, y1, x2, y2, conf = output[:5]
                conf = float(conf) / 255.0

                if conf >= self.confidence_threshold:
                    detections.append(
                        {
                            "class_id": int(output[5]),
                            "confidence": conf,
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                            },
                        }
                    )

        return detections
