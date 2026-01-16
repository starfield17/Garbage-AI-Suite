"""Camera adapter implementation."""

from typing import Optional

import cv2
import numpy as np

from garbage_deploy.application.ports import CameraPort
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class OpenCVCameraAdapter(CameraPort):
    def __init__(self, config: dict):
        self.device_index = config.get("device_index", 0)
        self.width = config.get("width", 640)
        self.height = config.get("height", 480)
        self.crop_region = config.get("crop_region")
        self._cap = None
        self._open_camera()

    def _open_camera(self):
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self.device_index}")

        log.info(
            "Camera opened",
            index=self.device_index,
            width=self.width,
            height=self.height,
        )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("Camera not opened")

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        if self.crop_region:
            x1, y1, x2, y2 = self.crop_region
            frame = frame[y1:y2, x1:x2]

        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Camera released")
