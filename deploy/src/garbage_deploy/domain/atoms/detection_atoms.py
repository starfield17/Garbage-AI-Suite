"""Domain atoms - pure functions for deploy module."""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class DetectionState:
    detected_type: Optional[int] = None
    position_frames: int = 0
    confidence_frames: int = 0
    last_count_time: float = 0.0


def check_detection_stability(
    state: DetectionState,
    detected_type: int,
    now: float,
    min_position_frames: int = 5,
    min_confidence_frames: int = 3,
) -> Tuple[DetectionState, bool]:
    if detected_type != state.detected_type:
        return DetectionState(), False

    state.detected_type = detected_type

    if state.detected_type is not None:
        state.position_frames += 1
        if state.position_frames >= min_position_frames:
            return state, True

    return state, False


def can_count_new_garbage(
    state: DetectionState,
    detected_type: int,
    now: float,
    cooldown_ms: int = 2000,
) -> bool:
    if detected_type != state.detected_type:
        return True

    time_since_last = now - state.last_count_time
    if time_since_last >= cooldown_ms / 1000.0:
        return True

    return False


def map_serial_payload(
    detection: dict,
    image_w: int,
    image_h: int,
    max_value: int = 255,
) -> Tuple[int, int, int]:
    bbox = detection["bbox"]
    x1, y1 = bbox["x1"], bbox["y1"]
    x2, y2 = bbox["x2"], bbox["y2"]

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    scaled_x = int((center_x / image_w) * max_value)
    scaled_y = int((center_y / image_h) * max_value)

    class_id = detection["class_id"]

    return class_id, scaled_x, scaled_y
