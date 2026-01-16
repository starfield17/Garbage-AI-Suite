"""Domain atoms - pure functions."""

from typing import List, Dict


def filter_by_confidence(detections: List[Dict], threshold: float) -> List[Dict]:
    filtered = [d for d in detections if d["confidence"] >= threshold]
    return filtered


def clip_bbox_to_image(detections: List[Dict], width: int, height: int) -> List[Dict]:
    result = []
    for det in detections:
        bbox = det["bbox"]
        clipped = {
            **det,
            "bbox": {
                "x1": max(0, min(bbox["x1"], width - 1)),
                "y1": max(0, min(bbox["y1"], height - 1)),
                "x2": max(0, min(bbox["x2"], width - 1)),
                "y2": max(0, min(bbox["y2"], height - 1)),
            },
        }
        result.append(clipped)
    return result


def map_class_names(detections: List[Dict], class_map: Dict[str, int]) -> List[Dict]:
    result = []
    for det in detections:
        mapped = {
            **det,
            "class_name": class_map.get(det["class_name"], det["class_name"]),
        }
        result.append(mapped)
    return result


def nms(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    sorted_dets = sorted(detections, key=lambda x: -x["confidence"])

    keep = []
    while sorted_dets:
        current = sorted_dets.pop(0)
        keep.append(current)

        remaining = []
        for det in sorted_dets:
            if _calculate_iou(current["bbox"], det["bbox"]) < iou_threshold:
                remaining.append(det)
        sorted_dets = remaining

    return keep


def _calculate_iou(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
    x1_min = max(bbox1["x1"], bbox2["x1"])
    y1_min = max(bbox1["y1"], bbox2["y1"])
    x2_max = min(bbox1["x2"], bbox2["x2"])
    y2_max = min(bbox1["y2"], bbox2["y2"])

    if x2_max <= x1_min or y2_max <= y1_min:
        return 0.0

    inter_area = (x2_max - x1_min) * (y2_max - y1_min)

    area1 = (bbox1["x2"] - bbox1["x1"]) * (bbox1["y2"] - bbox1["y1"])
    area2 = (bbox2["x2"] - bbox2["x1"]) * (bbox2["y2"] - bbox2["y1"])

    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
