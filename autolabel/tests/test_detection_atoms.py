"""Tests for autolabel domain atoms."""

from garbage_autolabel.domain.atoms import (
    filter_by_confidence,
    clip_bbox_to_image,
    map_class_names,
    nms,
)


def test_filter_by_confidence():
    detections = [
        {"class_id": 0, "class_name": "bottle", "confidence": 0.9, "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
        {"class_id": 1, "class_name": "can", "confidence": 0.6, "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
        {"class_id": 2, "class_name": "battery", "confidence": 0.4, "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
    ]

    filtered = filter_by_confidence(detections, 0.7)
    assert len(filtered) == 2
    assert all(d["class_name"] in ["bottle", "can"] for d in filtered)


def test_clip_bbox_to_image():
    detections = [
        {
            "class_id": 0,
            "confidence": 0.9,
            "bbox": {"x1": -10, "y1": -10, "x2": 110, "y2": 110},
        }
    ]

    clipped = clip_bbox_to_image(detections, 100, 100)
    assert clipped[0]["bbox"]["x1"] == 0
    assert clipped[0]["bbox"]["y1"] == 0
    assert clipped[0]["bbox"]["x2"] == 99
    assert clipped[0]["bbox"]["y2"] == 99


def test_map_class_names():
    detections = [{"class_id": 0, "class_name": "potato", "confidence": 0.9, "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}}]
    class_map = {"potato": 0, "bottle": 1}

    mapped = map_class_names(detections, class_map)
    assert mapped[0]["class_name"] == "potato"
    assert mapped[0]["class_id"] == 0


def test_nms():
    detections = [
        {"class_id": 0, "confidence": 0.9, "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
        {"class_id": 0, "confidence": 0.8, "bbox": {"x1": 5, "y1": 5, "x2": 105, "y2": 105}},
    ]

    result = nms(detections, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9


def test_nms_no_overlap():
    detections = [
        {"class_id": 0, "confidence": 0.9, "bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 50}},
        {"class_id": 1, "confidence": 0.8, "bbox": {"x1": 60, "y1": 60, "x2": 110, "y2": 110}},
    ]

    result = nms(detections, iou_threshold=0.3)
    assert len(result) == 2
