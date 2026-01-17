"""质量门禁领域服务"""

from dataclasses import dataclass
from typing import List

from shared_kernel.domain.annotation import Detection, LabelFile


@dataclass
class QualityReport:
    """质量报告"""
    is_passed: bool
    total_detections: int
    low_confidence_count: int
    out_of_bounds_count: int
    overlapping_count: int
    issues: List[str]


class QualityGate:
    """质量门禁
    
    职责:
    - 检查标注结果质量
    - 识别潜在问题（低置信度、越界、重叠）
    - 生成质量报告
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        min_box_size: float = 0.01
    ):
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._min_box_size = min_box_size
    
    def evaluate(self, label_file: LabelFile) -> QualityReport:
        """评估标注文件质量"""
        issues = []
        low_confidence_count = 0
        out_of_bounds_count = 0
        overlapping_count = 0
        
        detections = label_file.detections
        
        for det in detections:
            if not det.confidence.is_above_threshold(self._confidence_threshold):
                low_confidence_count += 1
        
        for det in detections:
            if self._is_out_of_bounds(det, label_file.image_width, label_file.image_height):
                out_of_bounds_count += 1
        
        overlapping_count = self._count_overlaps(detections)
        
        if overlapping_count > 0:
            issues.append(f"Found {overlapping_count} overlapping detections")
        
        is_passed = (
            low_confidence_count == 0 and
            out_of_bounds_count == 0 and
            overlapping_count == 0
        )
        
        return QualityReport(
            is_passed=is_passed,
            total_detections=len(detections),
            low_confidence_count=low_confidence_count,
            out_of_bounds_count=out_of_bounds_count,
            overlapping_count=overlapping_count,
            issues=issues
        )
    
    def _is_out_of_bounds(self, detection: Detection, img_width: int, img_height: int) -> bool:
        """检查检测是否越界"""
        bbox = detection.bounding_box
        return (
            bbox.x_center < 0 or bbox.y_center < 0 or
            bbox.x_center > 1 or bbox.y_center > 1 or
            bbox.width < self._min_box_size or bbox.height < self._min_box_size
        )
    
    def _count_overlaps(self, detections: List[Detection]) -> int:
        """计算重叠检测数量"""
        count = 0
        n = len(detections)
        for i in range(n):
            for j in range(i + 1, n):
                if self._calculate_iou(detections[i], detections[j]) > self._iou_threshold:
                    count += 1
        return count
    
    def _calculate_iou(self, det1: Detection, det2: Detection) -> float:
        """计算两个检测之间的 IoU"""
        box1 = det1.bounding_box
        box2 = det2.bounding_box
        
        x1 = max(box1.x_center - box1.width / 2, box2.x_center - box2.width / 2)
        y1 = max(box1.y_center - box1.height / 2, box2.y_center - box2.height / 2)
        x2 = min(box1.x_center + box1.width / 2, box2.x_center + box2.width / 2)
        y2 = min(box1.y_center + box1.height / 2, box2.y_center + box2.height / 2)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
