# train/src/train_context/domain/service/metric_policy.py
"""指标策略服务"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class MetricThreshold:
    """指标阈值"""
    min_map50: float = 0.5
    max_loss: float = 1.0
    min_precision: float = 0.6
    min_recall: float = 0.6


@dataclass
class MetricEvaluation:
    """指标评估结果"""
    is_acceptable: bool
    map50_score: float
    loss_score: float
    precision_score: float
    recall_score: float
    overall_score: float
    feedback: Dict[str, str]


class MetricPolicy:
    """指标策略服务
    
    职责:
    - 评估训练指标是否满足要求
    - 提供训练建议和反馈
    """
    
    def __init__(self, thresholds: Optional[MetricThreshold] = None):
        self._thresholds = thresholds or MetricThreshold()
    
    def evaluate(self, metrics: Dict[str, Any]) -> MetricEvaluation:
        """评估指标
        
        Args:
            metrics: 训练指标字典
        
        Returns:
            评估结果
        """
        map50 = metrics.get("mAP50", 0.0)
        loss = metrics.get("val_loss", float("inf"))
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        
        map50_score = min(1.0, map50 / self._thresholds.min_map50) if self._thresholds.min_map50 > 0 else 1.0
        loss_score = min(1.0, self._thresholds.max_loss / loss) if loss > 0 else 0.0
        precision_score = min(1.0, precision / self._thresholds.min_precision) if self._thresholds.min_precision > 0 else 1.0
        recall_score = min(1.0, recall / self._thresholds.min_recall) if self._thresholds.min_recall > 0 else 1.0
        
        overall_score = (map50_score + loss_score + precision_score + recall_score) / 4.0
        
        feedback = {}
        if map50 < self._thresholds.min_map50:
            feedback["mAP50"] = f"mAP50={map50:.4f} < threshold={self._thresholds.min_map50}"
        if loss > self._thresholds.max_loss:
            feedback["loss"] = f"val_loss={loss:.4f} > threshold={self._thresholds.max_loss}"
        if precision < self._thresholds.min_precision:
            feedback["precision"] = f"precision={precision:.4f} < threshold={self._thresholds.min_precision}"
        if recall < self._thresholds.min_recall:
            feedback["recall"] = f"recall={recall:.4f} < threshold={self._thresholds.min_recall}"
        
        return MetricEvaluation(
            is_acceptable=len(feedback) == 0,
            map50_score=map50_score,
            loss_score=loss_score,
            precision_score=precision_score,
            recall_score=recall_score,
            overall_score=overall_score,
            feedback=feedback
        )
    
    def get_recommendation(self, evaluation: MetricEvaluation) -> str:
        """获取训练建议"""
        if evaluation.is_acceptable:
            return "模型性能良好，可以考虑导出部署"
        
        suggestions = []
        for metric, message in evaluation.feedback.items():
            if metric == "mAP50":
                suggestions.append("增加训练轮次或调整学习率")
            elif metric == "loss":
                suggestions.append("检查数据标注质量或增加数据增强")
            elif metric == "precision":
                suggestions.append("减少误检：提高置信度阈值或增加负样本")
            elif metric == "recall":
                suggestions.append("减少漏检：降低置信度阈值或增加训练样本")
        
        return "建议: " + " | ".join(suggestions) if suggestions else "需要进一步分析"
