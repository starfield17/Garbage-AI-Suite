"""置信度值对象"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Confidence:
    """置信度值对象"""
    value: float
    
    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.value}")
    
    def is_above_threshold(self, threshold: float) -> bool:
        return self.value >= threshold
    
    def __repr__(self) -> str:
        return f"Confidence({self.value:.4f})"
