# train/src/train_context/infrastructure/callbacks/training_callbacks.py
"""训练回调系统"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from datetime import datetime


class TrainingCallback(ABC):
    """训练回调基类"""
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        """epoch 开始时调用"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """epoch 结束时调用"""
        pass
    
    @abstractmethod
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        """训练开始时调用"""
        pass
    
    @abstractmethod
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        """训练结束时调用"""
        pass


class CallbackManager:
    """回调管理器"""
    
    def __init__(self):
        self._callbacks: List[TrainingCallback] = []
    
    def add_callback(self, callback: TrainingCallback) -> None:
        """添加回调"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: TrainingCallback) -> None:
        """移除回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        for callback in self._callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        for callback in self._callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        for callback in self._callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        for callback in self._callbacks:
            callback.on_train_end(logs)


class ProgressLoggerCallback(TrainingCallback):
    """进度日志回调"""
    
    def __init__(self, log_interval: int = 10):
        self._log_interval = log_interval
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if epoch % self._log_interval == 0:
            print(f"Epoch {epoch}: loss={logs.get('loss', 0):.4f}")
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        print(f"Training started at {datetime.now().isoformat()}")
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        print(f"Training completed at {datetime.now().isoformat()}")


class EarlyStoppingCallback(TrainingCallback):
    """早停回调"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        metric: str = "loss"
    ):
        self._patience = patience
        self._min_delta = min_delta
        self._metric = metric
        self._best_value = None
        self._counter = 0
        self._should_stop = False
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        current_value = logs.get(self._metric, 0)
        
        if self._best_value is None:
            self._best_value = current_value
        elif abs(current_value - self._best_value) < self._min_delta:
            self._counter += 1
            if self._counter >= self._patience:
                self._should_stop = True
        else:
            self._best_value = current_value
            self._counter = 0
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        self._best_value = None
        self._counter = 0
        self._should_stop = False
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        if self._should_stop:
            print(f"Early stopping triggered at epoch {logs.get('epoch', 0)}")
    
    @property
    def should_stop(self) -> bool:
        return self._should_stop


class ModelCheckpointCallback(TrainingCallback):
    """模型检查点回调"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_best: bool = True,
        metric: str = "mAP50",
        mode: str = "max"
    ):
        self._checkpoint_dir = checkpoint_dir
        self._save_best = save_best
        self._metric = metric
        self._mode = mode
        self._best_value = None
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        current_value = logs.get(self._metric, 0)
        
        if self._save_best:
            if self._best_value is None:
                self._best_value = current_value
            elif (self._mode == "max" and current_value > self._best_value) or \
                 (self._mode == "min" and current_value < self._best_value):
                self._best_value = current_value
                print(f"Saving best model with {self._metric}={current_value:.4f}")
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        self._best_value = None
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        pass
