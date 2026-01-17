# train/tests/test_training_advanced.py
"""高级训练测试"""

import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../shared/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../train/src'))


def test_model_selector():
    """测试模型选择服务"""
    from train_context.domain.service.model_selector import ModelSelector
    
    selector = ModelSelector()
    
    models = selector.get_available_models()
    assert len(models) > 0
    print(f"Available models: {models}")
    
    info = selector.get_model_info("yolo", "n")
    assert info is not None
    assert info.parameters == 3.2
    print(f"YOLOv8n info: {info.variant}, {info.parameters}M params")
    
    recommended = selector.recommend_model(device="cpu", priority="speed")
    assert recommended is not None
    print(f"Recommended model for CPU speed: {recommended.variant}")
    
    print("Test 8: ModelSelector - PASSED")


def test_training_callbacks():
    """测试训练回调"""
    from train_context.infrastructure.callbacks.training_callbacks import (
        CallbackManager,
        ProgressLoggerCallback,
        EarlyStoppingCallback,
        ModelCheckpointCallback
    )
    
    manager = CallbackManager()
    
    progress_cb = ProgressLoggerCallback(log_interval=5)
    early_stop_cb = EarlyStoppingCallback(patience=5)
    checkpoint_cb = ModelCheckpointCallback("/tmp/checkpoints")
    
    manager.add_callback(progress_cb)
    manager.add_callback(early_stop_cb)
    manager.add_callback(checkpoint_cb)
    
    assert len(manager._callbacks) == 3
    
    manager.on_train_begin({})
    manager.on_epoch_begin(1, {"loss": 0.5})
    manager.on_epoch_end(1, {"loss": 0.5, "mAP50": 0.6})
    manager.on_train_end({})
    
    print("Test 9: TrainingCallbacks - PASSED")


def test_training_manifest():
    """测试训练清单"""
    from train_context.infrastructure.persistence.training_manifest import TrainingManifest
    
    manifest = TrainingManifest("/tmp/test_manifests")
    
    manifest.add_run("test_run_001", {
        "model_family": "yolo",
        "status": "completed",
        "epochs": 100
    })
    
    run = manifest.get_run("test_run_001")
    assert run is not None
    assert run["model_family"] == "yolo"
    
    runs = manifest.list_runs()
    assert len(runs) >= 1
    
    stats = manifest.get_statistics()
    assert "total_runs" in stats
    
    print("Test 10: TrainingManifest - PASSED")


def test_label_converter_advanced():
    """测试标签转换器高级功能"""
    from train_context.infrastructure.adapters.label_converter import LabelConverter
    
    converter = LabelConverter()
    
    annotations = converter.parse_yolo_label("/nonexistent/path.txt")
    assert isinstance(annotations, list)
    assert len(annotations) == 0
    
    validation = converter.validate_labels("/nonexistent/labels", "/nonexistent/images")
    assert "is_valid" in validation
    assert validation["total_labels"] == 0
    assert len(validation["missing_labels"]) == 0
    
    print("Test 11: LabelConverter Advanced - PASSED")


def test_metric_policy_advanced():
    """测试指标策略高级功能"""
    from train_context.domain.service.metric_policy import MetricPolicy, MetricThreshold
    
    thresholds = MetricThreshold(
        min_map50=0.6,
        max_loss=0.5,
        min_precision=0.7,
        min_recall=0.7
    )
    
    policy = MetricPolicy(thresholds)
    
    metrics = {
        "mAP50": 0.65,
        "val_loss": 0.4,
        "precision": 0.75,
        "recall": 0.72
    }
    
    evaluation = policy.evaluate(metrics)
    assert evaluation.is_acceptable
    assert evaluation.overall_score > 0.8
    
    recommendation = policy.get_recommendation(evaluation)
    assert recommendation
    
    print("Test 12: MetricPolicy Advanced - PASSED")


def test_dataset_splitter_advanced():
    """测试数据集分割高级功能"""
    from train_context.domain.service.dataset_splitter import DatasetSplitter
    
    splitter = DatasetSplitter(random_seed=42)
    
    try:
        result = splitter.split("/nonexistent/dataset", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    except ValueError as e:
        assert "No label files found" in str(e)
        print("Test 13: DatasetSplitter Advanced - PASSED (expected error)")
        return
    
    assert result.train_ratio == 0.7
    assert result.val_ratio == 0.2
    assert result.test_ratio == 0.1
    assert len(result.train_files) + len(result.val_files) + len(result.test_files) > 0
    
    print("Test 13: DatasetSplitter Advanced - PASSED")


if __name__ == "__main__":
    print("Running advanced tests...")
    test_model_selector()
    test_training_callbacks()
    test_training_manifest()
    test_label_converter_advanced()
    test_metric_policy_advanced()
    test_dataset_splitter_advanced()
    print("\nAll advanced tests passed!")
