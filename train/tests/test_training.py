import pytest


def test_training_run_creation():
    """测试训练运行创建"""
    from train_context.domain.model.aggregate.training_run import TrainingRun
    
    run = TrainingRun.create(
        model_family="yolo",
        model_variant="yolov8_n",
        dataset_path="/path/to/dataset",
        epochs=100,
        batch_size=16,
        learning_rate=0.01
    )
    
    assert run is not None
    assert run.model_spec.family == "yolo"
    assert run.model_spec.variant == "yolov8_n"
    assert run.hyper_params.epochs == 100


def test_run_id_generation():
    """测试运行 ID 生成"""
    from train_context.domain.model.value_object.run_id import RunId
    
    run_id = RunId.generate("yolo")
    
    assert run_id is not None
    assert run_id.model_family == "yolo"


def test_hyper_params_validation():
    """测试超参数验证"""
    from train_context.domain.model.value_object.hyper_params import HyperParams
    
    params = HyperParams(epochs=100, batch_size=16, learning_rate=0.01)
    
    assert params.epochs == 100
    assert params.batch_size == 16
    assert params.learning_rate == 0.01
