# train/tests/test_infrastructure.py
"""基础设施测试"""

import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../shared/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../train/src'))


def test_faster_rcnn_trainer():
    """测试 Faster R-CNN 训练器"""
    from train_context.infrastructure.trainer.faster_rcnn_trainer import FasterRcnnTrainer
    
    trainer = FasterRcnnTrainer(device="cpu")
    
    assert trainer is not None
    assert "pt" in trainer.supported_formats
    assert "onnx" in trainer.supported_formats
    
    print("Test 14: FasterRcnnTrainer - PASSED")


def test_local_artifact_store():
    """测试本地产物存储"""
    from train_context.infrastructure.persistence.local_artifact_store import LocalArtifactStore
    
    store = LocalArtifactStore("/tmp/test_artifacts")
    
    assert store is not None
    
    test_content = "test model content"
    with open("/tmp/test_model.pt", "w") as f:
        f.write(test_content)
    
    saved_path = store.save_artifact("test_run", "/tmp/test_model.pt")
    assert saved_path is not None
    
    retrieved = store.get_artifact("test_run")
    assert retrieved is not None
    
    exists = store.artifact_exists("test_run")
    assert exists
    
    print("Test 15: LocalArtifactStore - PASSED")


def test_manifest_repository():
    """测试清单仓储"""
    from train_context.infrastructure.persistence.manifest_repo import ManifestRepository
    
    repo = ManifestRepository("/tmp/test_manifests")
    
    datasets = repo.list_datasets()
    assert isinstance(datasets, list)
    
    validation = repo.validate_dataset("/nonexistent/path")
    assert validation is False
    
    print("Test 16: ManifestRepository - PASSED")


def test_training_result_dto():
    """测试训练结果 DTO"""
    from train_context.application.dto.training_dto import TrainingResultDTO
    
    dto = TrainingResultDTO(
        success=True,
        run_id="test_001",
        status="completed",
        model_family="yolo",
        model_variant="n",
        epochs=100,
        final_metrics={"mAP50": 0.8},
        best_model_path="/path/to/model.pt"
    )
    
    assert dto.success
    assert dto.run_id == "test_001"
    assert dto.final_metrics["mAP50"] == 0.8
    
    print("Test 17: TrainingResultDTO - PASSED")


def test_export_result_dto():
    """测试导出结果 DTO"""
    from train_context.application.dto.training_dto import ExportResultDTO
    
    dto = ExportResultDTO(
        success=True,
        run_id="test_001",
        original_path="/path/to/original.pt",
        exported_path="/path/to/exported.onnx",
        format="onnx"
    )
    
    assert dto.success
    assert dto.format == "onnx"
    assert dto.exported_path.endswith(".onnx")
    
    print("Test 18: ExportResultDTO - PASSED")


def test_convert_result_dto():
    """测试转换结果 DTO"""
    from train_context.application.dto.training_dto import ConvertResultDTO
    
    dto = ConvertResultDTO(
        success=True,
        input_path="/path/to/input",
        output_path="/path/to/output",
        source_format="yolo",
        target_format="coco",
        details={"images_converted": 100}
    )
    
    assert dto.success
    assert dto.source_format == "yolo"
    assert dto.target_format == "coco"
    assert dto.details["images_converted"] == 100
    
    print("Test 19: ConvertResultDTO - PASSED")


def test_training_assembler():
    """测试训练汇编器"""
    from train_context.application.assembler.training_assembler import TrainingAssembler
    from train_context.application.command.start_training_cmd import StartTrainingCmd
    
    cmd = StartTrainingCmd(
        model_family="yolo",
        model_variant="n",
        dataset_path="/path/to/dataset",
        epochs=100
    )
    
    run = TrainingAssembler.cmd_to_run(cmd)
    assert run is not None
    assert run.model_spec.family == "yolo"
    
    summary = TrainingAssembler.result_to_summary({
        "success": True,
        "final_epoch": 100,
        "best_model": "/path/to/model.pt"
    })
    assert summary["success"]
    assert summary["epochs"] == 100
    
    print("Test 20: TrainingAssembler - PASSED")


if __name__ == "__main__":
    print("Running infrastructure tests...")
    test_faster_rcnn_trainer()
    test_local_artifact_store()
    test_manifest_repository()
    test_training_result_dto()
    test_export_result_dto()
    test_convert_result_dto()
    test_training_assembler()
    print("\nAll infrastructure tests passed!")
