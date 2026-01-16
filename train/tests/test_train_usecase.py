"""Tests for train module."""

from unittest.mock import Mock, patch

import pytest

from garbage_train.application.usecases import TrainAndExportModelUseCase
from garbage_shared.contracts_models import ModelManifestDTO


@pytest.fixture
def use_case():
    return TrainAndExportModelUseCase()


@pytest.fixture
def mock_trainer():
    trainer = Mock()
    trainer.train.return_value = {
        "success": True,
        "metrics": {"map50": 0.85},
        "model_path": "/fake/path/to/model.pt",
    }
    return trainer


@pytest.fixture
def mock_exporter():
    exporter = Mock()
    exporter.export_to_onnx.return_value = {"success": True, "output_path": "/fake/path/to/model.onnx"}
    exporter.export_to_torchscript.return_value = {"success": True, "output_path": "/fake/path/to/model.torchscript"}
    return exporter


@pytest.fixture
def mock_storage():
    storage = Mock()
    storage.save.return_value = "/fake/path/to/manifest.json"
    return storage


@pytest.fixture
def mock_dataset():
    dataset = Mock()
    dataset.prepare_dataset.return_value = {"images": 100, "labels": 100}
    return dataset


def test_train_success(use_case, mock_trainer, mock_exporter, mock_storage, mock_dataset, tmp_path):
    use_case.config_loader.load_yaml.return_value = {
        "train_profiles": {
            "yolo_v12n_e300": {
                "model_family": "yolo",
                "dataset_path": tmp_path,
                "hyperparameters": {"epochs": 10},
                "export_targets": ["onnx"],
                "class_mapping": {"0": "Kitchen_waste"},
            }
        }
    }

    (tmp_path / "configs").mkdir()
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry" / "train_profiles.yaml").write_text(
        "train_profiles:\n  yolo_v12n_e300:\n    model_family: yolo\n"
    )

    with patch.object(use_case, "_prepare_dataset", return_value=mock_dataset):
        with patch.object(use_case, "_train_model", return_value=mock_trainer):
            with patch.object(use_case, "_export_model", return_value=mock_exporter):
                with patch.object(use_case, "_generate_manifest"):
                    result = use_case.execute(
                        train_id="yolo_v12n_e300",
                        output_dir=str(tmp_path / "output"),
                    )

    assert result["success"] is True
    assert "manifest_path" in result


def test_train_failure(use_case, tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry" / "train_profiles.yaml").write_text("train_profiles:\n  invalid:\n")

    result = use_case.execute(train_id="invalid", output_dir=str(tmp_path / "output"))

    assert result["success"] is False
    assert "invalid" in result.get("error", "").lower()
