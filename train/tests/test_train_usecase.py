"""Tests for train module."""

from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import pytest

from garbage_train.application.usecases import TrainAndExportModelUseCase


@pytest.fixture
def mock_config_loader():
    loader = Mock()
    return loader


@pytest.fixture
def mock_artifact_storage():
    storage = Mock()
    storage.save.return_value = "/fake/path/to/manifest.json"
    return storage


@pytest.fixture
def mock_dataset():
    dataset = Mock()
    dataset.prepare_dataset.return_value = {"images": 100, "labels": 100}
    return dataset


@pytest.fixture
def use_case(mock_config_loader, mock_artifact_storage, mock_dataset):
    use_case = TrainAndExportModelUseCase.__new__(TrainAndExportModelUseCase)
    use_case.config_loader = mock_config_loader
    use_case.artifact_storage = mock_artifact_storage
    use_case.dataset_preparation = mock_dataset
    return use_case


def test_train_success(use_case):
    """Test successful training execution path."""
    mock_config = {
        "train_profiles": {
            "yolo_v12n_e300": {
                "model_family": "yolo",
                "dataset_path": "/fake/path",
                "hyperparameters": {"epochs": 10},
                "export_targets": ["onnx"],
                "class_mapping": {"0": "Kitchen_waste"},
                "input_spec": {"width": 640, "height": 480, "channels": 3},
            }
        }
    }
    use_case.config_loader.load_yaml.return_value = mock_config

    # Mock the internal methods to avoid actual training
    mock_train_result = {
        "success": True,
        "metrics": {"map50": 0.85},
        "model_path": "/fake/path/to/model.pt",
    }
    mock_export_result = {"onnx": {"success": True, "output_path": "/fake/model.onnx"}}
    mock_manifest = {
        "model_name": "test_model",
        "model_family": "yolo",
        "train_id": "yolo_v12n_e300",
        "classes": [{"id": 0, "name": "Kitchen_waste"}],
        "input": {"width": 640, "height": 480, "channels": 3},
        "export_targets": ["onnx"],
        "files": [],
    }

    with patch.object(use_case, "_prepare_dataset", return_value={"images": 100}):
        with patch.object(use_case, "_train_model", return_value=mock_train_result):
            with patch.object(
                use_case, "_export_model", return_value=mock_export_result
            ):
                with patch.object(
                    use_case, "_generate_manifest", return_value=mock_manifest
                ):
                    result = use_case.execute(
                        train_id="yolo_v12n_e300",
                        output_dir="/fake/output",
                    )

    assert result["success"] is True
    assert "manifest_path" in result


def test_train_failure_invalid_profile(use_case):
    """Test failure when train_id doesn't exist in config."""
    use_case.config_loader.load_yaml.return_value = {"train_profiles": {}}

    result = use_case.execute(train_id="invalid", output_dir="/fake/output")

    assert result["success"] is False
    assert "invalid" in result.get("error", "").lower()


def test_train_failure_config_error(use_case):
    """Test failure when config loading fails."""
    use_case.config_loader.load_yaml.side_effect = Exception("Config file not found")

    result = use_case.execute(train_id="test", output_dir="/fake/output")

    assert result["success"] is False
