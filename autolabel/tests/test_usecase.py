"""Tests for autolabel use case with mocked ports."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from garbage_autolabel.application.usecases import AutoLabelDatasetUseCase
from garbage_autolabel.contracts_models import BBoxLabelDTO, BBoxEntryDTO


@pytest.fixture
def use_case():
    return AutoLabelDatasetUseCase()


@pytest.fixture
def mock_adapter():
    adapter = Mock()
    adapter.predict.return_value = [
        {
            "class_id": 0,
            "class_name": "bottle",
            "confidence": 0.9,
            "bbox": {"x1": 10, "y1": 10, "x2": 110, "y2": 110},
        }
    ]
    adapter.get_class_map.return_value = {"bottle": 0, "can": 1}
    adapter.get_image_size.return_value = (640, 480)
    return adapter


@pytest.fixture
def mock_writer():
    writer = Mock()
    return writer


@pytest.fixture
def mock_scanner(tmp_path):
    from garbage_autolabel.infrastructure.storage import DatasetScanner

    scanner = DatasetScanner()
    scanner.list_images.return_value = [tmp_path / "img1.jpg"]
    return scanner


def test_auto_label_success(use_case, mock_adapter, mock_writer, mock_scanner, tmp_path):
    config = {
        "model_path": "test.pt",
        "device": "cpu",
        "model_type": "mobilenet_v3",
    }

    (tmp_path / "img1.jpg").touch()
    (tmp_path / "config.yaml").write_text("model_type: yolo\nmodel_path: test.pt\n")

    with patch.object(use_case, "_get_adapter", return_value=mock_adapter):
        with patch.object(use_case, "_get_writer", return_value=mock_writer):
            with patch.object(use_case, "scanner", return_value=mock_scanner):
                result = use_case.execute(
                    config_path=tmp_path / "config.yaml",
                    model_type="yolo",
                    input_dir=tmp_path,
                    output_dir=tmp_path / "output",
                )

    assert result["success"] is True
    assert mock_writer.write.call_count == 1
    assert result["stats"]["processed"] == 1


def test_auto_label_failure(use_case, tmp_path):
    (tmp_path / "config.yaml").write_text("model_type: yolo\nmodel_path: non_existent.pt\n")

    result = use_case.execute(
        config_path=tmp_path / "config.yaml",
        model_type="yolo",
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
    )

    assert result["success"] is False
    assert "model not found" in result["error"].lower()
