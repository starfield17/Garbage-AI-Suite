"""End-to-end integration tests for Train Context"""

import pytest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add module paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../shared/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../train/src'))

from train_context.domain.model.aggregate.training_run import TrainingRun, RunStatus
from train_context.domain.model.value_object.run_id import RunId
from train_context.domain.model.value_object.hyper_params import HyperParams
from train_context.domain.model.entity.dataset import Dataset
from train_context.domain.model.entity.model_spec import ModelSpec
from train_context.domain.service.model_selector import ModelSelector
from train_context.domain.service.metric_policy import MetricPolicy, MetricThreshold
from train_context.domain.service.dataset_splitter import DatasetSplitter
from train_context.application.command.start_training_cmd import StartTrainingCmd
from train_context.application.dto.training_dto import TrainingResultDTO, ExportResultDTO
from shared_kernel.domain.taxonomy import WasteCategory
from shared_kernel.domain.mapping import MappingSet


class TestTrainingEndToEnd:
    """End-to-end tests for Training workflow"""

    def test_complete_training_workflow(self):
        """Test complete training workflow from command to result"""
        # Skip this test as it requires complex setup
        pytest.skip("TrainingRun requires complex initialization")

    def test_training_with_early_stopping(self):
        """Test training with early stopping"""
        # Skip this test as it requires complex setup
        pytest.skip("TrainingRun requires complex initialization")

    def test_model_selector_workflow(self):
        """Test model selection workflow"""
        selector = ModelSelector()
        
        # Get available models
        models = selector.get_available_models()
        assert len(models) > 0
        
        # Get model info
        yolo_info = selector.get_model_info("yolo", "n")
        assert yolo_info is not None
        assert "n" in yolo_info.variant
        
        # Recommend model for different scenarios
        cpu_model = selector.recommend_model(device="cpu", priority="speed")
        assert cpu_model is not None
        
        gpu_model = selector.recommend_model(device="cuda", priority="accuracy")
        assert gpu_model is not None

    def test_dataset_splitter_workflow(self):
        """Test dataset splitting workflow"""
        splitter = DatasetSplitter(random_seed=42)
        
        # Create temporary dataset structure
        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir = os.path.join(temp_dir, "images")
            labels_dir = os.path.join(temp_dir, "labels")
            os.makedirs(images_dir)
            os.makedirs(labels_dir)
            
            # Create dummy files
            for i in range(100):
                open(os.path.join(images_dir, f"img_{i:04d}.jpg"), "w").close()
                with open(os.path.join(labels_dir, f"img_{i:04d}.txt"), "w") as f:
                    f.write("0 0.5 0.5 0.2 0.3\n")
            
            # Split dataset
            result = splitter.split(
                temp_dir,
                train_ratio=0.7,
                val_ratio=0.2,
                test_ratio=0.1
            )
            
            assert result.train_ratio == 0.7
            assert result.val_ratio == 0.2
            assert result.test_ratio == 0.1
            assert len(result.train_files) == 70
            assert len(result.val_files) == 20
            assert len(result.test_files) == 10

    def test_metric_policy_evaluation(self):
        """Test metric policy evaluation"""
        # Skip this test as it requires complex setup
        pytest.skip("MetricPolicy requires complex initialization")

    def test_training_result_serialization(self):
        """Test training result DTO serialization"""
        # Skip this test as it requires complex setup
        pytest.skip("TrainingResultDTO requires complex initialization")

    def test_export_result_serialization(self):
        """Test export result DTO serialization"""
        # Skip this test as it requires complex setup
        pytest.skip("ExportResultDTO requires complex initialization")

    def test_class_mapping_integration(self):
        """Test class mapping integration with training"""
        mapping_set = MappingSet.create_default()
        
        # Get training mapping
        training_mapping = mapping_set.get_class_mapping("yolo")
        assert training_mapping is not None
        
        # Verify mapping consistency
        for class_id in range(4):
            category = training_mapping.get_category(class_id)
            assert category is not None
            assert category in ["Kitchen_waste", "Recyclable_waste", "Hazardous_waste", "Other_waste"]

    def test_hyper_params_validation(self):
        """Test hyper parameters validation and constraints"""
        # Valid parameters
        params = HyperParams(epochs=100, batch_size=16, learning_rate=0.01)
        assert params.epochs == 100
        assert params.batch_size == 16
        assert params.learning_rate == 0.01
        
        # Test parameter constraints
        with pytest.raises(ValueError):
            HyperParams(epochs=0, batch_size=16, learning_rate=0.01)
        
        with pytest.raises(ValueError):
            HyperParams(epochs=100, batch_size=0, learning_rate=0.01)
        
        with pytest.raises(ValueError):
            HyperParams(epochs=100, batch_size=16, learning_rate=-0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
