"""Use case for training models."""

from pathlib import Path
from datetime import datetime

from garbage_train.application.ports import (
    TrainerPort,
    ExporterPort,
    ArtifactStorePort,
    DatasetPort,
)
from garbage_train.infrastructure.trainers import YOLOTrainer, FasterRCNNTrainer
from garbage_train.infrastructure.exporters import (
    ONNXExporter,
    TorchScriptExporter,
    RKNNExporter,
)
from garbage_train.infrastructure.artifact_storage import ArtifactStorage
from garbage_train.infrastructure.dataset_preparation import GarbageDataset
from garbage_shared.config_loader import ConfigLoader
from garbage_shared.contracts_models import ModelManifestDTO, ModelFileDTO
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class TrainAndExportModelUseCase:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.artifact_storage = ArtifactStorage()
        self.dataset_preparation = GarbageDataset()

    def execute(
        self,
        train_id: str,
        output_dir: str,
    ) -> dict:
        try:
            log.info("Starting training", train_id=train_id)

            train_profile = self.config_loader.load_yaml(
                f"configs/registry/train_profiles.yaml"
            )
            profile = train_profile["train_profiles"][train_id]

            self._prepare_dataset(profile)
            result = self._train_model(profile)

            if not result["success"]:
                return result

            export_results = self._export_model(profile, result.get("model_path", ""))
            manifest = self._generate_manifest(profile, result, export_results)

            manifest_path = self.artifact_storage.save(manifest, Path(output_dir))

            final_result = {
                "success": True,
                "manifest_path": str(manifest_path),
                "export_results": export_results,
            }

            log.info("Training and export completed", result=final_result)
            return final_result

        except Exception as e:
            log.error("Training failed", error=str(e))
            return {"success": False, "error": str(e)}

    def _prepare_dataset(self, profile: dict):
        from garbage_train.infrastructure.dataset_preparation import GarbageDataset

        dataset_path = profile.get("dataset_path", "./datasets/garbage")
        self.dataset_preparation = GarbageDataset(dataset_path)
        stats = {
            "images": len(self.dataset_preparation),
            "labels": len(self.dataset_preparation),
        }

        log.info("Dataset prepared", stats=stats)
        return stats

    def _train_model(self, profile: dict) -> dict:
        model_family = profile.get("model_family", "yolo")
        hyperparameters = profile.get("hyperparameters", {})

        if model_family == "yolo":
            trainer = YOLOTrainer(hyperparameters)
            result = trainer.train(hyperparameters)
        elif model_family == "faster_rcnn":
            trainer = FasterRCNNTrainer(hyperparameters)
            result = trainer.train(hyperparameters)
        else:
            return {
                "success": False,
                "error": f"Unsupported model family: {model_family}",
            }

        log.info("Model trained", result=result)
        return result

    def _export_model(self, profile: dict, model_path: str) -> dict:
        export_targets = profile.get("export_targets", ["onnx"])

        export_results = {}
        for target in export_targets:
            exporter_config = profile.get("export_config", {})

            if target == "onnx":
                exporter = ONNXExporter(exporter_config)
                export_results["onnx"] = exporter.export_to_onnx(
                    Path(model_path), exporter_config
                )
            elif target == "torchscript":
                exporter = TorchScriptExporter(exporter_config)
                export_results["torchscript"] = exporter.export_to_torchscript(
                    Path(model_path), exporter_config
                )
            elif target == "rknn":
                exporter = RKNNExporter(exporter_config)
                export_results["rknn"] = exporter.export_to_rknn(
                    Path(model_path), exporter_config
                )

        return export_results

    def _generate_manifest(
        self, profile: dict, train_result: dict, export_results: dict
    ) -> dict:
        now = datetime.now().isoformat()

        files = []

        model_path = train_result.get("model_path", "")

        files.append(
            ModelFileDTO(
                path="weights/best.pt",
                type="model",
                size_bytes=Path(model_path).stat().st_size,
            )
        )

        for export_type, export_result in export_results.items():
            if export_result.get("success"):
                output_path = export_result.get("output_path", "")
                files.append(
                    ModelFileDTO(
                        path=Path(output_path).name,
                        type=export_type,
                        size_bytes=Path(output_path).stat().st_size,
                    )
                )

        return {
            "model_name": f"{profile.get('model_family', 'unknown')}_garbage_model",
            "model_family": profile.get("model_family", "unknown"),
            "train_id": profile.get("train_id", "unknown"),
            "created_at": now,
            "classes": self._get_classes(profile),
            "input": profile.get(
                "input_spec", {"width": 640, "height": 480, "channels": 3}
            ),
            "export_targets": profile.get("export_targets", []),
            "files": files,
        }

    def _get_classes(self, profile: dict) -> list:
        class_mapping = profile.get("class_mapping", {})

        return [{"id": idx, "name": name} for idx, name in class_mapping.items()]
