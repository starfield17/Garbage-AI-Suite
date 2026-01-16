"""Trainer adapters for YOLO and Fast R-CNN."""

from typing import Dict

import torch
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from garbage_train.application.ports import TrainerPort
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class YOLOTrainer(TrainerPort):
    def __init__(self, config: Dict):
        self.model_path = config.get("model_path", "")
        self.dataset_path = config.get("dataset_path", "./datasets/garbage")
        self.output_dir = Path(config.get("output_dir", "./outputs/train"))
        self.hyperparameters = config.get("hyperparameters", {})
        self.device = config.get("device", "auto")
        self._model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path or not Path(self.model_path).exists():
            raise ValueError(f"YOLO model not found: {self.model_path}")

        log.info("Loading YOLO base model", path=self.model_path)
        self._model = YOLO(self.model_path)

    def train(self, profile: Dict, progress_callback=None) -> Dict:
        epochs = self.hyperparameters.get("epochs", 100)
        batch_size = self.hyperparameters.get("batch_size", 10)
        imgsz = self.hyperparameters.get("image_size", 640)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = self._model.train(
                data=self.dataset_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                device=self.device,
                project=str(self.output_dir),
                exist_ok=True,
            )

            log.info("Training completed", metrics=results.results_dict)

            return {
                "success": True,
                "metrics": results.results_dict,
                "model_path": str(self.output_dir / "weights" / "best.pt"),
            }

        except Exception as e:
            log.error("Training failed", error=str(e))
            return {"success": False, "error": str(e)}

    def finetune(self, base_model: str, profile: Dict, progress_callback=None) -> Dict:
        base_path = Path(base_model)
        if not base_path.exists():
            raise ValueError(f"Base model not found: {base_model}")

        log.info("Finetuning from base model", base=base_model)

        self._model = YOLO(str(base_path))

        return self.train(profile, progress_callback)


class FasterRCNNTrainer(TrainerPort):
    def __init__(self, config: Dict):
        self.model_type = config.get("model_type", "resnet50_fpn")
        self.dataset_path = config.get("dataset_path", "./datasets/garbage")
        self.output_dir = Path(config.get("output_dir", "./outputs/train"))
        self.hyperparameters = config.get("hyperparameters", {})
        self.device = config.get("device", "auto")
        self._model = None
        self._load_model()

    def _load_model(self):
        if self.model_type == "resnet50_fpn":
            log.info("Initializing Fast R-CNN", model_type=self.model_type)
            self._model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=4)
            self._model.to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, profile: Dict, progress_callback=None) -> Dict:
        num_epochs = self.hyperparameters.get("num_epochs", 100)
        batch_size = self.hyperparameters.get("batch_size", 8)
        learning_rate = self.hyperparameters.get("learning_rate", 0.005)
        momentum = self.hyperparameters.get("momentum", 0.9)
        weight_decay = self.hyperparameters.get("weight_decay", 0.0005)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            params = [
                {"params": self._model.backbone.parameters(), "lr": learning_rate},
                {
                    "params": self._model.rpn_head.parameters(),
                    "lr": learning_rate * 10,
                },
            ]

            optimizer = torch.optim.SGD(
                params,
                momentum=momentum,
                weight_decay=weight_decay,
            )

            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
            )

            best_val_loss = float("inf")
            patience_counter = 0
            min_delta = self.hyperparameters.get("min_delta", 0.001)

            for epoch in range(num_epochs):
                if progress_callback:
                    progress_callback(
                        f"Epoch {epoch + 1}/{num_epochs}",
                        f"Training...",
                    )

                train_loss = self._train_one_epoch(epoch, optimizer, self.device)
                val_loss = self._validate_epoch(self.device)

                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, "best")
                else:
                    patience_counter += 1

                if patience_counter >= 10:
                    log.info(f"Early stopping at epoch {epoch}")
                    break

                lr_scheduler.step(val_loss)

            log.info("Training completed", best_loss=best_val_loss)

            return {
                "success": True,
                "best_val_loss": best_val_loss,
                "checkpoint_path": str(self.output_dir / "model_final.pth"),
            }

        except Exception as e:
            log.error("Training failed", error=str(e))
            return {"success": False, "error": str(e)}

    def _train_one_epoch(self, epoch, optimizer, device):
        self._model.train()
        total_loss = 0.0

        for batch_idx, (images, targets) in enumerate(self._dataloader()):
            optimizer.zero_grad()
            loss_dict = self._model(images, targets)
            loss = sum(loss_dict.values())

            loss.backward()
            optimizer.step()

            total_loss += loss

            if (batch_idx + 1) % 100 == 0:
                log.debug(f"Epoch {epoch} - Batch {batch_idx} - Loss: {loss:.4f}")

        return total_loss / (batch_idx + 1)

    def _dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self._dataset,
            batch_size=self.hyperparameters.get("batch_size", 8),
            shuffle=True,
            num_workers=self.hyperparameters.get("num_workers", 4),
        )

    @property
    def _dataset(self):
        from garbage_train.infrastructure.dataset_preparation import GarbageDataset

        return GarbageDataset(self.dataset_path)

    def _validate_epoch(self, device):
        self._model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in self._dataloader():
                loss_dict = self._model(images, targets)
                val_loss += sum(loss_dict.values())

        return val_loss / len(self._dataloader())

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint_name = (
            "model_best.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        )
        checkpoint_path = self.output_dir / checkpoint_name

        torch.save(
            {
                "model": self._model.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": self._best_val_loss if is_best else None,
            },
            checkpoint_path,
        )

    def finetune(self, base_model: str, profile: Dict, progress_callback=None) -> Dict:
        log.info("Loading pretrained model for finetuning", base=base_model)

        self._load_model()

        return self.train(profile, progress_callback)
