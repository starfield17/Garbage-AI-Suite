"""Artifact storage utilities."""

import hashlib
import json
from pathlib import Path
from typing import Dict

import torch
from garbage_train.application.ports import ArtifactStorePort
from garbage_shared.contracts_models import ModelManifestDTO, ModelFileDTO
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class ArtifactStorage(ArtifactStorePort):
    def __init__(self):
        pass

    def save(self, manifest: Dict, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_dto = ModelManifestDTO(**manifest)

        file_entries = []

        for file_entry in manifest.get("files", []):
            file_path = output_dir / file_entry["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if file_entry["type"] in ["model", "weights"]:
                torch.save(
                    self._get_model_state(manifest),
                    str(file_path),
                )

                if file_entry["type"] in ["config", "onnx", "rknn"]:
                    with open(file_path, "wb") as f:
                        f.write(file_entry.get("data", b""))

            file_hash = self._calculate_file_hash(file_path)

            file_entries.append(
                ModelFileDTO(
                    path=str(file_path.relative_to(output_dir)),
                    sha256=file_hash,
                    type=file_entry["type"],
                    size_bytes=file_path.stat().st_size,
                )
            )

        manifest_dto.files = file_entries

        manifest_path = output_dir / "model_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest_dto.model_dump_json(indent=2))

        log.info("Manifest saved", path=str(manifest_path))
        return manifest_path

    def _get_model_state(self, manifest: Dict) -> Dict:
        model_type = manifest.get("model_family", "yolo")

        if model_type == "yolo":
            return {"model": "state_dict"}
        elif model_type == "faster_rcnn":
            return {"model": "state_dict"}
        else:
            return {"model": "state_dict"}

    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096)):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()
