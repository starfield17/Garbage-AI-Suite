# train/src/train_context/infrastructure/persistence/manifest_repo.py
"""训练清单仓储实现"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ...domain.repository.i_dataset_repository import IDatasetRepository


class ManifestRepository(IDatasetRepository):
    """训练清单仓储
    
    管理数据集清单和训练记录
    """
    
    def __init__(self, manifest_dir: str = "manifests"):
        """初始化清单仓储
        
        Args:
            manifest_dir: 清单目录
        """
        self._manifest_dir = Path(manifest_dir)
        self._manifest_dir.mkdir(parents=True, exist_ok=True)
        
        self._dataset_manifest = self._manifest_dir / "datasets.json"
        self._training_manifest = self._manifest_dir / "training.json"
    
    def list_datasets(self) -> List[str]:
        """列出可用数据集"""
        if not self._dataset_manifest.exists():
            return []
        
        with open(self._dataset_manifest, "r") as f:
            data = json.load(f)
        
        return list(data.get("datasets", {}).keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        if not self._dataset_manifest.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {self._dataset_manifest}")
        
        with open(self._dataset_manifest, "r") as f:
            data = json.load(f)
        
        datasets = data.get("datasets", {})
        if dataset_name not in datasets:
            raise KeyError(f"Dataset not found: {dataset_name}")
        
        return datasets[dataset_name]
    
    def validate_dataset(self, dataset_path: str) -> bool:
        """验证数据集完整性"""
        path = Path(dataset_path)
        
        required_dirs = ["images", "labels"]
        for dir_name in required_dirs:
            if not (path / dir_name).exists():
                return False
        
        return True
    
    def get_dataset_path(self, dataset_name: str) -> Optional[str]:
        """获取数据集路径"""
        try:
            info = self.get_dataset_info(dataset_name)
            return info.get("path")
        except (FileNotFoundError, KeyError):
            return None
    
    def register_dataset(self, name: str, path: str, description: str = "") -> None:
        """注册数据集"""
        data = {}
        if self._dataset_manifest.exists():
            with open(self._dataset_manifest, "r") as f:
                data = json.load(f)
        
        if "datasets" not in data:
            data["datasets"] = {}
        
        data["datasets"][name] = {
            "name": name,
            "path": path,
            "description": description,
            "registered_at": str(Path(path).stat().st_mtime) if Path(path).exists() else ""
        }
        
        with open(self._dataset_manifest, "w") as f:
            json.dump(data, f, indent=2)
    
    def save_training_record(self, run_id: str, record: Dict[str, Any]) -> None:
        """保存训练记录"""
        data = {}
        if self._training_manifest.exists():
            with open(self._training_manifest, "r") as f:
                data = json.load(f)
        
        if "trainings" not in data:
            data["trainings"] = {}
        
        data["trainings"][run_id] = record
        
        with open(self._training_manifest, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_training_record(self, run_id: str) -> Optional[Dict[str, Any]]:
        """获取训练记录"""
        if not self._training_manifest.exists():
            return None
        
        with open(self._training_manifest, "r") as f:
            data = json.load(f)
        
        return data.get("trainings", {}).get(run_id)
