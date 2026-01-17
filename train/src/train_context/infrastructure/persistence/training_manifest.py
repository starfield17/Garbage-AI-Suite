# train/src/train_context/infrastructure/persistence/training_manifest.py
"""训练清单管理器"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class TrainingManifest:
    """训练清单管理器
    
    负责管理训练运行的历史记录和元数据
    """
    
    def __init__(self, manifest_dir: str = "manifests"):
        """初始化清单管理器
        
        Args:
            manifest_dir: 清单目录路径
        """
        self._manifest_dir = Path(manifest_dir)
        self._manifest_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_file = self._manifest_dir / "training_manifest.json"
        self._initialize_manifest()
    
    def _initialize_manifest(self) -> None:
        """初始化清单文件"""
        if not self._manifest_file.exists():
            self._save_manifest({"version": "1.0", "runs": []})
    
    def _load_manifest(self) -> Dict[str, Any]:
        """加载清单"""
        with open(self._manifest_file, "r") as f:
            return json.load(f)
    
    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        """保存清单"""
        with open(self._manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
    
    def add_run(self, run_id: str, run_info: Dict[str, Any]) -> None:
        """添加训练运行记录
        
        Args:
            run_id: 运行 ID
            run_info: 运行信息
        """
        manifest = self._load_manifest()
        
        run_info["created_at"] = datetime.utcnow().isoformat()
        
        manifest["runs"].append({
            "run_id": run_id,
            **run_info
        })
        
        self._save_manifest(manifest)
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """获取运行记录
        
        Args:
            run_id: 运行 ID
        
        Returns:
            运行信息字典
        """
        manifest = self._load_manifest()
        
        for run in manifest["runs"]:
            if run["run_id"] == run_id:
                return run
        
        return None
    
    def list_runs(
        self,
        model_family: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """列出运行记录
        
        Args:
            model_family: 按模型族过滤
            status: 按状态过滤
            limit: 返回数量限制
        
        Returns:
            运行记录列表
        """
        manifest = self._load_manifest()
        
        runs = manifest["runs"]
        
        if model_family:
            runs = [r for r in runs if r.get("model_family") == model_family]
        
        if status:
            runs = [r for r in runs if r.get("status") == status]
        
        return sorted(runs, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]
    
    def update_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """更新运行记录
        
        Args:
            run_id: 运行 ID
            updates: 更新内容
        
        Returns:
            是否更新成功
        """
        manifest = self._load_manifest()
        
        for i, run in enumerate(manifest["runs"]):
            if run["run_id"] == run_id:
                manifest["runs"][i].update(updates)
                manifest["runs"][i]["updated_at"] = datetime.utcnow().isoformat()
                self._save_manifest(manifest)
                return True
        
        return False
    
    def delete_run(self, run_id: str) -> bool:
        """删除运行记录
        
        Args:
            run_id: 运行 ID
        
        Returns:
            是否删除成功
        """
        manifest = self._load_manifest()
        
        original_count = len(manifest["runs"])
        manifest["runs"] = [r for r in manifest["runs"] if r["run_id"] != run_id]
        
        if len(manifest["runs"]) < original_count:
            self._save_manifest(manifest)
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        manifest = self._load_manifest()
        
        runs = manifest["runs"]
        
        total_runs = len(runs)
        completed_runs = len([r for r in runs if r.get("status") == "completed"])
        failed_runs = len([r for r in runs if r.get("status") == "failed"])
        
        model_families = {}
        for run in runs:
            family = run.get("model_family", "unknown")
            model_families[family] = model_families.get(family, 0) + 1
        
        return {
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "model_distribution": model_families,
            "success_rate": completed_runs / total_runs if total_runs > 0 else 0
        }
