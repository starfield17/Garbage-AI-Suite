# train/src/train_context/infrastructure/persistence/local_artifact_store.py
"""本地产物存储实现"""

import os
import shutil
from pathlib import Path
from typing import Optional

from shared_kernel.config.loader import ConfigLoader

from ...domain.repository.i_artifact_store import IArtifactStore


class LocalArtifactStore(IArtifactStore):
    """本地产物存储实现
    
    将训练产物存储在本地文件系统中
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """初始化本地产物存储
        
        Args:
            base_dir: 存储根目录
        """
        config_loader = ConfigLoader()
        self._base_dir = Path(base_dir) if base_dir else Path("artifacts")
        self._base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_artifact(self, run_id: str, artifact_path: str) -> str:
        """保存训练产物"""
        run_dir = self._base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        artifact_name = Path(artifact_path).name
        dest_path = run_dir / artifact_name
        
        if os.path.isdir(artifact_path):
            shutil.copytree(artifact_path, dest_path)
        else:
            shutil.copy2(artifact_path, dest_path)
        
        return str(dest_path)
    
    def get_artifact(self, run_id: str) -> Optional[str]:
        """获取训练产物路径"""
        run_dir = self._base_dir / run_id
        
        if not run_dir.exists():
            return None
        
        weights_dir = run_dir / "weights"
        if weights_dir.exists():
            best_pt = weights_dir / "best.pt"
            if best_pt.exists():
                return str(best_pt)
        
        for f in sorted(run_dir.glob("*"), key=lambda x: x.stat().st_mtime):
            if f.suffix == ".pt":
                return str(f)
        
        return None
    
    def artifact_exists(self, run_id: str) -> bool:
        """检查产物是否存在"""
        return self.get_artifact(run_id) is not None
    
    def delete_artifact(self, run_id: str) -> bool:
        """删除产物"""
        run_dir = self._base_dir / run_id
        
        if run_dir.exists():
            shutil.rmtree(run_dir)
            return True
        
        return False
    
    def list_artifacts(self) -> list:
        """列出所有产物"""
        return [d.name for d in self._base_dir.iterdir() if d.is_dir()]
