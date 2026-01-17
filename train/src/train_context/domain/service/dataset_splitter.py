# train/src/train_context/domain/service/dataset_splitter.py
"""数据集分割服务"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any


@dataclass
class SplitResult:
    """分割结果"""
    train_files: List[str]
    val_files: List[str]
    test_files: List[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float


class DatasetSplitter:
    """数据集分割服务
    
    职责:
    - 将数据集按比例分割为训练集、验证集、测试集
    - 支持分层采样确保类别分布均衡
    """
    
    def __init__(self, random_seed: int = 42):
        self._random_seed = random_seed
        random.seed(random_seed)
    
    def split(
        self,
        dataset_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        stratified: bool = True
    ) -> SplitResult:
        """分割数据集
        
        Args:
            dataset_path: 数据集路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            stratified: 是否分层采样
        
        Returns:
            分割结果
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        labels_dir = Path(dataset_path) / "labels"
        label_files = sorted([f.name for f in labels_dir.glob("*.txt")])
        
        if not label_files:
            raise ValueError(f"No label files found in {labels_dir}")
        
        random.shuffle(label_files)
        
        total = len(label_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_files = label_files[:train_end]
        val_files = label_files[train_end:val_end]
        test_files = label_files[val_end:]
        
        return SplitResult(
            train_files=[str(f) for f in train_files],
            val_files=[str(f) for f in val_files],
            test_files=[str(f) for f in test_files],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
    
    def create_split_files(self, dataset_path: str, split_result: SplitResult) -> None:
        """创建分割文件列表"""
        base_path = Path(dataset_path)
        
        for split_name, files in [
            ("train", split_result.train_files),
            ("val", split_result.val_files),
            ("test", split_result.test_files)
        ]:
            split_file = base_path / f"{split_name}.txt"
            with open(split_file, "w") as f:
                for file in files:
                    image_name = Path(file).stem + Path(file).suffix.replace(".txt", ".jpg")
                    f.write(f"images/{image_name}\n")
