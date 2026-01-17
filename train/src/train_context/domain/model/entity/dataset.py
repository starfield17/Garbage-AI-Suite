# train/src/train_context/domain/model/entity/dataset.py
"""数据集实体"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Dataset:
    """数据集实体
    
    封装数据集路径和基本信息
    """
    name: str
    path: str
    image_count: int = 0
    annotation_count: int = 0
    categories: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
    
    @classmethod
    def from_path(cls, dataset_path: str) -> "Dataset":
        """从路径创建数据集"""
        path = Path(dataset_path)
        name = path.name
        
        image_count = 0
        annotation_count = 0
        categories = []
        
        if path.exists():
            images_dir = path / "images"
            labels_dir = path / "labels"
            
            if images_dir.exists():
                image_count = len(list(images_dir.glob("*.jpg"))) + \
                             len(list(images_dir.glob("*.jpeg"))) + \
                             len(list(images_dir.glob("*.png")))
            
            if labels_dir.exists():
                annotation_count = len(list(labels_dir.glob("*.txt")))
            
            classes_file = path / "classes.txt"
            if classes_file.exists():
                with open(classes_file, "r") as f:
                    categories = [line.strip() for line in f if line.strip()]
        
        return cls(
            name=name,
            path=dataset_path,
            image_count=image_count,
            annotation_count=annotation_count,
            categories=categories
        )
    
    def get_images_dir(self) -> str:
        return str(Path(self.path) / "images")
    
    def get_labels_dir(self) -> str:
        return str(Path(self.path) / "labels")
    
    def validate(self) -> bool:
        """验证数据集完整性"""
        path = Path(self.path)
        return (path / "images").exists() and (path / "labels").exists()
