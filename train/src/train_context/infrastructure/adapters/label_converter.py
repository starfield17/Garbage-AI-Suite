# train/src/train_context/infrastructure/adapters/label_converter.py
"""标签格式转换适配器"""

import json
from pathlib import Path
from typing import List, Dict, Any


class LabelConverter:
    """标签格式转换器
    
    支持多种标注格式之间的转换:
    - YOLO format (class_id x_center y_center width height)
    - COCO format (JSON with bounding boxes)
    - VOC format (XML files)
    """
    
    def __init__(self, class_mapping: Dict[int, str] = None):
        """初始化标签转换器
        
        Args:
            class_mapping: 类别映射 {class_id: category_name}
        """
        self._class_mapping = class_mapping or {
            0: "Kitchen_waste",
            1: "Recyclable_waste",
            2: "Hazardous_waste",
            3: "Other_waste"
        }
    
    def yolo_to_coco(
        self,
        yolo_dir: str,
        output_path: str,
        images_info: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """将 YOLO 格式转换为 COCO 格式
        
        Args:
            yolo_dir: YOLO 标签目录
            output_path: 输出 JSON 路径
            images_info: 图片信息列表
        
        Returns:
            COCO 格式数据
        """
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        for class_id, class_name in self._class_mapping.items():
            coco_data["categories"].append({
                "id": class_id + 1,
                "name": class_name,
                "supercategory": "garbage"
            })
        
        annotation_id = 1
        yolo_dir = Path(yolo_dir)
        
        for label_file in sorted(yolo_dir.glob("*.txt")):
            image_name = label_file.stem + ".jpg"
            
            if images_info:
                image_info = next((img for img in images_info if img["file_name"] == image_name), None)
                if not image_info:
                    continue
            else:
                image_info = {
                    "id": annotation_id,
                    "file_name": image_name,
                    "width": 640,
                    "height": 480
                }
            
            coco_data["images"].append(image_info)
            
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        x_min = (x_center - width / 2) * image_info["width"]
                        y_min = (y_center - height / 2) * image_info["height"]
                        w = width * image_info["width"]
                        h = height * image_info["height"]
                        
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_info["id"],
                            "category_id": class_id + 1,
                            "bbox": [x_min, y_min, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        
                        annotation_id += 1
        
        with open(output_path, "w") as f:
            json.dump(coco_data, f, indent=2)
        
        return coco_data
    
    def coco_to_yolo(
        self,
        coco_path: str,
        output_dir: str,
        image_width: int = 640,
        image_height: int = 480
    ) -> None:
        """将 COCO 格式转换为 YOLO 格式
        
        Args:
            coco_path: COCO JSON 文件路径
            output_dir: 输出 YOLO 标签目录
            image_width: 图片宽度
            image_height: 图片高度
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(coco_path, "r") as f:
            coco_data = json.load(f)
        
        annotations_by_image = {}
        for ann in coco_data.get("annotations", []):
            image_id = ann["image_id"]
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        for image in coco_data.get("images", []):
            image_id = image["id"]
            image_name = Path(image["file_name"]).stem
            
            output_file = Path(output_dir) / f"{image_name}.txt"
            
            with open(output_file, "w") as f:
                if image_id in annotations_by_image:
                    for ann in annotations_by_image[image_id]:
                        bbox = ann["bbox"]
                        x_min, y_min, w, h = bbox
                        
                        x_center = (x_min + w / 2) / image_width
                        y_center = (y_min + h / 2) / image_height
                        norm_w = w / image_width
                        norm_h = h / image_height
                        
                        category_id = ann["category_id"] - 1
                        
                        f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    
    def parse_yolo_label(self, label_path: str) -> List[Dict[str, Any]]:
        """解析 YOLO 格式标签文件
        
        Args:
            label_path: YOLO 标签文件路径
        
        Returns:
            解析后的标注列表
        """
        annotations = []
        
        import os
        if not os.path.exists(label_path):
            return annotations
        
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    annotations.append({
                        "class_id": int(parts[0]),
                        "class_name": self._class_mapping.get(int(parts[0]), "unknown"),
                        "x_center": float(parts[1]),
                        "y_center": float(parts[2]),
                        "width": float(parts[3]),
                        "height": float(parts[4])
                    })
        
        return annotations
    
    def validate_labels(self, label_dir: str, image_dir: str) -> Dict[str, Any]:
        """验证标签文件
        
        Args:
            label_dir: 标签目录
            image_dir: 图片目录
        
        Returns:
            验证结果
        """
        label_path = Path(label_dir)
        image_path = Path(image_dir)
        
        validation_result = {
            "total_labels": 0,
            "total_images": 0,
            "missing_labels": [],
            "empty_labels": [],
            "class_distribution": {},
            "is_valid": True
        }
        
        image_files = set()
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.update([f.stem for f in image_path.glob(ext)])
        
        validation_result["total_images"] = len(image_files)
        
        label_files = list(label_path.glob("*.txt"))
        validation_result["total_labels"] = len(label_files)
        
        for label_file in label_files:
            stem = label_file.stem
            
            if stem not in image_files:
                validation_result["missing_labels"].append(stem)
                validation_result["is_valid"] = False
            
            with open(label_file, "r") as f:
                content = f.read().strip()
                if not content:
                    validation_result["empty_labels"].append(stem)
                
                for line in content.split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        validation_result["class_distribution"][class_id] = \
                            validation_result["class_distribution"].get(class_id, 0) + 1
        
        return validation_result
