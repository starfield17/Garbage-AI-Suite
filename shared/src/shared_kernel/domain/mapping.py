"""映射值对象 - 训练/部署编号映射"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ClassMapping:
    """训练类别映射
    
    用于将模型输出的类别编号映射到 WasteCategory
    支持多模型族配置
    """
    model_family: str              # 模型族标识，如 "yolo", "faster_rcnn"
    id_to_category: Dict[int, str] # {class_id: category_name}
    
    def get_category(self, class_id: int) -> Optional[str]:
        """根据类别 ID 获取分类名称"""
        return self.id_to_category.get(class_id)


@dataclass(frozen=True)
class ProtocolMapping:
    """部署协议映射
    
    用于将 WasteCategory 映射到串口协议字节
    支持多协议配置
    """
    protocol_name: str             # 协议名称，如 "stm32", "arduino"
    category_to_byte: Dict[int, int]  # {category_index: byte_code}
    empty_byte: int = 0            # 无检测时发送的字节
    
    def encode(self, category_index: int) -> int:
        """编码分类为串口字节"""
        return self.category_to_byte.get(category_index, self.empty_byte)


@dataclass(frozen=True)
class MappingSet:
    """映射集合 - 包含所有映射配置"""
    class_mappings: Dict[str, ClassMapping]   # {model_family: mapping}
    protocol_mappings: Dict[str, ProtocolMapping]  # {protocol_name: mapping}
    
    def get_class_mapping(self, model_family: str) -> Optional[ClassMapping]:
        """获取指定模型族的类别映射"""
        return self.class_mappings.get(model_family)
    
    def get_protocol_mapping(self, protocol_name: str) -> Optional[ProtocolMapping]:
        """获取指定协议的映射"""
        return self.protocol_mappings.get(protocol_name)
    
    @classmethod
    def create_default(cls) -> "MappingSet":
        """创建默认映射集"""
        # 默认四分类映射
        default_class = ClassMapping(
            model_family="default",
            id_to_category={
                0: "Kitchen_waste",
                1: "Recyclable_waste",
                2: "Hazardous_waste",
                3: "Other_waste",
            }
        )
        
        # YOLO 映射
        yolo_class = ClassMapping(
            model_family="yolo",
            id_to_category=default_class.id_to_category.copy()
        )
        
        # Faster R-CNN 映射（COCO 从 1 开始）
        faster_rcnn_class = ClassMapping(
            model_family="faster_rcnn",
            id_to_category={
                1: "Kitchen_waste",
                2: "Recyclable_waste",
                3: "Hazardous_waste",
                4: "Other_waste",
            }
        )
        
        # 默认协议映射
        default_protocol = ProtocolMapping(
            protocol_name="default",
            category_to_byte={
                0: 1,  # Kitchen_waste -> 0x01
                1: 2,  # Recyclable_waste -> 0x02
                2: 3,  # Hazardous_waste -> 0x03
                3: 4,  # Other_waste -> 0x04
            },
            empty_byte=0
        )
        
        # STM32 协议映射
        stm32_protocol = ProtocolMapping(
            protocol_name="stm32",
            category_to_byte=default_protocol.category_to_byte.copy(),
            empty_byte=0
        )
        
        # Arduino 协议映射（不同字节值）
        arduino_protocol = ProtocolMapping(
            protocol_name="arduino",
            category_to_byte={
                0: 10,
                1: 20,
                2: 30,
                3: 40,
            },
            empty_byte=0
        )
        
        return cls(
            class_mappings={
                "default": default_class,
                "yolo": yolo_class,
                "faster_rcnn": faster_rcnn_class,
            },
            protocol_mappings={
                "default": default_protocol,
                "stm32": stm32_protocol,
                "arduino": arduino_protocol,
            }
        )
