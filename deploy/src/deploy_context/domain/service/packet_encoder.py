"""串口协议编码领域服务"""

from typing import Dict, Optional

from shared_kernel.config.loader import ConfigLoader

from ..model.value_object import SerialPacket


class PacketEncoder:
    """串口协议编码服务
    
    职责:
    - 将检测结果编码为串口数据包
    - 管理协议映射配置
    - 处理特殊协议规则
    """
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self._config_loader = config_loader or ConfigLoader()
        self._protocol_map: Dict[int, int] = {}
        self._empty_value: int = 0
    
    def load_protocol_mapping(self, protocol: str = "default") -> None:
        """加载协议映射配置
        
        Args:
            protocol: 协议类型（default/stm32/arduino）
        """
        mapping = self._config_loader.get_deploy_class_map(protocol)
        self._protocol_map = mapping
        self._empty_value = mapping.get("empty", 0)
    
    def encode(
        self,
        category_id: int,
        x_normalized: float,
        y_normalized: float
    ) -> SerialPacket:
        """编码检测结果为串口数据包
        
        Args:
            category_id: 分类编号
            x_normalized: 归一化X坐标
            y_normalized: 归一化Y坐标
            
        Returns:
            SerialPacket: 串口数据包
        """
        protocol_id = self._protocol_map.get(category_id, category_id)
        return SerialPacket.from_normalized(protocol_id, x_normalized, y_normalized)
    
    def encode_empty(self) -> SerialPacket:
        """编码空检测"""
        return SerialPacket.empty()
    
    def get_protocol_id(self, category_id: int) -> int:
        """获取协议编号"""
        return self._protocol_map.get(category_id, category_id)
    
    def is_valid_category(self, category_id: int) -> bool:
        """检查分类是否有效"""
        return category_id in self._protocol_map or category_id in [0, 1, 2, 3]
    
    def get_all_protocol_ids(self) -> Dict[int, int]:
        """获取所有协议映射"""
        return self._protocol_map.copy()
