"""仓储接口模块导出"""

from .i_runtime_model import IInferenceRuntime
from .i_device_io import ICamera, ISerialDevice

__all__ = ["IInferenceRuntime", "ICamera", "ISerialDevice"]
