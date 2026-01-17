# 系统架构文档

本文档详细描述了 Garbage AI 系统的架构设计，包括 DDD 分层、模块划分、交互模式等。

## 目录

- [架构概览](#架构概览)
- [领域驱动设计](#领域驱动设计)
- [上下文划分](#上下文划分)
- [模块详解](#模块详解)
- [基础设施](#基础设施)
- [配置管理](#配置管理)
- [部署架构](#部署架构)

## 架构概览

### 整体架构

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           Garbage AI System                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         API / CLI Layer                               │  │
│  │            (autolabel, train, deploy 命令行入口)                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│  ┌──────────────────────────────────▼──────────────────────────────────┐  │
│  │                        Application Layer                              │  │
│  │        (用例编排、服务编排、DTO 转换、事件处理)                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│  ┌──────────────────────────────────▼──────────────────────────────────┐  │
│  │                           Domain Layer                                │  │
│  │         (聚合根、实体、值对象、领域服务、领域事件)                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│  ┌──────────────────────────────────▼──────────────────────────────────┐  │
│  │                      Infrastructure Layer                             │  │
│  │         (持久化、外部服务、消息队列、文件系统、串口通信)                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         Shared Kernel                                  │  │
│  │         (公共领域模型、类型定义、配置加载工具)                             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 技术栈

- **语言**: Python 3.10+
- **框架**: 纯 Python，无重型框架依赖
- **ML 框架**: Ultralytics YOLO, PyTorch, TorchVision
- **配置**: YAML + Pydantic
- **测试**: pytest
- **部署**: Docker

## 领域驱动设计

### DDD 核心概念

#### 聚合根 (Aggregate Root)

聚合根是领域模型的核心，每个聚合根维护其内部实体的一致性。

```python
class AutoLabelJob(AggregateRoot):
    """自动标注任务聚合根"""
    
    def __init__(self, job_id, engine_type, image_items, confidence_threshold):
        self._job_id = job_id
        self._engine_type = engine_type
        self._image_items = image_items  # 子实体
        self._status = JobStatus.PENDING
        self._results: List[LabelResult] = []
        self._statistics = JobStatistics(total_images=len(image_items))
```

#### 实体 (Entity)

实体具有唯一标识，包含业务行为。

```python
class ImageItem(Entity):
    """图片项实体"""
    
    def __init__(self, path: str):
        self._path = path
        self._is_processed = False
        self._processing_error = None
    
    def mark_processed(self):
        self._is_processed = True
    
    def mark_failed(self, error: str):
        self._is_processed = True
        self._processing_error = error
```

#### 值对象 (Value Object)

值对象是不可变的，用于描述领域概念的属性。

```python
@dataclass(frozen=True)
class BoundingBox:
    """边界框值对象"""
    x_center: float
    y_center: float
    width: float
    height: float
    
    def to_xyxy(self, img_width: int, img_height: int):
        """转换为 XYXY 格式"""
        x1 = int((self.x_center - self.width / 2) * img_width)
        y1 = int((self.y_center - self.height / 2) * img_height)
        x2 = int((self.x_center + self.width / 2) * img_width)
        y2 = int((self.y_center + self.height / 2) * img_height)
        return x1, y1, x2, y2
```

#### 领域服务 (Domain Service)

领域服务用于处理跨实体的业务逻辑。

```python
class StabilityJudge:
    """稳定性判断服务"""
    
    def __init__(self, policy: StabilityPolicy):
        self._policy = policy
        self._history = {}
    
    def evaluate(self, frame: DetectionFrame) -> StabilityReport:
        """评估检测帧的稳定性"""
        # 业务逻辑实现
```

### 分层规则

#### 依赖规则

```
API Layer → Application Layer → Domain Layer ← Infrastructure Layer
                     ↑                    ↑
                     └──── 依赖倒置 ──────┘
```

- **上层依赖下层**: 应用层依赖领域层
- **基础设施层依赖下层**: 基础设施层实现领域层定义的接口
- **领域层不依赖任何其他层**: 这是核心原则

#### 导入规则

```
Domain Layer:
  ✓ from shared_kernel.domain...
  ✓ from .value_object...
  ✓ from .entity...
  ✗ from application...
  ✗ from infrastructure...
  ✗ from autolabel_context... (其他上下文)
```

## 上下文划分

### Context 1: AutoLabel Context

**职责**: 自动标注图片

#### 领域模型

```
AutoLabelJob (聚合根)
├── ImageItem (实体)
├── LabelResult (实体)
└── JobStatistics (值对象)
```

#### 基础设施

- `YoloEngine`: YOLO 检测引擎
- `FasterRcnnEngine`: Faster R-CNN 检测引擎
- `VlmEngine`: VLM 标注引擎
- `FileLabelStore`: 文件标签存储

#### 应用服务

- `RunAutolabelHandler`: 处理自动标注命令
- `LabelAssembler`: DTO 与领域对象转换

### Context 2: Train Context

**职责**: 训练分类模型

#### 领域模型

```
TrainingRun (聚合根)
├── Dataset (实体)
├── ModelSpec (实体)
└── TrainingMetrics (值对象)
```

#### 基础设施

- `YoloTrainer`: YOLO 训练器
- `FasterRcnnTrainer`: Faster R-CNN 训练器
- `LabelConverter`: 标签格式转换
- `LocalArtifactStore`: 模型文件存储

#### 应用服务

- `StartTrainingHandler`: 处理训练命令
- `TrainingAssembler`: DTO 与领域对象转换

### Context 3: Deploy Context

**职责**: 部署推理和分拣控制

#### 领域模型

```
SortingSession (聚合根)
├── DetectionFrame (实体)
├── Counter (实体)
└── SessionStatistics (值对象)
```

#### 基础设施

- `YoloRuntime`: YOLO 推理运行时
- `RknnRuntime`: RK3588 NPU 推理运行时
- `CameraOpencv`: OpenCV 相机接口
- `SerialPyserial`: 串口通信

#### 应用服务

- `SessionHandler`: 处理会话命令
- `PacketAssembler`: DTO 与领域对象转换

### Shared Kernel

**职责**: 跨上下文的共享代码

#### 领域模型

```
WasteCategory (枚举)
BoundingBox (值对象)
Confidence (值对象)
LabelFile (值对象)
ClassMapping (值对象)
ProtocolMapping (值对象)
```

#### 基础设施

- `ConfigLoader`: 配置加载器
- `LoggingSetup`: 日志配置

## 模块详解

### 领域层结构

```
domain/
├── model/
│   ├── aggregate/         # 聚合根
│   ├── entity/           # 实体
│   └── value_object/     # 值对象
├── service/              # 领域服务
├── repository/           # 仓储接口
└── event/                # 领域事件
```

### 应用层结构

```
application/
├── command/              # 命令
├── dto/                  # 数据传输对象
├── handler/              # 命令处理器
└── assembler/            # DTO 转换器
```

### 基础设施层结构

```
infrastructure/
├── engine/               # 检测引擎
├── trainer/              # 训练器
├── runtime/              # 推理运行时
├── persistence/          # 持久化
├── device/               # 设备通信
└── callbacks/            # 训练回调
```

## 基础设施

### 持久化

使用仓储模式隔离领域层和持久化层：

```python
# 领域层定义接口
class ILabelStore(IRepository):
    def save(self, label_file: LabelFile): ...
    def load(self, file_id: str) -> LabelFile: ...

# 基础设施层实现
class FileLabelStore(ILabelStore):
    def save(self, label_file: LabelFile):
        with open(f"labels/{label_file.file_id}.json", "w") as f:
            json.dump(label_file.to_dict(), f)
```

### 配置管理

使用 ConfigLoader 统一管理配置：

```python
from shared_kernel.config.loader import ConfigLoader

config = ConfigLoader()
config.load("config/logging.yaml")
config.load("config/models/yolo.yaml")

model_config = config.get("yolo")
```

### 日志配置

使用 YAML 配置日志：

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: standard

loggers:
  shared_kernel:
    level: INFO
    handlers: [console]
```

## 配置管理

### 配置文件结构

```
config/
├── logging.yaml          # 日志配置
├── taxonomy/
│   └── waste_categories.yaml  # 垃圾分类定义
├── models/
│   ├── yolo.yaml         # YOLO 模型配置
│   ├── faster_rcnn.yaml  # Faster R-CNN 模型配置
│   └── vlm_qwen.yaml     # VLM 模型配置
├── mappings/
│   ├── train_class_map.yaml    # 训练类别映射
│   └── deploy_class_map.yaml   # 部署类别映射
├── profiles/
│   └── device_rk3588.yaml      # RK3588 设备配置
└── prompts/
    └── vlm_prompts.yaml        # VLM 提示词
```

### 配置加载顺序

1. 默认配置 (`config/`)
2. 环境变量覆盖
3. 命令行参数

## 部署架构

### 单机部署

```
┌─────────────────────────────────────────────────────┐
│                     Host Machine                     │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │              Docker Container                  │  │
│  │                                               │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐       │  │
│  │  │Autolabel│  │  Train   │  │  Deploy │       │  │
│  │  └─────────┘  └─────────┘  └─────────┘       │  │
│  │                                               │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │              Shared Kernel               │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  │                                               │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │              Volume Mounts                     │  │
│  │  - models/                                    │  │
│  │  - datasets/                                  │  │
│  │  - config/                                    │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 分布式部署

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cloud Server                              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Autolabel   │  │   Train     │  │      Deploy             │  │
│  │  Service    │──▶│  Service    │──▶│  Service               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│         │               │                    │                   │
│         └───────────────┴────────────────────┘                   │
│                           │                                      │
│                    ┌──────▼──────┐                               │
│                    │   MinIO /   │                               │
│                    │  S3 Bucket  │                               │
│                    └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### 硬件平台支持

#### RK3588 NPU

```yaml
device:
  type: npu
  model: rk3588
  npu_id: 0
  
inference:
  backend: rknn
  num_threads: 4
  
serial:
  port: /dev/ttyUSB0
  baudrate: 115200
```

#### Jetson (GPU)

```yaml
device:
  type: gpu
  model: jetson_orin
  cuda_id: 0
  
inference:
  backend: tensorrt
  precision: fp16
```

#### x86 CPU

```yaml
device:
  type: cpu
  
inference:
  backend: onnxruntime
  num_threads: 8
```

## 演进路线

### 短期目标

- [ ] 完成所有模块的单元测试
- [ ] 添加集成测试
- [ ] 优化推理性能
- [ ] 支持更多硬件平台

### 中期目标

- [ ] 添加 Web 管理界面
- [ ] 支持模型版本管理
- [ ] 添加监控和告警
- [ ] 支持分布式训练

### 长期目标

- [ ] 支持边缘设备增量学习
- [ ] 添加联邦学习支持
- [ ] 支持多语言界面
- [ ] 云边协同部署
