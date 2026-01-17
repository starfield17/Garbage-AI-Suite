# Garbage AI - 智能垃圾分类检测系统

基于深度学习的垃圾分类检测与自动分拣系统，支持多种检测模型（YOLO、Faster R-CNN、VLM）和部署平台（RK3588、 Jetson、x86 CPU/GPU）。

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [使用指南](#使用指南)
- [配置说明](#配置说明)
- [测试](#测试)
- [贡献](#贡献)
- [许可证](#许可证)

## 功能特性

### 核心功能

- **自动标注（AutoLabel）**: 使用 AI 模型自动标注垃圾图片
  - 支持多种检测引擎：YOLO、Faster R-CNN、VLM
  - 批量处理图片
  - 质量门禁控制
  - 支持 API 密钥配置

- **模型训练（Train）**: 训练自定义垃圾分类模型
  - 支持 YOLO、Faster R-CNN 架构
  - 超参数调优
  - 指标监控与早停
  - 模型导出（ONNX、PyTorch）

- **部署推理（Deploy）**: 实时垃圾分类推理
  - 支持多种硬件平台
  - 串口通信协议
  - 稳定性检测
  - 实时统计

### 技术特性

- **DDD 架构**: 采用领域驱动设计，代码结构清晰
- **配置外部化**: 所有配置通过 YAML 文件管理
- **类型安全**: 使用 Python 类型提示
- **易于测试**: 完善的单元测试和集成测试

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Garbage AI System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   AutoLabel  │  │    Train     │  │    Deploy    │          │
│  │   Context    │  │   Context    │  │   Context    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └────────────────┼─────────────────┘                   │
│                          │                                     │
│                    ┌─────▼─────┐                               │
│                    │   Shared  │                               │
│                    │  Kernel   │                               │
│                    │ (DDD Base)│                               │
│                    └───────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Context 说明

- **AutoLabel Context**: 负责图片自动标注
- **Train Context**: 负责模型训练
- **Deploy Context**: 负责模型部署和推理
- **Shared Kernel**: 共享的领域模型和基础设施

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+ (可选，用于 GPU 加速)
- FFmpeg (用于视频帧提取)

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装共享内核
pip install -e shared/

# 安装训练模块
pip install -e train/

# 安装部署模块
pip install -e deploy/

# 安装自动标注模块
pip install -e autolabel/
```

### 使用 Docker

```bash
# 构建所有镜像
docker build -f docker/train.Dockerfile -t garbage-ai:train .
docker build -f docker/deploy.Dockerfile -t garbage-ai:deploy .
docker build -f docker/autolabel.Dockerfile -t garbage-ai:autolabel .

# 运行自动标注
docker run --rm \
    -v $(pwd)/images:/data/images \
    -v $(pwd)/labels:/data/labels \
    -v $(pwd)/models:/data/models \
    -v $(pwd)/config:/data/config \
    -e OPENAI_API_KEY=${OPENAI_API_KEY} \
    garbage-ai:autolabel autolabel run --engine vlm --images /data/images/
```

## 项目结构

```
garbage-ai/
├── shared/                 # 共享内核
│   ├── src/
│   │   └── shared_kernel/
│   │       ├── domain/     # 领域模型
│   │       │   ├── base.py
│   │       │   ├── taxonomy.py
│   │       │   ├── annotation.py
│   │       │   └── mapping.py
│   │       └── config/     # 配置加载
│   └── tests/              # 测试
├── autolabel/              # 自动标注模块
│   ├── src/
│   │   └── autolabel_context/
│   │       ├── domain/     # 领域层
│   │       ├── application/# 应用层
│   │       └── infrastructure/# 基础设施层
│   └── tests/              # 测试
├── train/                  # 训练模块
│   ├── src/
│   │   └── train_context/
│   └── tests/              # 测试
├── deploy/                 # 部署模块
│   ├── src/
│   │   └── deploy_context/
│   └── tests/              # 测试
├── config/                 # 配置文件
│   ├── models/             # 模型配置
│   ├── taxonomy/           # 分类定义
│   ├── mappings/           # 类别映射
│   └── profiles/           # 设备配置
├── scripts/                # 脚本工具
├── docker/                 # Docker 文件
└── tests/                  # 集成测试
```

## 使用指南

### 自动标注

```bash
# 使用 VLM 进行标注（需要 API Key）
autolabel run --engine vlm --images ./images --output ./labels --api-key YOUR_KEY

# 使用 YOLO 进行标注
autolabel run --engine yolo --images ./images --output ./labels --model ./models/yolo.pt

# 使用 Faster R-CNN 进行标注
autolabel run --engine faster_rcnn --images ./images --output ./labels
```

### 模型训练

```bash
# 训练 YOLOv8n 模型
train run --model yolov8_n --data ./dataset --epochs 100 --batch 16

# 训练 Faster R-CNN 模型
train run --model faster_rcnn --data ./dataset --epochs 50 --batch 8

# 导出模型
train export --model ./models/yolo.pt --format onnx
```

### 部署推理

```bash
# 启动部署服务
deploy run --profile rk3588 --model ./models/yolo.pt

# 启动分拣会话
deploy session start --device /dev/ttyUSB0 --baudrate 115200
```

## 配置说明

### 垃圾分类配置

垃圾分类定义在 `config/taxonomy/waste_categories.yaml`:

```yaml
categories:
  kitchen_waste:
    id: 0
    name: "Kitchen_waste"
    display_name_zh: "厨余垃圾"
    display_name_en: "Kitchen Waste"
    color: "#56B4E9"
  
  recyclable_waste:
    id: 1
    name: "Recyclable_waste"
    display_name_zh: "可回收垃圾"
    display_name_en: "Recyclable Waste"
    color: "#E69F00"
  
  hazardous_waste:
    id: 2
    name: "Hazardous_waste"
    display_name_zh: "有害垃圾"
    display_name_en: "Hazardous Waste"
    color: "#F02720"
  
  other_waste:
    id: 3
    name: "Other_waste"
    display_name_zh: "其他垃圾"
    display_name_en: "Other Waste"
    color: "#009E73"
```

### 类别映射配置

类别映射配置在 `config/mappings/` 目录:

- `train_class_map.yaml`: 训练时的类别映射
- `deploy_class_map.yaml`: 部署时的类别映射

### 模型配置

模型配置在 `config/models/` 目录:

- `yolo.yaml`: YOLO 模型配置
- `faster_rcnn.yaml`: Faster R-CNN 模型配置
- `vlm_qwen.yaml`: VLM 模型配置

## 测试

### 运行所有测试

```bash
# 运行单元测试
pytest tests/ -v

# 运行集成测试
pytest autolabel/tests/ train/tests/ deploy/tests/ -v

# 运行性能测试
pytest tests/test_performance.py -v -s
```

### 测试覆盖率

```bash
pytest --cov=shared --cov=autolabel --cov=train --cov=deploy --cov-report=html
```

## 贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请查看 [LICENSE](LICENSE) 文件。

## 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目主页: [https://github.com/yourusername/garbage-ai](https://github.com/yourusername/garbage-ai)
