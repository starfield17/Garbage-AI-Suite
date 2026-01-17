# 配置参考文档

本文档提供了 Garbage AI 系统的完整配置参考，包括所有配置文件、参数说明和使用示例。

## 目录

- [配置文件概览](#配置文件概览)
- [日志配置](#日志配置)
- [垃圾分类配置](#垃圾分类配置)
- [模型配置](#模型配置)
- [类别映射配置](#类别映射配置)
- [设备配置](#设备配置)
- [VLM 提示词配置](#vlm-提示词配置)
- [环境变量](#环境变量)

## 配置文件概览

### 配置文件结构

```
config/
├── logging.yaml              # 日志配置
├── taxonomy/
│   └── waste_categories.yaml # 垃圾分类定义
├── models/
│   ├── yolo.yaml             # YOLO 模型配置
│   ├── faster_rcnn.yaml      # Faster R-CNN 模型配置
│   └── vlm_qwen.yaml         # VLM 模型配置
├── mappings/
│   ├── train_class_map.yaml  # 训练类别映射
│   └── deploy_class_map.yaml # 部署类别映射
├── profiles/
│   └── device_rk3588.yaml    # RK3588 设备配置
└── prompts/
    └── vlm_prompts.yaml      # VLM 提示词配置
```

### 配置加载顺序

配置按以下顺序加载，后加载的配置会覆盖先前的配置：

1. **默认配置** (`config/` 目录下的文件)
2. **环境变量覆盖**
3. **命令行参数**

---

## 日志配置

### logging.yaml

```yaml
version: 1

disable_existing_loggers: false

formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"
  
  detailed:
    format: "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d] - %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"

handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: detailed
    filename: "logs/garbage_ai.log"
    mode: "a"
    encoding: "utf-8"

loggers:
  shared_kernel:
    level: INFO
    handlers: [console, file]
    propagate: no

  autolabel_context:
    level: DEBUG
    handlers: [console, file]
    propagate: no

  train_context:
    level: DEBUG
    handlers: [console, file]
    propagate: no

  deploy_context:
    level: DEBUG
    handlers: [console, file]
    propagate: no

root:
  level: INFO
  handlers: [console, file]
```

### 配置项说明

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `version` | int | 1 | 配置版本号 |
| `disable_existing_loggers` | bool | false | 是否禁用已存在的日志器 |
| `formatters.*.format` | str | - | 日志格式字符串 |
| `formatters.*.datefmt` | str | - | 日期格式 |
| `handlers.*.level` | str | DEBUG | 日志级别 |
| `handlers.*.filename` | str | - | 日志文件路径 |
| `loggers.*.level` | str | - | 日志器级别 |
| `loggers.*.propagate` | bool | - | 是否向上传播 |
| `root.level` | str | INFO | 根日志器级别 |

### 日志级别

- `DEBUG`: 调试信息
- `INFO`: 一般信息
- `WARNING`: 警告
- `ERROR`: 错误
- `CRITICAL`: 严重错误

---

## 垃圾分类配置

### waste_categories.yaml

```yaml
version: "1.0.0"

categories:
  kitchen_waste:
    id: 0
    name: "Kitchen_waste"
    display_name_zh: "厨余垃圾"
    display_name_en: "Kitchen Waste"
    color: "#56B4E9"
    examples:
      - "剩饭剩菜"
      - "果皮果核"
      - "蛋壳"
      - "茶叶渣"
  
  recyclable_waste:
    id: 1
    name: "Recyclable_waste"
    display_name_zh: "可回收垃圾"
    display_name_en: "Recyclable Waste"
    color: "#E69F00"
    examples:
      - "塑料瓶"
      - "纸张"
      - "金属罐"
      - "玻璃瓶"
  
  hazardous_waste:
    id: 2
    name: "Hazardous_waste"
    display_name_zh: "有害垃圾"
    display_name_en: "Hazardous Waste"
    color: "#F02720"
    examples:
      - "电池"
      - "灯管"
      - "药品"
      - "油漆"
  
  other_waste:
    id: 3
    name: "Other_waste"
    display_name_zh: "其他垃圾"
    display_name_en: "Other Waste"
    color: "#009E73"
    examples:
      - "卫生纸"
      - "烟蒂"
      - "陶瓷碎片"
      - "一次性餐具"
```

### 配置项说明

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `version` | str | 配置版本 |
| `categories.*.id` | int | 类别 ID (0-3) |
| `categories.*.name` | str | 类别名称 (英文) |
| `categories.*.display_name_zh` | str | 类别名称 (中文) |
| `categories.*.display_name_en` | str | 类别名称 (英文) |
| `categories.*.color` | str | 颜色代码 (用于可视化) |
| `categories.*.examples` | list[str] | 类别示例 |

---

## 模型配置

### yolo.yaml

```yaml
model_path: "models/yolo/best.pt"
confidence_threshold: 0.5
iou_threshold: 0.45
device: "auto"  # auto/cpu/cuda:0
imgsz: 640

# 类别映射（可选，覆盖全局映射）
class_mapping:
  0: "Kitchen_waste"
  1: "Recyclable_waste"
  2: "Hazardous_waste"
  3: "Other_waste"
```

### faster_rcnn.yaml

```yaml
model_path: "models/faster_rcnn/faster_rcnn_resnet50.pth"
confidence_threshold: 0.5
iou_threshold: 0.5
device: "cuda:0"
num_classes: 4

class_mapping:
  1: "Kitchen_waste"
  2: "Recyclable_waste"
  3: "Hazardous_waste"
  4: "Other_waste"
```

### vlm_qwen.yaml

```yaml
model_path: "Qwen/Qwen-VL-Chat"
confidence_threshold: 0.5
max_new_tokens: 256
temperature: 0.7
top_p: 0.9

# API 配置（如果使用云端 API）
api:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4-vision-preview"
  api_key_env: "OPENAI_API_KEY"

# 本地推理配置
local:
  device: "cuda:0"
  load_in_8bit: false
```

### 模型配置项说明

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `model_path` | str | - | 模型文件路径 |
| `confidence_threshold` | float | 0.5 | 置信度阈值 |
| `iou_threshold` | float | 0.45 | NMS IoU 阈值 |
| `device` | str | "auto" | 计算设备 |
| `imgsz` | int | 640 | 输入图片尺寸 |
| `num_classes` | int | 4 | 类别数量 |
| `max_new_tokens` | int | 256 | 最大生成 token 数 |
| `temperature` | float | 0.7 | 采样温度 |
| `top_p` | float | 0.9 | Top-p 采样参数 |

---

## 类别映射配置

### train_class_map.yaml

```yaml
# 训练时的类别映射配置
# 将模型输出的类别 ID 映射到标准类别名称

mappings:
  yolo:
    0: "Kitchen_waste"
    1: "Recyclable_waste"
    2: "Hazardous_waste"
    3: "Other_waste"
  
  faster_rcnn:
    1: "Kitchen_waste"
    2: "Recyclable_waste"
    3: "Hazardous_waste"
    4: "Other_waste"
  
  custom:
    0: "Kitchen_waste"
    1: "Recyclable_waste"
    2: "Hazardous_waste"
    3: "Other_waste"
```

### deploy_class_map.yaml

```yaml
# 部署时的类别映射配置
# 将标准类别映射到串口协议字节

mappings:
  default:
    Kitchen_waste: 1
    Recyclable_waste: 2
    Hazardous_waste: 3
    Other_waste: 4
    # 无检测时发送的字节
    empty: 0
  
  stm32:
    Kitchen_waste: 1
    Recyclable_waste: 2
    Hazardous_waste: 3
    Other_waste: 4
    empty: 0
  
  arduino:
    Kitchen_waste: 10
    Recyclable_waste: 20
    Hazardous_waste: 30
    Other_waste: 40
    empty: 0
```

### 映射配置项说明

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `mappings.*.模型名.类别ID` | int | 模型输出的类别 ID |
| `mappings.*.模型名.类别名称` | int | 映射到的类别 ID 或字节值 |
| `mappings.*.empty` | int | 无检测时发送的值 |

---

## 设备配置

### device_rk3588.yaml

```yaml
# RK3588 NPU 设备配置

device:
  type: npu
  model: rk3588
  npu_id: 0

# 相机配置
camera:
  type: usb
  width: 1280
  height: 720
  fps: 30
  fourcc: "mjpeg"

# 推理配置
inference:
  backend: rknn
  num_threads: 4
  precision: int8

# 串口配置
serial:
  port: "/dev/ttyUSB0"
  baudrate: 115200
  data_bits: 8
  stop_bits: 1
  parity: "none"

# 分拣配置
sorting:
  # 稳定性检测
  stability:
    threshold_ms: 1000
    min_detection_count: 3
    position_tolerance: 0.05
  
  # 冷却策略
  cooldown:
    min_interval_ms: 100
    max_queue_size: 10
  
  # 类别映射
  class_mapping: "stm32"
```

### 设备配置项说明

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `device.type` | str | - | 设备类型 (cpu/gpu/npu) |
| `device.model` | str | - | 设备型号 |
| `device.npu_id` | int | 0 | NPU 编号 |
| `camera.type` | str | - | 相机类型 |
| `camera.width` | int | - | 图像宽度 |
| `camera.height` | int | - | 图像高度 |
| `camera.fps` | int | - | 帧率 |
| `inference.backend` | str | - | 推理后端 |
| `inference.num_threads` | int | - | 线程数 |
| `inference.precision` | str | - | 精度 (fp16/int8) |
| `serial.port` | str | - | 串口路径 |
| `serial.baudrate` | int | - | 波特率 |
| `stability.threshold_ms` | int | - | 稳定性判定时间 |
| `stability.min_detection_count` | int | - | 最少检测次数 |
| `stability.position_tolerance` | float | - | 位置容差 |
| `cooldown.min_interval_ms` | int | - | 最小发送间隔 |
| `cooldown.max_queue_size` | int | - | 最大队列大小 |

---

## VLM 提示词配置

### vlm_prompts.yaml

```yaml
# VLM 提示词配置

prompts:
  # 图片分类提示词
  classification: |
    你是一个垃圾分类专家。请分析图片中的垃圾类型。
    
    垃圾分类标准：
    1. 厨余垃圾 (Kitchen_waste): 剩饭剩菜、果皮果核、蛋壳、茶叶渣等
    2. 可回收垃圾 (Recyclable_waste): 塑料瓶、纸张、金属罐、玻璃瓶等
    3. 有害垃圾 (Hazardous_waste): 电池、灯管、药品、油漆等
    4. 其他垃圾 (Other_waste): 卫生纸、烟蒂、陶瓷碎片、一次性餐具等
    
    请仅输出垃圾所属的类别名称，不要输出其他内容。
    类别名称必须是以下之一：Kitchen_waste, Recyclable_waste, Hazardous_waste, Other_waste

  # 检测提示词
  detection: |
    你是一个垃圾分类专家。请检测图片中的所有垃圾物品。
    
    对于每个检测到的垃圾，请输出：
    - 类别名称
    - 置信度 (0-1)
    - 边界框坐标 (相对坐标，0-1)
    
    输出格式（JSON数组）：
    [
      {
        "category": "Kitchen_waste",
        "confidence": 0.95,
        "bbox": {"x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.3}
      }
    ]

  # 批量处理提示词
  batch: |
    你是一个垃圾分类专家。请批量处理以下图片。
    
    对于每张图片，输出其垃圾类别。
    
    输出格式：
    图片1: Kitchen_waste
    图片2: Recyclable_waste
    ...

system_prompts:
  classification: |
    你是一个专业的垃圾分类助手。
    你的任务是根据图片准确识别垃圾类型。
    请始终使用英文输出类别名称。

post_processing:
  # 后处理配置
  category_mapping:
    "厨余垃圾": "Kitchen_waste"
    "可回收垃圾": "Recyclable_waste"
    "有害垃圾": "Hazardous_waste"
    "其他垃圾": "Other_waste"
    "湿垃圾": "Kitchen_waste"
    "干垃圾": "Other_waste"
```

### 提示词配置项说明

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `prompts.classification` | str | 单图分类提示词 |
| `prompts.detection` | str | 单图检测提示词 |
| `prompts.batch` | str | 批量处理提示词 |
| `system_prompts.*` | str | 系统提示词 |
| `post_processing.category_mapping` | dict | 中文类别映射 |

---

## 环境变量

### 必需的环境变量

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `OPENAI_API_KEY` | OpenAI API Key | `sk-xxx...` |
| `CUDA_VISIBLE_DEVICES` | 可见的 GPU 设备 | `0,1` |

### 可选的环境变量

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GARBAGE_AI_CONFIG_DIR` | `config/` | 配置目录路径 |
| `GARBAGE_AI_MODELS_DIR` | `models/` | 模型目录路径 |
| `GARBAGE_AI_DATA_DIR` | `data/` | 数据目录路径 |
| `GARBAGE_AI_LOG_DIR` | `logs/` | 日志目录路径 |
| `GARBAGE_AI_CACHE_DIR` | `.cache/` | 缓存目录路径 |

### 设置环境变量

#### Linux/macOS

```bash
export OPENAI_API_KEY="sk-xxx..."
export CUDA_VISIBLE_DEVICES="0"
```

#### Windows

```bash
set OPENAI_API_KEY=sk-xxx...
set CUDA_VISIBLE_DEVICES=0
```

#### Docker

```bash
docker run -e OPENAI_API_KEY=sk-xxx... -e CUDA_VISIBLE_DEVICES=0 garbage-ai:deploy
```

---

## 配置最佳实践

### 开发环境

```yaml
# config/development.yaml
logging:
  level: DEBUG
  
inference:
  device: cpu
  num_threads: 4
```

### 生产环境

```yaml
# config/production.yaml
logging:
  level: INFO
  handlers:
    file:
      filename: /var/log/garbage_ai.log
      
inference:
  device: cuda:0
  num_threads: 8
  precision: fp16
```

### 多环境配置

使用环境变量切换配置：

```bash
# 使用生产配置
export GARBAGE_AI_ENV=production

# 或使用命令行参数
deploy run --config config/production.yaml
```

---

## 故障排除

### 配置不生效

1. 检查配置文件路径是否正确
2. 检查 YAML 语法是否正确
3. 检查环境变量是否已设置
4. 查看日志输出调试信息

### 配置冲突

配置按以下优先级覆盖：

1. 命令行参数 (最高)
2. 环境变量
3. 用户配置文件 (`~/.config/garbage-ai/`)
4. 项目配置文件 (`config/`)
5. 默认配置 (最低)

### 常见配置错误

```yaml
# 错误：YAML 缩进不正确
handlers:
  console:
class: logging.StreamHandler  # 缩进错误

# 正确写法
handlers:
  console:
    class: logging.StreamHandler  # 正确缩进
```
