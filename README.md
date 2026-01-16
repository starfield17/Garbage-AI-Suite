# Garbage AI Suite

AI-based garbage detection and classification system with modular architecture for training, deployment, and auto-labeling.

## Overview

This project provides a complete solution for garbage detection and classification using computer vision and deep learning. The system is built with a **clean architecture** approach, separating concerns into distinct modules that can be developed, tested, and deployed independently.

### Key Features

- **Multi-Model Support**: YOLO, Faster R-CNN, and Vision Language Models (Qwen-VL)
- **Real-Time Inference**: Optimized deployment for edge devices (RK3588, Raspberry Pi)
- **Auto-Labeling**: Semi-automated dataset labeling using AI models
- **Portable Exports**: ONNX, TorchScript, and RKNN format support
- **Structured Logging**: Observability with correlation IDs

## Architecture

```
garbage-ai-suite/
├── contracts/              # Data contracts and schemas
│   ├── labels/            # Label format schemas (bbox, COCO, YOLO)
│   ├── artifacts/         # Model manifest schemas
│   └── io/                # Inference request/response schemas
├── configs/               # Configuration files
│   └── registry/          # Profiles for train/deploy/autolabel
├── shared/                # Shared utilities
│   └── src/garbage_shared/
│       ├── config_loader/     # YAML configuration loading
│       ├── workflow_engine/   # Workflow orchestration
│       ├── observability/     # Structured logging
│       ├── contracts_models/  # Pydantic DTOs
│       └── utils/             # Helper utilities
├── train/                 # Training module
│   └── src/garbage_train/
│       ├── application/       # Use cases and ports
│       ├── domain/            # Business logic atoms
│       ├── infrastructure/    # Adapters (trainers, exporters)
│       └── entry/             # CLI entry point
├── deploy/                # Deployment module
│   └── src/garbage_deploy/
│       ├── application/       # Use cases and ports
│       ├── domain/            # Stability and detection logic
│       ├── infrastructure/    # Runtime/camera/serial adapters
│       └── entry/             # CLI entry point
├── autolabel/             # Auto-labeling module
│   └── src/garbage_autolabel/
│       ├── application/       # Use cases and ports
│       ├── domain/            # Detection atoms
│       ├── infrastructure/    # Model/label format adapters
│       └── entry/             # CLI entry point
├── scripts/               # Utility scripts
│   ├── sh/                # Shell scripts
│   ├── python_tools/      # Python utility tools
│   ├── devices/           # Device-specific scripts
│   ├── sys/               # System scripts
│   └── architect/         # Architecture validation
├── docs/                  # Documentation
│   ├── adr/               # Architecture Decision Records
│   └── workflows/         # Workflow definitions
└── .github/workflows/     # CI/CD pipelines
```

### Design Principles

1. **Port-Based Architecture**: Each module defines abstract ports that infrastructure adapters implement
2. **Domain Atoms**: Pure functions for core business logic (filtering, clipping, stability checking)
3. **Dependency Inversion**: High-level modules depend on abstractions, not implementations
4. **Unidirectional Data Flow**: Entry -> Application -> Domain -> Infrastructure

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, for training)
- Linux/macOS/Windows

### Install All Modules

```bash
# Install in development mode
pip install -e shared/
pip install -e train/
pip install -e deploy/
pip install -e autolabel/

# Or install all at once
pip install -e shared/ train/ deploy/ autolabel/
```

### Development Dependencies

```bash
pip install -e "shared/[dev]"
pip install -e "train/[dev]"
pip install -e "deploy/[dev]"
pip install -e "autolabel/[dev]"
```

## Usage

### Training Models

```bash
# Train using a profile from configs/registry/train_profiles.yaml
python -m garbage_train train \
    --train-id yolo_v12n_e300 \
    --out ./outputs/

# Available train profiles:
# - yolo_v12n_e300: YOLOv12 nano, 300 epochs
# - yolo_v12s_large: YOLOv12 small, 150 epochs
# - fasterrcnn_mobilenet: Fast R-CNN MobileNet
# - fasterrcnn_resnet50: Fast R-CNN ResNet50
```

### Deploying for Inference

```bash
# Run real-time inference
python -m garbage_deploy run \
    --deploy-id raspi_yolo_v12n \
    --manifest path/to/model_manifest.json \
    --dry-run  # Optional: test without hardware

# Available deploy profiles:
# - raspi_yolo_v12n: Raspberry Pi with YOLO
# - x86_fast_rcnn: x86 with Fast R-CNN
# - rk3588_yolo_onnx: RK3588 with ONNX
# - test_mode: Dry run for testing
```

### Auto-Labeling

```bash
# Auto-label dataset using AI model
python -m garbage_autolabel label \
    --config configs/autolabel/default.yaml \
    --model yolo \
    --input ./datasets/images \
    --out ./labels \
    --format bbox \
    --confidence 0.7

# Available models:
# - yolo: Ultralytics YOLO
# - faster_rcnn: torchvision Faster R-CNN
# - qwen_vl: Qwen Vision Language Model (requires VLM_API_KEY)
```

## Configuration

### Registry Files

Configuration profiles are stored in `configs/registry/`:

- **class_map.yaml**: Class ID to name mappings (4 waste categories)
- **train_profiles.yaml**: Training configurations per model
- **deploy_profiles.yaml**: Deployment configurations per device
- **mapping.yaml**: Train-to-deploy profile mappings

### Example Configuration

```yaml
# configs/registry/train_profiles.yaml
train_profiles:
  yolo_v12n_e300:
    model_family: yolo
    base_model: yolov12n.pt
    dataset_path: ./datasets/garbage_dataset
    hyperparameters:
      epochs: 300
      batch_size: 10
      image_size: 640
    export_targets:
      - onnx
      - rknn
```

## Project Structure Details

### Shared Module

The `shared` module contains cross-cutting concerns used by all other modules:

- **ConfigLoader**: YAML configuration loading with validation
- **WorkflowEngine**: YAML-based workflow orchestration
- **Observability**: Structured logging with correlation IDs
- **Contracts Models**: Pydantic models for all data contracts

### Domain Atoms

Pure functions implementing core business logic:

```python
# From autolabel
filter_by_confidence(detections, threshold)
clip_bbox_to_image(detections, width, height)
map_class_names(detections, class_map)
nms(detections, iou_threshold)

# From deploy
check_detection_stability(state, detected_type, now, ...)
can_count_new_garbage(state, detected_type, now, ...)
map_serial_payload(detection, image_w, image_h, max_value)
```

### Ports and Adapters

Each module defines abstract ports that infrastructure adapters implement:

| Module | Port | Implementations |
|--------|------|-----------------|
| train | TrainerPort | YOLOTrainer, FasterRCNNTrainer |
| train | ExporterPort | ONNXExporter, TorchScriptExporter, RKNNExporter |
| deploy | RuntimePort | TorchRuntimeAdapter, RKNNRuntimeAdapter |
| deploy | CameraPort | OpenCVCameraAdapter |
| deploy | SerialPort | PySerialAdapter |

## Development

### Running Tests

```bash
# Run all tests
pytest shared/ train/ deploy/ autolabel/tests/ -v

# Run specific module tests
pytest autolabel/tests/
pytest deploy/tests/
pytest train/tests/
```

### Code Quality

```bash
# Lint with ruff
ruff check shared/ train/ deploy/ autolabel/

# Format with black
black shared/ train/ deploy/ autolabel/

# Type checking with mypy
mypy shared/ train/ deploy/ autolabel/

# Architecture validation
python scripts/architect/check_architecture.py
```

### CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.pipeline.yaml`):
- Ruff linting
- Black formatting check
- MyPy type checking
- Pytest execution

## Data Formats

### BBox Label Format

```json
{
  "labels": [
    {
      "name": "bottle",
      "x1": 100,
      "y1": 50,
      "x2": 200,
      "y2": 150,
      "confidence": 0.95
    }
  ]
}
```

### Model Manifest

```json
{
  "model_name": "yolov12n_garbage_model",
  "model_family": "yolo",
  "train_id": "yolo_v12n_e300",
  "created_at": "2024-01-01T00:00:00",
  "classes": [
    {"id": 0, "name": "Kitchen_waste"},
    {"id": 1, "name": "Recyclable_waste"},
    {"id": 2, "name": "Hazardous_waste"},
    {"id": 3, "name": "Other_waste"}
  ],
  "input": {"width": 640, "height": 480, "channels": 3},
  "export_targets": ["onnx", "rknn"],
  "files": [...]
}
```

## Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `VLM_API_KEY` | API key for VLM models | Qwen-VL autolabeling |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | Training with GPU |

## License

MIT License

## Contributing

1. Follow the architecture layering rules
2. Add tests for new functionality
3. Run linting and type checking before submitting
4. Update documentation as needed
