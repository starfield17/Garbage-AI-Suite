# Architecture Document

This document describes the architecture principles, constraints, and design patterns used in Garbage-AI-Suite.

## 1. Core Principles

### 1.1 Contracts-First Development

All data formats that cross module boundaries must be defined as JSON Schema contracts before implementation. This ensures:

- **Interoperability**: Modules can evolve independently as long as contracts are respected
- **Documentation**: Contracts serve as living documentation for data structures
- **Validation**: Runtime validation against schemas catches integration errors early

**Contract Locations**:
- `contracts/labels/`: Label format schemas (COCO, YOLO, bbox)
- `contracts/artifacts/`: Model artifact schemas (manifest, export targets)
- `contracts/io/`: Inference I/O schemas (request, response)

### 1.2 Unidirectional Dependency

The architecture follows a strict unidirectional dependency flow:

```
Entry → Application → Domain → Infrastructure
```

**Rules**:
- Entry points (CLI) depend on Application
- Application contains use cases and defines ports
- Domain contains pure business logic (atoms)
- Infrastructure implements Application ports
- **NO backward dependencies**: Infrastructure cannot import Application or Entry

### 1.3 Port-Based Architecture

Each module (train, deploy, autolabel) defines abstract ports that infrastructure adapters implement:

| Module | Port | Implementations |
|--------|------|-----------------|
| train | TrainerPort | YOLOTrainer, FasterRCNNTrainer |
| train | ExporterPort | ONNXExporter, TorchScriptExporter, RKNNExporter |
| deploy | RuntimePort | TorchRuntimeAdapter, RKNNRuntimeAdapter |
| deploy | CameraPort | OpenCVCameraAdapter |
| deploy | SerialPort | PySerialAdapter |
| autolabel | ModelAdapterPort | YOLOAdapter, FasterRCNNAdapter, QwenVLAdapter |
| autolabel | LabelWriterPort | BBoxWriter, COCOWriter, YOLOWriter |

### 1.4 Configuration-Driven

All runtime behavior is driven by configuration files, not hardcoded paths or logic:

- **Registry Configs**: `configs/registry/{train,deploy,mapping,class}_profiles.yaml`
- **Profile IDs**: CLI commands use `--train-id`, `--deploy-id` to reference profiles
- **No Hardcoded Paths**: Profiles define all paths, devices, and thresholds

## 2. Module Structure

### 2.1 Train Module

```
train/
├── pyproject.toml          # Package configuration
├── README.md               # Module documentation
├── src/garbage_train/
│   ├── entry/              # CLI entry point
│   │   └── __main__.py     # Command-line interface
│   ├── application/        # Use cases and port definitions
│   │   ├── ports/          # Abstract interfaces (TrainerPort, ExporterPort)
│   │   └── usecases/       # Business workflow orchestration
│   ├── domain/             # Pure business logic (TODO: add domain atoms)
│   └── infrastructure/     # Adapter implementations
│       ├── trainers/       # YOLO, Faster R-CNN trainers
│       ├── exporters/      # ONNX, TorchScript, RKNN exporters
│       ├── dataset_preparation/
│       └── artifact_storage/
└── tests/                  # Unit and integration tests
```

### 2.2 Deploy Module

```
deploy/
├── pyproject.toml          # Package configuration
├── README.md               # Module documentation
├── src/garbage_deploy/
│   ├── entry/              # CLI entry point
│   │   └── __main__.py     # Command-line interface
│   ├── application/        # Use cases and port definitions
│   │   └── usecases/       # Inference workflow orchestration
│   ├── domain/             # Pure business logic
│   │   └── atoms/          # DetectionState, stability checking
│   └── infrastructure/     # Adapter implementations
│       ├── runtime_adapters/  # Model runtime (Torch, RKNN)
│       └── io/             # Camera, Serial, Clock adapters
└── tests/                  # Unit and integration tests
```

### 2.3 AutoLabel Module

```
autolabel/
├── pyproject.toml          # Package configuration
├── README.md               # Module documentation
├── src/garbage_autolabel/
│   ├── entry/              # CLI entry point
│   │   └── __main__.py     # Command-line interface
│   ├── application/        # Use cases and port definitions
│   │   └── usecases/       # Auto-labeling workflow
│   ├── domain/             # Pure business logic
│   │   └── atoms/          # filter_by_confidence, clip_bbox, NMS
│   └── infrastructure/     # Adapter implementations
│       ├── model_adapters/ # YOLO, Faster R-CNN, Qwen-VL
│       ├── label_formats/  # BBox, COCO, YOLO writers
│       └── storage/        # Dataset scanning
└── tests/                  # Unit and integration tests
```

### 2.4 Shared Module

```
shared/
├── pyproject.toml          # Package configuration
├── README.md               # Module documentation
└── src/garbage_shared/
    ├── config_loader/      # YAML configuration loading
    ├── workflow_engine/    # Workflow orchestration
    ├── observability/      # Structured logging with correlation IDs
    ├── contracts_models/   # Pydantic DTOs for contracts
    └── utils/              # Helper utilities
```

## 3. Data Flow

### 3.1 Training Flow

```
1. User: garbage-train run --train-id <id> --out <dir>
2. Entry: Parse CLI arguments
3. Application: Load train profile, orchestrate workflow
4. Infrastructure: Train model, export to targets
5. Infrastructure: Generate model_manifest.json
6. Application: Save manifest to output directory
```

### 3.2 Deployment Flow

```
1. User: garbage-deploy run --deploy-id <id> --manifest <path>
2. Entry: Parse CLI arguments
3. Application: Load deploy profile and manifest
4. Infrastructure: Initialize camera, serial, runtime
5. Domain: Check detection stability
6. Application: Run inference loop
7. Infrastructure: Send results via serial
```

### 3.3 AutoLabel Flow

```
1. User: garbage-autolabel run --model <type> --input <dir> --out <dir>
2. Entry: Parse CLI arguments
3. Application: Load config, scan dataset
4. Infrastructure: Load model adapter
5. Domain: Filter by confidence, clip bboxes, NMS
6. Infrastructure: Write labels in requested format
```

## 4. Configuration Files

### 4.1 Train Profiles (`configs/registry/train_profiles.yaml`)

```yaml
train_profiles:
  yolo_v12n_e300:
    model_family: yolo
    base_model: yolov12n.pt
    hyperparameters:
      epochs: 300
      batch_size: 10
    export_targets:
      - onnx
      - rknn
```

### 4.2 Deploy Profiles (`configs/registry/deploy_profiles.yaml`)

```yaml
deploy_profiles:
  raspi_yolo_v12n:
    runtime: torch
    device_type: raspi
    confidence_threshold: 0.9
    stability:
      min_position_frames: 5
      cooldown_ms: 2000
```

### 4.3 Mapping (`configs/registry/mapping.yaml`)

```yaml
links:
  - train_id: yolo_v12n_e300
    deploy_id: raspi_yolo_v12n
```

### 4.4 Class Map (`configs/registry/class_map.yaml`)

```yaml
classes:
  - id: 0
    name: Kitchen_waste
    aliases: [厨余垃圾, food_waste]
  - id: 1
    name: Recyclable_waste
    aliases: [可回收物, recycling]
```

## 5. Contracts

### 5.1 Model Manifest (`contracts/artifacts/model_manifest.schema.json`)

The model manifest is the **handshake mechanism** between train → deploy/autolabel:

```json
{
  "model_name": "yolov12n_garbage_model",
  "model_family": "yolo",
  "train_id": "yolo_v12n_e300",
  "classes": [
    {"id": 0, "name": "Kitchen_waste"}
  ],
  "input": {"width": 640, "height": 480, "channels": 3},
  "export_targets": ["onnx", "rknn"]
}
```

### 5.2 Inference Response (`contracts/io/inference_response.schema.json`)

```json
{
  "success": true,
  "detections": [
    {
      "class_id": 0,
      "class_name": "bottle",
      "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 150},
      "confidence": 0.95
    }
  ],
  "inference_time_ms": 15.5
}
```

## 6. Domain Atoms

Domain atoms are pure functions implementing core business logic:

### 6.1 Deploy Domain Atoms

```python
# Stability checking
check_detection_stability(state, detected_type, now, min_position_frames)
can_count_new_garbage(state, detected_type, now, cooldown_ms)

# Serial payload mapping
map_serial_payload(detection, image_w, image_h, max_value)
```

### 6.2 AutoLabel Domain Atoms

```python
# Detection processing
filter_by_confidence(detections, threshold)
clip_bbox_to_image(detections, width, height)
map_class_names(detections, class_map)
nms(detections, iou_threshold)
```

## 7. Workflows

Workflows define explicit orchestration steps:

- `docs/workflows/train.workflow.yaml`: prepare → train → eval → export → manifest
- `docs/workflows/deploy.workflow.yaml`: resolve_artifact → load → warmup → infer → post → report
- `docs/workflows/autolabel.workflow.yaml`: load_models → run → fuse/fallback → sample → export_labels

## 8. Scripts

Scripts are **NOT** allowed to contain core business logic. They only perform:

- System operations (service management, environment setup)
- Utility conversions (JSON transformation, mapping)
- Device-specific scripts (NPU config, UART management)

See `scripts/README.md` for detailed script documentation.

## 9. Import Boundaries

### 9.1 Allowed Imports

| From | To | Allowed |
|------|-----|---------|
| entry | application | ✅ Yes |
| application | domain | ✅ Yes |
| application | infrastructure | ✅ Yes (via ports) |
| domain | infrastructure | ❌ No |
| any module | shared | ✅ Yes |

### 9.2 Cross-Module Imports

```
train → deploy: ❌ No
deploy → train: ❌ No
autolabel → train: ❌ No
train/autolabel/deploy → shared: ✅ Yes
```

The only handshake between modules is through `model_manifest.json` files and configuration profiles.

## 10. Quality Gates

### 10.1 CI/CD Pipeline

The project uses GitHub Actions for continuous integration:

- **ruff**: Linting
- **black**: Code formatting
- **mypy**: Type checking
- **pytest**: Unit and integration tests
- **architecture**: Import boundary validation

### 10.2 Test Coverage Requirements

- All domain atoms must have unit tests
- Each module must have at least one smoke test
- Tests must run in CI before merge

## 11. Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `VLM_API_KEY` | API key for VLM models | Qwen-VL autolabeling |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | Training with GPU |

## 12. Versioning and Changelog

- Semantic versioning: MAJOR.MINOR.PATCH
- Changelog maintained in CHANGELOG.md
- Git tags for releases

## 13. Contributing

1. Follow the architecture layering rules
2. Add tests for new functionality
3. Run linting and type checking before submitting
4. Update documentation as needed
5. Create ADR for architectural decisions

## 14. References

- [Architecture Decision Records](adr/)
- [Workflow Definitions](workflows/)
- [Migration Map](migration_map.md)
