# Migration Map

This document maps old project components to the new Garbage-AI-Suite architecture.

## Overview

| Old Path / Component | New Module | New Path | Action |
|---------------------|------------|----------|--------|
| `YOLO_model/` | train/ | `train/src/garbage_train/infrastructure/trainers/yolo_trainer.py` | Migrated & Refactored |
| `Fast_R_CNN_model/` | train/ | `train/src/garbage_train/infrastructure/trainers/faster_rcnn_trainer.py` | Migrated & Refactored |
| `TorchVision/` | Deleted | N/A | **DELETED** (Deprecated) |
| `common/*.py` | scripts/python_tools/ | `scripts/python_tools/` | Migrated (glue code only) |
| `common/sh/*.sh` | scripts/sh/ | `scripts/sh/` | Migrated |
| `common/sys/*.sh` | scripts/sys/ | `scripts/sys/` | Migrated |
| `common/rk3588/*.sh` | scripts/devices/ | `scripts/devices/` | Migrated |
| `qwen2.5VL_with_serial/` | autolabel/ | `autolabel/src/garbage_autolabel/infrastructure/model_adapters/qwen_vl_adapter.py` | Migrated & Refactored |

---

## Detailed Mapping

### Training Components

| Old File | New Location | Notes |
|----------|--------------|-------|
| `YOLO_model/train/train4class_yolovX_easydata.py` | `train/src/garbage_train/infrastructure/trainers/yolo_trainer.py` | YOLO training logic refactored |
| `YOLO_model/train/finetune.py` | `train/src/garbage_train/infrastructure/dataset_preparation/garbage_dataset.py` | Dataset preparation logic |
| `YOLO_model/train/convert_to_onnx.py` | `train/src/garbage_train/infrastructure/exporters/exporters.py` | Export logic consolidated |
| `YOLO_model/train/convert_to_rknn.py` | `train/src/garbage_train/infrastructure/exporters/exporters.py` | RKNN export consolidated |
| `Fast_R_CNN_model/train/FAST_R_CNN_train.py` | `train/src/garbage_train/infrastructure/trainers/faster_rcnn_trainer.py` | Faster R-CNN training logic |
| `Fast_R_CNN_model/train/convert-checkpoint.py` | `train/src/garbage_train/infrastructure/exporters/exporters.py` | Checkpoint conversion |

### Deployment Components

| Old File | New Location | Notes |
|----------|--------------|-------|
| `YOLO_model/deploy/yolo4class_raspi_mod.py` | `deploy/src/garbage_deploy/infrastructure/runtime_adapters/torch_adapter.py` | Runtime adapter |
| `YOLO_model/deploy/y12e_rebuild.py` | `deploy/src/garbage_deploy/application/usecases/run_realtime_inference.py` | Inference use case |
| `Fast_R_CNN_model/deploy/FAST_R_CNN_deploy.py` | `deploy/src/garbage_deploy/infrastructure/runtime_adapters/torch_adapter.py` | Runtime adapter |

### Auto-Labeling Components

| Old File | New Location | Notes |
|----------|--------------|-------|
| `common/yolo_autolabeling.py` | `autolabel/src/garbage_autolabel/infrastructure/model_adapters/yolo_adapter.py` | YOLO adapter |
| `common/fast_R_CNN_autolabel.py` | `autolabel/src/garbage_autolabel/infrastructure/model_adapters/faster_rcnn_adapter.py` | Faster R-CNN adapter |
| `common/qwen_autolabeling.py` | `autolabel/src/garbage_autolabel/infrastructure/model_adapters/qwen_vl_adapter.py` | Qwen-VL adapter |
| `common/json-converter.py` | `scripts/python_tools/json-converter.py` | Utility script |
| `common/convert_mapping.py` | `scripts/python_tools/convert_mapping.py` | Utility script |

### Serial Communication

| Old File | New Location | Notes |
|----------|--------------|-------|
| `common/reptile.py` | `deploy/src/garbage_deploy/infrastructure/io/serial_adapter.py` | Serial adapter |
| `common/toolbox.py` | `shared/src/garbage_shared/utils/file_helpers.py` | Shared utilities |
| `common/uart-manager.sh` | `scripts/sh/uart-manager.sh` | Shell script |

### Deleted Components

| Old Path | Reason |
|----------|--------|
| `TorchVision/` | Deprecated, replaced by unified train/deploy/autolabel modules |

---

## Migration Principles

1. **No Core Logic in Scripts**: Scripts only contain glue code, system operations, and utilities
2. **Contract-First**: All data formats defined in `contracts/` before implementation
3. **One-Way Dependency**: Entry → Application → Domain → Infrastructure
4. **Port-Based Architecture**: Each module defines ports, infrastructure implements them
5. **Configuration-Driven**: All profiles in `configs/registry/`, CLI uses IDs not paths

---

## Unidirectional Data Flow

```
Old Architecture:
  train.py → model → deploy.py (direct coupling)

New Architecture:
  train/ ─→ contracts/ ←─ deploy/
       │              │
       └── model_manifest.json ─→ autolabel/
```

---

## Key Changes

### Before (Monolithic)
- Training and deployment tightly coupled
- Hardcoded paths and model configurations
- No standardized data contracts
- Scripts contained business logic

### After (Modular)
- Train/Deploy/AutoLabel independent packages
- Config-driven via `train_id`/`deploy_id`
- JSON Schema contracts for all data
- Scripts are pure utilities (no business logic)

---

## Verification

To verify migration completeness:

```bash
# Check no old paths referenced in new code
grep -r "YOLO_model\|Fast_R_CNN_model\|TorchVision" Garbage-AI-Suite/ --include="*.py"

# Verify all contracts exist
ls -la Garbage-AI-Suite/contracts/labels/
ls -la Garbage-AI-Suite/contracts/artifacts/
ls -la Garbage-AI-Suite/contracts/io/

# Verify registry configs
ls -la Garbage-AI-Suite/configs/registry/
```
