# Garbage Train Module

Training module for garbage detection models.

## Features

- **Multi-Model Support**: YOLO, Fast R-CNN
- **Export Formats**: ONNX, TorchScript, RKNN
- **Dataset Preparation**: Automatic validation and preparation
- **Artifact Management**: Model manifests and tracking

## CLI Usage

```bash
python -m garbage_train train \
    --train-id yolo_v12n_e300 \
    --out ./outputs/
```

## Configuration

Training profiles are defined in `configs/registry/train_profiles.yaml`.

## Architecture

```
garbage_train/
├── entry/
│   └── __main__.py
├── application/
│   ├── usecases/
│   │   └── train_model.py
│   └── ports/
│       ├── __init__.py
│       ├── trainer_port.py
│       ├── exporter_port.py
│       ├── artifact_store_port.py
│       └── dataset_port.py
└── infrastructure/
    ├── trainers/
    │   ├── yolo_trainer.py
    │   └── __init__.py
    ├── exporters/
    │   ├── exporters.py
    │   └── __init__.py
    ├── dataset_preparation/
    │   ├── garbage_dataset.py
    │   └── __init__.py
    └── artifact_storage/
        ├── artifact_storage.py
        └── __init__.py
```

## Testing

```bash
pytest train/tests/
```

## Development

```bash
pip install -e train/
```
