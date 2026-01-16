"""README for autolabel module."""

# Garbage Autolabel Module

Multi-model auto-labeling support for garbage detection.

## Features

- **Multi-Model Support**: YOLO, Faster R-CNN, Qwen-VL (Vision Language Model)
- **Flexible Output Formats**: BBox JSON, COCO, YOLO
- **Domain Atoms**: Pure functions for filtering, clipping, mapping, NMS
- **Port-Based Architecture**: Pluggable model adapters and label writers

## CLI Usage

```bash
python -m garbage_autolabel label \
    --config configs/autolabel/default.yaml \
    --model yolo \
    --input ./datasets/images \
    --out ./labels \
    --format bbox \
    --confidence 0.7
```

## Configuration

See `configs/autolabel/` for configuration examples.

## Model Adapters

| Model | Adapter Class | Config Keys |
|--------|---------------|-------------|
| YOLO | `YOLOAdapter` | `model_path`, `device` |
| Fast R-CNN | `FasterRCNNAdapter` | `model_path`, `device`, `model_type` |
| Qwen-VL | `QwenVLAdapter` | `model`, `base_url` (uses `VLM_API_KEY` env var) |

## Architecture

```
garbage_autolabel/
├── entry/
│   └── __main__.py
├── application/
│   ├── usecases/
│   │   └── auto_label_dataset.py
│   └── ports/
│       ├── __init__.py
│       ├── model_adapter_port.py
│       ├── label_writer_port.py
│       ├── dataset_scanner_port.py
│       └── converter_port.py
├── domain/
│   ├── atoms/
│   │   └── detection_atoms.py
│   └── molecules/
└── infrastructure/
    ├── model_adapters/
    │   ├── yolo_adapter.py
    │   ├── faster_rcnn_adapter.py
    │   └── qwen_vl_adapter.py
    ├── label_formats/
    │   ├── bbox_writer.py
    │   ├── coco_writer.py
    │   └── yolo_writer.py
    └── storage/
        └── dataset_scanner.py
```

## Testing

```bash
pytest autolabel/tests/
```

## Development

```bash
pip install -e autolabel/
```
