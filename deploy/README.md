# Garbage Deploy Module

Real-time inference and deployment module for garbage detection.

## Features

- **Multi-Runtime Support**: PyTorch, RKNN
- **Camera/Serial Integration**: OpenCV camera and pyserial communication
- **Stability Logic**: Anti-duplicate counting with cooldown
- **Dry Run Mode**: Test without hardware
- **Debug Window**: Real-time visualization

## CLI Usage

```bash
python -m garbage_deploy run \
    --deploy-id raspi_yolo_v12n \
    --manifest path/to/model_manifest.json \
    --dry-run
```

## Deploy Profiles

See `configs/registry/deploy_profiles.yaml` for available configurations.

## Architecture

```
garbage_deploy/
├── entry/
│   └── __main__.py
├── application/
│   ├── usecases/
│   │   └── run_realtime_inference.py
│   └── ports/
│       ├── __init__.py
│       ├── runtime_port.py
│       ├── camera_port.py
│       ├── serial_port.py
│       └── clock_port.py
├── domain/
│   └── atoms/
│       └── detection_atoms.py
└── infrastructure/
    ├── runtime_adapters/
    │   ├── torch_adapter.py
    │   └── __init__.py
    └── io/
        ├── camera_adapter.py
        ├── serial_adapter.py
        ├── clock_adapter.py
        └── __init__.py
```

## Testing

```bash
pytest deploy/tests/
```

## Development

```bash
pip install -e deploy/
```
