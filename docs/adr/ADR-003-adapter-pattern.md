# ADR-003: Adapter Pattern for Model Runtimes

## Status

Accepted

## Context

The system needs to support multiple model runtimes (PyTorch, ONNX, RKNN, TensorRT) across different deployment targets (x86, Raspberry Pi, RK3588). Each runtime has different APIs and loading mechanisms.

## Decision

Use the **Adapter Pattern** to abstract model runtime operations behind a common port interface. Each runtime is implemented as an adapter that conforms to the `RuntimePort` interface.

## Consequences

### Positive

1. **Runtime abstraction**: Application code works with any runtime
2. **Testability**: Mock adapters enable testing without actual models
3. **Extensibility**: New runtimes can be added without changing application
4. **Device-specific optimization**: Each adapter can optimize for its target

### Negative

1. **Interface design**: Must anticipate all required operations
2. **Adapter maintenance**: Each runtime needs its own adapter
3. **Performance overhead**: Abstraction may add small overhead
4. **Feature gaps**: Some runtimes may not support all operations

### Neutral

1. **Learning curve**: Understanding adapter pattern
2. **Testing complexity**: More test cases for adapter combinations

## Port Interface

```python
# garbage_deploy/application/ports.py

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path

class RuntimePort(ABC):
    """Abstract port for model runtime operations."""
    
    @abstractmethod
    def load_model(self, manifest_path: str, device: str) -> None:
        """Load model from manifest."""
        pass
    
    @abstractmethod
    def infer(self, image) -> List[Dict]:
        """Run inference on single image."""
        pass
    
    @abstractmethod
    def get_input_size(self) -> tuple[int, int]:
        """Return model input size (width, height)."""
        pass
    
    @abstractmethod
    def warmup(self, num_iterations: int = 10) -> float:
        """Run warmup inference and return avg time."""
        pass
```

## Adapter Implementations

### Torch Runtime Adapter

```python
# garbage_deploy/infrastructure/runtime_adapters/torch_adapter.py

from garbage_deploy.application.ports import RuntimePort
from ultralytics import YOLO
import torch

class TorchRuntimeAdapter(RuntimePort):
    def __init__(self, config: dict):
        self.config = config
        self._model = None
    
    def load_model(self, manifest_path: str, device: str):
        import yaml
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        model_path = manifest["files"][0]["path"]
        self._model = YOLO(model_path)
        self._model.to(device)
    
    def infer(self, image):
        results = self._model(image, verbose=False)
        return self._parse_results(results)
    
    def get_input_size(self):
        return (640, 640)  # YOLO standard size
    
    def warmup(self, num_iterations=10):
        import time
        start = time.time()
        for _ in range(num_iterations):
            self._model(self._dummy_image)
        return (time.time() - start) / num_iterations
```

### RKNN Runtime Adapter

```python
# garbage_deploy/infrastructure/runtime_adapters/rknn_adapter.py

from garbage_deploy.application.ports import RuntimePort
from rknn.api import RKNN

class RKNNRuntimeAdapter(RuntimePort):
    def __init__(self, config: dict):
        self.config = config
        self._rknn = None
        self._model = None
    
    def load_model(self, manifest_path: str, device: str):
        import yaml
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        rknn_path = [f for f in manifest["files"] if f["type"] == "rknn"][0]["path"]
        self._rknn = RKNN()
        self._rknn.load_rknn(rknn_path)
        self._rknn.init_runtime(target_device=device)
    
    def infer(self, image):
        outputs = self._rknn.inference(inputs=[image])
        return self._parse_outputs(outputs)
```

## Usage in Application

```python
# garbage_deploy/application/usecases/run_realtime_inference.py

from garbage_deploy.application.ports import RuntimePort

class RunRealtimeInferenceUseCase:
    def __init__(self, runtime: RuntimePort):
        self.runtime = runtime
    
    def execute(self, deploy_id: str, manifest_path: str, dry_run: bool):
        # Load model through adapter
        self.runtime.load_model(manifest_path, device)
        
        # Warmup
        avg_time = self.runtime.warmup()
        
        # Inference loop
        while True:
            frame = self.camera.read()
            detections = self.runtime.infer(frame)
```

## Adding New Runtimes

To add a new runtime (e.g., TensorRT):

1. Create `infrastructure/runtime_adapters/tensorrt_adapter.py`
2. Implement `RuntimePort` interface
3. Add to `RuntimePort.__subclasses__()` registry
4. Update `deploy_profiles.yaml` with new runtime option

## Testing with Mock Adapter

```python
# tests/test_inference.py

from unittest.mock import Mock
from garbage_deploy.application.ports import RuntimePort

class MockRuntimeAdapter(RuntimePort):
    """Mock adapter for testing without actual models."""
    
    def __init__(self):
        self._model = None
        self.call_count = 0
    
    def load_model(self, manifest_path: str, device: str):
        self._model = Mock()
    
    def infer(self, image):
        self.call_count += 1
        return [{"class_id": 0, "confidence": 0.9, "bbox": {...}}]
```

## References

- [Port-Based Architecture](architecture.md#13-port-based-architecture)
- [Deploy Infrastructure](../deploy/src/garbage_deploy/infrastructure/runtime_adapters/)
- [RuntimePort Definition](../deploy/src/garbage_deploy/application/ports/)
