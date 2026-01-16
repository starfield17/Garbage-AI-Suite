# Garbage Shared

Shared utilities, configuration, and contracts for the garbage AI suite.

## Features

- **ConfigLoader**: YAML configuration loading with validation
- **WorkflowEngine**: YAML-based workflow orchestration
- **Observability**: Structured logging with correlation IDs
- **Contracts Models**: Pydantic models for all data contracts

## Installation

```bash
pip install -e shared/
```

## Usage

```python
from garbage_shared.config_loader import ConfigLoader
from garbage_shared.workflow_engine import WorkflowEngine
from garbage_shared.contracts_models import BBoxLabelDTO
```

## Development

```bash
cd shared/
pip install -e ".[dev]"
pytest
```
