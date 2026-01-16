# ADR-002: Model Manifest as Handshake Mechanism

## Status

Accepted

## Context

The training module produces trained models in various formats (PyTorch, ONNX, RKNN). The deployment and auto-labeling modules need to consume these models. Without a standardized handshake, each module must know about the internal details of how models are trained and exported.

## Decision

Training will produce a `model_manifest.json` file that serves as the **single source of truth** for deploying or auto-labeling with a trained model. Deploy and auto-label modules will read this manifest to configure themselves.

## Consequences

### Positive

1. **Loose coupling**: Deploy/autolabel don't need to know training internals
2. **Reproducibility**: Manifest contains all metadata needed to recreate a deployment
3. **Versioning**: Manifest tracks which train_id produced the model
4. **Validation**: Deploy/autolabel can validate compatibility before loading
5. **Audit trail**: Manifest records when/how model was trained

### Negative

1. **Manifest management**: Must ensure manifest is always generated and valid
2. **Schema evolution**: Changing manifest schema requires migration strategy
3. **File dependency**: Extra file to manage alongside model weights

### Neutral

1. **Pipeline complexity**: Additional step in training workflow
2. **Storage overhead**: Small JSON file per model

## Manifest Schema

```json
{
  "model_name": "yolov12n_garbage_model",
  "model_family": "yolo",
  "train_id": "yolo_v12n_e300",
  "git_sha": "abc123def",
  "created_at": "2024-01-15T10:30:00Z",
  "classes": [
    {"id": 0, "name": "Kitchen_waste"},
    {"id": 1, "name": "Recyclable_waste"},
    {"id": 2, "name": "Hazardous_waste"},
    {"id": 3, "name": "Other_waste"}
  ],
  "input": {
    "width": 640,
    "height": 480,
    "channels": 3
  },
  "export_targets": ["onnx", "rknn"],
  "metrics": {
    "mAP": 0.85,
    "precision": 0.82,
    "recall": 0.88
  },
  "files": [
    {"path": "weights/best.pt", "type": "pytorch", "size_bytes": 1234567},
    {"path": "weights/best.onnx", "type": "onnx", "size_bytes": 2345678}
  ]
}
```

## Usage in Modules

### Train Module

```python
def _generate_manifest(profile, train_result, export_results):
    return {
        "model_name": f"{profile['model_family']}_garbage_model",
        "model_family": profile["model_family"],
        "train_id": profile.get("train_id"),
        "classes": self._get_classes(profile),
        "input": profile.get("input_spec"),
        "export_targets": profile.get("export_targets"),
        "files": export_results,
    }
```

### Deploy Module

```python
def _initialize_components(self, profile, manifest, dry_run):
    # Validate manifest
    assert manifest["model_family"] == profile["expected_family"]
    
    # Load model from manifest
    model_path = manifest["files"][0]["path"]
    self.runtime.load_model(model_path)
    
    # Configure thresholds from manifest classes
    self.num_classes = len(manifest["classes"])
```

### AutoLabel Module

```python
def _get_adapter(self, model_type, config, manifest=None):
    if manifest:
        # Use manifest to configure adapter
        adapter_class = ADAPTER_MAP[manifest["model_family"]]
        return adapter_class(config, manifest)
```

## Validation

Deploy and auto-label modules should validate:

1. **Required fields**: All required schema fields are present
2. **Model family compatibility**: Expected model family matches
3. **Class consistency**: Class IDs match known class map
4. **File existence**: All files referenced in manifest exist

## References

- [Model Manifest Schema](../contracts/artifacts/model_manifest.schema.json)
- [Contract-First Development](ADR-001-contracts-first.md)
