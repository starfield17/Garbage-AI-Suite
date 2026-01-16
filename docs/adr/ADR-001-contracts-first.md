# ADR-001: Contracts-First Development

## Status

Accepted

## Context

In a modular ML system with multiple components (train, deploy, autolabel), data flows between modules through various formats: model manifests, inference requests/responses, and label files. Without explicit contracts, changes to one module can break others silently.

## Decision

We will define all data formats that cross module boundaries as JSON Schema contracts before implementation.

## Consequences

### Positive

1. **Early validation**: Runtime validation against schemas catches integration errors
2. **Documentation**: Contracts serve as self-documenting APIs
3. **Independent evolution**: Modules can be updated as long as contracts are respected
4. **Tooling**: Schema validation can be automated in CI/CD
5. **Interoperability**: External tools can validate against schemas

### Negative

1. **Upfront design**: Contracts must be designed before implementation
2. **Schema maintenance**: Changes require updating schemas and all consumers
3. **Boilerplate**: Additional files for each data format

### Neutral

1. **Learning curve**: Team must understand JSON Schema
2. **Validation overhead**: Runtime validation adds latency (can be disabled in production)

## Implementation

Contract files are located in `contracts/` with subdirectories by domain:

```
contracts/
├── labels/           # COCO, YOLO, bbox label schemas
├── artifacts/        # Model manifest, export targets
└── io/               # Inference request/response
```

Each contract is a valid JSON Schema file that can be used for:

- Code generation (Pydantic models, TypeScript interfaces)
- Runtime validation
- Documentation generation
- API specification

## Example

```json
// contracts/artifacts/model_manifest.schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Model Manifest",
  "type": "object",
  "properties": {
    "model_name": {"type": "string"},
    "model_family": {"type": "string", "enum": ["yolo", "faster_rcnn", "vlm"]},
    "train_id": {"type": "string"},
    "classes": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "integer"},
          "name": {"type": "string"}
        },
        "required": ["id", "name"]
      }
    }
  },
  "required": ["model_name", "model_family", "train_id", "classes"]
}
```

## References

- [JSON Schema](https://json-schema.org/)
- [Contracts Directory](../contracts/)
- [Model Manifest Schema](../contracts/artifacts/model_manifest.schema.json)
