"""Domain Layer - Core business objects"""

from .base import (
    ValueObject,
    Entity,
    AggregateRoot,
    DomainEvent,
    IRepository,
)

from .taxonomy import (
    WasteCategory,
    LabelAlias,
    TaxonomyVersion,
)

from .annotation import (
    BoundingBox,
    Confidence,
    DetectionSource,
    Detection,
    LabelFile,
)

from .mapping import (
    MappingSet,
    ClassMapping,
    ProtocolMapping,
)

__all__ = [
    "ValueObject",
    "Entity",
    "AggregateRoot",
    "DomainEvent",
    "IRepository",
    "WasteCategory",
    "LabelAlias",
    "TaxonomyVersion",
    "BoundingBox",
    "Confidence",
    "DetectionSource",
    "Detection",
    "LabelFile",
    "MappingSet",
    "ClassMapping",
    "ProtocolMapping",
]
