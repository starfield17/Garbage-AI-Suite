"""Configuration loader with YAML validation."""

import os
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings

T = TypeVar("T", bound=BaseModel)


class ConfigLoader:
    """Load and validate YAML configuration files with Pydantic models."""

    def __init__(self, base_path: Path | str | None = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def load_yaml(self, config_path: str | Path) -> dict[str, Any]:
        full_path = self.base_path / config_path
        if not full_path.exists():
            raise FileNotFoundError(f"Config file not found: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_model(
        self, config_path: str | Path, model_class: type[T]
    ) -> T:
        raw_config = self.load_yaml(config_path)

        try:
            return model_class(**raw_config)
        except ValidationError as e:
            raise ValueError(f"Config validation failed: {e}") from e

    def load_with_env_override(
        self,
        config_path: str | Path,
        env_prefix: str = "",
    ) -> dict[str, Any]:
        config = self.load_yaml(config_path)

        if env_prefix:
            for key in config.keys():
                env_var = f"{env_prefix}_{key.upper()}"
                if env_var in os.environ:
                    value = os.environ[env_var]
                    config[key] = self._cast_value(config[key], value)

        return config

    @staticmethod
    def _cast_value(original: Any, new: str) -> Any:
        if isinstance(original, bool):
            return new.lower() in ("true", "1", "yes")
        if isinstance(original, int):
            return int(new)
        if isinstance(original, float):
            return float(new)
        return new


class BaseConfig(BaseSettings):
    """Base configuration class with env var support."""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
