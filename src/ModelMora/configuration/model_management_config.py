from typing import ClassVar

from pydantic import Field

from modelmora.configuration.base_config import BaseConfig


class ModelManagementConfig(BaseConfig):
    """Model management configuration settings."""

    config_name: ClassVar[str] = "ModelMoraModelManagement"

    model_registry_path: str = Field(
        default="/app/configuration/ModelRegistry.yaml", description="Path to the model registry directory."
    )
    model_cache_dir: str = Field(default="/models", description="Directory to cache downloaded models.")
    max_model_memory_mb: int = Field(default=8192, gt=0, description="Maximum memory (in MB) allocated for each model.")
    auto_warmup: bool = Field(default=True, description="Flag to enable or disable automatic model warmup on startup.")
