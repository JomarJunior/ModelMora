from typing import ClassVar

from pydantic import Field

from modelmora.configuration.base_config import BaseConfig


class GPUConfig(BaseConfig):
    """GPU configuration settings."""

    config_name: ClassVar[str] = "ModelMoraGPU"

    gpu_enabled: bool = Field(default=False, description="Enable GPU acceleration for model inference.")
    gpu_device_id: int = Field(default=0, ge=0, description="The GPU device ID to use for inference.")
