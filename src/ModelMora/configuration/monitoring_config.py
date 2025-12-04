from typing import ClassVar

from pydantic import Field

from modelmora.configuration.base_config import BaseConfig


class MonitoringConfig(BaseConfig):
    """Configuration for monitoring settings."""

    config_name: ClassVar[str] = "ModelMoraMonitoring"

    prometheus_port: int = Field(
        default=9090,
        ge=0,
        le=65535,
        description="Port number for Prometheus metrics exposure.",
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level for monitoring.",
    )

    enable_tracing: bool = Field(
        default=False,
        description="Enable or disable tracing for monitoring.",
    )
