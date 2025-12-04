from typing import ClassVar

from pydantic import Field

from modelmora.configuration.base_config import BaseConfig
from modelmora.configuration.connection_config import ConnectionConfig
from modelmora.configuration.gpu_config import GPUConfig
from modelmora.configuration.kafka_config import KafkaConfig
from modelmora.configuration.model_management_config import ModelManagementConfig
from modelmora.configuration.monitoring_config import MonitoringConfig
from modelmora.configuration.performance_config import PerformanceConfig


class AppConfig(BaseConfig):
    """Application configuration settings."""

    config_name: ClassVar[str] = "ModelMora"

    app_name: str = Field(default="ModelMora", description="The name of the application.")
    app_version: str = Field(default="0.1.0", description="The version of the application.")
    debug_mode: bool = Field(default=False, description="Flag to enable or disable debug mode.")

    connection_config: ConnectionConfig = Field(
        default_factory=ConnectionConfig.from_env, description="The connection configuration settings."
    )

    performance_config: PerformanceConfig = Field(
        default_factory=PerformanceConfig.from_env, description="The performance configuration settings."
    )

    model_management_config: ModelManagementConfig = Field(
        default_factory=ModelManagementConfig.from_env, description="The model management configuration settings."
    )

    gpu_config: GPUConfig = Field(default_factory=GPUConfig.from_env, description="The GPU configuration settings.")

    kafka_config: KafkaConfig = Field(
        default_factory=KafkaConfig.from_env, description="The Kafka configuration settings."
    )

    monitoring_config: MonitoringConfig = Field(
        default_factory=MonitoringConfig.from_env, description="The monitoring configuration settings."
    )
