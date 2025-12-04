from typing import ClassVar

from pydantic import Field

from modelmora.configuration.base_config import BaseConfig


class ConnectionConfig(BaseConfig):
    """Server ports configuration settings."""

    config_name: ClassVar[str] = "ModelMoraConnection"

    grpc_port: int = Field(default=50051, ge=0, le=65535, description="Port number for the gRPC server.")
    http_port: int = Field(default=8080, ge=0, le=65535, description="Port number for the HTTP API server.")
    http_host: str = Field(default="0.0.0.0", description="Host address for the HTTP API server.")
