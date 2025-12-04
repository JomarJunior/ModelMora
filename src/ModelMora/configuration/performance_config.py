from typing import ClassVar

from pydantic import Field

from modelmora.configuration.base_config import BaseConfig


class PerformanceConfig(BaseConfig):
    """Performance configuration settings."""

    config_name: ClassVar[str] = "ModelMoraPerformance"

    max_concurrent_requests: int = Field(
        default=100, gt=0, description="Maximum number of concurrent requests allowed."
    )
    request_timeout_seconds: int = Field(default=30, gt=0, description="Timeout duration for requests in seconds.")
    batch_size: int = Field(default=32, gt=0, description="Number of items to process in a single batch.")
    num_workers: int = Field(default=4, gt=0, description="Number of worker threads for processing.")
