from typing import ClassVar, List

from pydantic import Field

from modelmora.configuration.base_config import BaseConfig


class KafkaConfig(BaseConfig):
    """Kafka configuration settings."""

    config_name: ClassVar[str] = "ModelMoraKafka"

    kafka_bootstrap_servers: str = Field(default="localhost:9092", description="Kafka bootstrap servers address.")
    kafka_consumer_group: str = Field(default="modelmora_consumer", description="Kafka consumer group ID.")
    subscribe_to_events: List[str] = Field(default_factory=list, description="List of events to subscribe to.")
