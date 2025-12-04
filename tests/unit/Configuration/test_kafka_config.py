import os
from unittest.mock import patch

import pytest

from modelmora.configuration.kafka_config import KafkaConfig


class TestKafkaConfig:
    """Test cases for KafkaConfig model."""

    def test_initialize_with_default_values_should_set_correct_defaults(self):
        """Test that KafkaConfig initializes with correct default values."""
        # Arrange & Act
        config = KafkaConfig()

        # Assert
        assert config.kafka_bootstrap_servers == "localhost:9092"
        assert config.kafka_consumer_group == "modelmora_consumer"
        assert config.subscribe_to_events == []
        assert config.config_name == "ModelMoraKafka"

    def test_initialize_with_custom_values_should_set_correct_values(self):
        """Test that KafkaConfig initializes with custom values correctly."""
        # Arrange
        custom_servers = "kafka1:9092,kafka2:9092"
        custom_group = "custom_group"
        custom_events = ["event1", "event2"]

        # Act
        config = KafkaConfig(
            kafka_bootstrap_servers=custom_servers,
            kafka_consumer_group=custom_group,
            subscribe_to_events=custom_events,
        )

        # Assert
        assert config.kafka_bootstrap_servers == custom_servers
        assert config.kafka_consumer_group == custom_group
        assert config.subscribe_to_events == custom_events

    @patch.dict(
        os.environ,
        {
            "MODELMORAKAFKA_KAFKA_BOOTSTRAP_SERVERS": "prod-kafka:9092",
            "MODELMORAKAFKA_KAFKA_CONSUMER_GROUP": "prod_consumer",
            "MODELMORAKAFKA_SUBSCRIBE_TO_EVENTS": '["image.metadata.registered", "model.loaded"]',
        },
    )
    def test_from_env_with_all_environment_variables_should_set_correct_values(self):
        """Test that from_env creates instance with environment variable values."""
        # Act
        config = KafkaConfig.from_env()

        # Assert
        assert config.kafka_bootstrap_servers == "prod-kafka:9092"
        assert config.kafka_consumer_group == "prod_consumer"

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_with_no_environment_variables_should_use_defaults(self):
        """Test that from_env uses default values when no environment variables set."""
        # Act
        config = KafkaConfig.from_env()

        # Assert
        assert config.kafka_bootstrap_servers == "localhost:9092"
        assert config.kafka_consumer_group == "modelmora_consumer"
        assert config.subscribe_to_events == []

    def test_initialize_with_empty_bootstrap_servers_should_accept_value(self):
        """Test that initialization accepts empty bootstrap servers string."""
        # Act
        config = KafkaConfig(kafka_bootstrap_servers="")

        # Assert
        assert config.kafka_bootstrap_servers == ""

    def test_initialize_with_multiple_bootstrap_servers_should_accept_value(self):
        """Test that initialization accepts multiple comma-separated servers."""
        # Arrange
        servers = "kafka1:9092,kafka2:9093,kafka3:9094"

        # Act
        config = KafkaConfig(kafka_bootstrap_servers=servers)

        # Assert
        assert config.kafka_bootstrap_servers == servers

    def test_initialize_with_empty_consumer_group_should_accept_value(self):
        """Test that initialization accepts empty consumer group string."""
        # Act
        config = KafkaConfig(kafka_consumer_group="")

        # Assert
        assert config.kafka_consumer_group == ""

    def test_initialize_with_single_event_should_create_list(self):
        """Test that initialization with single event creates proper list."""
        # Arrange
        single_event = ["image.uploaded"]

        # Act
        config = KafkaConfig(subscribe_to_events=single_event)

        # Assert
        assert len(config.subscribe_to_events) == 1
        assert config.subscribe_to_events[0] == "image.uploaded"

    def test_initialize_with_multiple_events_should_preserve_order(self):
        """Test that initialization with multiple events preserves order."""
        # Arrange
        events = ["event1", "event2", "event3"]

        # Act
        config = KafkaConfig(subscribe_to_events=events)

        # Assert
        assert len(config.subscribe_to_events) == 3
        assert config.subscribe_to_events == events

    def test_str_representation_should_include_kafka_settings(self):
        """Test that __str__ includes Kafka configuration details."""
        # Arrange
        config = KafkaConfig(subscribe_to_events=["event1", "event2"])

        # Act
        result = str(config)

        # Assert
        assert "ModelMoraKafka Configuration:" in result
        assert "kafka_bootstrap_servers: localhost:9092" in result
        assert "kafka_consumer_group: modelmora_consumer" in result
        assert "subscribe_to_events:" in result
