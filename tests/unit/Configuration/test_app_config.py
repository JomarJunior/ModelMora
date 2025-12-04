import os
from unittest.mock import MagicMock, patch

import pytest

from modelmora.configuration.app_config import AppConfig


class TestAppConfig:
    """Test cases for AppConfig application configuration."""

    def test_initialize_with_default_values_should_set_correct_defaults(self):
        """Test that AppConfig initializes with correct default values."""
        # Arrange & Act
        config = AppConfig()

        # Assert
        assert config.app_name == "ModelMora"
        assert config.app_version == "0.1.0"
        assert config.debug_mode is False
        assert config.config_name == "ModelMora"
        assert config.connection_config is not None
        assert config.performance_config is not None
        assert config.model_management_config is not None
        assert config.gpu_config is not None
        assert config.kafka_config is not None

    def test_initialize_with_custom_values_should_set_correct_values(self):
        """Test that AppConfig initializes with custom values correctly."""
        # Arrange
        custom_name = "CustomApp"
        custom_version = "1.0.0"
        custom_debug = True

        # Act
        config = AppConfig(app_name=custom_name, app_version=custom_version, debug_mode=custom_debug)

        # Assert
        assert config.app_name == custom_name
        assert config.app_version == custom_version
        assert config.debug_mode == custom_debug

    @patch.dict(
        os.environ,
        {
            "MODELMORA_APP_NAME": "EnvApp",
            "MODELMORA_APP_VERSION": "2.0.0",
            "MODELMORA_DEBUG_MODE": "true",
        },
        clear=True,
    )
    def test_from_env_with_app_environment_variables_should_set_correct_values(self):
        """Test that from_env creates instance with application environment variables."""
        # Act
        config = AppConfig.from_env()

        # Assert
        assert config.app_name == "EnvApp"
        assert config.app_version == "2.0.0"
        assert config.debug_mode is True

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_with_no_environment_variables_should_use_defaults(self):
        """Test that from_env uses default values when no environment variables set."""
        # Act
        config = AppConfig.from_env()

        # Assert
        assert config.app_name == "ModelMora"
        assert config.app_version == "0.1.0"
        assert config.debug_mode is False

    def test_connection_config_should_have_default_ports(self):
        """Test that nested ConnectionConfig has correct default values."""
        # Act
        config = AppConfig()

        # Assert
        assert config.connection_config.grpc_port == 50051
        assert config.connection_config.http_port == 8080

    def test_performance_config_should_have_default_values(self):
        """Test that nested PerformanceConfig has correct default values."""
        # Act
        config = AppConfig()

        # Assert
        assert config.performance_config.max_concurrent_requests == 100
        assert config.performance_config.request_timeout_seconds == 30
        assert config.performance_config.batch_size == 32
        assert config.performance_config.num_workers == 4

    def test_model_management_config_should_have_default_values(self):
        """Test that nested ModelManagementConfig has correct default values."""
        # Act
        config = AppConfig()

        # Assert
        assert config.model_management_config.model_registry_path == "/app/configuration/ModelRegistry.yaml"
        assert config.model_management_config.model_cache_dir == "/models"
        assert config.model_management_config.max_model_memory_mb == 8192
        assert config.model_management_config.auto_warmup is True

    def test_gpu_config_should_have_default_values(self):
        """Test that nested GPUConfig has correct default values."""
        # Act
        config = AppConfig()

        # Assert
        assert config.gpu_config.gpu_enabled is False
        assert config.gpu_config.gpu_device_id == 0

    def test_kafka_config_should_have_default_values(self):
        """Test that nested KafkaConfig has correct default values."""
        # Act
        config = AppConfig()

        # Assert
        assert config.kafka_config.kafka_bootstrap_servers == "localhost:9092"
        assert config.kafka_config.kafka_consumer_group == "modelmora_consumer"
        assert config.kafka_config.subscribe_to_events == []

    @patch.dict(
        os.environ,
        {
            "MODELMORACONNECTION_GRPC_PORT": "60051",
            "MODELMORAPERFORMANCE_MAX_CONCURRENT_REQUESTS": "200",
        },
        clear=True,
    )
    def test_from_env_with_nested_config_environment_variables_should_propagate(self):
        """Test that from_env propagates environment variables to nested configs."""
        # Act
        config = AppConfig.from_env()

        # Assert
        assert config.connection_config.grpc_port == 60051
        assert config.performance_config.max_concurrent_requests == 200

    def test_initialize_with_empty_app_name_should_accept_value(self):
        """Test that initialization accepts empty app name string."""
        # Act
        config = AppConfig(app_name="")

        # Assert
        assert config.app_name == ""

    def test_initialize_with_empty_version_should_accept_value(self):
        """Test that initialization accepts empty version string."""
        # Act
        config = AppConfig(app_version="")

        # Assert
        assert config.app_version == ""

    @patch.dict(os.environ, {"MODELMORA_DEBUG_MODE": "false"})
    def test_from_env_with_string_false_should_disable_debug(self):
        """Test that from_env correctly parses string 'false' as boolean."""
        # Act
        config = AppConfig.from_env()

        # Assert
        assert config.debug_mode is False

    @patch.dict(os.environ, {"MODELMORA_DEBUG_MODE": "1"})
    def test_from_env_with_numeric_one_should_enable_debug(self):
        """Test that from_env correctly parses numeric 1 as boolean True."""
        # Act
        config = AppConfig.from_env()

        # Assert
        assert config.debug_mode is True

    def test_str_representation_should_include_all_configurations(self):
        """Test that __str__ includes all nested configuration details."""
        # Arrange
        config = AppConfig()

        # Act
        result = str(config)

        # Assert
        assert "ModelMora Configuration:" in result
        assert "app_name: ModelMora" in result
        assert "app_version: 0.1.0" in result
        assert "debug_mode: False" in result
        assert "connection_config:" in result
        assert "performance_config:" in result
        assert "model_management_config:" in result
        assert "gpu_config:" in result
        assert "kafka_config:" in result

    def test_nested_configs_should_be_independent_instances(self):
        """Test that nested config instances are independent."""
        # Act
        config1 = AppConfig()
        config2 = AppConfig()

        # Assert
        assert config1.connection_config is not config2.connection_config
        assert config1.performance_config is not config2.performance_config
        assert config1.model_management_config is not config2.model_management_config
        assert config1.gpu_config is not config2.gpu_config
        assert config1.kafka_config is not config2.kafka_config

    def test_modify_nested_config_should_not_affect_other_instances(self):
        """Test that modifying nested config in one instance doesn't affect others."""
        # Arrange
        config1 = AppConfig()
        config2 = AppConfig()

        # Act
        config1.connection_config.grpc_port = 60051

        # Assert
        assert config1.connection_config.grpc_port == 60051
        assert config2.connection_config.grpc_port == 50051
