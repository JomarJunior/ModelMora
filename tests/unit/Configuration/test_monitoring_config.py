import os
from unittest.mock import patch

import pytest

from modelmora.configuration.monitoring_config import MonitoringConfig


class TestMonitoringConfig:
    """Test cases for MonitoringConfig model."""

    def test_initialize_with_default_values_should_set_correct_defaults(self):
        """Test that MonitoringConfig initializes with correct default values."""
        # Arrange & Act
        config = MonitoringConfig()

        # Assert
        assert config.prometheus_port == 9090
        assert config.log_level == "INFO"
        assert config.enable_tracing is False
        assert config.config_name == "ModelMoraMonitoring"

    def test_initialize_with_custom_values_should_set_correct_values(self):
        """Test that MonitoringConfig initializes with custom values correctly."""
        # Arrange
        custom_port = 9091
        custom_level = "DEBUG"
        custom_tracing = True

        # Act
        config = MonitoringConfig(prometheus_port=custom_port, log_level=custom_level, enable_tracing=custom_tracing)

        # Assert
        assert config.prometheus_port == custom_port
        assert config.log_level == custom_level
        assert config.enable_tracing == custom_tracing

    @patch.dict(
        os.environ,
        {
            "MODELMORAMONITORING_PROMETHEUS_PORT": "8080",
            "MODELMORAMONITORING_LOG_LEVEL": "ERROR",
            "MODELMORAMONITORING_ENABLE_TRACING": "true",
        },
    )
    def test_from_env_with_all_environment_variables_should_set_correct_values(self):
        """Test that from_env creates instance with environment variable values."""
        # Act
        config = MonitoringConfig.from_env()

        # Assert
        assert config.prometheus_port == 8080
        assert config.log_level == "ERROR"
        assert config.enable_tracing is True

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_with_no_environment_variables_should_use_defaults(self):
        """Test that from_env uses default values when no environment variables set."""
        # Act
        config = MonitoringConfig.from_env()

        # Assert
        assert config.prometheus_port == 9090
        assert config.log_level == "INFO"
        assert config.enable_tracing is False

    def test_initialize_with_valid_log_levels_should_accept_values(self):
        """Test that initialization accepts various valid log level strings."""
        # Arrange
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        # Act & Assert
        for level in valid_levels:
            config = MonitoringConfig(log_level=level)
            assert config.log_level == level

    def test_initialize_with_lowercase_log_level_should_accept_value(self):
        """Test that initialization accepts lowercase log level strings."""
        # Act
        config = MonitoringConfig(log_level="debug")

        # Assert
        assert config.log_level == "debug"

    def test_initialize_with_negative_port_should_raise_validation_error(self):
        """Test that initialization raises validation error with negative port."""
        # Act & Assert
        with pytest.raises(ValueError):
            MonitoringConfig(prometheus_port=-1)

    def test_initialize_with_zero_port_should_accept_value(self):
        """Test that initialization accepts zero as valid port (OS assigns port)."""
        # Act
        config = MonitoringConfig(prometheus_port=0)

        # Assert
        assert config.prometheus_port == 0

    def test_initialize_with_port_above_65535_should_raise_validation_error(self):
        """Test that initialization raises validation error with port above valid range."""
        # Act & Assert
        with pytest.raises(ValueError):
            MonitoringConfig(prometheus_port=65536)

    @patch.dict(os.environ, {"MODELMORAMONITORING_PROMETHEUS_PORT": "not_a_number"})
    def test_from_env_with_invalid_port_should_raise_validation_error(self):
        """Test that from_env raises validation error with non-numeric port."""
        # Act & Assert
        with pytest.raises(ValueError):
            MonitoringConfig.from_env()

    @patch.dict(os.environ, {"MODELMORAMONITORING_ENABLE_TRACING": "False"})
    def test_from_env_with_string_false_should_disable_tracing(self):
        """Test that from_env correctly parses string 'False' as boolean."""
        # Act
        config = MonitoringConfig.from_env()

        # Assert
        assert config.enable_tracing is False

    @patch.dict(os.environ, {"MODELMORAMONITORING_ENABLE_TRACING": "1"})
    def test_from_env_with_numeric_one_should_enable_tracing(self):
        """Test that from_env correctly parses numeric 1 as boolean True."""
        # Act
        config = MonitoringConfig.from_env()

        # Assert
        assert config.enable_tracing is True

    def test_initialize_with_empty_log_level_should_accept_value(self):
        """Test that initialization accepts empty log level string."""
        # Act
        config = MonitoringConfig(log_level="")

        # Assert
        assert config.log_level == ""

    def test_str_representation_should_include_monitoring_settings(self):
        """Test that __str__ includes monitoring configuration details."""
        # Arrange
        config = MonitoringConfig(log_level="DEBUG", enable_tracing=True)

        # Act
        result = str(config)

        # Assert
        assert "ModelMoraMonitoring Configuration:" in result
        assert "prometheus_port: 9090" in result
        assert "log_level: DEBUG" in result
        assert "enable_tracing: True" in result
