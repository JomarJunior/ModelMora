import os
from unittest.mock import patch

import pytest

from modelmora.configuration.connection_config import ConnectionConfig


class TestConnectionConfig:
    """Test cases for ConnectionConfig model."""

    def test_initialize_with_default_values_should_set_correct_defaults(self):
        """Test that ConnectionConfig initializes with correct default values."""
        # Arrange & Act
        config = ConnectionConfig()

        # Assert
        assert config.grpc_port == 50051
        assert config.http_port == 8080
        assert config.config_name == "ModelMoraConnection"

    def test_initialize_with_custom_values_should_set_correct_values(self):
        """Test that ConnectionConfig initializes with custom values correctly."""
        # Arrange
        custom_grpc_port = 9090
        custom_http_port = 3000

        # Act
        config = ConnectionConfig(grpc_port=custom_grpc_port, http_port=custom_http_port)

        # Assert
        assert config.grpc_port == custom_grpc_port
        assert config.http_port == custom_http_port

    @patch.dict(
        os.environ,
        {
            "MODELMORACONNECTION_GRPC_PORT": "60051",
            "MODELMORACONNECTION_HTTP_PORT": "9000",
        },
    )
    def test_from_env_with_all_environment_variables_should_set_correct_values(self):
        """Test that from_env creates instance with environment variable values."""
        # Act
        config = ConnectionConfig.from_env()

        # Assert
        assert config.grpc_port == 60051
        assert config.http_port == 9000

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_with_no_environment_variables_should_use_defaults(self):
        """Test that from_env uses default values when no environment variables set."""
        # Act
        config = ConnectionConfig.from_env()

        # Assert
        assert config.grpc_port == 50051
        assert config.http_port == 8080

    @patch.dict(os.environ, {"MODELMORACONNECTION_GRPC_PORT": "65536"})
    def test_from_env_with_invalid_port_number_should_raise_validation_error(self):
        """Test that from_env raises validation error with port out of valid range."""
        # Act & Assert
        with pytest.raises(ValueError):
            ConnectionConfig.from_env()

    def test_initialize_with_negative_port_should_raise_validation_error(self):
        """Test that initialization raises validation error with negative port."""
        # Act & Assert
        with pytest.raises(ValueError):
            ConnectionConfig(grpc_port=-1)

    def test_initialize_with_zero_port_should_accept_value(self):
        """Test that initialization accepts zero as valid port (OS assigns port)."""
        # Act
        config = ConnectionConfig(grpc_port=0, http_port=0)

        # Assert
        assert config.grpc_port == 0
        assert config.http_port == 0

    def test_str_representation_should_include_port_values(self):
        """Test that __str__ includes connection configuration details."""
        # Arrange
        config = ConnectionConfig()

        # Act
        result = str(config)

        # Assert
        assert "ModelMoraConnection Configuration:" in result
        assert "grpc_port: 50051" in result
        assert "http_port: 8080" in result
