import os
from unittest.mock import patch

import pytest

from modelmora.configuration.gpu_config import GPUConfig


class TestGPUConfig:
    """Test cases for GPUConfig model."""

    def test_initialize_with_default_values_should_set_correct_defaults(self):
        """Test that GPUConfig initializes with correct default values."""
        # Arrange & Act
        config = GPUConfig()

        # Assert
        assert config.gpu_enabled is False
        assert config.gpu_device_id == 0
        assert config.config_name == "ModelMoraGPU"

    def test_initialize_with_custom_values_should_set_correct_values(self):
        """Test that GPUConfig initializes with custom values correctly."""
        # Arrange
        custom_enabled = True
        custom_device_id = 2

        # Act
        config = GPUConfig(gpu_enabled=custom_enabled, gpu_device_id=custom_device_id)

        # Assert
        assert config.gpu_enabled == custom_enabled
        assert config.gpu_device_id == custom_device_id

    @patch.dict(
        os.environ,
        {
            "MODELMORAGPU_GPU_ENABLED": "true",
            "MODELMORAGPU_GPU_DEVICE_ID": "3",
        },
    )
    def test_from_env_with_all_environment_variables_should_set_correct_values(self):
        """Test that from_env creates instance with environment variable values."""
        # Act
        config = GPUConfig.from_env()

        # Assert
        assert config.gpu_enabled is True
        assert config.gpu_device_id == 3

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_with_no_environment_variables_should_use_defaults(self):
        """Test that from_env uses default values when no environment variables set."""
        # Act
        config = GPUConfig.from_env()

        # Assert
        assert config.gpu_enabled is False
        assert config.gpu_device_id == 0

    @patch.dict(os.environ, {"MODELMORAGPU_GPU_ENABLED": "True"})
    def test_from_env_with_string_true_should_enable_gpu(self):
        """Test that from_env correctly parses string 'True' as boolean."""
        # Act
        config = GPUConfig.from_env()

        # Assert
        assert config.gpu_enabled is True

    @patch.dict(os.environ, {"MODELMORAGPU_GPU_ENABLED": "false"})
    def test_from_env_with_string_false_should_disable_gpu(self):
        """Test that from_env correctly parses string 'false' as boolean."""
        # Act
        config = GPUConfig.from_env()

        # Assert
        assert config.gpu_enabled is False

    def test_initialize_with_negative_device_id_should_raise_validation_error(self):
        """Test that initialization raises validation error with negative device ID."""
        # Act & Assert
        with pytest.raises(ValueError):
            GPUConfig(gpu_device_id=-1)

    def test_initialize_with_large_device_id_should_accept_value(self):
        """Test that initialization accepts large device ID values."""
        # Arrange
        large_device_id = 7

        # Act
        config = GPUConfig(gpu_device_id=large_device_id)

        # Assert
        assert config.gpu_device_id == large_device_id

    @patch.dict(os.environ, {"MODELMORAGPU_GPU_DEVICE_ID": "not_a_number"})
    def test_from_env_with_invalid_device_id_should_raise_validation_error(self):
        """Test that from_env raises validation error with non-numeric device ID."""
        # Act & Assert
        with pytest.raises(ValueError):
            GPUConfig.from_env()

    def test_str_representation_should_include_gpu_settings(self):
        """Test that __str__ includes GPU configuration details."""
        # Arrange
        config = GPUConfig(gpu_enabled=True, gpu_device_id=1)

        # Act
        result = str(config)

        # Assert
        assert "ModelMoraGPU Configuration:" in result
        assert "gpu_enabled: True" in result
        assert "gpu_device_id: 1" in result
