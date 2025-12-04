import os
from unittest.mock import patch

import pytest

from modelmora.configuration.model_management_config import ModelManagementConfig


class TestModelManagementConfig:
    """Test cases for ModelManagementConfig model."""

    def test_initialize_with_default_values_should_set_correct_defaults(self):
        """Test that ModelManagementConfig initializes with correct default values."""
        # Arrange & Act
        config = ModelManagementConfig()

        # Assert
        assert config.model_registry_path == "/app/configuration/ModelRegistry.yaml"
        assert config.model_cache_dir == "/models"
        assert config.max_model_memory_mb == 8192
        assert config.auto_warmup is True
        assert config.config_name == "ModelMoraModelManagement"

    def test_initialize_with_custom_values_should_set_correct_values(self):
        """Test that ModelManagementConfig initializes with custom values correctly."""
        # Arrange
        custom_registry = "/custom/path/registry.yaml"
        custom_cache = "/custom/cache"
        custom_memory = 16384
        custom_warmup = False

        # Act
        config = ModelManagementConfig(
            model_registry_path=custom_registry,
            model_cache_dir=custom_cache,
            max_model_memory_mb=custom_memory,
            auto_warmup=custom_warmup,
        )

        # Assert
        assert config.model_registry_path == custom_registry
        assert config.model_cache_dir == custom_cache
        assert config.max_model_memory_mb == custom_memory
        assert config.auto_warmup == custom_warmup

    @patch.dict(
        os.environ,
        {
            "MODELMORAMODELMANAGEMENT_MODEL_REGISTRY_PATH": "/env/registry.yaml",
            "MODELMORAMODELMANAGEMENT_MODEL_CACHE_DIR": "/env/cache",
            "MODELMORAMODELMANAGEMENT_MAX_MODEL_MEMORY_MB": "4096",
            "MODELMORAMODELMANAGEMENT_AUTO_WARMUP": "false",
        },
    )
    def test_from_env_with_all_environment_variables_should_set_correct_values(self):
        """Test that from_env creates instance with environment variable values."""
        # Act
        config = ModelManagementConfig.from_env()

        # Assert
        assert config.model_registry_path == "/env/registry.yaml"
        assert config.model_cache_dir == "/env/cache"
        assert config.max_model_memory_mb == 4096
        assert config.auto_warmup is False

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_with_no_environment_variables_should_use_defaults(self):
        """Test that from_env uses default values when no environment variables set."""
        # Act
        config = ModelManagementConfig.from_env()

        # Assert
        assert config.model_registry_path == "/app/configuration/ModelRegistry.yaml"
        assert config.model_cache_dir == "/models"
        assert config.max_model_memory_mb == 8192
        assert config.auto_warmup is True

    def test_initialize_with_negative_memory_should_raise_validation_error(self):
        """Test that initialization raises validation error with negative memory."""
        # Act & Assert
        with pytest.raises(ValueError):
            ModelManagementConfig(max_model_memory_mb=-1)

    def test_initialize_with_zero_memory_should_raise_validation_error(self):
        """Test that initialization raises validation error with zero memory."""
        # Act & Assert
        with pytest.raises(ValueError):
            ModelManagementConfig(max_model_memory_mb=0)

    def test_initialize_with_minimum_valid_memory_should_accept_value(self):
        """Test that initialization accepts minimum valid memory value."""
        # Arrange
        min_memory = 1

        # Act
        config = ModelManagementConfig(max_model_memory_mb=min_memory)

        # Assert
        assert config.max_model_memory_mb == min_memory

    def test_initialize_with_large_memory_value_should_accept_value(self):
        """Test that initialization accepts large memory values."""
        # Arrange
        large_memory = 65536  # 64GB

        # Act
        config = ModelManagementConfig(max_model_memory_mb=large_memory)

        # Assert
        assert config.max_model_memory_mb == large_memory

    @patch.dict(os.environ, {"MODELMORAMODELMANAGEMENT_MAX_MODEL_MEMORY_MB": "not_a_number"})
    def test_from_env_with_invalid_memory_value_should_raise_validation_error(self):
        """Test that from_env raises validation error with non-numeric memory."""
        # Act & Assert
        with pytest.raises(ValueError):
            ModelManagementConfig.from_env()

    def test_initialize_with_empty_registry_path_should_accept_value(self):
        """Test that initialization accepts empty registry path string."""
        # Act
        config = ModelManagementConfig(model_registry_path="")

        # Assert
        assert config.model_registry_path == ""

    def test_initialize_with_relative_paths_should_accept_values(self):
        """Test that initialization accepts relative paths."""
        # Arrange
        relative_registry = "./config/registry.yaml"
        relative_cache = "./cache/models"

        # Act
        config = ModelManagementConfig(model_registry_path=relative_registry, model_cache_dir=relative_cache)

        # Assert
        assert config.model_registry_path == relative_registry
        assert config.model_cache_dir == relative_cache

    @patch.dict(os.environ, {"MODELMORAMODELMANAGEMENT_AUTO_WARMUP": "1"})
    def test_from_env_with_numeric_boolean_should_enable_warmup(self):
        """Test that from_env correctly parses numeric 1 as boolean True."""
        # Act
        config = ModelManagementConfig.from_env()

        # Assert
        assert config.auto_warmup is True

    def test_str_representation_should_include_management_settings(self):
        """Test that __str__ includes model management configuration details."""
        # Arrange
        config = ModelManagementConfig()

        # Act
        result = str(config)

        # Assert
        assert "ModelMoraModelManagement Configuration:" in result
        assert "model_registry_path:" in result
        assert "model_cache_dir:" in result
        assert "max_model_memory_mb: 8192" in result
        assert "auto_warmup: True" in result
