import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from modelmora.configuration.base_config import BaseConfig


class TestBaseConfig:
    """Test cases for BaseConfig base class."""

    def test_initialize_with_default_values_should_create_instance(self):
        """Test that BaseConfig initializes with default values."""
        # Arrange & Act
        config = BaseConfig()

        # Assert
        assert isinstance(config, BaseConfig)
        assert config.config_name == "BaseConfig"

    def test_from_env_with_no_environment_variables_should_create_default_instance(self):
        """Test that from_env creates instance with defaults when no env vars set."""
        # Arrange
        with patch.dict(os.environ, {}, clear=True):
            # Act
            config = BaseConfig.from_env()

            # Assert
            assert isinstance(config, BaseConfig)

    def test_from_yaml_with_valid_file_should_load_configuration(self):
        """Test that from_yaml loads configuration from valid YAML file."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
            yaml_content = {}
            yaml.dump(yaml_content, temp_file)
            temp_file_path = temp_file.name

        try:
            # Act
            config = BaseConfig.from_yaml(temp_file_path)

            # Assert
            assert isinstance(config, BaseConfig)
        finally:
            # Cleanup
            Path(temp_file_path).unlink(missing_ok=True)

    def test_from_yaml_with_nonexistent_file_should_raise_file_not_found_error(self):
        """Test that from_yaml raises FileNotFoundError with nonexistent file."""
        # Arrange
        nonexistent_path = "/nonexistent/path/config.yaml"

        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            BaseConfig.from_yaml(nonexistent_path)

        assert "Configuration file not found" in str(exc_info.value)
        assert nonexistent_path in str(exc_info.value)

    def test_from_yaml_with_empty_file_should_raise_value_error(self):
        """Test that from_yaml raises ValueError with empty YAML file."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
            temp_file.write("")
            temp_file_path = temp_file.name

        try:
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                BaseConfig.from_yaml(temp_file_path)

            assert "empty or invalid" in str(exc_info.value)
        finally:
            # Cleanup
            Path(temp_file_path).unlink(missing_ok=True)

    def test_from_yaml_with_invalid_yaml_should_raise_value_error(self):
        """Test that from_yaml raises ValueError with malformed YAML."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
            temp_file.write("invalid: yaml: content: [\n")
            temp_file_path = temp_file.name

        try:
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                BaseConfig.from_yaml(temp_file_path)

            assert "Error parsing YAML file" in str(exc_info.value)
        finally:
            # Cleanup
            Path(temp_file_path).unlink(missing_ok=True)

    def test_from_json_with_valid_file_should_load_configuration(self):
        """Test that from_json loads configuration from valid JSON file."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            json_content = {}
            json.dump(json_content, temp_file)
            temp_file_path = temp_file.name

        try:
            # Act
            config = BaseConfig.from_json(temp_file_path)

            # Assert
            assert isinstance(config, BaseConfig)
        finally:
            # Cleanup
            Path(temp_file_path).unlink(missing_ok=True)

    def test_from_json_with_nonexistent_file_should_raise_file_not_found_error(self):
        """Test that from_json raises FileNotFoundError with nonexistent file."""
        # Arrange
        nonexistent_path = "/nonexistent/path/config.json"

        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            BaseConfig.from_json(nonexistent_path)

        assert "Configuration file not found" in str(exc_info.value)
        assert nonexistent_path in str(exc_info.value)

    def test_from_json_with_invalid_json_should_raise_value_error(self):
        """Test that from_json raises ValueError with malformed JSON."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            temp_file.write("{invalid json content")
            temp_file_path = temp_file.name

        try:
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                BaseConfig.from_json(temp_file_path)

            assert "Error parsing JSON file" in str(exc_info.value)
        finally:
            # Cleanup
            Path(temp_file_path).unlink(missing_ok=True)

    def test_str_representation_should_return_readable_format(self):
        """Test that __str__ returns readable configuration string."""
        # Arrange
        config = BaseConfig()

        # Act
        result = str(config)

        # Assert
        assert "BaseConfig Configuration:" in result
        assert isinstance(result, str)
        assert len(result) > 0
