import os
from unittest.mock import patch

import pytest

from modelmora.configuration.performance_config import PerformanceConfig


class TestPerformanceConfig:
    """Test cases for PerformanceConfig model."""

    def test_initialize_with_default_values_should_set_correct_defaults(self):
        """Test that PerformanceConfig initializes with correct default values."""
        # Arrange & Act
        config = PerformanceConfig()

        # Assert
        assert config.max_concurrent_requests == 100
        assert config.request_timeout_seconds == 30
        assert config.batch_size == 32
        assert config.num_workers == 4
        assert config.config_name == "ModelMoraPerformance"

    def test_initialize_with_custom_values_should_set_correct_values(self):
        """Test that PerformanceConfig initializes with custom values correctly."""
        # Arrange
        custom_requests = 200
        custom_timeout = 60
        custom_batch = 64
        custom_workers = 8

        # Act
        config = PerformanceConfig(
            max_concurrent_requests=custom_requests,
            request_timeout_seconds=custom_timeout,
            batch_size=custom_batch,
            num_workers=custom_workers,
        )

        # Assert
        assert config.max_concurrent_requests == custom_requests
        assert config.request_timeout_seconds == custom_timeout
        assert config.batch_size == custom_batch
        assert config.num_workers == custom_workers

    @patch.dict(
        os.environ,
        {
            "MODELMORAPERFORMANCE_MAX_CONCURRENT_REQUESTS": "500",
            "MODELMORAPERFORMANCE_REQUEST_TIMEOUT_SECONDS": "120",
            "MODELMORAPERFORMANCE_BATCH_SIZE": "128",
            "MODELMORAPERFORMANCE_NUM_WORKERS": "16",
        },
    )
    def test_from_env_with_all_environment_variables_should_set_correct_values(self):
        """Test that from_env creates instance with environment variable values."""
        # Act
        config = PerformanceConfig.from_env()

        # Assert
        assert config.max_concurrent_requests == 500
        assert config.request_timeout_seconds == 120
        assert config.batch_size == 128
        assert config.num_workers == 16

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_with_no_environment_variables_should_use_defaults(self):
        """Test that from_env uses default values when no environment variables set."""
        # Act
        config = PerformanceConfig.from_env()

        # Assert
        assert config.max_concurrent_requests == 100
        assert config.request_timeout_seconds == 30
        assert config.batch_size == 32
        assert config.num_workers == 4

    def test_initialize_with_negative_concurrent_requests_should_raise_validation_error(self):
        """Test that initialization raises validation error with negative concurrent requests."""
        # Act & Assert
        with pytest.raises(ValueError):
            PerformanceConfig(max_concurrent_requests=-1)

    def test_initialize_with_zero_concurrent_requests_should_raise_validation_error(self):
        """Test that initialization raises validation error with zero concurrent requests."""
        # Act & Assert
        with pytest.raises(ValueError):
            PerformanceConfig(max_concurrent_requests=0)

    def test_initialize_with_minimum_valid_concurrent_requests_should_accept_value(self):
        """Test that initialization accepts minimum valid concurrent requests."""
        # Act
        config = PerformanceConfig(max_concurrent_requests=1)

        # Assert
        assert config.max_concurrent_requests == 1

    def test_initialize_with_negative_timeout_should_raise_validation_error(self):
        """Test that initialization raises validation error with negative timeout."""
        # Act & Assert
        with pytest.raises(ValueError):
            PerformanceConfig(request_timeout_seconds=-1)

    def test_initialize_with_zero_timeout_should_raise_validation_error(self):
        """Test that initialization raises validation error with zero timeout."""
        # Act & Assert
        with pytest.raises(ValueError):
            PerformanceConfig(request_timeout_seconds=0)

    def test_initialize_with_large_timeout_should_accept_value(self):
        """Test that initialization accepts large timeout values."""
        # Arrange
        large_timeout = 3600  # 1 hour

        # Act
        config = PerformanceConfig(request_timeout_seconds=large_timeout)

        # Assert
        assert config.request_timeout_seconds == large_timeout

    def test_initialize_with_negative_batch_size_should_raise_validation_error(self):
        """Test that initialization raises validation error with negative batch size."""
        # Act & Assert
        with pytest.raises(ValueError):
            PerformanceConfig(batch_size=-1)

    def test_initialize_with_zero_batch_size_should_raise_validation_error(self):
        """Test that initialization raises validation error with zero batch size."""
        # Act & Assert
        with pytest.raises(ValueError):
            PerformanceConfig(batch_size=0)

    def test_initialize_with_small_batch_size_should_accept_value(self):
        """Test that initialization accepts small batch size values."""
        # Act
        config = PerformanceConfig(batch_size=1)

        # Assert
        assert config.batch_size == 1

    def test_initialize_with_large_batch_size_should_accept_value(self):
        """Test that initialization accepts large batch size values."""
        # Arrange
        large_batch = 512

        # Act
        config = PerformanceConfig(batch_size=large_batch)

        # Assert
        assert config.batch_size == large_batch

    def test_initialize_with_negative_num_workers_should_raise_validation_error(self):
        """Test that initialization raises validation error with negative workers."""
        # Act & Assert
        with pytest.raises(ValueError):
            PerformanceConfig(num_workers=-1)

    def test_initialize_with_zero_num_workers_should_raise_validation_error(self):
        """Test that initialization raises validation error with zero workers."""
        # Act & Assert
        with pytest.raises(ValueError):
            PerformanceConfig(num_workers=0)

    def test_initialize_with_single_worker_should_accept_value(self):
        """Test that initialization accepts single worker configuration."""
        # Act
        config = PerformanceConfig(num_workers=1)

        # Assert
        assert config.num_workers == 1

    @patch.dict(os.environ, {"MODELMORAPERFORMANCE_MAX_CONCURRENT_REQUESTS": "not_a_number"})
    def test_from_env_with_invalid_concurrent_requests_should_raise_validation_error(self):
        """Test that from_env raises validation error with non-numeric concurrent requests."""
        # Act & Assert
        with pytest.raises(ValueError):
            PerformanceConfig.from_env()

    @patch.dict(os.environ, {"MODELMORAPERFORMANCE_NUM_WORKERS": "invalid"})
    def test_from_env_with_invalid_num_workers_should_raise_validation_error(self):
        """Test that from_env raises validation error with non-numeric workers."""
        # Act & Assert
        with pytest.raises(ValueError):
            PerformanceConfig.from_env()

    def test_str_representation_should_include_performance_settings(self):
        """Test that __str__ includes performance configuration details."""
        # Arrange
        config = PerformanceConfig()

        # Act
        result = str(config)

        # Assert
        assert "ModelMoraPerformance Configuration:" in result
        assert "max_concurrent_requests: 100" in result
        assert "request_timeout_seconds: 30" in result
        assert "batch_size: 32" in result
        assert "num_workers: 4" in result
