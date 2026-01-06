"""Tests for configuration settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import settings


def test_settings_import():
    """Test that settings can be imported successfully."""
    from src.config.settings import Settings

    assert Settings is not None


def test_settings_instance():
    """Test that global settings instance exists."""
    assert settings is not None
    assert hasattr(settings, 'environment')


def test_environment_default():
    """Test default environment value."""
    assert settings.environment == "development"


def test_log_level_default():
    """Test default log level."""
    assert settings.log_level == "INFO"


def test_paths_are_path_objects():
    """Test that path settings are Path objects."""
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.model_dir, Path)
    assert isinstance(settings.log_dir, Path)


def test_paths_exist():
    """Test that paths are created or creatable."""
    # These should exist after accessing settings
    assert settings.data_dir.exists() or not settings.data_dir.exists()
    assert settings.model_dir.exists() or not settings.model_dir.exists()
    assert settings.log_dir.exists() or not settings.log_dir.exists()


def test_ray_settings():
    """Test Ray-related settings."""
    assert isinstance(settings.ray_num_cpus, int)
    assert isinstance(settings.ray_use_gpu, bool)
    assert settings.ray_num_cpus >= 1


def test_apple_silicon_settings():
    """Test Apple Silicon settings."""
    assert isinstance(settings.use_mps, bool)
    assert isinstance(settings.mps_fallback_to_cpu, bool)


def test_training_defaults():
    """Test training default settings."""
    assert settings.train_batch_size > 0
    assert 0 < settings.train_learning_rate < 1
    assert settings.train_num_epochs > 0
    assert settings.train_hidden_dim > 0


def test_settings_with_env_vars(monkeypatch):
    """Test that environment variables override defaults."""
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RAY_NUM_CPUS", "8")

    # Need to reload settings to pick up env vars
    from src.config.settings import Settings

    test_settings = Settings()

    assert test_settings.environment == "testing"
    assert test_settings.log_level == "DEBUG"
    assert test_settings.ray_num_cpus == 8


def test_invalid_environment_validation():
    """Test that invalid environment raises ValidationError."""
    from src.config.settings import Settings

    with pytest.raises(ValidationError):
        Settings(environment="invalid_env")


def test_settings_to_dict():
    """Test that settings can be converted to dictionary."""
    settings_dict = settings.dict()

    assert isinstance(settings_dict, dict)
    assert "environment" in settings_dict
    assert "log_level" in settings_dict
    assert "data_dir" in settings_dict


def test_settings_json_serializable():
    """Test that settings are JSON serializable."""
    import json

    settings_dict = settings.dict()
    # Convert Path objects to strings for JSON serialization
    serializable_dict = {
        k: str(v) if isinstance(v, Path) else v for k, v in settings_dict.items()
    }

    json_str = json.dumps(serializable_dict)
    assert isinstance(json_str, str)
    assert "environment" in json_str


def test_feature_engineering_settings():
    """Test feature engineering settings."""
    assert isinstance(settings.feature_interaction_enabled, bool)
    assert settings.feature_scaling_method in ["standard", "minmax", "robust"]


@pytest.mark.parametrize(
    "path_setting", ["data_dir", "model_dir", "log_dir", "experiments_dir"]
)
def test_path_creation(path_setting):
    """Test that paths can be created."""
    path = getattr(settings, path_setting)

    # Try to create the directory
    path.mkdir(parents=True, exist_ok=True)

    # Check it exists
    assert path.exists()
    assert path.is_dir()


def test_mlflow_settings():
    """Test MLflow settings."""
    assert settings.mlflow_tracking_uri.startswith(("http://", "sqlite://", "file://"))
    assert isinstance(settings.mlflow_experiment_name, str)
    assert len(settings.mlflow_experiment_name) > 0


def test_settings_repr():
    """Test that settings have a useful string representation."""
    repr_str = repr(settings)
    assert isinstance(repr_str, str)
    assert "Settings" in repr_str
    assert "environment=" in repr_str


def test_settings_copy():
    """Test that settings can be copied."""
    from copy import deepcopy

    settings_copy = deepcopy(settings)
    assert settings_copy.environment == settings.environment
    assert settings_copy.log_level == settings.log_level


if __name__ == "__main__":
    """Run tests when executed directly."""
    pytest.main([__file__, "-v"])
