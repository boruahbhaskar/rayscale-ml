"""Pydantic settings configuration for the ML platform."""

from pathlib import Path
from typing import Any, Optional, Dict

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Environment
    environment: str = "development"
    log_level: str = "INFO"

    # Paths
    data_dir: Path = Path("data/")
    model_dir: Path = Path("models/")
    log_dir: Path = Path("logs/")
    experiments_dir: Path = Path("experiments/")

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db" #"http://localhost:5000"
    mlflow_experiment_name: str = "rayscale-ml"
    mlflow_registry_uri: str = "sqlite:///mlflow.db"
    
    # For remote servers, add authentication
    mlflow_username: Optional[str] = None
    mlflow_password: Optional[str] = None
    mlflow_token: Optional[str] = None

    # Ray
    ray_address: str | None = None
    ray_num_cpus: int = 4
    ray_use_gpu: bool = False

    # Apple Silicon
    use_mps: bool = True
    mps_fallback_to_cpu: bool = True

    # Training defaults
    train_batch_size: int = 1024
    train_learning_rate: float = 0.001
    train_num_epochs: int = 10
    train_hidden_dim: int = 64

    # Feature engineering
    feature_interaction_enabled: bool = True
    feature_scaling_method: str = "standard"

    @field_validator(
        "data_dir", "model_dir", "log_dir", "experiments_dir", mode="before"
    )
    @classmethod  # Required in V2 for validators
    def validate_paths(cls, v: Any) -> Path:
        if isinstance(v, str):
            path = Path(v)
        elif isinstance(v, Path):
            path = v
        else:
            raise ValueError(f"Invalid path type: {type(v)}")

        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = ["development", "staging", "production", "test"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v

    # Use the modern ConfigDict for V2 and Mypy compatibility
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
