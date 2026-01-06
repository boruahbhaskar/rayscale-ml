"""Constants used throughout the ML platform."""

from enum import Enum
from pathlib import Path


class DatasetSplit(str, Enum):
    """Dataset split types."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    ALL = "all"


class FeatureType(str, Enum):
    """Feature types."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IMAGE = "image"
    EMBEDDING = "embedding"


class ModelFramework(str, Enum):
    """Model frameworks."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


# Path constants
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
FEATURES_PATH = Path("data/processed/features")
MODELS_PATH = Path("models")
LOGS_PATH = Path("logs")

# Schema constants
FEATURE_COLUMNS = ["feature_1_norm", "feature_2_norm", "feature_3", "feature_4"]
TARGET_COLUMN = "target"
ID_COLUMN = "id"
TIMESTAMP_COLUMN = "timestamp"  # Add this


# Default values
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_EPOCHS = 10
DEFAULT_HIDDEN_DIM = 64

# MLflow constants
MLFLOW_EXPERIMENT_NAME = "rayscale-ml"
MLFLOW_RUN_NAME_PREFIX = "run"
MLFLOW_ARTIFACT_PATH = "artifacts"
