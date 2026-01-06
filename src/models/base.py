"""Base model interface for the ML platform."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass
class ModelMetadata:
    """Metadata for a model."""
    
    name: str
    version: str
    framework: str
    created_at: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    description: Optional[str] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "framework": self.framework,
            "created_at": self.created_at,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "description": self.description,
            "tags": self.tags
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BaseModel(ABC):
    """Abstract base class for models."""
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize model.
        
        Args:
            name: Model name.
            **kwargs: Model hyperparameters.
        """
        self.name = name
        self.hyperparameters = kwargs
        self.metadata: Optional[ModelMetadata] = None
        self._is_trained = False
        
    @abstractmethod
    def build(self) -> Any:
        """
        Build model architecture.
        
        Returns:
            Model instance.
        """
        pass
    
    @abstractmethod
    def train(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train model.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            **kwargs: Additional training arguments.
            
        Returns:
            Training metrics.
        """
        pass
    
    @abstractmethod
    def predict(self, data: Any) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data: Input data.
            
        Returns:
            Predictions.
        """
        pass
    
    @abstractmethod
    def evaluate(self, data: Any, labels: Any) -> Dict[str, float]:
        """
        Evaluate model.
        
        Args:
            data: Test data.
            labels: True labels.
            
        Returns:
            Evaluation metrics.
        """
        pass
    
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model.
        """
        raise NotImplementedError("Subclasses must implement save method")
    
    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """
        Load model from disk.
        
        Args:
            path: Path to load model from.
            
        Returns:
            Loaded model instance.
        """
        raise NotImplementedError("Subclasses must implement load method")
    
    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata.
        
        Returns:
            Model metadata.
            
        Raises:
            ValueError: If model has no metadata.
        """
        if self.metadata is None:
            raise ValueError("Model has no metadata")
        return self.metadata
    
    def update_metadata(self, **kwargs) -> None:
        """
        Update model metadata.
        
        Args:
            **kwargs: Metadata fields to update.
        """
        if self.metadata is None:
            self.metadata = ModelMetadata(
                name=self.name,
                version="1.0.0",
                framework=self.__class__.__name__,
                created_at=...,
                hyperparameters=self.hyperparameters,
                metrics={}
            )
        
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
            else:
                logger.warning(f"Invalid metadata field: {key}")


class PyTorchModel(BaseModel):
    """Base class for PyTorch models."""
    
    def __init__(self, name: str, **kwargs):
        """Initialize PyTorch model."""
        super().__init__(name, **kwargs)
        self.framework = "pytorch"
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """
        Get appropriate device for Apple Silicon.
        
        Returns:
            Device string.
        """
        import torch
        from src.config import settings
        
        if settings.use_mps and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def build(self) -> Any:
        """Build PyTorch model."""
        import torch.nn as nn
        
        # Example: Simple MLP
        class SimpleMLP(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, output_size: int):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, output_size)
                )
            
            def forward(self, x):
                return self.network(x)
        
        input_size = self.hyperparameters.get("input_size", 4)
        hidden_size = self.hyperparameters.get("hidden_size", 64)
        output_size = self.hyperparameters.get("output_size", 1)
        
        model = SimpleMLP(input_size, hidden_size, output_size)
        return model.to(self.device)


class TensorFlowModel(BaseModel):
    """Base class for TensorFlow models."""
    
    def __init__(self, name: str, **kwargs):
        """Initialize TensorFlow model."""
        super().__init__(name, **kwargs)
        self.framework = "tensorflow"
        
    def build(self) -> Any:
        """Build TensorFlow model."""
        import tensorflow as tf
        
        # Example: Simple MLP
        input_size = self.hyperparameters.get("input_size", 4)
        hidden_size = self.hyperparameters.get("hidden_size", 64)
        output_size = self.hyperparameters.get("output_size", 1)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_size,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(hidden_size // 2, activation='relu'),
            tf.keras.layers.Dense(output_size)
        ])
        
        return model


class SklearnModel(BaseModel):
    """Base class for scikit-learn models."""
    
    def __init__(self, name: str, model_type: str = "random_forest", **kwargs):
        """
        Initialize scikit-learn model.
        
        Args:
            name: Model name.
            model_type: Type of sklearn model.
            **kwargs: Model hyperparameters.
        """
        super().__init__(name, **kwargs)
        self.framework = "sklearn"
        self.model_type = model_type
        self.model = self.build()
        
    def build(self) -> Any:
        """Build scikit-learn model."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.svm import SVR
        
        model_map = {
            "random_forest": RandomForestRegressor,
            "linear_regression": LinearRegression,
            "ridge": Ridge,
            "svr": SVR
        }
        
        if self.model_type not in model_map:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Must be one of: {list(model_map.keys())}"
            )
        
        model_class = model_map[self.model_type]
        return model_class(**self.hyperparameters)
    
    def train(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train sklearn model.
        
        Args:
            train_data: Training data as tuple (X, y).
            val_data: Validation data as tuple (X, y).
            **kwargs: Additional arguments.
            
        Returns:
            Training metrics.
        """
        X_train, y_train = train_data
        
        self.model.fit(X_train, y_train)
        self._is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        metrics = self._calculate_metrics(y_train, train_pred, "train")
        
        # Calculate validation metrics if provided
        if val_data is not None:
            X_val, y_val = val_data
            val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred, "val")
            metrics.update(val_metrics)
        
        return metrics
    
    def predict(self, data: Any) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data: Input features.
            
        Returns:
            Predictions.
            
        Raises:
            ValueError: If model is not trained.
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(data)
    
    def evaluate(self, data: Any, labels: Any) -> Dict[str, float]:
        """
        Evaluate model.
        
        Args:
            data: Test features.
            labels: True labels.
            
        Returns:
            Evaluation metrics.
        """
        predictions = self.predict(data)
        return self._calculate_metrics(labels, predictions, "test")
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            prefix: Prefix for metric names.
            
        Returns:
            Dictionary of metrics.
        """
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score
        )
        
        metrics = {
            f"{prefix}_mse": float(mean_squared_error(y_true, y_pred)),
            f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
            f"{prefix}_r2": float(r2_score(y_true, y_pred))
        }
        
        return metrics
    
    def save(self, path: Path) -> None:
        """
        Save sklearn model.
        
        Args:
            path: Path to save model.
        """
        import joblib
        
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        
        # Save metadata
        metadata_path = path.with_suffix(".json")
        if self.metadata:
            with open(metadata_path, "w") as f:
                f.write(self.metadata.to_json())
    
    @classmethod
    def load(cls, path: Path) -> "SklearnModel":
        """
        Load sklearn model.
        
        Args:
            path: Path to load model from.
            
        Returns:
            Loaded model instance.
        """
        import joblib
        
        model = joblib.load(path)
        
        # Load metadata
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
        
        # Create model instance
        # Note: This is simplified - you'd need to reconstruct properly
        instance = cls(name=metadata_dict.get("name", "loaded_model"))
        instance.model = model
        instance._is_trained = True
        
        return instance