"""Model loading and inference management."""

import json
import time
from datetime import datetime
from threading import Lock
from typing import Any, Optional

import torch
from loguru import logger

from src.artifacts.mlflow_client import MLflowClient
from src.config import settings
from src.utils.mps_utils import get_device


class ModelLoadingError(Exception):
    """Error raised when model loading fails."""

    pass


class ModelManager:
    """Manages model loading, caching, and inference."""

    _instance: Optional["ModelManager"] = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize model manager."""
        if not hasattr(self, "_initialized"):
            self.models: dict[str, dict[str, Any]] = {}
            self.current_version: str | None = None
            self.model_registry_path = settings.model_dir
            self.mlflow_client = MLflowClient()
            self._initialized = True
            self._start_time = time.time()

    def load_model(
        self,
        model_name: str = "tabular_nn",
        version: str = "latest",
        force_reload: bool = False,
    ) -> None:
        """
        Load model into memory.

        Args:
            model_name: Name of the model to load.
            version: Model version ("latest" or specific version).
            force_reload: Whether to force reload even if already loaded.

        Raises:
            ModelLoadingError: If model loading fails.
        """
        with self._lock:
            try:
                # Check if model is already loaded
                model_key = f"{model_name}_{version}"
                if model_key in self.models and not force_reload:
                    logger.info(f"Model {model_key} already loaded")
                    return

                logger.info(f"Loading model {model_name} version {version}")

                # Resolve version if "latest"
                if version == "latest":
                    version = self._get_latest_version(model_name)

                # Load model from MLflow
                model_uri = f"models:/{model_name}/{version}"
                model = self.mlflow_client.load_model(model_uri)

                # Store model
                self.models[model_key] = {
                    "model": model,
                    "version": version,
                    "loaded_at": datetime.now().isoformat(),
                    "metadata": self._load_model_metadata(model_name, version),
                }

                # Update current version
                self.current_version = version

                logger.info(f"Successfully loaded model {model_name} version {version}")

            except Exception as e:
                error_msg = (
                    f"Failed to load model {model_name} version {version}: {str(e)}"
                )
                logger.error(error_msg)
                raise ModelLoadingError(error_msg)

    def _get_latest_version(self, model_name: str) -> str:
        """
        Get latest model version from MLflow.

        Args:
            model_name: Model name.

        Returns:
            Latest version string.
        """
        try:
            # Get model versions from MLflow
            versions = self.mlflow_client.list_model_versions(model_name)

            if not versions:
                raise ModelLoadingError(f"No versions found for model {model_name}")

            # Get latest version
            latest_version = max(versions, key=lambda v: int(v.version))
            return latest_version.version

        except Exception as e:
            logger.warning(f"Could not get latest version from MLflow: {e}")

            # Fallback to local registry
            model_dir = self.model_registry_path / model_name
            if not model_dir.exists():
                raise ModelLoadingError(f"Model directory not found: {model_dir}")

            # Get all versions
            versions = []
            for version_dir in model_dir.iterdir():
                if version_dir.name.startswith("v="):
                    versions.append(version_dir.name.split("=")[1])

            if not versions:
                raise ModelLoadingError(
                    f"No local versions found for model {model_name}"
                )

            return sorted(versions)[-1]

    def _load_model_metadata(self, model_name: str, version: str) -> dict[str, Any]:
        """
        Load model metadata.

        Args:
            model_name: Model name.
            version: Model version.

        Returns:
            Model metadata.
        """
        metadata_path = (
            self.model_registry_path / model_name / f"v={version}" / "metadata.json"
        )

        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)

        # Try to get metadata from MLflow
        try:
            run = self.mlflow_client.get_run_by_model(model_name, version)
            return {
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
        except:
            return {}

    def predict(
        self,
        features: dict[str, float],
        model_name: str = "tabular_nn",
        version: str | None = None,
        **kwargs,
    ) -> float:
        """
        Make prediction with loaded model.

        Args:
            features: Input features.
            model_name: Model name.
            version: Model version (None for current version).
            **kwargs: Additional prediction arguments.

        Returns:
            Prediction.

        Raises:
            ValueError: If model is not loaded.
        """
        if version is None:
            version = self.current_version

        model_key = f"{model_name}_{version}"

        if model_key not in self.models:
            # Try to load model
            self.load_model(model_name, version)

        model_info = self.models[model_key]
        model = model_info["model"]

        # Prepare input
        input_tensor = self._prepare_input(features)

        # Make prediction
        with torch.no_grad():
            if hasattr(model, 'predict'):
                # scikit-learn style
                prediction = model.predict(input_tensor.numpy())
            else:
                # PyTorch style
                model.eval()
                device = get_device()
                model.to(device)
                input_tensor = input_tensor.to(device)
                prediction = model(input_tensor)
                prediction = prediction.cpu().numpy()

        return float(prediction[0])

    def _prepare_input(self, features: dict[str, float]) -> torch.Tensor:
        """
        Prepare input tensor from features.

        Args:
            features: Input features.

        Returns:
            Input tensor.
        """
        # Define expected feature order
        expected_features = [
            "feature_1_norm",
            "feature_2_norm",
            "feature_3",
            "feature_4",
        ]

        # Extract features in correct order
        feature_values = []
        for feature in expected_features:
            if feature in features:
                feature_values.append(features[feature])
            else:
                # Try alternative names
                alt_feature = feature.replace("_norm", "")
                if alt_feature in features:
                    feature_values.append(features[alt_feature])
                else:
                    raise ValueError(f"Missing feature: {feature}")

        # Convert to tensor
        return torch.FloatTensor([feature_values])

    def batch_predict(
        self,
        features_list: list[dict[str, float]],
        model_name: str = "tabular_nn",
        version: str | None = None,
        **kwargs,
    ) -> list[float]:
        """
        Make batch predictions.

        Args:
            features_list: List of input features.
            model_name: Model name.
            version: Model version.
            **kwargs: Additional prediction arguments.

        Returns:
            List of predictions.
        """
        predictions = []

        for features in features_list:
            prediction = self.predict(features, model_name, version, **kwargs)
            predictions.append(prediction)

        return predictions

    def get_model_info(
        self, model_name: str = "tabular_nn", version: str | None = None
    ) -> dict[str, Any]:
        """
        Get model information.

        Args:
            model_name: Model name.
            version: Model version.

        Returns:
            Model information.
        """
        if version is None:
            version = self.current_version

        model_key = f"{model_name}_{version}"

        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not loaded")

        model_info = self.models[model_key]

        return {
            "model_name": model_name,
            "model_version": version,
            "loaded_at": model_info["loaded_at"],
            "metadata": model_info["metadata"],
            "input_size": self._get_input_size(model_info["model"]),
            "output_size": self._get_output_size(model_info["model"]),
            "framework": self._get_framework(model_info["model"]),
        }

    def _get_input_size(self, model: Any) -> int | None:
        """Get model input size."""
        if hasattr(model, 'n_features_in_'):
            return model.n_features_in_
        elif hasattr(model, 'input_size'):
            return model.input_size
        return None

    def _get_output_size(self, model: Any) -> int | None:
        """Get model output size."""
        if hasattr(model, 'n_outputs_'):
            return model.n_outputs_
        elif hasattr(model, 'output_size'):
            return model.output_size
        return None

    def _get_framework(self, model: Any) -> str:
        """Get model framework."""
        model_type = type(model).__module__

        if 'torch' in model_type:
            return 'pytorch'
        elif 'sklearn' in model_type:
            return 'sklearn'
        elif 'tensorflow' in model_type:
            return 'tensorflow'
        else:
            return 'unknown'

    def get_loaded_models(self) -> list[str]:
        """Get list of loaded models."""
        return list(self.models.keys())

    def unload_model(self, model_name: str, version: str | None = None) -> bool:
        """
        Unload model from memory.

        Args:
            model_name: Model name.
            version: Model version.

        Returns:
            True if model was unloaded.
        """
        with self._lock:
            if version is None:
                # Unload all versions of this model
                keys_to_remove = [
                    key
                    for key in self.models.keys()
                    if key.startswith(f"{model_name}_")
                ]
            else:
                model_key = f"{model_name}_{version}"
                keys_to_remove = [model_key] if model_key in self.models else []

            for key in keys_to_remove:
                del self.models[key]
                logger.info(f"Unloaded model: {key}")

            # Update current version if needed
            if self.current_version in keys_to_remove:
                self.current_version = None
                if self.models:
                    # Set to first available model
                    self.current_version = list(self.models.values())[0]["version"]

            return len(keys_to_remove) > 0

    def get_uptime(self) -> float:
        """Get manager uptime in seconds."""
        return time.time() - self._start_time


# Global model manager instance
def get_model_manager() -> ModelManager:
    """
    Get global model manager instance.

    Returns:
        ModelManager instance.
    """
    return ModelManager()
