"""Feature server for serving and transforming features."""

import json
from datetime import datetime
from threading import Lock
from typing import Any, Optional

import numpy as np
from loguru import logger

from src.config import settings
from src.data.feature_store import get_feature_store
from src.data.preprocessing import Preprocessor


class FeatureServer:
    """Server for feature serving and transformation."""

    _instance: Optional["FeatureServer"] = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize feature server."""
        if not hasattr(self, "_initialized"):
            self.feature_store = get_feature_store()
            self.preprocessor: Preprocessor | None = None
            self.feature_stats: dict[str, dict[str, float]] = {}
            self._initialized = True
            self._load_preprocessor()

    def _load_preprocessor(self) -> None:
        """Load preprocessor from disk."""
        preprocessor_path = settings.data_dir / "processed" / "preprocessor.pkl"

        if preprocessor_path.exists():
            try:
                self.preprocessor = Preprocessor.load(preprocessor_path)
                self.feature_stats = self.preprocessor._stats
                logger.info("Loaded preprocessor from disk")
            except Exception as e:
                logger.warning(f"Failed to load preprocessor: {e}")
                self.preprocessor = None
        else:
            logger.info("No preprocessor found on disk")

    def transform_features(
        self,
        features: dict[str, Any],
        dataset_name: str = "features",
        dataset_version: str = "latest",
    ) -> dict[str, float]:
        """
        Transform raw features using stored statistics.

        Args:
            features: Raw features.
            dataset_name: Dataset name for statistics.
            dataset_version: Dataset version.

        Returns:
            Transformed features.
        """
        try:
            # Load feature statistics if not loaded
            if not self.feature_stats:
                self._load_feature_stats(dataset_name, dataset_version)

            transformed = features.copy()

            # Apply transformations based on statistics
            for feature_name, stats in self.feature_stats.items():
                base_name = feature_name.replace("_norm", "")

                if base_name in transformed:
                    # Apply standardization
                    if stats.get("std", 0) > 0:
                        transformed[f"{base_name}_norm"] = (
                            transformed[base_name] - stats["mean"]
                        ) / stats["std"]
                    else:
                        transformed[f"{base_name}_norm"] = 0.0

            # Add interaction feature if enabled
            if settings.feature_interaction_enabled:
                if "feature_1_norm" in transformed and "feature_2_norm" in transformed:
                    transformed["feature_interaction"] = (
                        transformed["feature_1_norm"] * transformed["feature_2_norm"]
                    )

            return transformed

        except Exception as e:
            logger.error(f"Feature transformation failed: {e}")
            # Return original features if transformation fails
            return features

    def _load_feature_stats(self, dataset_name: str, dataset_version: str) -> None:
        """
        Load feature statistics from feature store.

        Args:
            dataset_name: Dataset name.
            dataset_version: Dataset version.
        """
        try:
            _, metadata = self.feature_store.read_dataset(dataset_name, dataset_version)

            self.feature_stats = metadata.feature_stats
            logger.info(f"Loaded feature statistics for {dataset_name}")

        except Exception as e:
            logger.warning(f"Could not load feature statistics: {e}")
            self.feature_stats = {}

    def get_feature_schema(
        self, dataset_name: str = "features", dataset_version: str = "latest"
    ) -> dict[str, Any]:
        """
        Get feature schema.

        Args:
            dataset_name: Dataset name.
            dataset_version: Dataset version.

        Returns:
            Feature schema.
        """
        try:
            _, metadata = self.feature_store.read_dataset(dataset_name, dataset_version)

            return {
                "schema": metadata.schema,
                "feature_stats": metadata.feature_stats,
                "num_rows": metadata.num_rows,
                "created_at": metadata.created_at,
            }

        except Exception as e:
            logger.error(f"Failed to get feature schema: {e}")
            return {}

    def validate_features(
        self, features: dict[str, Any], schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Validate features against schema.

        Args:
            features: Features to validate.
            schema: Feature schema (optional).

        Returns:
            Validation result.
        """
        validation_result = {"valid": True, "errors": [], "warnings": []}

        try:
            # Load schema if not provided
            if schema is None:
                schema_info = self.get_feature_schema()
                schema = schema_info.get("schema", {})
                feature_stats = schema_info.get("feature_stats", {})
            else:
                feature_stats = schema.get("feature_stats", {})

            # Check required features
            required_features = list(schema.keys())
            for feature in required_features:
                if feature not in features:
                    validation_result["errors"].append(
                        f"Missing required feature: {feature}"
                    )
                    validation_result["valid"] = False

            # Check feature ranges if statistics available
            for feature_name, stats in feature_stats.items():
                if feature_name in features:
                    value = features[feature_name]

                    # Check for outliers (beyond 3 standard deviations)
                    if "mean" in stats and "std" in stats:
                        z_score = abs((value - stats["mean"]) / max(stats["std"], 1e-8))
                        if z_score > 3:
                            validation_result["warnings"].append(
                                f"Feature {feature_name} is an outlier (z-score: {z_score:.2f})"
                            )

            # Check for NaN or infinite values
            for feature_name, value in features.items():
                if isinstance(value, (int, float)):
                    if np.isnan(value):
                        validation_result["errors"].append(
                            f"Feature {feature_name} is NaN"
                        )
                        validation_result["valid"] = False
                    elif np.isinf(value):
                        validation_result["errors"].append(
                            f"Feature {feature_name} is infinite"
                        )
                        validation_result["valid"] = False

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")

        return validation_result

    def compute_feature_stats(
        self, features_list: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """
        Compute statistics for features.

        Args:
            features_list: List of feature dictionaries.

        Returns:
            Feature statistics.
        """
        if not features_list:
            return {}

        # Initialize statistics
        stats = {}

        # Get all feature names
        feature_names = set()
        for features in features_list:
            feature_names.update(features.keys())

        # Compute statistics for each feature
        for feature_name in feature_names:
            values = []
            for features in features_list:
                if feature_name in features:
                    value = features[feature_name]
                    if isinstance(value, (int, float)):
                        values.append(value)

            if values:
                stats[feature_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }

        return stats

    def save_feature_stats(
        self,
        stats: dict[str, dict[str, float]],
        name: str = "custom_features",
        description: str | None = None,
    ) -> str:
        """
        Save feature statistics.

        Args:
            stats: Feature statistics.
            name: Statistics name.
            description: Description.

        Returns:
            Saved path.
        """
        stats_path = settings.data_dir / "feature_stats" / f"{name}.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        stats_data = {
            "stats": stats,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "num_features": len(stats),
        }

        with open(stats_path, "w") as f:
            json.dump(stats_data, f, indent=2)

        logger.info(f"Saved feature statistics to {stats_path}")
        return str(stats_path)


# Global feature server instance
def get_feature_server() -> FeatureServer:
    """
    Get global feature server instance.

    Returns:
        FeatureServer instance.
    """
    return FeatureServer()
