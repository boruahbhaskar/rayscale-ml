"""Ray-specific training pipeline implementation."""

from typing import Any

import ray
import ray.data as rd
from loguru import logger

from src.config import settings
from src.data.feature_store import get_feature_store
from src.models.architectures.tabular_nn import TabularNNModel
from src.models.trainers.ray_trainer import RayModelTrainer
from src.training.pipelines.base_pipeline import BaseTrainingPipeline, PipelineConfig
from src.utils.mps_utils import configure_for_mps


class RayTrainingPipeline(BaseTrainingPipeline):
    """Training pipeline using Ray for distributed training."""

    def __init__(self, config: PipelineConfig):
        """Initialize Ray pipeline."""
        super().__init__(config)
        self._init_ray()

    def _init_ray(self) -> None:
        """Initialize Ray runtime."""
        import platform

        runtime_env = {
            "env_vars": {
                "OMP_NUM_THREADS": "1",
                "RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER": "1",
                "PYTHONUNBUFFERED": "1",
            }
        }

        # Apple Silicon specific configuration
        if platform.processor() == "arm" and configure_for_mps():
            runtime_env["env_vars"].update(
                {"PYTORCH_ENABLE_MPS_FALLBACK": "1", "MPS_DEVICE": "mps"}
            )

        # Initialize Ray
        ray.init(
            runtime_env=runtime_env,
            ignore_reinit_error=True,
            include_dashboard=settings.environment == "development",
            logging_level=(
                "info" if settings.environment == "development" else "warning"
            ),
            num_cpus=settings.ray_num_cpus,
        )

        logger.info(
            f"Ray initialized with {settings.ray_num_cpus} CPUs, "
            f"MPS: {configure_for_mps()}"
        )

    def load_data(self) -> rd.Dataset:
        """Load data from feature store."""

        logger.info("Loading data from feature store")

        # Get data source configuration
        data_config = self.config.data_config
        dataset_name = data_config.get("dataset_name", "features")
        dataset_version = data_config.get("dataset_version", "latest")

        # Load from feature store
        feature_store = get_feature_store()
        dataset, metadata = feature_store.read_dataset(
            name=dataset_name, version=dataset_version
        )

        logger.info(
            f"Loaded dataset '{dataset_name}' version {metadata.version} "
            f"with {metadata.num_rows} rows"
        )

        # Apply any data transformations
        transformations = data_config.get("transformations", [])
        for transform in transformations:
            dataset = self._apply_transformation(dataset, transform)

        return dataset

    def _apply_transformation(
        self, dataset: rd.Dataset, transform: dict[str, Any]
    ) -> rd.Dataset:
        """Apply data transformation."""
        transform_type = transform.get("type")

        if transform_type == "filter":
            # Filter rows based on condition
            column = transform.get("column")
            condition = transform.get("condition")

            if column and condition:
                if condition == "not_null":
                    dataset = dataset.filter(lambda row: row[column] is not None)
                elif condition.startswith("gt:"):
                    value = float(condition.split(":")[1])
                    dataset = dataset.filter(lambda row: row[column] > value)
                elif condition.startswith("lt:"):
                    value = float(condition.split(":")[1])
                    dataset = dataset.filter(lambda row: row[column] < value)

        elif transform_type == "map":
            # Apply map function
            function = transform.get("function")
            if function:
                # Note: In production, you'd want to properly serialize/deserialize
                # the function or use a registry of predefined functions
                pass

        return dataset

    def create_model(self) -> TabularNNModel:
        """Create tabular neural network model."""
        from src.config.constants import FEATURE_COLUMNS

        logger.info("Creating tabular neural network model")

        model_config = self.config.model_config
        architecture = model_config.get("architecture", "mlp")

        # Create model
        model = TabularNNModel(
            name=model_config.get("name", "tabular_nn"),
            architecture=architecture,
            **model_config.get("hyperparameters", {}),
        )

        # Update hyperparameters with inferred values
        model.hyperparameters.update(
            {"input_size": len(FEATURE_COLUMNS), "framework": "pytorch"}
        )

        logger.info(f"Created model: {model.name} with architecture: {architecture}")

        return model

    def create_trainer(self) -> RayModelTrainer:
        """Create Ray model trainer."""
        logger.info("Creating Ray model trainer")

        training_config = self.config.training_config

        # Create trainer
        trainer = RayModelTrainer(
            self.model, **training_config.get("trainer_config", {})
        )

        # Add callbacks if specified
        callbacks_config = training_config.get("callbacks", [])
        for callback_config in callbacks_config:
            callback = self._create_callback(callback_config)
            if callback:
                trainer.add_callback(callback)

        logger.info("Created Ray model trainer")
        return trainer

    def _create_callback(self, config: dict[str, Any]) -> Any:
        """Create training callback."""
        from src.models.trainers.base_trainer import (
            EarlyStopping,
            LearningRateScheduler,
        )

        callback_type = config.get("type")

        if callback_type == "early_stopping":
            return EarlyStopping(
                patience=config.get("patience", 10),
                min_delta=config.get("min_delta", 1e-4),
                restore_best_weights=config.get("restore_best_weights", True),
            )

        elif callback_type == "learning_rate_scheduler":
            return LearningRateScheduler(
                initial_lr=config.get("initial_lr", 0.001),
                decay_factor=config.get("decay_factor", 0.5),
                patience=config.get("patience", 5),
                min_lr=config.get("min_lr", 1e-6),
            )

        return None

    def train(self) -> Any:
        """Execute training with Ray."""
        logger.info("Starting training with Ray")

        # Load data
        data = self.load_data()

        # Train
        training_config = self.config.training_config
        result = self.trainer.train(data, **training_config.get("training_kwargs", {}))

        return result

    def cleanup(self) -> None:
        """Cleanup Ray resources."""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray resources cleaned up")
