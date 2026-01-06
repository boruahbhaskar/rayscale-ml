"""Ray-specific model trainer implementation."""

from typing import Dict, Any, Optional, Tuple
import time
from pathlib import Path

import ray
import ray.data as rd
import torch
import torch.nn as nn
import torch.optim as optim
from ray import train
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig
from loguru import logger

from src.models.trainers.base_trainer import BaseTrainer, TrainingResult
from src.models.base import BaseModel
from src.config import settings
from src.utils.mps_utils import configure_for_mps, get_device


class RayModelTrainer(BaseTrainer):
    """Model trainer using Ray for distributed training."""
    
    def __init__(self, model: BaseModel, **kwargs):
        """
        Initialize Ray trainer.
        
        Args:
            model: Model to train.
            **kwargs: Trainer configuration.
        """
        super().__init__(model, **kwargs)
        self.ray_trainer: Optional[TorchTrainer] = None
        
    def prepare_data(
        self,
        data: rd.Dataset,
        validation_split: float = 0.2,
        seed: int = 42
    ) -> Tuple[rd.Dataset, Optional[rd.Dataset]]:
        """
        Prepare data for training with Ray.
        
        Args:
            data: Ray dataset.
            validation_split: Validation split ratio.
            seed: Random seed.
            
        Returns:
            Tuple of (train_data, validation_data).
        """
        logger.info("Preparing data for Ray training")
        
        # Split data
        if validation_split > 0:
            train_data, val_data = data.train_test_split(
                test_size=validation_split,
                seed=seed
            )
            logger.info(
                f"Split data: {train_data.count()} train, "
                f"{val_data.count()} validation"
            )
            return train_data, val_data
        else:
            logger.info(f"Using all data for training: {data.count()} samples")
            return data, None
    
    def train(
        self,
        data: rd.Dataset,
        **kwargs
    ) -> TrainingResult:
        """
        Train model using Ray.
        
        Args:
            data: Training data.
            **kwargs: Additional training arguments.
            
        Returns:
            Training result.
        """
        logger.info("Starting Ray training")
        start_time = time.time()
        
        # Prepare data
        train_data, val_data = self.prepare_data(
            data,
            validation_split=kwargs.get("validation_split", 0.2)
        )
        
        # Configure Ray
        scaling_config = self._get_scaling_config()
        run_config = self._get_run_config()
        
        # Create Ray trainer
        train_loop_config = {
            "model_config": self.model.hyperparameters,
            "training_config": {
                "learning_rate": kwargs.get("learning_rate", 0.001),
                "batch_size": kwargs.get("batch_size", 1024),
                "num_epochs": kwargs.get("num_epochs", 10),
                "weight_decay": kwargs.get("weight_decay", 0.0),
            }
        }
        
        self.ray_trainer = TorchTrainer(
            train_loop_per_worker=self._create_train_loop(train_data, val_data),
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            run_config=run_config
        )
        
        # Execute training
        logger.info("Executing Ray training")
        result = self.ray_trainer.fit()
        
        # Extract results
        metrics = result.metrics
        checkpoint = result.checkpoint
        
        # Load model from checkpoint
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                self.model = self._load_model_from_checkpoint(
                    Path(checkpoint_dir),
                    self.model
                )
        
        training_time = time.time() - start_time
        
        # Create training result
        self.training_result = TrainingResult(
            model=self.model,
            metrics=metrics,
            training_time=training_time,
            artifacts={"checkpoint": checkpoint} if checkpoint else {},
            metadata={
                "ray_version": ray.__version__,
                "scaling_config": scaling_config.to_dict(),
                "training_config": train_loop_config
            }
        )
        
        logger.info(
            f"Ray training completed in {training_time:.2f}s. "
            f"Final metrics: {metrics}"
        )
        
        return self.training_result
    
    def _create_train_loop(self, train_data: rd.Dataset, val_data: Optional[rd.Dataset]):
        """
        Create training loop for Ray workers.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            
        Returns:
            Training loop function.
        """
        from src.config.constants import FEATURE_COLUMNS, TARGET_COLUMN
        
        def train_loop_per_worker(config: Dict[str, Any]):
            """Training loop executed on each Ray worker."""
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from ray.train.torch import get_device
            
            # Configure device
            device = get_device()
            logger.info(f"Training on device: {device}")
            
            # Get data
            train_shard = train.get_dataset_shard("train")
            val_shard = train.get_dataset_shard("val") if val_data else None
            
            # Build model
            model_config = config["model_config"]
            training_config = config["training_config"]
            
            # Create model instance (simplified - in production, use registry)
            from src.models.architectures.tabular_nn import TabularMLP
            model = TabularMLP(
                input_size=model_config.get("input_size", len(FEATURE_COLUMNS)),
                hidden_sizes=model_config.get("hidden_sizes", [64, 32]),
                dropout_rate=model_config.get("dropout_rate", 0.2)
            ).to(device)
            
            # Setup optimizer and loss
            optimizer = optim.Adam(
                model.parameters(),
                lr=training_config["learning_rate"],
                weight_decay=training_config["weight_decay"]
            )
            criterion = nn.MSELoss()
            
            # Training loop
            model.train()
            for epoch in range(training_config["num_epochs"]):
                epoch_loss = 0.0
                num_batches = 0
                
                # Training
                for batch in train_shard.iter_torch_batches(
                    batch_size=training_config["batch_size"],
                    device=device,
                    dtypes=torch.float32
                ):
                    # Prepare features and target
                    x = torch.stack([batch[col] for col in FEATURE_COLUMNS], dim=1)
                    y = batch[TARGET_COLUMN].unsqueeze(1)
                    
                    # Forward pass
                    predictions = model(x)
                    loss = criterion(predictions, y)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_train_loss = epoch_loss / max(num_batches, 1)
                
                # Validation
                if val_shard:
                    val_loss = 0.0
                    val_batches = 0
                    
                    model.eval()
                    with torch.no_grad():
                        for batch in val_shard.iter_torch_batches(
                            batch_size=training_config["batch_size"],
                            device=device,
                            dtypes=torch.float32
                        ):
                            x = torch.stack([batch[col] for col in FEATURE_COLUMNS], dim=1)
                            y = batch[TARGET_COLUMN].unsqueeze(1)
                            
                            predictions = model(x)
                            loss = criterion(predictions, y)
                            
                            val_loss += loss.item()
                            val_batches += 1
                    
                    avg_val_loss = val_loss / max(val_batches, 1)
                    model.train()
                    
                    # Report metrics
                    train.report({
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "epoch": epoch + 1
                    })
                    
                    logger.info(
                        f"Epoch {epoch + 1}: "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}"
                    )
                else:
                    train.report({
                        "train_loss": avg_train_loss,
                        "epoch": epoch + 1
                    })
                    
                    logger.info(
                        f"Epoch {epoch + 1}: "
                        f"Train Loss: {avg_train_loss:.4f}"
                    )
                
                # Create checkpoint (save every few epochs)
                if (epoch + 1) % 5 == 0:
                    checkpoint_dir = train.get_context().get_trial_dir()
                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    train.report({"checkpoint": checkpoint}, checkpoint=checkpoint)
            
            # Final checkpoint
            checkpoint_dir = train.get_context().get_trial_dir()
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report({"final_checkpoint": checkpoint}, checkpoint=checkpoint)
        
        return train_loop_per_worker
    
    def _get_scaling_config(self) -> ScalingConfig:
        """Get Ray scaling configuration."""
        # Apple Silicon specific configuration
        use_gpu = settings.ray_use_gpu and configure_for_mps()
        
        return ScalingConfig(
            num_workers=self.config.get("num_workers", 2),
            use_gpu=use_gpu,
            resources_per_worker={
                "CPU": self.config.get("cpu_per_worker", 2),
                "GPU": 0.5 if use_gpu else 0
            },
            trainer_resources={
                "CPU": self.config.get("trainer_cpus", 1)
            }
        )
    
    def _get_run_config(self) -> RunConfig:
        """Get Ray run configuration."""
        from ray.air.integrations.mlflow import MLflowLoggerCallback
        
        callbacks = []
        
        # Add MLflow callback if enabled
        if self.config.get("enable_mlflow", True):
            mlflow_callback = MLflowLoggerCallback(
                tracking_uri=settings.mlflow_tracking_uri,
                experiment_name=settings.mlflow_experiment_name,
                tags={
                    "framework": "ray",
                    "model": self.model.name,
                    "environment": settings.environment
                }
            )
            callbacks.append(mlflow_callback)
        
        return RunConfig(
            name=self.config.get("run_name", f"{self.model.name}-ray"),
            callbacks=callbacks,
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min"
            ),
            storage_path=str(settings.experiments_dir),
            failure_config=None,
            verbose=self.config.get("verbose", 1)
        )
    
    def _load_model_from_checkpoint(
        self,
        checkpoint_dir: Path,
        model: BaseModel
    ) -> BaseModel:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_dir: Checkpoint directory.
            model: Model instance.
            
        Returns:
            Loaded model.
        """
        import torch
        
        # Find model file
        model_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))
        
        if not model_files:
            logger.warning(f"No model files found in {checkpoint_dir}")
            return model
        
        model_file = model_files[0]
        
        # Load model state
        if isinstance(model.model, torch.nn.Module):
            model.model.load_state_dict(
                torch.load(model_file, map_location=get_device())
            )
            model._is_trained = True
            logger.info(f"Loaded model from checkpoint: {model_file}")
        
        return model
    
    def save(self, path: Path) -> None:
        """
        Save trainer state.
        
        Args:
            path: Path to save trainer.
        """
        raise NotImplementedError("Ray trainer state saving not implemented")
    
    @classmethod
    def load(cls, path: Path) -> "RayModelTrainer":
        """
        Load trainer state.
        
        Args:
            path: Path to load trainer from.
            
        Returns:
            Loaded trainer instance.
        """
        raise NotImplementedError("Ray trainer state loading not implemented")