"""Hyperparameter tuning pipeline using Ray Tune."""

from typing import Dict, Any, Optional
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.air.config import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from loguru import logger

from src.training.pipelines.base_pipeline import BaseTrainingPipeline, PipelineConfig
from src.config import settings
from src.utils.mps_utils import configure_for_mps


class TunePipeline(BaseTrainingPipeline):
    """Hyperparameter tuning pipeline using Ray Tune."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize tuning pipeline."""
        super().__init__(config)
        self.tuner = None
        self.best_result = None
        
    def load_data(self) -> Any:
        """Load data for tuning."""
        from src.data.feature_store import get_feature_store
        
        logger.info("Loading data for hyperparameter tuning")
        
        data_config = self.config.data_config
        dataset_name = data_config.get("dataset_name", "features")
        dataset_version = data_config.get("dataset_version", "latest")
        
        feature_store = get_feature_store()
        dataset, metadata = feature_store.read_dataset(
            name=dataset_name,
            version=dataset_version
        )
        
        logger.info(
            f"Loaded dataset for tuning: {dataset_name} "
            f"with {metadata.num_rows} rows"
        )
        
        return dataset
    
    def create_model(self) -> Any:
        """Create model factory for tuning."""
        # For tuning, we create a model factory that can create
        # models with different hyperparameters
        logger.info("Creating model factory for tuning")
        return None  # Model will be created in training loop
    
    def create_trainer(self) -> Any:
        """Create trainer for tuning."""
        # Trainer is created within the tuning configuration
        logger.info("Creating tuner configuration")
        return None
    
    def train(self) -> Any:
        """Execute hyperparameter tuning."""
        logger.info("Starting hyperparameter tuning with Ray Tune")
        
        # Get data
        data = self.load_data()
        
        # Configure search space
        search_space = self._create_search_space()
        
        # Create tuner
        self.tuner = self._create_tuner(data, search_space)
        
        # Run tuning
        logger.info("Running hyperparameter tuning")
        results = self.tuner.fit()
        
        # Get best result
        self.best_result = results.get_best_result(
            metric=self.config.training_config.get("metric", "val_loss"),
            mode=self.config.training_config.get("mode", "min")
        )
        
        logger.info(
            f"Tuning completed. Best config: {self.best_result.config}, "
            f"Best metric: {self.best_result.metrics}"
        )
        
        return self.best_result
    
    def _create_search_space(self) -> Dict[str, Any]:
        """Create hyperparameter search space."""
        tune_config = self.config.training_config.get("tune_config", {})
        
        search_space = {
            "learning_rate": tune.loguniform(
                tune_config.get("lr_min", 1e-4),
                tune_config.get("lr_max", 1e-2)
            ),
            "batch_size": tune.choice(
                tune_config.get("batch_sizes", [64, 128, 256, 512])
            ),
            "hidden_dim": tune.choice(
                tune_config.get("hidden_dims", [32, 64, 128, 256])
            ),
            "dropout_rate": tune.uniform(
                tune_config.get("dropout_min", 0.1),
                tune_config.get("dropout_max", 0.5)
            ),
            "num_layers": tune.choice(
                tune_config.get("num_layers", [2, 3, 4])
            )
        }
        
        return search_space
    
    def _create_tuner(self, data: Any, search_space: Dict[str, Any]) -> tune.Tuner:
        """Create Ray Tune tuner."""
        from src.config.constants import FEATURE_COLUMNS, TARGET_COLUMN
        
        # Training loop for tuning
        def trainable(config: Dict[str, Any]):
            """Training function for each trial."""
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import mlflow
            
            # Initialize MLflow
            mlflow.start_run()
            
            # Get data shards
            train_shard = train.get_dataset_shard("train")
            val_shard = train.get_dataset_shard("val")
            
            # Create model
            from src.models.architectures.tabular_nn import TabularMLP
            
            # Build model architecture based on config
            hidden_sizes = [config["hidden_dim"]] * config["num_layers"]
            model = TabularMLP(
                input_size=len(FEATURE_COLUMNS),
                hidden_sizes=hidden_sizes,
                dropout_rate=config["dropout_rate"]
            )
            
            # Move to device
            device = torch.device("mps" if configure_for_mps() else "cpu")
            model.to(device)
            
            # Setup optimizer and loss
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
            criterion = nn.MSELoss()
            
            # Training loop
            model.train()
            num_epochs = config.get("num_epochs", 5)
            
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                # Training
                for batch in train_shard.iter_torch_batches(
                    batch_size=config["batch_size"],
                    device=device,
                    dtypes=torch.float32
                ):
                    x = torch.stack([batch[col] for col in FEATURE_COLUMNS], dim=1)
                    y = batch[TARGET_COLUMN].unsqueeze(1)
                    
                    predictions = model(x)
                    loss = criterion(predictions, y)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_train_loss = epoch_loss / max(num_batches, 1)
                
                # Validation
                val_loss = 0.0
                val_batches = 0
                
                model.eval()
                with torch.no_grad():
                    for batch in val_shard.iter_torch_batches(
                        batch_size=config["batch_size"],
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
                
                # Report metrics to Tune
                train.report({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "epoch": epoch + 1
                })
                
                # Log to MLflow
                mlflow.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss
                }, step=epoch)
            
            # Log parameters to MLflow
            mlflow.log_params(config)
            mlflow.end_run()
        
        # Create trainer
        trainer = TorchTrainer(
            train_loop_per_worker=trainable,
            scaling_config=ScalingConfig(
                num_workers=2,
                use_gpu=settings.ray_use_gpu and configure_for_mps(),
                resources_per_worker={
                    "CPU": 2,
                    "GPU": 0.5 if settings.ray_use_gpu and configure_for_mps() else 0
                }
            )
        )
        
        # Configure Tune
        tune_config = self.config.training_config.get("tune_config", {})
        
        tuner = tune.Tuner(
            trainer,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric=tune_config.get("metric", "val_loss"),
                mode=tune_config.get("mode", "min"),
                search_alg=OptunaSearch(),
                num_samples=tune_config.get("num_samples", 10),
                max_concurrent_trials=tune_config.get("max_concurrent_trials", 2)
            ),
            run_config=RunConfig(
                name=tune_config.get("run_name", "ray-tune-experiment"),
                storage_path=str(settings.experiments_dir),
                verbose=1
            )
        )
        
        return tuner
    
    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Get best hyperparameter configuration."""
        if self.best_result is None:
            return None
        
        return self.best_result.config
    
    def get_best_model(self) -> Any:
        """Get best model from tuning."""
        if self.best_result is None:
            return None
        
        # In production, you would load the best model from checkpoint
        # For now, return the result
        return self.best_result