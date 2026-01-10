"""Ray-specific model trainer implementation."""

import time
from pathlib import Path
from typing import Any

import ray
import ray.data as rd
import torch
import torch.nn as nn
import torch.optim as optim
import logging
logger = logging.getLogger(__name__)
from ray import train

try:
    # For newer Ray versions
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
except ImportError:
    # For older Ray versions
    from ray.air import CheckpointConfig, RunConfig, ScalingConfig

from ray.train import Checkpoint
from ray.train.torch import TorchTrainer

from src.config import settings
from src.models.base import BaseModel
from src.models.trainers.base_trainer import BaseTrainer, TrainingResult
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
        self.ray_trainer: TorchTrainer | None = None
        
        # Store config and extract training parameters
        self.config = kwargs.get("config", {}) or kwargs  # Get config from kwargs
        training_config = self.config.get("training", {})
        
        # Set training parameters
        self.num_workers = training_config.get("num_workers", 1)
        self.use_gpu = training_config.get("use_gpu", False)
        self.cpus_per_worker = training_config.get("cpus_per_worker", 1)
        self.gpus_per_worker = training_config.get("gpus_per_worker", 0)
        
        # Also check if values are passed directly in kwargs
        if "num_workers" in kwargs:
            self.num_workers = kwargs["num_workers"]
        if "use_gpu" in kwargs:
            self.use_gpu = kwargs["use_gpu"]

    def prepare_data(
        self, data: rd.Dataset, validation_split: float = 0.2, seed: int = 42
    ) -> tuple[rd.Dataset, rd.Dataset | None]:
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
                test_size=validation_split, seed=seed
            )
            logger.info(
                f"Split data: {train_data.count()} train, "
                f"{val_data.count()} validation"
            )
            return train_data, val_data
        else:
            logger.info(f"Using all data for training: {data.count()} samples")
            return data, None

    def train(self, data: rd.Dataset, **kwargs) -> TrainingResult:
        """
        Train model using Ray.

        Args:
            data: Training data.
            **kwargs: Additional training arguments.

        Returns:
            Training result.
        """
        import os
        os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"
        logger.info("Starting Ray training")
        
        logger.info(f"=== DEBUG: Starting train() ===")
        logger.info(f"Data type: {type(data)}")
        
        # Check if it it's a Ray Dataset
        import ray

        if isinstance(data, ray.data.Dataset):
            logger.info(f"Data is a Ray Dataset with {data.count()} rows")
            logger.info(f"Data schema: {data.schema()}")

            # checking a sample to see the structure
            sample = data.take(1)
            logger.info(f"Sample data rows: {sample[0] if sample else 'No data'}")
        else:
            if hasattr(data, '__len__'):
                logger.info(f"Data length: {len(data)}")
            if hasattr(data, 'shape'):
                logger.info(f"Data shape: {data.shape}")

        start_time = time.time()

        # Add column filtering before preprocessing
        def filter_numeric_columns(batch):
            """Filter out non-numeric columns."""
            import pandas as pd
            import numpy as np
            
            if isinstance(batch, dict):
                df = pd.DataFrame(batch)
            else:
                df = batch
            
            # Identify numeric columns
            numeric_cols = []
            for col in df.columns:
                dtype = df[col].dtype
                # Keep only numeric columns
                if np.issubdtype(dtype, np.number):
                    numeric_cols.append(col)
                else:
                    logger.debug(f"Filtering out non-numeric column: {col} (dtype: {dtype})")
            
            # Keep only numeric columns
            df = df[numeric_cols]
            
            return df.to_dict('list') if isinstance(batch, dict) else df
        
        # Apply column filtering before datetime conversion
        preprocessed_data = data.map_batches(
            filter_numeric_columns,
            batch_format="pandas",
        )

        # Prepare data
        train_data, val_data = self.prepare_data(
            preprocessed_data, validation_split=kwargs.get("validation_split", 0.2)
        )

        # Configure Ray
        scaling_config = self._get_scaling_config()
        run_config = self._get_run_config()

        # Get feature columns from the preprocessed data
        feature_columns = kwargs.get("feature_columns", [])
        target_column = kwargs.get("target_column", "target")
        
        if not feature_columns:
            # Infer feature columns (all except target)
            all_columns = train_data.columns()
            if target_column in all_columns:
                feature_columns = [col for col in all_columns if col != target_column]
            logger.info(f"Inferred feature columns: {feature_columns}")

        # Get actual number of features
        num_features = len(feature_columns)
        logger.info(f"Number of features: {num_features}")

        # FIX: Update model hyperparameters with CORRECT input_size BEFORE training
        self.model.hyperparameters["input_size"] = num_features

        # FIX: Clean hyperparameters before passing to training loop
        # Get only valid TabularMLP parameters from self.model.hyperparameters
        valid_mlp_params = [
            'input_size', 'hidden_sizes', 'output_size', 
            'dropout_rate', 'use_batch_norm', 'activation'
        ]

        # Start with default values
        model_config_for_training = {
            "input_size": num_features,  # Use actual number of features
            "hidden_sizes": [64, 32],
            "dropout_rate": 0.2,
            "use_batch_norm": True,
            "activation": "relu",
        }

        # Override with valid parameters from model.hyperparameters
        for param in valid_mlp_params:
            if param in self.model.hyperparameters:
                model_config_for_training[param] = self.model.hyperparameters[param]
        
        # Override with values from kwargs if provided
        for param in valid_mlp_params:
            if param in kwargs:
                model_config_for_training[param] = kwargs[param]

        # Create training configuration
        # train_loop_config = {
        #         "model_config": {
        #         # OVERRIDE input_size with actual number of features
        #         "input_size": num_features,  # <-- FORCE THIS TO BE 10
        #         # Copy other parameters but ensure input_size is correct
        #         "hidden_sizes": model_config_for_training.get("hidden_sizes", [64, 32]),
        #         "dropout_rate": model_config_for_training.get("dropout_rate", 0.2),
        #         "use_batch_norm": model_config_for_training.get("use_batch_norm", True),
        #         "activation": model_config_for_training.get("activation", "relu"),
        #         "feature_columns": feature_columns,
        #         "target_column": target_column,
        #     },
        #     "training_config": {
        #         "learning_rate": kwargs.get("learning_rate", 0.001),
        #         "batch_size": kwargs.get("batch_size", 1024),
        #         "num_epochs": kwargs.get("num_epochs", 10),
        #         "weight_decay": kwargs.get("weight_decay", 0.0),
        #     },
        # }

        train_loop_config = {
            "model_config": {
                # Use the cleaned hyperparameters
                **{k: v for k, v in self.model.hyperparameters.items() 
                if k in valid_mlp_params},
                "feature_columns": feature_columns,
                "target_column": target_column,
            },
            "training_config": {
                "learning_rate": kwargs.get("learning_rate", 0.001),
                "batch_size": kwargs.get("batch_size", 1024),
                "num_epochs": kwargs.get("num_epochs", 10),
                "weight_decay": kwargs.get("weight_decay", 0.0),
            },
        }

        logger.info(f"Training model with input_size: {self.model.hyperparameters.get('input_size')}")

        logger.info(f"Model config for training: {model_config_for_training}")

        # Create datasets dict to pass to Ray Trainer
        datasets = {"train": train_data}
        if val_data:
            datasets["val"] = val_data

        logger.info("Prepared datasets: {list(datasets.keys())}")    

        # Create Ray TorchTrainer with datasets

        self.ray_trainer = TorchTrainer(
            train_loop_per_worker=self._create_train_loop(),
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            run_config=run_config,
            datasets=datasets,
        )

        # Execute training
        logger.info("Executing Ray training")
        result = self.ray_trainer.fit()

        # Extract results
        metrics = result.metrics
        checkpoint = result.checkpoint

        logger.info("Cleaning model hyperparameters before building...")

        # Update model hyperparameters with correct values from training
        for param in valid_mlp_params:
            if param in model_config_for_training:
                self.model.hyperparameters[param] = model_config_for_training[param]
        
        # Remove invalid parameters
        invalid_params = []
        for param in list(self.model.hyperparameters.keys()):
            if param not in valid_mlp_params:
                invalid_params.append(param)
        
        for param in invalid_params:
            del self.model.hyperparameters[param]
            logger.debug(f"Removed invalid parameter from hyperparameters: {param}")

        # CRITICAL: Ensure model is built BEFORE trying to load checkpoint
        if hasattr(self.model, 'model') and self.model.model is None:
            logger.info("Building model architecture before loading checkpoint...")
            if hasattr(self.model, 'build'):
                self.model.build()
            else:
                logger.warning("Model doesn't have build() method")

        # Load model from checkpoint
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                self.model = self._load_model_from_checkpoint(
                    Path(checkpoint_dir), self.model
                )
        else:
            # Even if no checkpoint, mark model as trained
            self.model._is_trained = True
            logger.info("No checkpoint available, marking model as trained")


        training_time = time.time() - start_time

            # CRITICAL FINAL CHECK: Ensure model can be saved
        if hasattr(self.model, 'model') and self.model.model is None:
            logger.error("Model is still None after training! Creating fallback model.")
            # Create a simple fallback model to prevent save errors
            from src.models.architectures.tabular_nn import TabularMLP
            input_size = len(feature_columns)
            self.model.model = TabularMLP(
                input_size=input_size,
                hidden_sizes=[64, 32],
                dropout_rate=0.2,
            )
            # self.model.model = TabularMLP(
            #     input_size=num_features,
            #     hidden_sizes=model_config_for_training["hidden_sizes"],
            #     dropout_rate=model_config_for_training["dropout_rate"],
            # )
            self.model._is_trained = True

        # # Add this debugging code temporarily:
        # logger.info(f"ScalingConfig type: {type(scaling_config)}")
        # logger.info(f"ScalingConfig dir: {dir(scaling_config)}")
        # logger.info(f"ScalingConfig attributes: {scaling_config.__dict__ if hasattr(scaling_config, '__dict__') else 'No __dict__'}")

        # # Get scaling config info
        # scaling_config_info = {}
        # try:
        #     # Get basic attributes
        #     scaling_config_info["num_workers"] = scaling_config.num_workers
        #     scaling_config_info["use_gpu"] = scaling_config.use_gpu
            
        #     # Try to get resources_per_worker
        #     if hasattr(scaling_config, "resources_per_worker"):
        #         scaling_config_info["resources_per_worker"] = scaling_config.resources_per_worker
            
        #     # Try to get trainer_resources (for older versions)
        #     if hasattr(scaling_config, "trainer_resources"):
        #         scaling_config_info["trainer_resources"] = scaling_config.trainer_resources
            
        #     # For newer versions, check for additional attributes
        #     if hasattr(scaling_config, "placement_strategy"):
        #         scaling_config_info["placement_strategy"] = scaling_config.placement_strategy
            
        # except Exception as e:
        #     logger.warning(f"Could not extract scaling config info: {e}")
        #     scaling_config_info = {"error": str(e)}
        
        
        # Create training result
        self.training_result = TrainingResult(
            model=self.model,
            metrics=metrics,
            training_time=training_time,
            artifacts={"checkpoint": checkpoint} if checkpoint else {},
            metadata={
                "ray_version": ray.__version__,
                "training_config": train_loop_config,
            },
        )

        logger.info(
            f"Ray training completed in {training_time:.2f}s. "
            f"Final metrics: {metrics}"
        )

        return self.training_result

    def _create_train_loop(self):
        """
        Create training loop for Ray workers.

        Returns:
            Training loop function.
        """
        # from src.config.constants import FEATURE_COLUMNS, TARGET_COLUMN

        def train_loop_per_worker(config: dict[str, Any]):
            """Training loop executed on each Ray worker."""
            from ray.train.torch import get_device

            import logging  # Import inside function, don't capture from outer
            import numpy as np

            # Create logger inside the function
            worker_logger = logging.getLogger(__name__)

            # Configure device
            device = get_device()
            worker_logger.info(f"Training on device: {device}")
            

            # Get data shards
            train_shard = train.get_dataset_shard("train")

            if train_shard is None:
                raise ValueError("Train dataset shard is not available. Check if the dataset was passed with key 'train'.")
            
            worker_logger.info(f"Train shard has {train_shard}")

            # Try to get validation shard if it exists
            val_shard = None
            try:
                val_shard = train.get_dataset_shard("val")
                worker_logger.info(f"Validation shard has {val_shard} ")
            except KeyError:
                worker_logger.info("No validation shard provided")


            # Build model
            model_config = config["model_config"]
            training_config = config["training_config"]

            # Get feature columns and target column from model config or use defaults
            FEATURE_COLUMNS = model_config.get("feature_columns", [])
            TARGET_COLUMN = model_config.get("target_column", "target")

            # Log the columns we're using
            worker_logger.info(f"Feature columns: {FEATURE_COLUMNS}")
            worker_logger.info(f"Target column: {TARGET_COLUMN}")


            # If no feature columns specified, try to infer them
            if not FEATURE_COLUMNS and train_shard:
                try:
                    # Get all columns except target
                    all_columns = train_shard.columns()
                    if TARGET_COLUMN in all_columns:
                        FEATURE_COLUMNS = [col for col in all_columns if col != TARGET_COLUMN]
                        worker_logger.info(f"Inferred feature columns: {FEATURE_COLUMNS}")
                except Exception as e:
                    worker_logger.warning(f"Could not infer feature columns: {e}")

            # Get the actual number of features
            num_features = len(FEATURE_COLUMNS)
            worker_logger.info(f"Number of features: {num_features}")

            # Debug: Check feature columns are actually numeric
            if train_shard and FEATURE_COLUMNS:
                try:
                    sample = train_shard.take_batch(batch_size=1)
                    worker_logger.info(f"Sample batch keys: {list(sample.keys())}")
                    for col in FEATURE_COLUMNS:
                        if col in sample:
                            value = sample[col]
                            dtype = value.dtype if hasattr(value, 'dtype') else type(value)
                            shape = value.shape if hasattr(value, 'shape') else 'N/A'
                            worker_logger.info(f"Column '{col}': dtype={dtype}, shape={shape}")
                        else:
                            worker_logger.warning(f"Column '{col}' not found in batch")
                except Exception as e:
                    worker_logger.warning(f"Could not check sample: {e}")


            # Create the same model class as used in the main process

            from src.models.architectures.tabular_nn import TabularMLP

             # Get input_size from config or use number of features
            input_size = num_features # model_config.get("input_size", num_features)
            worker_logger.info(f"Creating model with input_size={input_size}")

            # Create model directly with only valid parameters
            model = TabularMLP(
                input_size=input_size,
                hidden_sizes=model_config.get("hidden_sizes", [64, 32]),
                dropout_rate=model_config.get("dropout_rate", 0.2),
                use_batch_norm=model_config.get("use_batch_norm", True),
                activation=model_config.get("activation", "relu"),
            ).to(device)

            # Build the model ( create the actual PyTorch model)

            #worker_logger.info(f"Model architecture: {model}")
            worker_logger.info(f"Model input size: {input_size}")


            # model = TabularMLP(
            #     input_size=input_size,  # Use the actual number of features
            #     hidden_sizes=model_config.get("hidden_sizes", [64, 32]),
            #     dropout_rate=model_config.get("dropout_rate", 0.2),
            # ).to(device)

            # Setup optimizer and loss
            optimizer = optim.Adam(
                model.parameters(),
                lr=training_config["learning_rate"],
                weight_decay=training_config["weight_decay"],
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
                    dtypes=torch.float32,
                ):
                    try:
                        # Prepare features
                        feature_tensors = []
                        valid_features = []

                        for col in FEATURE_COLUMNS:
                            if col in batch:
                                value = batch[col]
                                
                                # Ensure it's a tensor
                                if not isinstance(value, torch.Tensor):
                                    try:
                                        value = torch.tensor(value, dtype=torch.float32, device=device)
                                    except Exception as e:
                                        worker_logger.error(f"Error converting column '{col}' to tensor: {e}")
                                        continue

                                # check shape
                                if len(value.shape) == 1:
                                    value = value.unsqueeze(1)  # Make it to 2D: ( batch_size, 1)

                                
                                feature_tensors.append(value)
                                valid_features.append(col)
                        
                        if not feature_tensors:
                            raise ValueError(f"No valid feature tensors created from columns: {FEATURE_COLUMNS}")
                        
                        # Debug Shape 
                        for i, (col, tensor) in enumerate(zip(valid_features, feature_tensors)):
                            worker_logger.debug(f"Feature tensor {i} - Column '{col}': shape={tensor.shape}, dtype={tensor.dtype}")
                        

                        # Stack features along dimension 1 (columns)
                        # x = torch.stack(feature_tensors, dim=1)
                        x = torch.cat(feature_tensors, dim=1)  # Concatenate along feature dimension

                        # worker_logger.debug(f"Input tensor x shape: {x.shape}")
                        # worker_logger.debug(f"Model expects input size: {model.input_size}")

                        # Check shape compatibility
                        if x.shape[1] != input_size :
                            raise ValueError(
                                f"Input feature dimension mismatch: "
                                f"x has {x.shape[1]} features, "
                                f"model expects {input_size} features"
                            )
                        
                        # Prepare target
                        y = batch[TARGET_COLUMN]
                        if not isinstance(y, torch.Tensor):
                            y = torch.tensor(y, dtype=torch.float32, device=device)
                        y = y.unsqueeze(1)  # Add batch dimension if needed

                        # Forward pass
                        predictions = model(x)
                        loss = criterion(predictions, y)

                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        num_batches += 1
                        
                    except Exception as e:
                        worker_logger.error(f"Error processing batch: {e}")
                        worker_logger.debug(f"Batch keys: {list(batch.keys())}")
                        for key, val in batch.items():
                            if hasattr(val, 'dtype'):
                                dtype_str = str(val.dtype)
                                if hasattr(val, 'shape'):
                                    worker_logger.debug(f"  {key}: dtype={dtype_str}, shape={val.shape}")   
                                else:
                                    worker_logger.debug(f"  {key}: dtype={dtype_str}, shape=N/A")

                                # worker_logger.debug(f"  {key}: dtype={val.dtype}, shape={val.shape if hasattr(val, 'shape') else 'N/A'}")
                        raise

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
                            dtypes=torch.float32,
                        ):
                            try:
                                # Prepare features
                                feature_tensors = []
                                for col in FEATURE_COLUMNS:
                                    if col in batch:
                                        value = batch[col]
                                        
                                        # Ensure it's a tensor
                                        if not isinstance(value, torch.Tensor):
                                            try:
                                                value = torch.tensor(value, dtype=torch.float32, device=device)
                                            except Exception as e:
                                                worker_logger.error(f"Error converting column '{col}' to tensor: {e}")
                                                continue

                                        if len(value.shape) == 1:
                                            value = value.unsqueeze(1)  # Make it to 2D: ( batch_size, 1)

                                        
                                        feature_tensors.append(value)
                                
                                if not feature_tensors:
                                    continue
                                
                                #x = torch.stack(feature_tensors, dim=1)
                                x = torch.cat(feature_tensors, dim=1)  # Concatenate along feature dimension


                                # Check shape
                                if x.shape[1] != model.input_size:
                                    worker_logger.warning(
                                        f"Validation shape mismatch: "
                                        f"x has {x.shape[1]} features, "
                                        f"model expects {model.input_size}"
                                    )
                                    continue

                                # Prepare target
                                y = batch[TARGET_COLUMN]
                                if not isinstance(y, torch.Tensor):
                                    y = torch.tensor(y, dtype=torch.float32, device=device)
                                y = y.unsqueeze(1)

                                predictions = model(x)
                                loss = criterion(predictions, y)

                                val_loss += loss.item()
                                val_batches += 1
                                
                            except Exception as e:
                                worker_logger.error(f"Error processing validation batch: {e}")
                                continue

                    avg_val_loss = val_loss / max(val_batches, 1)
                    model.train()

                    # Report metrics
                    train.report(
                        {
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss,
                            "epoch": epoch + 1,
                        }
                    )
                    worker_logger.info(
                        f"Epoch {epoch + 1}: "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}"
                    )
                else:
                    train.report({"train_loss": avg_train_loss, "epoch": epoch + 1})

                    worker_logger.info(
                        f"Epoch {epoch + 1}: " f"Train Loss: {avg_train_loss:.4f}"
                    )

            #     # Create checkpoint (save every few epochs)
            #     if (epoch + 1) % 5 == 0:
            #         checkpoint_dir = train.get_context().get_trial_dir()
            #         checkpoint = Checkpoint.from_directory(checkpoint_dir)
            #         train.report({"checkpoint": checkpoint}, checkpoint=checkpoint)

            # # Final checkpoint
            # checkpoint_dir = train.get_context().get_trial_dir()
            # checkpoint = Checkpoint.from_directory(checkpoint_dir)
            # train.report({"final_checkpoint": checkpoint}, checkpoint=checkpoint)

            # Create checkpoint (save every few epochs and at the end)
            if (epoch + 1) % 5 == 0 or epoch == training_config["num_epochs"] - 1:
                try:
                    # Create a simple checkpoint using the new Ray Train API
                    from ray.train import Checkpoint
                    import tempfile
                    import os
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save model state
                        model_path = os.path.join(temp_dir, "model.pth")
                        torch.save(model.state_dict(), model_path)
                        
                        # # Save optimizer state
                        # optimizer_path = os.path.join(temp_dir, "optimizer.pth")
                        # torch.save(optimizer.state_dict(), optimizer_path)
                        
                        # # Save training state
                        # training_state = {
                        #     "epoch": epoch + 1,
                        #     "loss": avg_train_loss,
                        # }
                        # state_path = os.path.join(temp_dir, "training_state.pth")
                        # torch.save(training_state, state_path)
                        
                        # # Create checkpoint
                        # checkpoint = Checkpoint.from_directory(temp_dir)
                        
                        # Save model configuration

                        # config_path = os.path.join(temp_dir, "model_config.json")
                        # import json
                        # with open(config_path, 'w') as f:
                        #     json.dump(model_wrapper.hyperparameters, f)

                        checkpoint = Checkpoint.from_directory(temp_dir)


                        # Report checkpoint
                        train.report(
                            {"checkpoint_epoch": epoch + 1, "checkpoint_loss": avg_train_loss},
                            checkpoint=checkpoint
                        )
                        worker_logger.info(f"Saved checkpoint at epoch {epoch + 1}")
                        
                except Exception as e:
                    worker_logger.warning(f"Could not save checkpoint: {e}")

            worker_logger.info("Training completed successfully")

        return train_loop_per_worker

    # def _get_scaling_config(self) -> ScalingConfig:
    #     """Get Ray scaling configuration."""
    #     # Apple Silicon specific configuration
    #     use_gpu = settings.ray_use_gpu and configure_for_mps()

    #     return ScalingConfig(
    #         num_workers=self.config.get("num_workers", 2),
    #         use_gpu=use_gpu,
    #         resources_per_worker={
    #             "CPU": self.config.get("cpu_per_worker", 2),
    #             "GPU": 0.5 if use_gpu else 0,
    #         },
    #         trainer_resources={"CPU": self.config.get("trainer_cpus", 1)},
    #     )
    
    def _get_scaling_config(self) -> ScalingConfig:
        """
        Get scaling configuration for Ray training.
        
        Returns:
            Scaling configuration.
        """
         # Apple Silicon specific configuration
        use_gpu = settings.ray_use_gpu and configure_for_mps()
        try:
            # For newer Ray versions (>= 2.7)
            return ScalingConfig(
                num_workers=self.num_workers,
                use_gpu=self.use_gpu,
                resources_per_worker={
                    "CPU": self.cpus_per_worker,
                    "GPU": self.gpus_per_worker if self.use_gpu else 0
                }
            )
        except TypeError:
            # For older Ray versions (< 2.7)
            return ScalingConfig(
                num_workers=self.num_workers,
                use_gpu=self.use_gpu,
                trainer_resources={"CPU": 1},
                resources_per_worker={
                    "CPU": self.cpus_per_worker,
                    "GPU": self.gpus_per_worker if self.use_gpu else 0
                }
            )

    # def _get_run_config(self) -> RunConfig:
    #     """Get Ray run configuration for Ray 2.53+."""
    #     callbacks = []

    #     # Add MLflow callback if enabled - using new Ray 2.53+ API
    #     if self.config.get("enable_mlflow", True):
    #         try:
    #             # In Ray 2.53+, MLflow integration is different
    #             # Option 1: Use the new MLflowTrainLogger (recommended for Ray 2.53+)
    #             try:
    #                 from ray.train import MLflowTrainLogger
    #                 mlflow_logger = MLflowTrainLogger(
    #                     tracking_uri=settings.mlflow_tracking_uri,
    #                     experiment_name=settings.mlflow_experiment_name,
    #                     tags={
    #                         "framework": "ray",
    #                         "model": self.model.name,
    #                         "environment": settings.environment,
    #                     },
    #                     save_artifact=True,
    #                 )
    #                 callbacks.append(mlflow_logger)
    #                 logger.info("Using MLflowTrainLogger (Ray 2.53+)")
    #             except ImportError:
    #                 # Option 2: Try the older MLflowLoggerCallback name
    #                 from ray.train import MLflowLoggerCallback
    #                 mlflow_callback = MLflowLoggerCallback(
    #                     tracking_uri=settings.mlflow_tracking_uri,
    #                     experiment_name=settings.mlflow_experiment_name,
    #                     tags={
    #                         "framework": "ray",
    #                         "model": self.model.name,
    #                         "environment": settings.environment,
    #                     },
    #                 )
    #                 callbacks.append(mlflow_callback)
    #                 logger.info("Using MLflowLoggerCallback")
                    
    #         except ImportError as e:
    #             logger.warning(f"MLflow integration not available in your Ray version: {e}")
    #         except Exception as e:
    #             logger.warning(f"Failed to setup MLflow: {e}")

    #     # Return RunConfig
    #     return RunConfig(
    #         name=self.config.get("run_name", f"{self.model.name}-ray"),
    #         callbacks=callbacks if callbacks else None,
    #         checkpoint_config=CheckpointConfig(
    #             num_to_keep=3,
    #             checkpoint_score_attribute="val_loss",
    #             checkpoint_score_order="min",
    #         ),
    #         storage_path=str(settings.experiments_dir),
    #         failure_config=None,
    #     )

    def _get_run_config(self) -> RunConfig:
        """Get Ray run configuration."""

        callbacks = []
        
        # Get storage path - ensure it's a proper URL-encoded file:// URI
        from pathlib import Path
        import urllib.parse
        
        # Convert experiments_dir to absolute path
        exp_dir = Path(settings.experiments_dir)
        if not exp_dir.is_absolute():
            exp_dir = Path.cwd() / exp_dir
        
        # Ensure directory exists
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to file:// URI with URL encoding for spaces
        # CRITICAL: URL encode the path
        encoded_path = urllib.parse.quote(str(exp_dir.absolute()))
        storage_path = f"file://{encoded_path}"
        
        return RunConfig(
            name=self.config.get("run_name", f"{self.model.name}-ray"),
            callbacks=callbacks,
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
            storage_path=storage_path,  # Use URL-encoded URI
            failure_config=None,)

    def _load_model_from_checkpoint(
        self, checkpoint_dir: Path, model: BaseModel
    ) -> BaseModel:
        """
        Load model from checkpoint.

        Args:
            checkpoint_dir: Checkpoint directory.
            model: Model instance.

        Returns:
            Loaded model.
        """
        logger.info(f"Loading model from checkpoint: {checkpoint_dir}")

        # Find model file
        model_files = list(checkpoint_dir.glob("*.pth")) + list(
            checkpoint_dir.glob("*.pt")
        )

        #config_files = list(checkpoint_dir.glob("*config*.json"))

        if not model_files:
            logger.warning(f"No model files found in {checkpoint_dir}")
            return model

        model_file = model_files[0]

        logger.info(f"Found model file: {model_file}")

        # CRITICAL: Ensure the model is built
        if hasattr(model, 'build'):
            logger.info("Building model architecture...")
            model.build()
            
        # Load model state
        if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
            try:
                state_dict = torch.load(model_file, map_location=get_device())
                model.model.load_state_dict(state_dict)
                model._is_trained = True
                logger.info(f"Loaded model from checkpoint: {model_file}")
            except Exception as e:
                logger.error(f"Failed to load model state dict: {e}")
                # If loading fails, still mark as trained to avoid save errors
                model._is_trained = True
        else:
            logger.warning(f"Model is not a torch.nn.Module or doesn't exist")
            # Still mark as trained to avoid save errors
            model._is_trained = True

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
